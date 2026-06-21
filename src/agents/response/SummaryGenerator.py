"""
SummaryGenerator — Map-reduce full-lecture summarization (Phase 2 / Item 1).

Why map-reduce?
---------------
A typical lecture after chunking can be 50-300 chunks totalling 40k-200k
characters.  The LLM context window (capped by ``process_text`` at
``INPUT_DAFAULT_MAX_CHARACTERS``, default 16 000 chars) means we cannot
send everything at once.  A single similarity-search retrieval step picks
only 15 reranked chunks — missing large parts of the lecture.

Map-reduce summary:
    MAP   — for each ordered batch of chunks, extract detailed notes
             (using ``summary_batch_system_prompt``).
    REDUCE — merge all batch-notes into one structured final summary
             (using ``summarize_system_prompt``).

If the lecture is small enough to fit in one batch, the reduce step is
skipped and the batch result is polished directly with the full
``summarize_system_prompt``.

The generator is called from:
  • ``ResponseFormatterAgent.execute()``  — for agent-pipeline (chat) summary requests
  • ``NLPController.generate_summary``    — for the classic /v1/nlp/summarize endpoint
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Characters per batch fed to the LLM (map step).
# Chosen to leave headroom inside the 16 000-char process_text limit for
# the system prompt (~2 000 chars) and the generated batch-notes output.
_BATCH_CHARS = 10_000

# Characters per merged-notes batch sent to the reduce step.
# Batch-notes are denser than raw chunks so we use a slightly larger budget.
_REDUCE_BATCH_CHARS = 12_000

# Maximum output tokens for map (batch-notes) and reduce (final summary) calls.
_MAP_MAX_TOKENS = 1_500
_REDUCE_MAX_TOKENS = 4_000


class SummaryGenerator:
    """
    Runs a map-reduce summarization over a list of ordered ``DataChunk``
    objects and returns the final structured study summary as a string.

    Parameters
    ----------
    generation_client:
        The app-level LLM provider (``CoHereProvider`` or similar).
    template_parser:
        The app-level ``TemplateParser`` — used to look up batch/merge
        prompt templates from ``en/rag.py`` or ``ar/rag.py``.
    language:
        ``"ar"`` or ``"en"`` — controls which locale's prompts are used.
    """

    def __init__(self, generation_client, template_parser, language: str = "en") -> None:
        self._llm = generation_client
        self._tp = template_parser
        self._lang = language if language in ("ar", "en") else "en"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: list, asset_ids: Optional[List[str]] = None) -> str:
        """
        Synchronous entry point (called from ``ResponseFormatterAgent`` /
        ``NLPController`` which are both synchronous at the
        generation-call level).

        ``chunks`` — ordered ``DataChunk`` ORM objects with ``.chunk_text``
                     and ``.chunk_metadata`` attributes.

        Returns the complete, structured summary string.
        """
        if not chunks:
            return "No lecture content was found to summarise."

        # ── Build ordered text blocks ───────────────────────────────────
        ordered_blocks = self._chunks_to_blocks(chunks)
        total_chars = sum(len(b) for b in ordered_blocks)
        logger.info(
            "SummaryGenerator: %d chunks, %d total chars, lang=%s",
            len(chunks), total_chars, self._lang,
        )

        # ── MAP step ─────────────────────────────────────────────────────
        batches = self._split_into_batches(ordered_blocks, _BATCH_CHARS)
        logger.info("SummaryGenerator: %d map batches", len(batches))

        if len(batches) == 1:
            # Small lecture — skip reduce, go straight to full summary prompt.
            return self._final_summary(batches[0])

        batch_notes: List[str] = []
        for i, batch_text in enumerate(batches, start=1):
            logger.info("SummaryGenerator: map batch %d/%d", i, len(batches))
            note = self._map_batch(batch_text, batch_num=i, total_batches=len(batches))
            if note:
                batch_notes.append(note)

        if not batch_notes:
            return "Summary generation failed: no notes extracted from lecture batches."

        # ── REDUCE step ──────────────────────────────────────────────────
        # If even the combined notes overflow one reduce call, merge in stages.
        reduce_batches = self._split_into_batches(batch_notes, _REDUCE_BATCH_CHARS)
        logger.info("SummaryGenerator: %d reduce batches", len(reduce_batches))

        if len(reduce_batches) == 1:
            return self._final_summary(reduce_batches[0])

        # Multi-stage reduce: merge pairs of note-batches until one remains.
        while len(reduce_batches) > 1:
            merged: List[str] = []
            for j in range(0, len(reduce_batches), 2):
                if j + 1 < len(reduce_batches):
                    combined = reduce_batches[j] + "\n\n---\n\n" + reduce_batches[j + 1]
                else:
                    combined = reduce_batches[j]
                intermediate = self._intermediate_merge(combined)
                merged.append(intermediate)
            reduce_batches = self._split_into_batches(merged, _REDUCE_BATCH_CHARS)

        return self._final_summary(reduce_batches[0])

    async def generate_stream(self, chunks: list, asset_ids: Optional[List[str]] = None):
        """
        Async-generator entry point — streams the final summary token by
        token.  Map steps are run synchronously (blocking) before
        streaming begins; only the final reduce/polish step is streamed.
        """
        if not chunks:
            yield "No lecture content was found to summarise."
            return

        ordered_blocks = self._chunks_to_blocks(chunks)
        batches = self._split_into_batches(ordered_blocks, _BATCH_CHARS)
        logger.info("SummaryGenerator.stream: %d map batches", len(batches))

        if len(batches) == 1:
            async for token in self._final_summary_stream(batches[0]):
                yield token
            return

        # MAP (blocking — no streaming for intermediate steps)
        batch_notes: List[str] = []
        for i, batch_text in enumerate(batches, start=1):
            logger.info("SummaryGenerator.stream: map batch %d/%d", i, len(batches))
            note = self._map_batch(batch_text, batch_num=i, total_batches=len(batches))
            if note:
                batch_notes.append(note)

        if not batch_notes:
            yield "Summary generation failed: no notes extracted from lecture batches."
            return

        # REDUCE — stream the final pass
        reduce_batches = self._split_into_batches(batch_notes, _REDUCE_BATCH_CHARS)
        while len(reduce_batches) > 1:
            merged: List[str] = []
            for j in range(0, len(reduce_batches), 2):
                if j + 1 < len(reduce_batches):
                    combined = reduce_batches[j] + "\n\n---\n\n" + reduce_batches[j + 1]
                else:
                    combined = reduce_batches[j]
                intermediate = self._intermediate_merge(combined)
                merged.append(intermediate)
            reduce_batches = self._split_into_batches(merged, _REDUCE_BATCH_CHARS)

        async for token in self._final_summary_stream(reduce_batches[0]):
            yield token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunks_to_blocks(self, chunks: list) -> List[str]:
        """
        Convert ordered DataChunk ORM rows into labelled text blocks.

        Each block is prefixed with a short label derived from the chunk's
        metadata (source/page/section) so the LLM can use that structure
        when building the notes.
        """
        blocks: List[str] = []
        for chunk in chunks:
            text = (chunk.chunk_text or "").strip()
            if not text:
                continue

            meta = chunk.chunk_metadata or {}
            parts: List[str] = []
            if meta.get("source"):
                parts.append(f"Lecture: {meta['source']}")
            if meta.get("page"):
                parts.append(f"Page: {meta['page']}")
            if meta.get("section"):
                parts.append(f"Section: {meta['section']}")
            if meta.get("chunk_index"):
                parts.append(f"Chunk: {meta['chunk_index']}")

            label = " | ".join(parts)
            block = f"[{label}]\n{text}" if label else text
            blocks.append(block)

        return blocks

    def _split_into_batches(self, blocks: List[str], max_chars: int) -> List[str]:
        """
        Group *blocks* into batches where each batch is at most
        *max_chars* characters long.  A single block larger than
        *max_chars* goes into its own batch (it will be truncated by
        ``process_text`` inside the LLM provider).
        """
        batches: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for block in blocks:
            block_len = len(block)
            separator_len = 2 if current_parts else 0  # "\n\n"

            if current_parts and current_len + separator_len + block_len > max_chars:
                batches.append("\n\n".join(current_parts))
                current_parts = [block]
                current_len = block_len
            else:
                current_parts.append(block)
                current_len += separator_len + block_len

        if current_parts:
            batches.append("\n\n".join(current_parts))

        return batches

    def _get_prompt(self, key: str, vars: dict = None) -> str:
        """Fetch a prompt template from the template parser."""
        try:
            result = self._tp.get("rag", key, vars or {})
            return result or ""
        except Exception as exc:
            logger.warning("SummaryGenerator: template '%s' not found: %s", key, exc)
            return ""

    def _map_batch(self, batch_text: str, batch_num: int, total_batches: int) -> str:
        """
        Extract detailed notes from one batch (MAP step).
        Returns the raw notes string, or "" on failure.
        """
        system_prompt = self._get_prompt("summary_batch_system_prompt")
        footer = self._get_prompt("summary_batch_footer_prompt")

        if not system_prompt:
            # Fallback: treat the batch text as its own notes
            return batch_text

        prompt = (
            f"--- Lecture excerpt {batch_num} of {total_batches} ---\n\n"
            f"{batch_text}"
            + (f"\n\n{footer}" if footer else "")
        )

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        try:
            result = self._llm.generate_text(
                prompt=prompt,
                chat_history=chat_history,
                max_output_tokens=_MAP_MAX_TOKENS,
            )
            return (result or "").strip()
        except Exception as exc:
            logger.error("SummaryGenerator: map batch %d failed: %s", batch_num, exc)
            # Fall back to raw text so we don't lose lecture content.
            return batch_text

    def _intermediate_merge(self, notes_text: str) -> str:
        """
        Lightly merge two sets of batch-notes into a single, still-detailed
        intermediate note set (used when the reduce step itself needs more
        than one pass).
        """
        system_prompt = self._get_prompt("summary_batch_system_prompt")
        prompt = self._get_prompt("summary_intermediate_merge_prompt", {"notes_text": notes_text})
        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt or "You are a careful note-merger.",
                role=self._llm.enums.SYSTEM.value,
            )
        ]
        try:
            result = self._llm.generate_text(
                prompt=prompt,
                chat_history=chat_history,
                max_output_tokens=_MAP_MAX_TOKENS,
            )
            return (result or notes_text).strip()
        except Exception as exc:
            logger.error("SummaryGenerator: intermediate merge failed: %s", exc)
            return notes_text

    def _final_summary(self, notes_text: str) -> str:
        """
        Polish the accumulated notes into the structured final summary
        (REDUCE / polish step — synchronous).
        """
        system_prompt = self._get_prompt("summarize_system_prompt")
        footer = self._get_prompt("summarize_footer_prompt")

        if not system_prompt:
            return notes_text  # Safety fallback

        prompt = notes_text + (f"\n\n{footer}" if footer else "")
        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        try:
            result = self._llm.generate_text(
                prompt=prompt,
                chat_history=chat_history,
                max_output_tokens=_REDUCE_MAX_TOKENS,
            )
            return (result or notes_text).strip()
        except Exception as exc:
            logger.error("SummaryGenerator: final summary failed: %s", exc)
            return notes_text

    async def _final_summary_stream(self, notes_text: str):
        """
        Streaming version of the final polish/reduce step.
        Yields tokens as they arrive from the LLM.
        """
        system_prompt = self._get_prompt("summarize_system_prompt")
        footer = self._get_prompt("summarize_footer_prompt")

        if not system_prompt:
            yield notes_text
            return

        prompt = notes_text + (f"\n\n{footer}" if footer else "")
        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        try:
            async for token in self._llm.generate_stream(
                prompt=prompt,
                chat_history=chat_history,
                max_output_tokens=_REDUCE_MAX_TOKENS,
            ):
                yield token
        except Exception as exc:
            logger.error("SummaryGenerator: stream failed: %s", exc)
            yield notes_text