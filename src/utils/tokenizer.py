"""
Lightweight token-counting helper.

The chunking pipeline (``ProcessController``) needs to size chunks in
*tokens* (roughly 700-1000 tokens per chunk with 100-150 tokens of
overlap) instead of raw characters, because chunk quality for retrieval
and summarization depends on how much context actually fits inside the
LLM's context window.

``tiktoken`` gives an accurate, fast token count for OpenAI/Cohere-style
BPE tokenizers, but its encoding tables are downloaded from the network
on first use. In environments without access to that endpoint we fall
back to a simple, well-tested heuristic (~4 characters per token) so the
pipeline never fails just because token counting isn't available.
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_encoder = None
_encoder_load_attempted = False

# Average characters-per-token for mixed English/Arabic technical text.
# Used as a fallback when tiktoken is unavailable.
_CHARS_PER_TOKEN = 4.0


def _get_encoder():
    """Lazily load a tiktoken encoder, caching success *and* failure."""
    global _encoder, _encoder_load_attempted

    if _encoder_load_attempted:
        return _encoder

    with _lock:
        if _encoder_load_attempted:
            return _encoder

        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning(
                "tiktoken encoder unavailable (%s) — falling back to a "
                "character-based token estimate.",
                exc,
            )
            _encoder = None
        finally:
            _encoder_load_attempted = True

    return _encoder


def count_tokens(text: str) -> int:
    """
    Return an approximate token count for *text*.

    Uses ``tiktoken`` (``cl100k_base``) when available; otherwise falls
    back to a character/word based heuristic that approximates BPE
    tokenization closely enough for chunk-sizing decisions.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover - defensive
            pass

    # Fallback heuristic: blend a character-based and word-based estimate.
    char_estimate = len(text) / _CHARS_PER_TOKEN
    word_estimate = len(re.findall(r"\S+", text)) * 1.3
    return max(1, int((char_estimate + word_estimate) / 2))


def tokens_to_chars(num_tokens: int) -> int:
    """Rough conversion used when an API needs a character budget."""
    return int(num_tokens * _CHARS_PER_TOKEN)
