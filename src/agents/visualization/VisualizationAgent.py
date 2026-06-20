"""
VisualizationAgent — Napkin AI visualization integration.

Reads ``state["final_response"]`` (the already-generated text answer),
sends it to the Napkin AI API to generate an illustrative diagram,
and writes the resulting image URLs into ``state["visualization_urls"]``.

This agent is stateless — it is called directly from the
``multimodal_query/stream`` route handler when the user requests
visualizations (``visualize=True``), not wired into the LangGraph pipeline.

If Napkin AI is not configured (no ``NAPKIN_API_KEY``) or the API call
fails for any reason, the agent silently returns the state unchanged so
the main answer is never lost.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VisualizationAgent:
    """
    Calls the Napkin AI API to generate diagrams from the final response text.

    Usage::

        agent = VisualizationAgent()
        state = await agent.execute(state)
        # state["visualization_urls"] is now populated (may be empty on failure)
    """

    _NAPKIN_CREATE_URL = "https://api.napkin.ai/v1/visual"
    _POLL_INTERVAL_S = 3
    _MAX_POLLS = 20

    def __init__(self) -> None:
        # Settings are loaded lazily so import errors don't break startup.
        self._api_key: Optional[str] = None
        self._loaded = False

    def _load_settings(self) -> None:
        if self._loaded:
            return
        try:
            from helpers.config import get_settings
            settings = get_settings()
            self._api_key = getattr(settings, "NAPKIN_API_KEY", None)
        except Exception as exc:
            logger.warning("VisualizationAgent: could not load settings: %s", exc)
        self._loaded = True

    async def execute(self, state: dict) -> dict:
        """
        Generate visualizations from ``state["final_response"]`` and append
        the image URLs to ``state["visualization_urls"]``.

        Returns the state dict unchanged if visualization is not configured
        or fails.
        """
        self._load_settings()

        if not self._api_key:
            logger.info(
                "VisualizationAgent: NAPKIN_API_KEY not set — skipping visualization."
            )
            return state

        text = (state.get("final_response") or "").strip()
        if not text:
            return state

        try:
            import asyncio
            import httpx

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 1: Create visualization request
                create_res = await client.post(
                    self._NAPKIN_CREATE_URL,
                    headers=headers,
                    json={"text": text},
                )

                if create_res.status_code not in (200, 201, 202):
                    logger.warning(
                        "VisualizationAgent: Napkin API create failed: %s %s",
                        create_res.status_code,
                        create_res.text[:200],
                    )
                    return state

                req_id = create_res.json().get("id")
                if not req_id:
                    logger.warning(
                        "VisualizationAgent: Napkin API returned no request ID."
                    )
                    return state

                # Step 2: Poll for completion
                status_state = "pending"
                status_data: dict = {}

                for _ in range(self._MAX_POLLS):
                    await asyncio.sleep(self._POLL_INTERVAL_S)
                    status_res = await client.get(
                        f"https://api.napkin.ai/v1/visual/{req_id}/status",
                        headers=headers,
                    )
                    status_data = status_res.json()
                    status_state = status_data.get("status", "pending")
                    if status_state != "pending":
                        break

                if status_state != "completed":
                    logger.warning(
                        "VisualizationAgent: Napkin did not complete in time "
                        "(status=%s).",
                        status_state,
                    )
                    return state

                # Step 3: Download generated images
                generated_files = status_data.get("generated_files", [])
                vis_urls: list = []

                for f in generated_files:
                    file_url = f.get("url")
                    if file_url:
                        vis_urls.append(file_url)

                if vis_urls:
                    existing = state.get("visualization_urls") or []
                    state["visualization_urls"] = list(existing) + vis_urls
                    logger.info(
                        "VisualizationAgent: added %d visualization URL(s).",
                        len(vis_urls),
                    )

        except Exception as exc:
            logger.error("VisualizationAgent.execute failed: %s", exc)

        return state