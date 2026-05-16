"""
Episode trace recorder — agent-side counterpart to mock_ivr.py's trace.

The mock IVR records what the IVR side did (DTMF received, prompts spoken).
This module records what the agent side did (STT transcripts, LLM responses,
tool calls, generated TTS, state changes).

After both sides finish, `score_episodes.py` (2.1d) reads both traces and
correlates them by job_id / episode_id / wall-clock timestamp.

Output file shape:
    voice-agent/episodes/agent_<job_id>.json

Schema:
    {
      "job_id": "AJ_xxx",
      "room": "mock-e2e-...",
      "agent_id": "rumacare",
      "started_at": <unix ts>,
      "ended_at": <unix ts>,
      "events": [
        {"ts": <seconds since start>, "type": "...", "payload": {...}},
        ...
      ]
    }

Hook into an AgentSession by calling:
    recorder = EpisodeRecorder(job_id=ctx.job.id, room=ctx.room.name)
    recorder.attach(session)
    # ... session runs ...
    recorder.dump(EPISODES_DIR)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from livekit.agents.voice import AgentSession
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

logger = logging.getLogger("rumacare.trace")


def _safe_dump(obj: Any) -> Any:
    """Best-effort JSON serialization. Pydantic models → dict; everything else → str."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json", exclude={"speech_handle"})
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    return repr(obj)


class EpisodeRecorder:
    """Subscribes to AgentSession events and writes a structured trace to disk."""

    def __init__(
        self,
        job_id: str,
        room: str,
        agent_name: str = "rumacare",
        model: str = "unknown",
    ) -> None:
        self.job_id = job_id
        self.room = room
        self.agent_name = agent_name
        # Stored in the trace so score_episodes.py --group-by model works.
        self.model = model
        self.started_at = time.time()
        self.events: list[dict] = []
        self._session: AgentSession | None = None
        # Track DTMF sends from the agent side too — episode_trace.py is the
        # right place to collect them since the JobContext has the room handle.
        self._dtmf_callback = self._make_dtmf_logger()

    def _log_event(self, type_: str, payload: dict) -> None:
        self.events.append(
            {
                "ts": round(time.time() - self.started_at, 3),
                "type": type_,
                "payload": payload,
            }
        )

    def attach(self, session: AgentSession) -> None:
        """Wire all the AgentSession event listeners. Call this BEFORE
        session.start() so we don't miss the initial state transitions."""
        self._session = session

        @session.on("user_input_transcribed")
        def _on_user_transcribed(ev: UserInputTranscribedEvent) -> None:
            # IVR or human voice transcribed by the agent's STT. We only log
            # finals to keep traces clean — partials are noise for evaluation.
            if not ev.is_final:
                return
            self._log_event(
                "agent_heard",
                {"transcript": ev.transcript, "language": ev.language},
            )

        @session.on("conversation_item_added")
        def _on_conv_item(ev: ConversationItemAddedEvent) -> None:
            item = ev.item
            payload: dict = {"item_type": type(item).__name__}
            for attr in ("role", "content", "text_content", "interrupted"):
                if hasattr(item, attr):
                    payload[attr] = _safe_dump(getattr(item, attr))
            self._log_event("conversation_item", payload)

        @session.on("function_tools_executed")
        def _on_tools_executed(ev: FunctionToolsExecutedEvent) -> None:
            calls = []
            for fc, fco in zip(ev.function_calls, ev.function_call_outputs):
                call_payload = {
                    "name": getattr(fc, "name", None),
                    "call_id": getattr(fc, "call_id", None),
                    "arguments": _safe_dump(getattr(fc, "arguments", None)),
                }
                if fco is not None:
                    call_payload["output"] = _safe_dump(
                        getattr(fco, "output", None)
                    )
                    call_payload["is_error"] = getattr(fco, "is_error", False)
                calls.append(call_payload)
            self._log_event("tools_executed", {"calls": calls})

        @session.on("speech_created")
        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            self._log_event(
                "agent_speech_created",
                {"source": ev.source, "user_initiated": ev.user_initiated},
            )

        @session.on("agent_state_changed")
        def _on_agent_state(ev: AgentStateChangedEvent) -> None:
            self._log_event(
                "agent_state",
                {"old": ev.old_state, "new": ev.new_state},
            )

        @session.on("user_state_changed")
        def _on_user_state(ev: UserStateChangedEvent) -> None:
            self._log_event(
                "user_state",
                {"old": ev.old_state, "new": ev.new_state},
            )

        @session.on("error")
        def _on_error(ev: ErrorEvent) -> None:
            self._log_event(
                "session_error",
                {"source": str(getattr(ev, "source", None)), "error": repr(ev.error)},
            )

        @session.on("close")
        def _on_close(ev: CloseEvent) -> None:
            self._log_event(
                "session_closed",
                {"reason": str(ev.reason), "error": repr(ev.error) if ev.error else None},
            )

    def _make_dtmf_logger(self):
        """Returns a callback the agent.py send_dtmf tool can call to log
        each DTMF transmit attempt with its arguments. We don't want to
        intercept LiveKit internals; the tool itself should call
        recorder.log_dtmf_sent(digits, success, error) explicitly."""
        def cb(digits: str, success: bool, error: str | None = None) -> None:
            self._log_event(
                "dtmf_sent",
                {"digits": digits, "success": success, "error": error},
            )
        return cb

    def log_dtmf_sent(self, digits: str, success: bool, error: str | None = None) -> None:
        """Public hook for tools (like send_dtmf in agent.py) to log their actions."""
        self._dtmf_callback(digits, success, error)

    def log_custom(self, type_: str, payload: dict) -> None:
        """Escape hatch for ad-hoc events that aren't standard AgentSession events."""
        self._log_event(type_, payload)

    def dump(self, episodes_dir: Path) -> Path:
        episodes_dir.mkdir(parents=True, exist_ok=True)
        out = episodes_dir / f"agent_{self.job_id}.json"
        trace = {
            "job_id": self.job_id,
            "room": self.room,
            "agent_name": self.agent_name,
            "model": self.model,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "events": self.events,
        }
        out.write_text(json.dumps(trace, indent=2, default=str))
        logger.info("Agent episode trace written to %s (%d events)", out, len(self.events))
        return out
