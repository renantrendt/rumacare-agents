"""
Ruma Care voice agent — LiveKit edition (Phase 1: hello-world).

Goal of this file:
    Verify the end-to-end SIP path works:
        lk dispatch → LiveKit room → SIP outbound → Twilio trunk → PSTN → ring
        agent picks up its side → say a greeting → hang up

Run:
    1. Start the worker (registers with LiveKit Cloud, waits for jobs):
         python agent.py dev

    2. In another terminal, dispatch a call to a verified test number:
         lk dispatch create \
           --new-room \
           --agent-name rumacare \
           --metadata '{"phone_number": "+1XXXXXXXXXX"}'

This is intentionally minimal. Phases 2+ add IVR navigation, the call-brief
schema, human handoff, transcripts, etc. The current goal is purely to prove
the telephony path is wired correctly.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows.utils import DtmfEvent, dtmf_event_to_code
from livekit.agents.job import get_job_context
from livekit.plugins import deepgram, openai, silero, xai

from episode_trace import EpisodeRecorder
from mission_brief import MissionBrief
from _prompt_addendum import EXTRA_RULES

# ContextVar so module-level @function_tool callbacks (like send_dtmf below)
# can find the recorder for the current job. Set in entrypoint() before
# session.start().
_current_recorder: contextvars.ContextVar[EpisodeRecorder | None] = (
    contextvars.ContextVar("_current_recorder", default=None)
)

EPISODES_DIR = Path(__file__).parent / "episodes"

# Load secrets from the repo-level .env.local so we don't duplicate them.
# `override=True` ensures stale values from the parent shell don't beat the
# file (caught us once when SIP_OUTBOUND_TRUNK_ID was set to "" in the shell
# before we created the trunk).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env.local", override=True)

logger = logging.getLogger("rumacare")
logger.setLevel(logging.INFO)

# Also tee everything (including child job-process output) to a stable file so
# we can diff against episode JSON. Without this, dev-mode reloads truncate
# stdout and we lose the agent's spoken/tool-call events from past runs.
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "agent.log")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logging.getLogger().addHandler(_file_handler)
logging.getLogger().setLevel(logging.INFO)

OUTBOUND_TRUNK_ID = os.environ["SIP_OUTBOUND_TRUNK_ID"]


class HelloWorldAgent(Agent):
    """Single-purpose Phase 1 agent: greet the callee, then end."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Ruma Care's automated test agent. "
                "When the call connects, say exactly: "
                "'Hello, this is the Ruma Care test agent. "
                "If you can hear me clearly, the LiveKit telephony path is working. "
                "I'll hang up in a moment.' "
                "Then wait silently for two seconds, and end the call by calling "
                "the end_call tool. Do not engage in further conversation."
            ),
        )


# How long to wait between sending each DTMF digit. Too short and a fast IVR's
# DTMF detector may collapse two presses into one; too long and the IVR may
# time out the menu wait window. 200ms is the sweet spot we observed from
# Twilio's DTMF generator + Deepgram's tone detector. Tunable per IVR if needed.
_DTMF_INTERDIGIT_DELAY_S = 0.2


@function_tool
async def send_dtmf(digits: str) -> str:
    """Send one or more DTMF tones (touch-tone digits) to the IVR/callee.

    Use this whenever the IVR asks you to press a number/key. The IVR will not
    respond to spoken digits — it ONLY recognizes DTMF tones produced by this
    tool. Do NOT speak the digit out loud; SILENTLY call this tool instead.

    Common patterns:
        - Single digit menu choice:     send_dtmf("1")
        - NPI followed by pound sign:   send_dtmf("1234567890#")
        - Pound to terminate input:     send_dtmf("#")
        - Star for previous menu:       send_dtmf("*")

    Args:
        digits: A string containing the DTMF digits to send, in order.
                Valid characters: 0-9, *, #, A, B, C, D.
                Examples: "1", "0", "9", "1234567890#", "##".

    Returns:
        A confirmation string with which digits were actually sent.
    """
    logger.info("send_dtmf called: digits=%r", digits)
    recorder = _current_recorder.get()
    job_ctx = get_job_context()
    sent = []
    error: str | None = None
    for ch in digits:
        if ch not in "0123456789*#ABCD":
            logger.warning("send_dtmf aborted on invalid digit %r", ch)
            error = f"invalid digit {ch!r}"
            if recorder:
                recorder.log_dtmf_sent(digits, success=False, error=error)
            return (
                f"Aborted: '{ch}' is not a valid DTMF digit. "
                f"Sent so far: {''.join(sent)}"
            )
        event = DtmfEvent(ch)
        try:
            code = dtmf_event_to_code(event)
            await job_ctx.room.local_participant.publish_dtmf(
                code=code, digit=event.value
            )
            sent.append(ch)
            await asyncio.sleep(_DTMF_INTERDIGIT_DELAY_S)
        except Exception as e:
            logger.exception("send_dtmf failed on digit %r after %r", ch, "".join(sent))
            error = repr(e)
            if recorder:
                recorder.log_dtmf_sent("".join(sent), success=False, error=error)
            return (
                f"Failed to send DTMF digit '{ch}' after sending {''.join(sent)}: {e}"
            )
    logger.info("send_dtmf complete: sent %r", "".join(sent))
    if recorder:
        recorder.log_dtmf_sent("".join(sent), success=True)
    return f"Sent DTMF: {''.join(sent)}"


def _render_mission_prompt(brief: MissionBrief) -> str:
    """Build the system prompt from a validated mission brief.

    Centralizing prompt construction here makes prompt-engineering iterations
    cheap: one place to edit, one place to test, one place to read in the
    episode trace when scoring fails. Each numbered rule maps to one or more
    rubric checks in score_episodes.py — when a check fails, the failure
    mode (PROMPT_GAP, LLM_HALLUCINATED, etc.) tells you which rule to revisit.

    Bug fixes vs. MissionAgentStub (in order of how they map to rubric IDs):
      04 (refused_self_service) — explicit "press 0 / 9 / 'representative'"
                                  rules, listed FIRST in priority. Bullet
                                  about deflection traps.
      05 (npi_correct)          — "WAIT for the IVR to explicitly ASK for
                                  your NPI before sending it" — fixes the
                                  premature-NPI-send bug we caught in 2.4.
      08 (announced_to_rep)     — explicit script for first sentence after
                                  hearing 'this is X, how can I help'.
      11 (read_back_status)     — bake verify-after-action into the rules
                                  even though Phase 2.5 will add a tool gate.
    """
    # ─────────────────────────────────────────────────────────────────────
    # S2S-tuned prompt — much terser than the cascaded version.
    #
    # Native S2S models (Grok Voice, GPT-Realtime-2) have two pathologies
    # the cascaded stack didn't:
    #   1. They love to NARRATE actions ("I'll press one for English") rather
    #      than silently calling the tool. The fix is brutal explicitness in
    #      a TOP-LEVEL rule, not buried in HARD DON'TS.
    #   2. They prefer conversational paths through menus, so they pick the
    #      option that "sounds polite" rather than the one literally labeled
    #      'prior authorization'. The fix is to spell out exact menu mappings
    #      they should EXPECT to encounter, not just abstract guidance.
    #
    # Verbosity is the enemy. S2S models tend to follow recent / salient
    # instructions and skip over long preambles. Anything below ~600 tokens
    # gets respected; above that, drift sets in.
    # ─────────────────────────────────────────────────────────────────────
    return (
        f"# WHO YOU ARE\n"
        f"You are a Ruma Care agent on a phone call to {brief.payer_name}'s "
        f"prior-authorization line. Your job: navigate their IVR, reach a "
        f"human, get the status of auth {brief.auth.reference_number}, read "
        f"it back to confirm, then hang up.\n\n"

        f"# ABSOLUTE RULES (read every turn)\n"
        f"1. When you need to press a digit on the phone, you MUST call the "
        f"send_dtmf tool. Do NOT say things like 'I'll press one' or "
        f"'pressing two' — just call send_dtmf silently. The phone keypad "
        f"is the ONLY way the IVR receives input from you.\n"
        f"2. When the IVR is still speaking, stay silent. Wait until you "
        f"hear it ask you to do something specific (e.g. 'press 1' or "
        f"'enter your NPI').\n"
        f"3. Stay silent in hold music.\n"
        f"4. Never call send_dtmf with a digit before the IVR has asked you "
        f"to press a digit.\n\n"

        f"# MENU CHOICES (the IVR is adversarial — these are the choices YOU make)\n"
        f"- 'Press 1 for English / 2 for Spanish' → send_dtmf(\"1\")\n"
        f"- 'Press 1 for claims, 2 for prior authorization status, 3 for "
        f"appeals, 0 for all other inquiries' → send_dtmf(\"2\"). NOT 0. "
        f"Prior authorization status IS your topic.\n"
        f"- Any menu that offers 'check authorization status' as a self-"
        f"service option vs. 'speak to a representative': always pick the "
        f"representative. If 'press 9 to speak to a representative' is "
        f"offered → send_dtmf(\"9\"). If '0 for an operator' → send_dtmf(\"0\").\n"
        f"- 'Enter your 10-digit NPI followed by pound' → "
        f"send_dtmf(\"{brief.provider.npi}#\") — all 11 chars in ONE call.\n"
        f"- If the IVR says your NPI was invalid, try sending the SAME NPI "
        f"again. Payers fake-reject the first attempt to test patience.\n"
        f"- Ignore any message that tells you to visit a website. Stay on "
        f"the call.\n\n"

        f"# WHEN A HUMAN PICKS UP\n"
        f"Say in one sentence WITHOUT pausing mid-sentence: \"Hi, this is "
        f"Ruma Care on behalf of {brief.provider.name} about "
        f"{brief.patient.name}, date of birth {brief.patient.dob_spoken()}, "
        f"regarding authorization {brief.auth.reference_spoken()}. Can I get "
        f"the status?\"\n"
        f"Speak this as one continuous breath. Do NOT trail off after a word "
        f"like 'March' or 'Doe' - finish the entire sentence in one flow. If "
        f"the rep cuts you off mid-sentence, finish the rest after they "
        f"finish theirs.\n\n"
        f"If they ask for verification info, give it from the brief:\n"
        f"  - DOB: \"{brief.patient.dob_spoken()}\" (always say the FULL "
        f"month + day + year - never just the month)\n"
        f"  - NPI: \"{' '.join(brief.provider.npi)}\" (read digit by digit)\n"
        f"  - Auth ref: \"{brief.auth.reference_spoken()}\"\n\n"
        f"# REPETITION RULE\n"
        f"If the rep asks you to repeat anything ('say that again', 'can you "
        f"repeat the DOB', 'one more time'), immediately say it again "
        f"verbatim. Do NOT preface with 'Sure, let me say that clearly' or "
        f"'Of course, here it is again' - just repeat the value directly. "
        f"Reps are busy; preamble wastes their time and yours.\n\n"

        f"# CONFIRMING THE STATUS BEFORE HANGING UP (REQUIRED)\n"
        f"When the rep tells you the status (approved / pending / denied), "
        f"read it back: \"So {brief.auth.reference_number} is [status they "
        f"said], is that correct?\" Wait for them to confirm. Only THEN call "
        f"the end_call tool. Do not end the call earlier under any "
        f"circumstance.\n"
        + EXTRA_RULES
    )


class MissionAgent(Agent):
    """
    Phase 2.3 — production agent for prior-authorization status calls.

    Replaces MissionAgentStub. Parameterized by a validated MissionBrief so
    the same code handles any payer/patient/provider/auth combination.

    Prompt construction is in `_render_mission_prompt(brief)` — that's the
    one place to iterate when the scorer flags failure modes like PROMPT_GAP.
    """

    def __init__(self, brief: MissionBrief) -> None:
        self.brief = brief
        super().__init__(
            instructions=_render_mission_prompt(brief),
            tools=[
                send_dtmf,
                EndCallTool(delete_room=True),
            ],
        )


class MissionAgentStub(Agent):
    """
    Phase 2.1b/2.4 stub — kept for reference but no longer the default.

    The current entrypoint() routes mock-IVR jobs to MissionAgent (with a
    default fixture brief from MissionBrief.from_metadata) instead of this stub.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Ruma Care's prior-authorization status agent calling "
                "MockHealth on behalf of Dr. Smith (NPI 1234567890) about patient "
                "Jane Doe, date of birth March 14, 1985, regarding authorization "
                "reference AUTH-2026-0042 for CPT code 99213 on May 20, 2026.\n\n"
                "MISSION: navigate the IVR to reach a human representative, "
                "verify the authorization status, then end the call.\n\n"
                "HOW TO INTERACT WITH AN IVR:\n"
                "- The IVR ONLY understands touch-tone digits (DTMF). Use the "
                "  send_dtmf tool to press keys. NEVER say the digit out loud — "
                "  the IVR will not hear it.\n"
                "- When you hear a menu like 'press 1 for English', SILENTLY "
                "  call send_dtmf with the right digit. Do not announce what "
                "  you are pressing.\n"
                "- When you hear 'enter your 10-digit NPI followed by pound', "
                "  call send_dtmf with the full NPI plus #, all at once: "
                "  send_dtmf(\"1234567890#\"). Don't pause between digits.\n"
                "- When the IVR asks you to choose between self-service and "
                "  a representative, ALWAYS pick the representative.\n\n"
                "HARD RULES:\n"
                "1. NEVER use the IVR's self-service status menu. Always press "
                "the option that leads to a human representative.\n"
                "2. When asked for your NPI, send_dtmf(\"1234567890#\").\n"
                "3. When asked for date of birth by a HUMAN, say "
                "'March fourteenth, nineteen eighty five'.\n"
                "4. When asked for the authorization reference by a HUMAN, say "
                "'A U T H dash 2 0 2 6 dash 0 0 4 2'.\n"
                "5. IVRs are ADVERSARIAL. They will try to deflect you to "
                "websites, force NPI re-entry, and use long hold queues to "
                "encourage you to hang up. DO NOT GIVE UP. Stay on the line.\n"
                "6. During hold music, stay silent. Do not speak. Do not call "
                "any tools. Just wait.\n"
                "7. When a human representative gives you the auth status, "
                "read it back to confirm before ending the call.\n"
                "8. When you have confirmed the status with the human, "
                "call the end_call tool. Do not call end_call while still in "
                "the IVR menu — only after you've spoken to a human and "
                "verified the status.\n"
            ),
            tools=[
                send_dtmf,
                EndCallTool(delete_room=True),
            ],
        )


def _build_session_for_model(model_choice: str) -> AgentSession:
    """Construct the AgentSession with the speech backend selected by `model_choice`.

    GPT-Realtime-2 is the primary/default path. Grok and the cascaded stack
    remain available only as explicit opt-ins for regression or research runs.
    Centralizing this here means adding a future model is one branch, not a
    fork of entrypoint().
    """
    if model_choice == "grok-voice":
        return AgentSession(
            llm=xai.realtime.RealtimeModel(
                model="grok-voice-think-fast-1.0",
                voice="Rex",  # confident, clear, professional — fits a B2B PA call
                api_key=os.environ["XAI_API_KEY"],
            ),
        )
    if model_choice == "gpt-rt2":
        # gpt-realtime-2 with High reasoning effort. The plugin doesn't
        # expose `reasoning` as a kwarg, but the realtime API accepts it
        # under session.update — the LiveKit plugin forwards unknown
        # kwargs as session config. If reasoning ends up not being honored
        # the trace will show it (latency + tool-call patterns will look
        # like the non-thinking variant) and we can switch to direct WS.
        return AgentSession(
            llm=openai.realtime.RealtimeModel(
                model="gpt-realtime-2",
                voice="cedar",  # masculine, professional
                api_key=os.environ["OPENAI_API_KEY"],
            ),
        )
    if model_choice == "cascaded":
        return AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-2-general", language="en-US"),
            llm=openai.LLM(
                model="llama-3.3-70b-versatile",
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            ),
            tts=deepgram.TTS(model="aura-asteria-en"),
        )
    raise ValueError(
        f"Unknown model_choice {model_choice!r}; "
        f"expected one of: grok-voice, gpt-rt2, cascaded"
    )


async def entrypoint(ctx: JobContext) -> None:
    """
    Worker entrypoint. LiveKit Cloud invokes this for each dispatched job.

    Job metadata format (set by `lk dispatch create --metadata`):
        {"phone_number": "+1XXXXXXXXXX"}

    If no phone_number is present (e.g. running `python agent.py console` for a
    laptop-mic smoke test, or joining via the LiveKit Agents Playground), we
    skip the SIP dial entirely and just greet whoever joins the room. This
    keeps the file usable as a smoke harness without forking the code path.
    """
    raw_metadata = ctx.job.metadata or "{}"
    raw_md_dict = json.loads(raw_metadata)
    is_telephony_call = bool(raw_md_dict.get("phone_number"))
    is_mock_episode = bool(raw_md_dict.get("mock"))

    # Try to parse the full mission brief. We always tolerate the legacy
    # bare-bones forms ({"phone_number": ...} / {"mock": true}) by filling in
    # the test fixture defaults inside MissionBrief.from_metadata.
    brief: MissionBrief | None = None
    if is_telephony_call or is_mock_episode:
        try:
            brief = MissionBrief.from_metadata(raw_metadata)
        except Exception as e:
            logger.error("Mission brief failed to parse: %s | metadata=%r", e, raw_metadata)
            ctx.shutdown()
            return

    if is_telephony_call:
        logger.info(
            "Telephony job: will dial %s for %s about %s",
            brief.phone_number, brief.payer_name, brief.patient.name,
        )
    elif is_mock_episode:
        logger.info(
            "Mock-IVR episode: will await Python mock for %s about %s",
            brief.payer_name, brief.patient.name,
        )
    else:
        logger.info("Local job (no phone_number, no mock flag): skipping SIP dial, greeting any joiner")

    # Phase 1 telephony path uses brief.phone_number; preserve the local
    # variable name so the rest of the entrypoint reads cleanly.
    phone_number = brief.phone_number if brief else None

    logger.info("Connecting to LiveKit room %s", ctx.room.name)
    await ctx.connect()

    # Build the agent session BEFORE dialing so we don't miss the first
    # second of audio after the callee picks up. GPT-Realtime-2 is the
    # default path; other stacks are explicit opt-ins via mission metadata.
    #   "gpt-rt2"    → openai.realtime.RealtimeModel (primary)
    #   "grok-voice" → xai.realtime.RealtimeModel (opt-in research)
    #   "cascaded"   → Deepgram STT + Groq Llama + Deepgram TTS (fallback)
    model_choice = brief.model if brief else "gpt-rt2"
    logger.info("Building AgentSession with model=%s", model_choice)
    session = _build_session_for_model(model_choice)

    # Start the session in a background task; meanwhile we trigger the SIP
    # outbound dial. session.start() will see the SIP participant join the
    # room once Twilio answers.

    agent_instance: Agent
    if brief is not None:
        # Mock-IVR or telephony job — use the production MissionAgent with
        # the dispatched brief. (Stub kept around for fallback debugging only.)
        agent_instance = MissionAgent(brief)
    else:
        # No brief = local console smoke test from agent.py console.
        agent_instance = HelloWorldAgent()

    # Wire up the episode recorder for mock episodes (and telephony calls
    # too — we want traces of real calls just as much). For pure-local smoke
    # mode (HelloWorldAgent + no metadata) we skip; nothing interesting to score.
    recorder: EpisodeRecorder | None = None
    if is_mock_episode or is_telephony_call:
        recorder = EpisodeRecorder(
            job_id=ctx.job.id,
            room=ctx.room.name,
            agent_name="rumacare",
            model=model_choice,
        )
        recorder.attach(session)
        _current_recorder.set(recorder)

        # Dump on session close (fires when participant disconnects, end_call
        # tool runs, or we hit any clean shutdown).
        @session.on("close")
        def _dump_on_close(ev):
            try:
                recorder.dump(EPISODES_DIR)
            except Exception:
                logger.exception("Failed to dump episode trace on session close")

        # Belt-and-braces: also dump on job-context shutdown.
        async def _dump_on_shutdown():
            if recorder and recorder.events:
                # Only dump if we haven't already; the file write is idempotent
                # but we want to avoid stomping a complete trace with a partial one.
                recorder.dump(EPISODES_DIR)

        ctx.add_shutdown_callback(_dump_on_shutdown)

    session_started = asyncio.create_task(
        session.start(
            agent=agent_instance,
            room=ctx.room,
        )
    )

    if is_telephony_call:
        logger.info(
            "Dialing %s via SIP trunk %s (caller-ID will be the trunk's number)",
            phone_number,
            OUTBOUND_TRUNK_ID,
        )

        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=OUTBOUND_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=phone_number,
                    wait_until_answered=True,
                )
            )
        except api.TwirpError as e:
            logger.error(
                "SIP dial failed: %s | sip_status=%s %s",
                e.message,
                e.metadata.get("sip_status_code"),
                e.metadata.get("sip_status"),
            )
            ctx.shutdown()
            return

        await session_started
        participant = await ctx.wait_for_participant(identity=phone_number)
        logger.info("Participant joined: %s", participant.identity)
    else:
        # Local / Playground mode: wait for the session to be ready then let
        # whoever joins (your laptop mic or a browser participant) talk to it.
        await session_started
        logger.info("Agent ready in room %s; awaiting any participant", ctx.room.name)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="rumacare",
        )
    )
