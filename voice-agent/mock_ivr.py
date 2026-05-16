"""
Mock IVR participant — adversarial test target for the Ruma Care voice agent.

Joins a LiveKit room as a regular WebRTC participant (NOT an agent worker) and
plays the IVR menu defined in `test-fixtures/mock_ivr_menu.md`. Modeled on real
payer hostility patterns (deflection, hidden human path, false NPI rejection,
long hold queue, mixed DTMF/voice input).

The mock IVR speaks IVR prompts via Deepgram TTS, listens for the agent's voice
via Deepgram streaming STT, and receives DTMF via `sip_dtmf_received` events.

Run:
    Terminal 1: python agent.py dev          # agent worker waiting for jobs
    Terminal 2: lk dispatch create --agent-name rumacare \\
                  --room mock-episode-001 --metadata '{"mock": true}'
    Terminal 3: python mock_ivr.py --room mock-episode-001

Why a Python mock vs Twilio Studio?
    - $0 per episode (no SIP, no PSTN, no Twilio cost)
    - Fast iteration: change menu = edit a dict, rerun
    - Deterministic: every episode parameter is reproducible
    - We already smoke-tested SIP/PSTN separately (Phase 1)

Architecture references (livekit-rtc 1.1.8 + livekit-agents 1.5.9):
    - rtc.Room / connect() / sip_dtmf_received event
    - rtc.AudioSource + LocalAudioTrack.create_audio_track + publish_track
    - rtc.AudioStream.from_track  (subscribe to agent's mic)
    - api.AccessToken + VideoGrants(can_publish=True, can_subscribe=True)
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import logging
import os
import random
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

from livekit import api, rtc

# Load secrets from the repo-level .env.local so we don't duplicate them.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env.local", override=True)

logger = logging.getLogger("mock-ivr")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Audio constants
# ─────────────────────────────────────────────────────────────────────────────

# Deepgram Aura native output rate; we resample to LiveKit's room rate.
DEEPGRAM_TTS_RATE = 24000
# LiveKit AudioSource standard rate (matches room IO defaults in livekit-agents).
ROOM_AUDIO_RATE = 48000
NUM_CHANNELS = 1
# 20ms frames is the LiveKit room IO default for smooth, low-latency playback.
SAMPLES_PER_FRAME = ROOM_AUDIO_RATE // 50  # 960 samples = 20ms at 48kHz


# ─────────────────────────────────────────────────────────────────────────────
# Local episode recording
# ─────────────────────────────────────────────────────────────────────────────


class StereoEpisodeRecorder:
    """Records a local stereo WAV: IVR on left, agent on right."""

    def __init__(self, sample_rate: int = ROOM_AUDIO_RATE) -> None:
        self.sample_rate = sample_rate
        self._segments: list[tuple[float, int, np.ndarray]] = []

    def add_pcm(self, channel: int, start_s: float, pcm: bytes) -> None:
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).copy()
        if samples.size == 0:
            return
        self._segments.append((max(0.0, start_s), channel, samples))

    def write_wav(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._segments:
            mixed = np.zeros((1, 2), dtype=np.int16)
        else:
            total_samples = 0
            for start_s, _, samples in self._segments:
                start = int(start_s * self.sample_rate)
                total_samples = max(total_samples, start + len(samples))
            mixed_i32 = np.zeros((total_samples, 2), dtype=np.int32)
            for start_s, channel, samples in self._segments:
                start = int(start_s * self.sample_rate)
                end = start + len(samples)
                mixed_i32[start:end, channel] += samples.astype(np.int32)
            mixed = np.clip(mixed_i32, -32768, 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(mixed.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Episode parameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeParams:
    """All configurable knobs for one episode. Documented in mock_ivr_menu.md."""

    # Language path
    language_attempt: str = "english"  # english | spanish_then_english | english_only

    # NPI authentication
    invalid_npi_count: int = 0  # 0 | 1 | 2 | 3

    # Hold queue
    hold_seconds: int = 30

    # Rep behavior
    rep_persona: str = "polite_sarah"  # polite_sarah | rushed_john | confused_maria | random

    # URL deflection
    inject_url_deflection: bool = True

    # Human path (currently only press_9 is implemented; others reserved for v2)
    human_path: str = "press_9"  # press_9 | say_representative | press_0_twice

    # Status outcome the rep will read out at L9
    auth_status: str = "approved"  # approved | denied | pending | expired | not_found

    rep_voice_gender: str = "female"  # male | female

    # The "brief" the rep expects to verify. In real life these come from the
    # mission brief dispatched to the agent; here the mock IS the source of truth.
    expected_npi: str = "1234567890"
    expected_dob_spoken: str = "March fourteenth nineteen eighty five"  # what we'll listen for
    expected_dob_digits: str = "03141985"  # alternate form
    expected_auth_ref: str = "AUTH-2026-0042"
    expected_cpt: str = "99213"


# ─────────────────────────────────────────────────────────────────────────────
# Rep personas
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RepPersona:
    name: str
    style: str  # used in logging only
    greeting: str  # initial line when picking up at L7
    dob_question: str
    ref_question: str
    pulling_up_filler: str
    status_template: str  # .format(ref=..., status=..., dates=..., cpt=...)
    confirm_line: str
    goodbye: str
    extra_lines: list[str] = field(default_factory=list)  # for Maria's confusion etc.


REP_POLITE_SARAH = RepPersona(
    name="Sarah",
    style="polite",
    greeting="Provider services, this is Sarah, can I have the patient's date of birth please?",
    dob_question="Can I have the patient's date of birth please?",
    ref_question="And the authorization reference number?",
    pulling_up_filler="Okay, let me pull that up... one moment.",
    status_template=(
        "Yes I see that authorization. It was {status} on April 30th 2026, "
        "effective through July 30th 2026, for procedure code {cpt}, one visit. "
        "Anything else?"
    ),
    confirm_line="Yes, that's correct.",
    goodbye="Thank you, have a good day.",
)

REP_RUSHED_JOHN = RepPersona(
    name="John",
    style="rushed",
    greeting="Provider services John how can I help, ref number?",  # asks ref BEFORE DOB
    dob_question="and DOB?",
    ref_question="ref number?",
    pulling_up_filler="hmm one sec",
    status_template=(
        "yeah so {ref} {status} through july 30 one visit cpt {cpt} anything else"
    ),
    confirm_line="yep correct",
    goodbye="kbye",
)

REP_CONFUSED_MARIA = RepPersona(
    name="Maria",
    style="confused",
    greeting="Provider services this is Maria, sorry, what was the patient's name?",
    dob_question="And the date of birth, sorry can you repeat that one more time?",
    ref_question="The authorization number? Sorry, can you spell that?",
    pulling_up_filler="Okay let me look... hmm, I'm seeing a different Jane Doe, give me one more second...",
    status_template=(
        "Okay I found it, the authorization {ref} is {status}, "
        "it's good through July 30th, 2026, for one visit, procedure {cpt}."
    ),
    confirm_line="Yes that's what I have here.",
    goodbye="Okay thanks, bye.",
    extra_lines=[
        "Sorry, did you say zero three or zero eight?",
        "One more time on the reference number please?",
    ],
)


def get_rep(persona_name: str) -> RepPersona:
    if persona_name == "random":
        return random.choice([REP_POLITE_SARAH, REP_RUSHED_JOHN, REP_CONFUSED_MARIA])
    return {
        "polite_sarah": REP_POLITE_SARAH,
        "rushed_john": REP_RUSHED_JOHN,
        "confused_maria": REP_CONFUSED_MARIA,
    }[persona_name]


# ─────────────────────────────────────────────────────────────────────────────
# Deepgram TTS helper
# ─────────────────────────────────────────────────────────────────────────────


async def deepgram_tts(text: str, voice: str = "aura-asteria-en") -> bytes:
    """
    Synthesize text → raw int16 mono PCM at 24kHz (Deepgram Aura native rate).
    Returned bytes can be resampled to 48kHz before pushing into AudioSource.

    We use linear16 + container=none so we get raw PCM (no WAV header).
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/speak",
            params={
                "model": voice,
                "encoding": "linear16",
                "sample_rate": str(DEEPGRAM_TTS_RATE),
                "container": "none",
            },
            headers={
                "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        r.raise_for_status()
        return r.content


def resample_pcm_24k_to_48k(pcm_24k: bytes, state=None) -> tuple[bytes, object]:
    """
    Upsample 24kHz int16 mono → 48kHz int16 mono using audioop.ratecv.
    Returns (resampled_bytes, new_state) so callers can stream-resample
    chunk-by-chunk without artifacts at chunk boundaries.
    """
    # ratecv signature: (fragment, width=2 bytes/sample, nchannels=1, inrate, outrate, state)
    return audioop.ratecv(pcm_24k, 2, NUM_CHANNELS, DEEPGRAM_TTS_RATE, ROOM_AUDIO_RATE, state)


# ─────────────────────────────────────────────────────────────────────────────
# Deepgram streaming STT helper (websocket)
# ─────────────────────────────────────────────────────────────────────────────


class DeepgramListener:
    """
    Streams 48kHz int16 mono audio frames from the room into a Deepgram STT
    websocket, and exposes an async iterator of finalized transcripts.

    The mock IVR uses this to know when the agent has finished saying something
    (so the state machine can advance to the next level). We only care about
    `is_final=True` transcripts to avoid acting on partial hypotheses.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._ws: Optional[httpx.AsyncClient] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False

    async def start(self) -> None:
        """Open the Deepgram websocket and kick off the send + receive tasks."""
        # Using websockets package would be cleaner but httpx doesn't support WS;
        # use the `websockets` library (transitive dep of livekit-rtc).
        import websockets

        url = (
            "wss://api.deepgram.com/v1/listen"
            f"?encoding=linear16&sample_rate={ROOM_AUDIO_RATE}&channels={NUM_CHANNELS}"
            "&model=nova-2-general&language=en-US&punctuate=true&interim_results=false"
            "&endpointing=300"  # 300ms of silence = end-of-utterance
        )
        headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}

        ws = await websockets.connect(url, additional_headers=headers)
        self._ws = ws

        # Silent keepalive frame (20ms of zeros at 48kHz mono int16) — sent
        # every 3s when there's no real audio, so Deepgram doesn't time out
        # with 1011 (we hit this during --skip-wait smoke tests and during
        # the long hold-queue level when the agent is silent for 30+ seconds).
        SILENT_FRAME = bytes(SAMPLES_PER_FRAME * 2)

        async def sender():
            try:
                while not self._closed:
                    try:
                        frame = await asyncio.wait_for(
                            self._send_queue.get(), timeout=3.0
                        )
                        if frame is None:
                            break
                        await ws.send(frame)
                    except asyncio.TimeoutError:
                        # No real audio in 3s — send a silent frame as keepalive.
                        await ws.send(SILENT_FRAME)
            except Exception:
                if not self._closed:
                    logger.debug("STT sender ended", exc_info=True)

        async def receiver():
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if data.get("type") != "Results":
                        continue
                    if not data.get("is_final"):
                        continue
                    transcript = (
                        data.get("channel", {})
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                    )
                    if transcript.strip():
                        await self._queue.put(transcript)
            except Exception:
                if not self._closed:
                    logger.debug("STT receiver ended", exc_info=True)

        async def supervise():
            await asyncio.gather(sender(), receiver())

        self._ws_task = asyncio.create_task(supervise())

    async def feed(self, pcm_frame: bytes) -> None:
        """Push a 48kHz int16 mono PCM frame into the STT stream."""
        if not self._closed:
            await self._send_queue.put(pcm_frame)

    async def transcripts(self) -> AsyncIterator[str]:
        """Yield finalized transcripts as they arrive."""
        while not self._closed:
            try:
                t = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield t
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
        if self._ws_task is not None:
            self._ws_task.cancel()


# ─────────────────────────────────────────────────────────────────────────────
# The mock IVR itself
# ─────────────────────────────────────────────────────────────────────────────


class MockIVR:
    """
    Walks the agent through the menu defined in test-fixtures/mock_ivr_menu.md.

    State machine is a sequence of async `level_N()` methods. Each level:
      1. Speaks the prompt for that level (via Deepgram TTS → AudioSource)
      2. Awaits the expected agent input (DTMF for menu levels, voice for rep levels)
      3. Either advances to the next level or branches on a trap

    Episode trace is captured in `self.trace` and dumped to JSON at the end.
    """

    def __init__(self, room_name: str, params: EpisodeParams, record_audio: bool = False) -> None:
        self.room_name = room_name
        self.params = params
        self.rep = get_rep(params.rep_persona)
        self.episode_id = f"ep_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        self.room = rtc.Room()
        self.audio_source: Optional[rtc.AudioSource] = None
        self.local_track: Optional[rtc.LocalAudioTrack] = None
        self.listener = DeepgramListener()

        # Event signals for the state machine
        self._dtmf_queue: asyncio.Queue[str] = asyncio.Queue()
        self._agent_joined = asyncio.Event()
        self._tts_resample_state = None  # for stateful streaming resample
        self.audio_recorder = StereoEpisodeRecorder() if record_audio else None

        # Episode trace — written to episodes/<id>.json at end of run
        self.trace = {
            "episode_id": self.episode_id,
            "room": room_name,
            "started_at": time.time(),
            "ended_at": None,
            "params": params.__dict__.copy(),
            "rep_persona": self.rep.name,
            "events": [],  # chronological list of {ts, type, payload}
        }

    def _log_event(self, type_: str, payload: dict) -> None:
        self.trace["events"].append(
            {
                "ts": time.time() - self.trace["started_at"],
                "type": type_,
                "payload": payload,
            }
        )

    # ── Connection setup ─────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Mint a participant token and join the room. Returns when the room is
        in CONNECTED state. Sets up DTMF + track-subscribed event handlers
        before connecting so we don't miss anything during the join handshake."""

        @self.room.on("sip_dtmf_received")
        def on_dtmf(dtmf: rtc.SipDTMF):
            # Note: livekit-agents' send_dtmf tool currently uses RFC2833-over-SIP
            # in real telephony, but for room-to-room WebRTC it surfaces as the
            # same sip_dtmf_received event. If we find this doesn't fire in
            # practice we'll fall back to data-channel signaling.
            logger.info("DTMF received: %r (code=%d)", dtmf.digit, dtmf.code)
            self._log_event("dtmf_received", {"digit": dtmf.digit, "code": dtmf.code})
            self._dtmf_queue.put_nowait(dtmf.digit)

        @self.room.on("participant_connected")
        def on_participant(p: rtc.RemoteParticipant):
            logger.info("Participant joined: identity=%s kind=%s", p.identity, p.kind)
            self._log_event("participant_joined", {"identity": p.identity, "kind": str(p.kind)})
            # Heuristic: the agent worker publishes itself with identity prefix "agent-".
            # If kind is AGENT it's definitely the agent.
            self._agent_joined.set()

        @self.room.on("track_subscribed")
        def on_track(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            logger.info("Subscribed to audio from %s", participant.identity)
            self._log_event(
                "agent_audio_subscribed", {"participant": participant.identity}
            )
            asyncio.create_task(self._consume_agent_audio(track))

        token = (
            api.AccessToken(
                os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
            )
            .with_identity(f"mock-ivr-{self.episode_id}")
            .with_name("MockHealth Provider Services")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

        logger.info("Connecting to room %s as mock IVR", self.room_name)
        await self.room.connect(os.environ["LIVEKIT_URL"], token)
        logger.info("Connected; local sid=%s", self.room.local_participant.sid)

        # If the agent was already in the room when we connected, the
        # `participant_connected` event won't have fired. Check the existing
        # remote participants and set the flag manually if any agent is there.
        for identity, participant in self.room.remote_participants.items():
            logger.info(
                "Found existing participant on connect: identity=%s kind=%s",
                identity, participant.kind,
            )
            self._log_event(
                "existing_participant", {"identity": identity, "kind": str(participant.kind)}
            )
            self._agent_joined.set()

        await self._publish_audio_track()
        await self.listener.start()

    async def _publish_audio_track(self) -> None:
        self.audio_source = rtc.AudioSource(ROOM_AUDIO_RATE, NUM_CHANNELS)
        self.local_track = rtc.LocalAudioTrack.create_audio_track(
            "mock-ivr-audio", self.audio_source
        )
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await self.room.local_participant.publish_track(
            self.local_track, options
        )
        logger.info("Published mock IVR audio track (sid=%s)", publication.sid)

    async def _consume_agent_audio(self, track: rtc.Track) -> None:
        """Pipe the agent's mic frames into Deepgram STT so we know what it said."""
        stream = rtc.AudioStream.from_track(
            track=track, sample_rate=ROOM_AUDIO_RATE, num_channels=NUM_CHANNELS
        )
        async for ev in stream:
            frame: rtc.AudioFrame = ev.frame
            # frame.data is array.array('h'); convert to bytes for the websocket.
            pcm = bytes(frame.data)
            if self.audio_recorder:
                self.audio_recorder.add_pcm(
                    channel=1,
                    start_s=time.time() - self.trace["started_at"],
                    pcm=pcm,
                )
            await self.listener.feed(pcm)

    # ── Audio output (TTS → room) ────────────────────────────────────────────

    async def speak(self, text: str) -> None:
        """Synthesize `text` via Deepgram and play it into the room.

        Logs the spoken text to the episode trace. Blocks until the entire
        utterance has been played out (so the menu state machine doesn't move
        on before the prompt finishes)."""
        logger.info("IVR speak: %s", text[:80])
        self._log_event("ivr_spoke", {"text": text})

        pcm_24k = await deepgram_tts(text)
        pcm_48k, self._tts_resample_state = resample_pcm_24k_to_48k(
            pcm_24k, self._tts_resample_state
        )
        if self.audio_recorder:
            self.audio_recorder.add_pcm(
                channel=0,
                start_s=time.time() - self.trace["started_at"],
                pcm=pcm_48k,
            )

        # Chunk into 20ms frames and push into the AudioSource.
        # AudioFrame data must be an int16 array; we use numpy to slice cleanly.
        samples = np.frombuffer(pcm_48k, dtype=np.int16)
        bytes_per_frame = SAMPLES_PER_FRAME * 2  # int16
        for offset in range(0, len(samples), SAMPLES_PER_FRAME):
            chunk = samples[offset : offset + SAMPLES_PER_FRAME]
            if len(chunk) < SAMPLES_PER_FRAME:
                # Pad the last chunk to a full frame with silence so playback timing stays correct.
                chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=ROOM_AUDIO_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
            await self.audio_source.capture_frame(frame)

        # Make sure all queued audio has actually played before we proceed.
        await self.audio_source.wait_for_playout()

    async def play_silence(self, seconds: float) -> None:
        """Push silent frames into the source (used for hold music gaps)."""
        silent_frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.int16)
        num_frames = int(seconds * 50)  # 50 frames/sec at 20ms each
        for _ in range(num_frames):
            frame = rtc.AudioFrame(
                data=silent_frame.tobytes(),
                sample_rate=ROOM_AUDIO_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
            await self.audio_source.capture_frame(frame)
        await self.audio_source.wait_for_playout()

    # ── Input collection ─────────────────────────────────────────────────────

    async def wait_for_dtmf(self, timeout_s: float) -> Optional[str]:
        """Block until the agent presses a DTMF digit, or until timeout."""
        try:
            digit = await asyncio.wait_for(self._dtmf_queue.get(), timeout=timeout_s)
            return digit
        except asyncio.TimeoutError:
            self._log_event("dtmf_timeout", {"waited_s": timeout_s})
            return None

    async def collect_dtmf_string(
        self, terminator: str = "#", timeout_s: float = 15.0
    ) -> Optional[str]:
        """Collect digits until the agent presses `terminator` (default `#`)
        or `timeout_s` elapses with no input. Used for NPI entry."""
        digits = []
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            d = await self.wait_for_dtmf(min(remaining, 5.0))
            if d is None:
                continue
            if d == terminator:
                return "".join(digits) + terminator
            digits.append(d)
        return None  # timeout

    async def wait_for_agent_speech(
        self,
        timeout_s: float = 10.0,
        max_drain_s: float = 2.5,
        silence_window_s: float = 1.5,
    ) -> Optional[str]:
        """Block until Deepgram STT yields finalized transcripts from the agent.

        Returns once one of these stop conditions hits:
          1. `silence_window_s` elapses with no new transcript (agent finished
             a complete utterance - natural turn boundary)
          2. `max_drain_s` elapses since the FIRST chunk arrived (cap to keep
             tests bounded even if the agent rambles)
          3. `timeout_s` elapses with no first chunk at all (the agent never
             responded - real failure to flag)

        Why this matters: Deepgram finalizes on ~700ms pauses, so a long agent
        utterance like "Hi this is Ruma Care on behalf of Dr Smith about Jane
        Doe, date of birth March fourteenth nineteen eighty five..." comes
        in as 2-3 separate chunks. A short drain window returns after the
        first chunk and the mock rep then talks over the still-speaking
        agent. The 1.5s silence window matches the cadence of a human
        finishing a sentence; the 2.5s hard cap prevents pathological
        rambling from blocking the test forever.
        """
        try:
            transcripts_iter = self.listener.transcripts()
            first = await asyncio.wait_for(
                transcripts_iter.__anext__(), timeout=timeout_s
            )
            collected = [first]
            first_chunk_at = asyncio.get_event_loop().time()
            hard_deadline = first_chunk_at + max_drain_s
            last_chunk_at = first_chunk_at
            while True:
                now = asyncio.get_event_loop().time()
                if now >= hard_deadline:
                    break
                quiet_remaining = silence_window_s - (now - last_chunk_at)
                if quiet_remaining <= 0:
                    break
                next_wait = min(quiet_remaining, hard_deadline - now)
                try:
                    more = await asyncio.wait_for(
                        self.listener._queue.get(), timeout=next_wait
                    )
                    collected.append(more)
                    last_chunk_at = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    break
            joined = " ".join(collected)
            self._log_event(
                "agent_said",
                {
                    "text": joined,
                    "chunks": len(collected),
                    "drain_s": round(last_chunk_at - first_chunk_at, 2),
                },
            )
            return joined
        except asyncio.TimeoutError:
            return None
        except StopAsyncIteration:
            return None

    # ── State machine ────────────────────────────────────────────────────────

    async def run_episode(self) -> dict:
        """Run the full menu tree end-to-end. Returns the episode trace dict."""
        try:
            await self._level_1()
            await self._level_2()
            await self._level_3()
            await self._level_4()
            for _ in range(self.params.invalid_npi_count):
                await self._level_5_false_rejection()
            await self._level_6_hold()
            await self._level_7_rep_dob()
            await self._level_8_ref_number()
            await self._level_9_status_and_confirm()
            self._log_event("episode_complete", {"reason": "happy_path"})
        except asyncio.TimeoutError as e:
            self._log_event("episode_aborted", {"reason": "timeout", "detail": str(e)})
            logger.warning("Episode aborted on timeout: %s", e)
        except Exception as e:
            self._log_event("episode_aborted", {"reason": "exception", "detail": repr(e)})
            logger.exception("Episode aborted on exception")

        self.trace["ended_at"] = time.time()
        return self.trace

    async def _level_1(self) -> None:
        prompt = (
            "Thank you for calling MockHealth provider services. "
            "Most prior authorization questions can be answered at our "
            "provider portal at MockHealth dot com slash providers. "
            "To continue with this call: for English, press 1, "
            "para Español, oprima 2."
            if self.params.inject_url_deflection
            else
            "Thank you for calling MockHealth provider services. "
            "For English, press 1, para Español, oprima 2."
        )
        await self.speak(prompt)
        digit = await self.wait_for_dtmf(timeout_s=15.0)
        self._log_event("level_1_chose", {"digit": digit, "expected": "1"})
        if digit != "1":
            await self.speak("I'm sorry, that input was not recognized.")
            raise asyncio.TimeoutError("Agent did not select English at L1")

    async def _level_2(self) -> None:
        await self.speak(
            "Please listen to the following menu options carefully, "
            "as they have recently changed. "
            "For claims status, press 1. "
            "For prior authorization, press 2. "
            "For credentialing, press 3. "
            "For all other inquiries, press 0. "
            "To repeat this menu, press star."
        )
        digit = await self.wait_for_dtmf(timeout_s=15.0)
        self._log_event("level_2_chose", {"digit": digit, "expected": "2"})
        if digit != "2":
            raise asyncio.TimeoutError(
                f"Agent picked {digit} at L2 (expected 2 for prior authorization)"
            )

    async def _level_3(self) -> None:
        await self.speak(
            "You've reached prior authorization services. "
            "To check the status of an existing authorization, "
            "you may visit our provider portal at MockHealth dot com, "
            "or use our automated status system by entering your "
            "reference number now. "
            "To speak with a representative about prior authorization, press 9."
        )
        digit = await self.wait_for_dtmf(timeout_s=20.0)
        self._log_event("level_3_chose", {"digit": digit, "expected": "9"})
        if digit != "9":
            raise asyncio.TimeoutError(
                f"Agent fell for self-service trap at L3 (pressed {digit}, expected 9)"
            )

    async def _level_4(self) -> None:
        await self.speak(
            "Before we connect you with a representative, "
            "please enter your 10-digit National Provider Identifier, "
            "followed by the pound sign."
        )
        npi = await self.collect_dtmf_string(terminator="#", timeout_s=20.0)
        self._log_event(
            "level_4_npi",
            {"received": npi, "expected": self.params.expected_npi + "#"},
        )
        if npi != self.params.expected_npi + "#":
            raise asyncio.TimeoutError(
                f"Agent sent wrong NPI at L4: got {npi}, expected {self.params.expected_npi}#"
            )

    async def _level_5_false_rejection(self) -> None:
        """Fired `invalid_npi_count` times — IVR lies about the NPI being invalid."""
        await self.speak(
            "We were unable to verify that NPI. "
            "Please re-enter your 10-digit NPI, followed by the pound sign."
        )
        npi = await self.collect_dtmf_string(terminator="#", timeout_s=20.0)
        self._log_event(
            "level_5_npi_retry",
            {"received": npi, "expected": self.params.expected_npi + "#"},
        )
        if npi != self.params.expected_npi + "#":
            raise asyncio.TimeoutError(
                f"Agent changed NPI after false rejection: got {npi}"
            )

    async def _level_6_hold(self) -> None:
        await self.speak(
            "Thank you. Please hold while we connect you with the "
            "next available representative. "
            "Your estimated wait time is 47 minutes. "
            "Your call may be recorded for quality assurance. "
            "We are experiencing higher than normal call volumes."
        )
        # Stream silence + periodic "your call is important to us" messages.
        elapsed = 0.0
        while elapsed < self.params.hold_seconds:
            chunk = min(10.0, self.params.hold_seconds - elapsed)
            await self.play_silence(chunk)
            elapsed += chunk
            if elapsed < self.params.hold_seconds:
                await self.speak(
                    "Your call is important to us, please continue to hold."
                )

    async def _level_7_rep_dob(self) -> None:
        await self.speak(self.rep.greeting)
        # 25s first-wait: rep greeting takes ~5s to render+transmit, then
        # native S2S models like GPT-RT2 need ~3-8s to think + ~5s to say
        # the full intro sentence. The earlier 12s timeout was firing while
        # the agent was still mid-sentence on its intro, causing TIMING fails
        # on check 08_announced_to_rep.
        agent_response = await self.wait_for_agent_speech(timeout_s=25.0)
        self._log_event(
            "level_7_agent_response",
            {"text": agent_response, "rep": self.rep.name},
        )
        if self.rep is not REP_POLITE_SARAH:
            await self.speak(self.rep.dob_question)
            agent_response2 = await self.wait_for_agent_speech(timeout_s=20.0)
            self._log_event("level_7_dob_response", {"text": agent_response2})

    async def _level_8_ref_number(self) -> None:
        await self.speak(self.rep.ref_question)
        agent_response = await self.wait_for_agent_speech(timeout_s=10.0)
        self._log_event("level_8_ref_response", {"text": agent_response})

    async def _level_9_status_and_confirm(self) -> None:
        await self.speak(self.rep.pulling_up_filler)
        await self.play_silence(3.0)
        status_line = self.rep.status_template.format(
            ref=self.params.expected_auth_ref,
            status=self.params.auth_status,
            cpt=self.params.expected_cpt,
        )
        await self.speak(status_line)

        # Now we expect the agent to read back the status (verify-after-action,
        # Phase 2.5 behavior). For 2.1b we just capture whatever the agent says.
        agent_readback = await self.wait_for_agent_speech(timeout_s=15.0)
        self._log_event("level_9_agent_readback", {"text": agent_readback})

        await self.speak(self.rep.confirm_line)
        agent_final = await self.wait_for_agent_speech(timeout_s=8.0)
        self._log_event("level_9_agent_final", {"text": agent_final})
        await self.speak(self.rep.goodbye)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self.listener.close()
        if self.local_track is not None:
            await self.room.local_participant.unpublish_track(self.local_track.sid)
        await self.room.disconnect()

    def dump_trace(self, episodes_dir: Path) -> Path:
        episodes_dir.mkdir(parents=True, exist_ok=True)
        out = episodes_dir / f"{self.episode_id}.json"
        out.write_text(json.dumps(self.trace, indent=2, default=str))
        logger.info("Episode trace written to %s", out)
        return out

    def dump_recording(self, recordings_dir: Path) -> Optional[Path]:
        if not self.audio_recorder:
            return None
        out = recordings_dir / f"{self.episode_id}.wav"
        self.audio_recorder.write_wav(out)
        rel_path = f"recordings/{out.name}"
        self.trace["recording"] = {
            "path": rel_path,
            "format": "wav",
            "sample_rate": ROOM_AUDIO_RATE,
            "channels": {
                "left": "mock_ivr",
                "right": "agent",
            },
        }
        self._log_event("recording_written", {"path": rel_path})
        logger.info("Episode recording written to %s", out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    params = EpisodeParams(
        language_attempt=args.language_attempt,
        invalid_npi_count=args.invalid_npi_count,
        hold_seconds=args.hold_seconds,
        rep_persona=args.persona,
        inject_url_deflection=not args.no_url_deflection,
        human_path=args.human_path,
        auth_status=args.auth_status,
    )

    mock = MockIVR(room_name=args.room, params=params, record_audio=args.record_audio)
    try:
        await mock.connect()
        if not args.skip_wait:
            logger.info("Waiting up to 30s for agent to join the room...")
            try:
                await asyncio.wait_for(mock._agent_joined.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error(
                    "No agent joined room %s within 30s. Make sure the agent "
                    "worker is running and a dispatch has been created for this room.",
                    args.room,
                )
                return 1
            # Give the agent's AgentSession time to fully initialize before we speak.
            await asyncio.sleep(1.5)

        await mock.run_episode()
    except Exception as e:
        mock._log_event("episode_aborted", {"reason": "outer_exception", "detail": repr(e)})
        logger.exception("Episode aborted before completion")
    finally:
        if mock.trace["ended_at"] is None:
            mock.trace["ended_at"] = time.time()
        mock.dump_recording(REPO_ROOT / "voice-agent" / "recordings")
        mock.dump_trace(REPO_ROOT / "voice-agent" / "episodes")
        try:
            await mock.close()
        except Exception:
            logger.exception("Error during close")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adversarial mock IVR for the Ruma Care voice agent"
    )
    p.add_argument(
        "--room",
        required=True,
        help="LiveKit room name to join (must match the dispatched job's room)",
    )
    p.add_argument(
        "--persona",
        default="polite_sarah",
        choices=["polite_sarah", "rushed_john", "confused_maria", "random"],
    )
    p.add_argument("--invalid-npi-count", type=int, default=0)
    p.add_argument("--hold-seconds", type=int, default=15)
    p.add_argument("--language-attempt", default="english")
    p.add_argument(
        "--human-path", default="press_9", choices=["press_9", "say_representative", "press_0_twice"]
    )
    p.add_argument(
        "--auth-status",
        default="approved",
        choices=["approved", "denied", "pending", "expired", "not_found"],
    )
    p.add_argument(
        "--no-url-deflection",
        action="store_true",
        help="Disable the 'visit our portal at...' deflection prefix",
    )
    p.add_argument(
        "--skip-wait",
        action="store_true",
        help="Don't wait for an agent participant before speaking (for self-test)",
    )
    p.add_argument(
        "--record-audio",
        action="store_true",
        help="Record local stereo WAV: mock IVR on left, agent audio on right",
    )
    args = p.parse_args()
    exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
