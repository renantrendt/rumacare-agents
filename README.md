# RumaCare Agents

Voice-agent experiments for automating prior authorization status-check calls.

The current implementation is a LiveKit Agents worker plus an adversarial mock IVR harness. It is designed for fast prompt-engineering and model evaluation before piloting against real payer phone trees.

## What This Does

- Runs a LiveKit voice agent that can join mock or SIP-backed calls.
- Uses GPT-Realtime-2 as the default speech-to-speech model path.
- Keeps Grok Voice and a cascaded STT/LLM/TTS stack as opt-in comparison paths.
- Provides an adversarial Python mock IVR with deflection menus, DTMF capture, hold behavior, and representative personas.
- Records mock episodes as JSON traces and optional local stereo WAV files.
- Scores episodes with deterministic checks and optional LLM-judged soft checks.
- Generates a local HTML dashboard with leaderboard, replay timeline, and audio playback.

## Repository Layout

```text
.
├── README.md
├── .env.example
├── voice-agent/
│   ├── agent.py              # LiveKit worker and mission agent
│   ├── mock_ivr.py           # adversarial mock IVR participant
│   ├── bench.py              # benchmark runner
│   ├── score_episodes.py     # scoring harness
│   ├── dashboard.py          # local HTML report generator
│   ├── episode_trace.py      # agent-side trace recorder
│   ├── mission_brief.py      # Pydantic mission schema
│   ├── requirements.txt
│   ├── .env.example
│   └── test-fixtures/
│       └── mock_ivr_menu.md
└── scripts/
```

Generated artifacts are intentionally ignored:

- `voice-agent/episodes/`
- `voice-agent/recordings/`
- `voice-agent/dashboard.html`
- `voice-agent/logs/`
- `.env.local`

## Quick Start

```bash
cd voice-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local env file from the template:

```bash
cp .env.example ../.env.local
```

Fill in the real provider keys in `../.env.local`. Never commit that file.

Start the LiveKit worker:

```bash
python agent.py dev
```

In another terminal, run one GPT-Realtime-2 mock IVR episode:

```bash
python bench.py \
  --models gpt-rt2 \
  --personas polite_sarah \
  --runs 1 \
  --hold-seconds 3 \
  --record-audio
```

Generate the dashboard:

```bash
python dashboard.py --out dashboard.html --open
```

## Model Paths

The mission metadata controls the backend:

- `gpt-rt2`: OpenAI GPT-Realtime-2, the default path.
- `grok-voice`: xAI Grok Voice Think Fast, available for comparison.
- `cascaded`: Deepgram STT, Groq Llama, and Deepgram TTS fallback.

`MissionBrief.model` defaults to `gpt-rt2`.

## Scoring

The active benchmark focuses on IVR navigation only. Human-representative checks are implemented but temporarily disabled until the human-in-the-loop process is defined.

Current hard checks include:

- Agent connected to the mock IVR.
- Chose English.
- Chose prior authorization.
- Refused the self-service trap and selected the representative path.
- Sent the expected NPI followed by `#`.
- Retried calmly after a false NPI rejection.
- Finished under five minutes.

## Audio Replay

When `--record-audio` is used, the mock harness writes a stereo WAV:

- Left channel: mock IVR audio.
- Right channel: agent audio received from LiveKit.

`dashboard.py` adds an audio player for recorded episodes and keeps the text replay below it for debugging.

## Safety Notes

- Do not commit `.env.local`, API keys, real phone numbers, production call recordings, or payer/patient data.
- Use synthetic mission briefs and mock IVR traces for public examples.
- Treat real payer pilots as private operational data.

## License

No license has been selected yet.
