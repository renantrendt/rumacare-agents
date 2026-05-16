# Ruma Care Voice Agent - LiveKit Edition

Outbound voice agent and mock IVR evaluation harness for prior authorization status checks.

The current default model path is GPT-Realtime-2. Grok Voice and a cascaded Deepgram/Groq/Deepgram path remain available for explicit comparison runs.

---

## Why LiveKit

The previous bespoke stack worked, but every precision improvement required more hand-rolled telephony and media infrastructure. LiveKit gives us a cleaner worker model, dispatching, WebRTC audio, SIP integration, and native speech-to-speech model support.

| Capability | Bespoke stack (legacy) | LiveKit |
|---|---|---|
| IVR testing | Manual call attempts | Python mock IVR participant |
| DTMF for IVR | Twilio TwiML updates | LiveKit room DTMF tool |
| Turn-taking + barge-in | Manual VAD wiring | `livekit.plugins.turn_detector` |
| Switching cascaded/native S2S | Major rewrite | Swap session model construction |
| Concurrent calls | 1 per process (FastAPI bound) | N per worker, dispatched on demand |
| Observability | Local logs only | traces, scoring, dashboard, LiveKit logs |

---

## Quick Start

```bash
# 1. Install
cd voice-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure env
cp .env.example ../.env.local
# Fill in real keys in ../.env.local. Do not commit it.

# 3. Run worker (registers with LiveKit Cloud, waits for dispatches)
python agent.py dev
```

In another terminal, run a mock IVR episode:

```bash
python bench.py --models gpt-rt2 --personas polite_sarah --runs 1 --record-audio
```

Generate the HTML dashboard:

```bash
python dashboard.py --out dashboard.html --open
```

---

## Required Env Vars

See `.env.example` for the full list with comments. All secrets live in `../.env.local`, which is gitignored at repo root.

---

## File Map

| File | Role |
|---|---|
| `agent.py` | LiveKit worker entrypoint, session routing, mission agent |
| `mission_brief.py` | Pydantic mission schema and metadata parser |
| `mock_ivr.py` | adversarial LiveKit participant for mock payer IVR calls |
| `bench.py` | sequential model/persona benchmark runner |
| `score_episodes.py` | deterministic and LLM-assisted episode scorer |
| `dashboard.py` | self-contained local HTML dashboard generator |
| `episode_trace.py` | agent-side trace recorder |
| `test-fixtures/mock_ivr_menu.md` | adversarial mock IVR menu spec |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Template for `../.env.local` |

Generated artifacts are ignored by git:

- `episodes/`
- `recordings/`
- `dashboard.html`
- `logs/`

---

## Architecture

```
┌──────────────┐      WebRTC       ┌─────────────────┐      WebRTC       ┌──────────────┐
│ mock_ivr.py  │◀────────────────▶│ LiveKit Cloud   │◀────────────────▶│ agent.py     │
│ payer IVR    │                   │ room / SFU      │                  │ MissionAgent │
└──────────────┘                   └─────────────────┘                  └──────────────┘
       │                                                                      │
       ├─ Deepgram TTS/STT for mock prompts/listening                         ├─ GPT-Realtime-2
       ├─ JSON mock trace                                                     ├─ DTMF tool calls
       └─ optional stereo WAV recording                                       └─ JSON agent trace
```

For real phone calls, LiveKit SIP connects through a Twilio Elastic SIP Trunk. Mock episodes do not use PSTN or SIP costs.

## Benchmarking

Run one model:

```bash
python bench.py --models gpt-rt2 --personas all --runs 1 --record-audio
```

Run an opt-in comparison:

```bash
python bench.py --models gpt-rt2,grok-voice --personas all --runs 1
```

Render the dashboard:

```bash
python dashboard.py --out dashboard.html --open
```

## Public Data Policy

Use only synthetic patients, provider identifiers, phone numbers, traces, and recordings in public examples. Do not commit `.env.local`, real call recordings, logs, or payer/patient data.
