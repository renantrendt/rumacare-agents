# Ruma Care Voice Agent - LiveKit Edition

Outbound voice agent and mock IVR evaluation harness for prior authorization status checks.

The public harness is GPT-first: the default model path is GPT-Realtime-2.

---

## Stack And Accounts

| Layer | Provider/tool | Env vars |
|---|---|---|
| Realtime rooms, dispatch, SIP | LiveKit Cloud | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `SIP_OUTBOUND_TRUNK_ID` |
| Primary S2S model | OpenAI GPT-Realtime-2 | `OPENAI_API_KEY` |
| Mock IVR TTS/STT | Deepgram | `DEEPGRAM_API_KEY` |
| PSTN/SIP carrier | Twilio Elastic SIP Trunk | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` |
| Optional scorer judge | Anthropic Claude | `ANTHROPIC_API_KEY`, `JUDGE_MODEL` |

For mock-IVR benchmarking, you need LiveKit, OpenAI, and Deepgram. Add Twilio when testing real outbound SIP/PSTN calls. Add Anthropic only if you run LLM-judged soft checks.

---

## Why LiveKit

The previous bespoke stack worked, but every precision improvement required more hand-rolled telephony and media infrastructure. LiveKit gives us a cleaner worker model, dispatching, WebRTC audio, SIP integration, and native speech-to-speech model support.

| Capability | Bespoke stack (legacy) | LiveKit |
|---|---|---|
| IVR testing | Manual call attempts | Python mock IVR participant |
| DTMF for IVR | Twilio TwiML updates | LiveKit room DTMF tool |
| Turn-taking + barge-in | Manual VAD wiring | `livekit.plugins.turn_detector` |
| Native S2S model path | Major rewrite | GPT-Realtime-2 session construction |
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

## Mock IVR Personas

The harness intentionally varies the representative after the IVR so prompt changes do not overfit to a single cooperative rep.

| Persona | CLI value | Behavior |
|---|---|---|
| Polite Sarah | `polite_sarah` | Clear baseline. Asks for DOB, then authorization reference, then reads status. |
| Rushed John | `rushed_john` | Fast and terse. Often asks for the reference first and uses shorter confirmations. |
| Confused Maria | `confused_maria` | Recovery test. Asks for repeats and may appear to mix up the patient before finding the authorization. |

Use `--personas all` to rotate all three:

```bash
python bench.py --models gpt-rt2 --personas all --runs 1 --record-audio
```

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

The public repo includes one synthetic demo run so people can inspect the output without running the harness first:

- `episodes/`
- `recordings/`
- `dashboard.html`
- `logs/`

Screenshots used by the root README live at the repo root and are intentionally trackable.

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

Render the dashboard:

```bash
python dashboard.py --out dashboard.html --open
```

The dashboard includes:

- Leaderboard by model.
- Per-check pass-rate matrix.
- Per-episode trace replay.
- Audio player for episodes recorded with `--record-audio`.

## Public Data Policy

Use only synthetic patients, provider identifiers, phone numbers, traces, and recordings in public examples. Do not commit `.env.local`, real call recordings, logs, or payer/patient data.
