# Mock IVR menu tree — MockHealth Provider Services

**Purpose:** Adversarial test target for the Ruma Care voice agent's IVR navigation.
Models real payer hostility patterns observed in Aetna / UHC / Anthem provider lines.

**Used by:**
- `voice-agent/mock_ivr.py` — Python implementation that joins LiveKit rooms as a participant (fast iteration, no SIP/PSTN cost)
- `voice-agent/test-fixtures/mock_ivr_flow.json` — (future) Twilio Studio flow export for final SIP-layer confirmation

**Used together with:** `episode_schema.py` (rubric checks) + `score_episodes.py` (Groq judge).

---

## Persona

**"MockHealth Provider Services"** — a fictional payer's provider services line.
Voice: female, neutral American English (matches "polite Sarah" rep persona at L7+).
Style: deliberately verbose, often deflects to web portal, hides the human option.

---

## Adversarial design principles

These are the real-world hostility patterns the agent MUST handle without giving up:

1. **Self-service is offered first at every level** (deflection toward portal/automated check)
2. **Human path requires effort** — typically 2+ DTMF presses, sometimes a spoken "representative"
3. **Authentication is asked multiple times** (NPI first via DTMF, DOB later via voice)
4. **One false "invalid input" prompt** at 10% rate to test recovery from wrongful rejection
5. **Hold queue with deliberately long announced wait** ("estimated 47 minutes") to encourage hang-ups
6. **Mixed input modes** — DTMF at some levels, spoken response at others
7. **URL deflection** — "most authorizations can be checked online at..." before EVERY menu
8. **Inconsistent menu conventions** — "press 0" might mean "main menu" not "operator" depending on level

---

## Menu tree

### L0 — Connect

```
RING → ANSWER
```

### L1 — Language selection (with URL deflection)

```
PROMPT (after 1s):
  "Thank you for calling MockHealth provider services.
   Most prior authorization questions can be answered
   at our provider portal at MockHealth dot com slash providers.
   To continue with this call:
   for English, press 1, para español, oprima 2."

WAIT: 10s for DTMF input
EXPECTED AGENT ACTION: presses 1
TRAP: if no input → re-prompts once → second timeout transfers to Spanish line
```

### L2 — Main menu (with "0 for all other" red herring)

```
PROMPT:
  "Please listen to the following menu options carefully,
   as they have recently changed."
  [pause 2s]
  "For claims status, press 1.
   For prior authorization, press 2.
   For credentialing, press 3.
   For all other inquiries, press 0.
   To repeat this menu, press star."

WAIT: 8s for DTMF
EXPECTED AGENT ACTION: presses 2
TRAP: "press 0 for all other inquiries" sounds like the
      legacy "press 0 for operator" convention but actually
      routes to a different deflection menu (not modeled in v1)
```

### L3 — Prior auth submenu (the self-service trap)

```
PROMPT:
  "You've reached prior authorization services.
   To check the status of an existing authorization,
   you may visit our provider portal at MockHealth dot com,
   or use our automated status system by entering your
   reference number now.
   To speak with a representative about prior authorization,
   press 9."

WAIT: 12s for DTMF (longer wait — the IVR is hoping you'll cave and enter the ref number)
EXPECTED AGENT ACTION: presses 9
TRAP 1: the automated status system is the obvious self-service path —
        agent MUST refuse per the hard rule "never trust self-service for PA status"
TRAP 2: agent might think to enter the ref number digit-by-digit since it has it
        in the brief; correct behavior is to ignore that option
```

### L4 — NPI authentication

```
PROMPT:
  "Before we connect you with a representative,
   please enter your 10-digit National Provider Identifier,
   followed by the pound sign."

WAIT: 15s for DTMF
EXPECTED AGENT ACTION: sends DTMF "<NPI from brief>#"
  e.g. for NPI 1234567890: agent sends "1234567890#"
```

### L5 — Random false NPI rejection (10% of episodes by default)

```
PROMPT (only fires in adversarial mode, ~10% of the time):
  "We were unable to verify that NPI.
   Please re-enter your 10-digit NPI, followed by the pound sign."

WAIT: 15s for DTMF
EXPECTED AGENT ACTION: re-sends the SAME NPI (NOT a different one)
TRAP 1: agent might try a slightly different NPI thinking the brief was wrong
TRAP 2: agent might give up and try to escalate to supervisor
TRAP 3: agent might enter the NPI without the # this time
```

### L6 — Hold queue

```
PROMPT:
  "Thank you. Please hold while we connect you with the
   next available representative.
   Your estimated wait time is 47 minutes.
   Your call may be recorded for quality assurance.
   We are experiencing higher than normal call volumes."

THEN: hold music plays for `hold_seconds` (default 30s, configurable per episode)
EVERY 10s during hold: "Your call is important to us, please continue to hold."

EXPECTED AGENT ACTION:
  - silence (no speech)
  - no DTMF
  - no hangup

TRAP 1: agent might say "are you there?" filling the silence, causing
        confusion when the rep finally picks up mid-sentence
TRAP 2: agent might hang up because of the "47 minute" announcement
TRAP 3: agent might be tricked by "we are experiencing higher than
        normal call volumes" into giving up
```

### L7 — Rep picks up + DOB request

```
REP (one of three personas — see "Rep personas" section):
  "Provider services, this is <name>, can I have the patient's
   date of birth please?"

WAIT: up to 10s for agent response
EXPECTED AGENT ACTION:
  1. brief greeting identifying agent + payer-context
  2. DOB digit-by-digit OR conversational, both acceptable
     - digit-by-digit: "zero three fourteen nineteen eighty five"
     - conversational: "March fourteenth, nineteen eighty five"
```

### L8 — Auth reference number request

```
REP: "And the authorization reference number?"

WAIT: up to 10s
EXPECTED AGENT ACTION: reads reference char-by-char or as a unit
  e.g. "A U T H dash 2 0 2 6 dash 0 0 4 2"
       OR "AUTH-2026-0042"
```

### L9 — Status read-out + verification gate

```
REP: "Okay, let me pull that up... one moment."
[pause 3s]
REP: "Yes I see that authorization. It was approved on April 30th, 2026,
      effective through July 30th, 2026, for procedure code 99213,
      one visit. Anything else?"

EXPECTED AGENT ACTION (verify-after-action, REQUIRED for Phase 2.5):
  Agent reads back the status BEFORE saying goodbye:
    "Got it — so AUTH-2026-0042 is approved, effective through
     July 30 2026, for one visit of CPT 99213. Correct?"

REP (after agent reads back): confirms ("Yes, that's right" or "Correct")

THEN agent: "Thank you for confirming. Have a good day."
```

### L10 — Clean hangup

```
EXPECTED AGENT ACTION: calls end_call tool
```

---

## Rep personas (one chosen per episode)

The Python mock rotates between these three to test conversational robustness.
Configured via `rep_persona` episode parameter (or `"random"` to pick uniformly).

### A. Polite Sarah (the easy case)

```
- Speaks at a normal pace, clear American English
- Asks questions in order (DOB → ref number → gives status)
- Waits for the agent to finish before speaking
- Gives the status when first asked
- Used as the baseline; if agent fails here, it'll fail elsewhere too
```

### B. Rushed John (turn-taking stress test)

```
- Talks fast, sometimes interrupts mid-question
- Asks for ref number BEFORE DOB (reversed order — agent shouldn't assume)
- Says "uh-huh, uh-huh" while agent is talking (false barge-ins)
- Gives status in one rushed sentence:
  "yeahso0042approvedthroughjuly30one99213anythingelse"
- Tests: agent's STT robustness, turn-taking model, ability to parse
        smushed-together speech
```

### C. Confused Maria (recovery stress test)

```
- Asks for DOB twice ("sorry, one more time?")
- Mishears digits ("did you say 0-3 or 0-8?")
- Pulls up wrong patient first ("oh wait that's a different Jane Doe...")
- Asks "can you repeat that?" multiple times
- Eventually gives the right status but takes 90+ seconds longer
- Tests: agent patience, willingness to repeat, no over-correcting
```

---

## Episode parameters (Python mock configurability)

Each episode is a configuration dict passed when spawning the mock:

```python
{
  # Language path
  "language_attempt": "english",      # english | spanish_then_english | english_only

  # NPI authentication
  "invalid_npi_count": 0,             # 0 | 1 | 2 | 3 — how many false rejections to inject

  # Hold queue
  "hold_seconds": 30,                 # 5 | 30 | 60 | 120

  # Rep behavior
  "rep_persona": "polite_sarah",      # polite_sarah | rushed_john | confused_maria | random

  # URL deflection
  "inject_url_deflection": True,      # plays "visit MockHealth.com" at L1 + L3

  # Human path
  "human_path": "press_9",            # press_9 | say_representative | press_0_twice

  # Status outcome
  "auth_status": "approved",          # approved | denied | pending | expired | not_found

  # Rep voice (for transcript debugging only)
  "rep_voice_gender": "female",       # male | female
}
```

---

## Scoring rubric (12 checks for batch 1)

Run after each episode by `score_episodes.py`. Hard fails block "batch 1 complete".

| ID | Check | Type | Hard fail? |
|---|---|---|---|
| `01_picked_up` | Agent connected to mock | event | YES |
| `02_chose_english` | Agent pressed `1` (not `2`) at L1 | dtmf | YES |
| `03_chose_prior_auth` | Agent pressed `2` (not `0`/`3`) at L2 | dtmf | YES |
| `04_refused_self_service` | Agent pressed `9` (not entered ref number) at L3 | dtmf | YES |
| `05_npi_correct` | DTMF at L4 matches brief's NPI + `#` | dtmf | YES |
| `06_retried_npi_calmly` | If L5 triggered, agent re-sent SAME NPI without different value | dtmf | YES |
| `07_silent_during_hold` | No agent speech during hold music | transcript | NO (warning only) |
| `08_announced_to_rep` | Within 5s of rep speaking at L7, agent identified self + patient name | llm-judge | YES |
| `09_gave_correct_dob` | DOB spoken/transcribed matches brief | llm-judge | YES |
| `10_gave_correct_auth_ref` | Auth ref spoken/transcribed matches brief | llm-judge | YES |
| `11_read_back_status` | Agent recited the auth status the rep gave, BEFORE calling end_call | llm-judge | YES |
| `12_under_5min` | Total call duration <300s | duration | NO (warning only) |

**Pass rate target for batch 1:** ≥80% of episodes pass ALL hard checks before moving to batch 2.

---

## Failure-mode taxonomy

When a check fails, the scorer categorizes the failure for prompt-engineering signal:

| Category | Meaning | Fix lever |
|---|---|---|
| `PROMPT_GAP` | Agent did something not covered in the system prompt | Add a rule to `MissionAgent.instructions` |
| `STT_MISHEARD` | Deepgram transcribed wrong words | Tune STT params (model, language, keywords) |
| `LLM_HALLUCINATED` | Agent invented a value not in the brief | Add explicit "never guess" instruction with example |
| `LLM_GAVE_UP` | Agent escalated or hung up when it shouldn't have | Reinforce "don't give up" + add specific scenario to prompt |
| `TOOL_MISUSE` | Agent called wrong tool / wrong args | Improve tool docstring + add few-shot example |
| `TIMING` | Agent acted too slow or too fast | Tune VAD / turn-detector params |
| `IVR_SIGNAL` | LiveKit's IVR detector missed an IVR / detected human as IVR | Adjust `ivr_detection` config, or add manual heuristic |

Each failed check in the episode trace gets one of these labels (Groq judge does the classification).
