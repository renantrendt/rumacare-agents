"""
Episode scorer — runs the 12-check rubric from mock_ivr_menu.md against the
correlated mock + agent traces for each episode, then prints a dashboard.

Usage:
    python score_episodes.py                # score all episodes
    python score_episodes.py --since 1h     # last hour only
    python score_episodes.py --episode <id> # one specific episode
    python score_episodes.py --json         # machine-readable output
    python score_episodes.py --judge groq   # default; can switch to claude later

Pairs episodes by matching `mock_trace.room == agent_trace.room`.

Failure modes are tagged with the taxonomy from mock_ivr_menu.md:
  PROMPT_GAP | STT_MISHEARD | LLM_HALLUCINATED | LLM_GAVE_UP |
  TOOL_MISUSE | TIMING | IVR_SIGNAL

We use the Groq Llama 3.3 70B model for the judge (per the user's choice in
2.1d planning — swap to Claude later by adding ANTHROPIC_API_KEY and
plumbing _llm_judge_with_claude).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import openai as oa
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env.local", override=True)

EPISODES_DIR = Path(__file__).parent / "episodes"

logger = logging.getLogger("scorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    id: str
    description: str
    passed: bool
    hard_fail: bool  # if True, failing this check fails the whole episode
    detail: str = ""
    failure_mode: Optional[str] = None  # taxonomy tag if failed


@dataclass
class EpisodeScore:
    episode_id: str
    room: str
    job_id: Optional[str]
    model: str  # which speech backend the agent used (grok-voice / gpt-rt2 / cascaded / unknown)
    rep_persona: str
    duration_s: float
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def hard_pass(self) -> bool:
        """True if ALL hard-fail checks passed."""
        return all(c.passed for c in self.checks if c.hard_fail)

    @property
    def hard_score(self) -> tuple[int, int]:
        """(passed, total) for hard-fail checks."""
        hard = [c for c in self.checks if c.hard_fail]
        return sum(1 for c in hard if c.passed), len(hard)

    @property
    def all_score(self) -> tuple[int, int]:
        return sum(1 for c in self.checks if c.passed), len(self.checks)


# ─────────────────────────────────────────────────────────────────────────────
# Trace loading + correlation
# ─────────────────────────────────────────────────────────────────────────────


def load_episodes(
    episodes_dir: Path, since_seconds: Optional[float] = None
) -> list[tuple[dict, Optional[dict]]]:
    """Load (mock_trace, agent_trace) pairs from `episodes_dir`.

    Pairs by matching room name. If an agent trace has no matching mock trace
    (e.g. real telephony call), its mock side will be None. Same for orphan mocks.
    """
    cutoff = (time.time() - since_seconds) if since_seconds else 0
    mocks: dict[str, dict] = {}
    agents: dict[str, dict] = {}

    for path in episodes_dir.glob("ep_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping malformed mock trace %s", path)
            continue
        if data.get("started_at", 0) < cutoff:
            continue
        mocks[data.get("room", "")] = data

    for path in episodes_dir.glob("agent_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping malformed agent trace %s", path)
            continue
        if data.get("started_at", 0) < cutoff:
            continue
        agents[data.get("room", "")] = data

    pairs: list[tuple[dict, Optional[dict]]] = []
    for room, mock in mocks.items():
        pairs.append((mock, agents.get(room)))
    # Sort by mock start time so dashboard is chronological.
    pairs.sort(key=lambda p: p[0].get("started_at", 0))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to extract events
# ─────────────────────────────────────────────────────────────────────────────


def mock_events(mock: dict, type_: str) -> list[dict]:
    return [e for e in mock.get("events", []) if e.get("type") == type_]


def agent_events(agent: Optional[dict], type_: str) -> list[dict]:
    if not agent:
        return []
    return [e for e in agent.get("events", []) if e.get("type") == type_]


def agent_dtmf_sends(agent: Optional[dict]) -> list[str]:
    """Extract just the DTMF digits the agent attempted to send, in order."""
    digits = []
    for e in agent_events(agent, "dtmf_sent"):
        if e["payload"].get("success"):
            digits.append(e["payload"]["digits"])
    return digits


def agent_transcript(agent: Optional[dict]) -> str:
    """Concatenated transcript of what the agent's STT heard (the IVR/rep speaking)."""
    if not agent:
        return ""
    return "\n".join(
        e["payload"]["transcript"] for e in agent_events(agent, "agent_heard")
    )


def agent_spoken_text(agent: Optional[dict]) -> str:
    """What the agent actually said back, extracted from conversation_item events
    where role=='assistant'. Best-effort — content may be a list of parts."""
    if not agent:
        return ""
    parts = []
    for e in agent_events(agent, "conversation_item"):
        p = e["payload"]
        if p.get("role") != "assistant":
            continue
        content = p.get("content") or p.get("text_content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict) and "text" in c:
                    parts.append(c["text"])
    return " ".join(parts)


def mock_dtmf_received(mock: dict) -> list[str]:
    """Digits the mock IVR actually saw arrive (from agent's send_dtmf)."""
    return [e["payload"]["digit"] for e in mock_events(mock, "dtmf_received")]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic checks (1-7, 12)
# ─────────────────────────────────────────────────────────────────────────────


def check_picked_up(mock: dict, agent: Optional[dict]) -> CheckResult:
    has_existing = bool(mock_events(mock, "existing_participant")) or bool(
        mock_events(mock, "participant_joined")
    )
    return CheckResult(
        id="01_picked_up",
        description="Agent connected to the mock IVR",
        passed=has_existing,
        hard_fail=True,
        detail="agent participant detected"
        if has_existing
        else "no agent participant joined the room",
        failure_mode=None if has_existing else "TIMING",
    )


def _first_dtmf_at_or_before_level(mock: dict, level_event_type: str) -> Optional[str]:
    """Returns the first DTMF the mock received before or at the given level event,
    if any. Used to validate per-level menu choices."""
    level_evs = mock_events(mock, level_event_type)
    if not level_evs:
        return None
    level_ts = level_evs[0]["ts"]
    for e in mock_events(mock, "dtmf_received"):
        if e["ts"] <= level_ts:
            return e["payload"]["digit"]
    return None


def check_chose_english(mock: dict, agent: Optional[dict]) -> CheckResult:
    level_chose = mock_events(mock, "level_1_chose")
    if not level_chose:
        return CheckResult(
            id="02_chose_english",
            description="Agent pressed 1 (not 2) at L1",
            passed=False,
            hard_fail=True,
            detail="L1 never recorded a choice (timeout)",
            failure_mode="TIMING",
        )
    digit = level_chose[0]["payload"]["digit"]
    passed = digit == "1"
    return CheckResult(
        id="02_chose_english",
        description="Agent pressed 1 (not 2) at L1",
        passed=passed,
        hard_fail=True,
        detail=f"agent pressed {digit!r}",
        failure_mode=None if passed else ("TOOL_MISUSE" if digit else "TIMING"),
    )


def check_chose_prior_auth(mock: dict, agent: Optional[dict]) -> CheckResult:
    level_chose = mock_events(mock, "level_2_chose")
    if not level_chose:
        return CheckResult(
            id="03_chose_prior_auth",
            description="Agent pressed 2 (not 0/3) at L2",
            passed=False,
            hard_fail=True,
            detail="L2 never reached",
            failure_mode="TIMING",
        )
    digit = level_chose[0]["payload"]["digit"]
    passed = digit == "2"
    return CheckResult(
        id="03_chose_prior_auth",
        description="Agent pressed 2 (not 0/3) at L2",
        passed=passed,
        hard_fail=True,
        detail=f"agent pressed {digit!r}",
        failure_mode=None if passed else "TOOL_MISUSE",
    )


def check_refused_self_service(mock: dict, agent: Optional[dict]) -> CheckResult:
    level_chose = mock_events(mock, "level_3_chose")
    if not level_chose:
        return CheckResult(
            id="04_refused_self_service",
            description="Agent pressed 9 (not entered ref number) at L3",
            passed=False,
            hard_fail=True,
            detail="L3 never reached",
            failure_mode="TIMING",
        )
    digit = level_chose[0]["payload"]["digit"]
    passed = digit == "9"
    return CheckResult(
        id="04_refused_self_service",
        description="Agent pressed 9 (not entered ref number) at L3",
        passed=passed,
        hard_fail=True,
        detail=f"agent pressed {digit!r}",
        # Pressing the wrong digit at the deflection trap is the canonical
        # PROMPT_GAP failure — the prompt didn't make 'representative' salient enough.
        failure_mode=None if passed else "PROMPT_GAP",
    )


def check_npi_correct(mock: dict, agent: Optional[dict]) -> CheckResult:
    level_npi = mock_events(mock, "level_4_npi")
    if not level_npi:
        return CheckResult(
            id="05_npi_correct",
            description="Agent sent correct NPI + #",
            passed=False,
            hard_fail=True,
            detail="L4 never reached",
            failure_mode="TIMING",
        )
    received = level_npi[0]["payload"]["received"]
    expected = level_npi[0]["payload"]["expected"]
    passed = received == expected
    return CheckResult(
        id="05_npi_correct",
        description="Agent sent correct NPI + #",
        passed=passed,
        hard_fail=True,
        detail=f"received {received!r}, expected {expected!r}",
        failure_mode=None if passed else "LLM_HALLUCINATED",
    )


def check_npi_retry_calm(mock: dict, agent: Optional[dict]) -> CheckResult:
    """Only meaningful if L5 (false rejection) actually fired."""
    retries = mock_events(mock, "level_5_npi_retry")
    if not retries:
        return CheckResult(
            id="06_retried_npi_calmly",
            description="Agent re-sent same NPI after false rejection",
            passed=True,
            hard_fail=True,
            detail="L5 not triggered this episode (skipped)",
        )
    last = retries[-1]["payload"]
    passed = last["received"] == last["expected"]
    return CheckResult(
        id="06_retried_npi_calmly",
        description="Agent re-sent same NPI after false rejection",
        passed=passed,
        hard_fail=True,
        detail=f"received {last['received']!r}, expected {last['expected']!r}",
        failure_mode=None if passed else "LLM_GAVE_UP",
    )


def check_silent_during_hold(mock: dict, agent: Optional[dict]) -> CheckResult:
    """Heuristic: did the agent send any DTMF or speak between the hold-start
    prompt and the rep's greeting? We approximate by checking agent_speech_created
    events between L6 and L7 timestamps."""
    if not agent:
        return CheckResult(
            id="07_silent_during_hold",
            description="Agent stayed silent during hold music",
            passed=True,
            hard_fail=False,
            detail="no agent trace available",
        )
    hold_start = next(
        (e for e in mock_events(mock, "ivr_spoke")
         if "hold while we connect" in e["payload"]["text"].lower()),
        None,
    )
    rep_greeting = next(
        (e for e in mock_events(mock, "ivr_spoke")
         if "provider services" in e["payload"]["text"].lower() and "this is" in e["payload"]["text"].lower()),
        None,
    )
    if not hold_start or not rep_greeting:
        return CheckResult(
            id="07_silent_during_hold",
            description="Agent stayed silent during hold music",
            passed=True,
            hard_fail=False,
            detail="hold/rep boundaries not found (episode aborted before hold)",
        )
    # Find agent speech events between those mock-side timestamps. Mock and agent
    # use different `started_at`, so we use wall-clock for correlation.
    mock_start = mock["started_at"]
    agent_start = agent["started_at"]
    hold_wall = mock_start + hold_start["ts"]
    rep_wall = mock_start + rep_greeting["ts"]
    agent_spoke_during_hold = [
        e for e in agent_events(agent, "agent_speech_created")
        if hold_wall < (agent_start + e["ts"]) < rep_wall
    ]
    passed = len(agent_spoke_during_hold) == 0
    return CheckResult(
        id="07_silent_during_hold",
        description="Agent stayed silent during hold music",
        passed=passed,
        hard_fail=False,
        detail=f"{len(agent_spoke_during_hold)} speech event(s) during hold",
        failure_mode=None if passed else "PROMPT_GAP",
    )


def check_under_5min(mock: dict, agent: Optional[dict]) -> CheckResult:
    duration = (mock.get("ended_at") or mock.get("started_at", 0)) - mock.get("started_at", 0)
    passed = duration < 300
    return CheckResult(
        id="12_under_5min",
        description="Total call duration <300s",
        passed=passed,
        hard_fail=False,
        detail=f"{duration:.1f}s",
        failure_mode=None if passed else "TIMING",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge for soft checks (8-11)
# ─────────────────────────────────────────────────────────────────────────────

_judge_client: Optional[oa.OpenAI] = None
_judge_initialized = False
_judge_provider: Optional[str] = None
_judge_model: Optional[str] = None
# Once we hit a rate-limit error we short-circuit subsequent calls in the
# same process to avoid wasting time on guaranteed-to-fail requests.
# Dashboards over 20+ episodes were spending most of their time waiting
# for sequential 429s before this guard.
_judge_disabled_reason: Optional[str] = None


# Each entry maps `provider` → (base_url, api_key_env_var, default_model).
# openai/xai/groq expose OpenAI-compatible /chat/completions; anthropic uses
# the Claude Messages API and is handled separately in _call_anthropic_judge().
# Override with `JUDGE_MODEL=provider/specific-model` to change it.
_JUDGE_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai": ("https://api.openai.com/v1",      "OPENAI_API_KEY", "gpt-4o-mini"),
    "xai":    ("https://api.x.ai/v1",            "XAI_API_KEY",    "grok-4.20-fast"),
    "groq":   ("https://api.groq.com/openai/v1", "GROQ_API_KEY",   "llama-3.3-70b-versatile"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY", "claude-opus-4-7"),
}


def _resolve_judge_spec(spec: str) -> tuple[str, str]:
    """Parse a `provider/model` or `provider` spec into (provider, model)."""
    if "/" in spec:
        provider, model = spec.split("/", 1)
    else:
        provider, model = spec, ""
    provider = provider.strip().lower()
    if provider not in _JUDGE_PROVIDERS:
        raise ValueError(
            f"Unknown judge provider {provider!r}. "
            f"Valid: {list(_JUDGE_PROVIDERS.keys())}"
        )
    if not model:
        model = _JUDGE_PROVIDERS[provider][2]
    return provider, model


def _get_judge_config() -> tuple[Optional[oa.OpenAI], str, str]:
    """Lazy-init the configured judge. Returns (client, provider, model_id).

    Configuration: `JUDGE_MODEL` env var, default "anthropic/claude-opus-4-7".
    Format: "provider" or "provider/model". Valid providers in _JUDGE_PROVIDERS.

    For OpenAI-compatible providers, `client` is an oa.OpenAI instance. For
    Anthropic, `client` is None because we call the Messages API directly.
    """
    global _judge_client, _judge_initialized, _judge_provider, _judge_model
    if not _judge_initialized:
        spec = os.environ.get("JUDGE_MODEL", "anthropic/claude-opus-4-7")
        provider, model = _resolve_judge_spec(spec)
        base_url, key_env, _ = _JUDGE_PROVIDERS[provider]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"Judge configured as {spec!r} requires env var {key_env}. "
                f"Set it or change JUDGE_MODEL."
            )
        if provider != "anthropic":
            _judge_client = oa.OpenAI(api_key=api_key, base_url=base_url)
        _judge_provider = provider
        _judge_model = model
        _judge_initialized = True
        logger.info("Judge: provider=%s model=%s", provider, model)
    return _judge_client, _judge_provider or "", _judge_model or ""


_JUDGE_SYSTEM = """You are an evaluator grading a voice agent's call to an
insurance company's prior-authorization line. You will receive a specific
yes/no question and the relevant transcript snippets. Be strict but fair.

Respond ONLY with valid JSON in this exact schema:
{"passed": true|false, "evidence": "short quote from transcript", "reason": "1-2 sentence explanation"}

Do not include markdown, prose, or anything else outside the JSON."""


def _parse_judge_json(raw: str) -> dict:
    """Parse a judge JSON response with a small repair path.

    Claude generally follows the "JSON only" instruction, but if a provider
    returns surrounding whitespace or text, recover the first {...} object.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])
    return {
        "passed": bool(parsed.get("passed", False)),
        "evidence": str(parsed.get("evidence", ""))[:200],
        "reason": str(parsed.get("reason", ""))[:300],
    }


def _call_anthropic_judge(model: str, question: str, context: str) -> dict:
    """Call Claude via Anthropic's Messages API.

    We intentionally use raw httpx instead of adding the anthropic SDK: httpx is
    already in this environment for the mock IVR, and this keeps the scorer's
    dependency surface small.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Judge configured as anthropic requires env var ANTHROPIC_API_KEY."
        )
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 200,
            "system": _JUDGE_SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nTRANSCRIPT:\n{context}",
                }
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text_parts = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return _parse_judge_json("\n".join(text_parts))


def _llm_judge(question: str, context: str) -> dict:
    """Ask the configured judge the question. Returns
    {"passed": bool, "evidence": str, "reason": str}.

    Short-circuits on rate limits to avoid wasting time on guaranteed-fail calls.
    """
    global _judge_disabled_reason
    if _judge_disabled_reason is not None:
        return {
            "passed": False,
            "evidence": "",
            "reason": f"judge skipped: {_judge_disabled_reason}",
        }
    if not context.strip():
        return {"passed": False, "evidence": "", "reason": "no transcript available"}
    try:
        client, provider, model = _get_judge_config()
        if provider == "anthropic":
            return _call_anthropic_judge(model, question, context)
        if client is None:
            raise RuntimeError(f"Judge provider {provider!r} has no client")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nTRANSCRIPT:\n{context}",
                },
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        return _parse_judge_json(raw)
    except oa.RateLimitError:
        _judge_disabled_reason = (
            f"{_judge_provider}/{_judge_model} rate-limited — "
            f"swap JUDGE_MODEL to re-enable"
        )
        return {
            "passed": False,
            "evidence": "",
            "reason": f"judge skipped: {_judge_disabled_reason}",
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            _judge_disabled_reason = (
                f"{_judge_provider}/{_judge_model} rate-limited — "
                f"swap JUDGE_MODEL to re-enable"
            )
            return {
                "passed": False,
                "evidence": "",
                "reason": f"judge skipped: {_judge_disabled_reason}",
            }
        return {
            "passed": False,
            "evidence": "",
            "reason": f"judge error: Anthropic HTTP {e.response.status_code}: {e.response.text[:200]}",
        }
    except Exception as e:
        return {
            "passed": False,
            "evidence": "",
            "reason": f"judge error: {e!r}",
        }


def check_announced_to_rep(mock: dict, agent: Optional[dict]) -> CheckResult:
    """Did the agent identify itself + the patient within the first thing it said
    after the rep picked up?"""
    if not agent:
        return CheckResult(
            id="08_announced_to_rep",
            description="Agent identified self + patient when rep picked up",
            passed=False,
            hard_fail=True,
            detail="no agent trace",
            failure_mode="TIMING",
        )
    # Find the rep-greeting event in mock, then look at agent's spoken text after that wall-clock.
    rep_evs = [
        e for e in mock_events(mock, "ivr_spoke")
        if "this is" in e["payload"]["text"].lower()
        and "provider services" in e["payload"]["text"].lower()
    ]
    if not rep_evs:
        return CheckResult(
            id="08_announced_to_rep",
            description="Agent identified self + patient when rep picked up",
            passed=False,
            hard_fail=True,
            detail="L7 (rep greeting) never reached",
            failure_mode="TIMING",
        )
    spoken = agent_spoken_text(agent)
    if not spoken:
        return CheckResult(
            id="08_announced_to_rep",
            description="Agent identified self + patient when rep picked up",
            passed=False,
            hard_fail=True,
            detail="agent never spoke",
            failure_mode="LLM_GAVE_UP",
        )
    judge = _llm_judge(
        question=(
            "Did the agent introduce itself as 'Ruma Care' (or similar) AND mention "
            "the patient name 'Jane Doe' AND provide context (DOB or auth ref) within "
            "the first response after the human representative picked up? Be strict — "
            "all three are required for a pass."
        ),
        context=f"AGENT SAID:\n{spoken[:2000]}",
    )
    return CheckResult(
        id="08_announced_to_rep",
        description="Agent identified self + patient when rep picked up",
        passed=judge["passed"],
        hard_fail=True,
        detail=f"{judge['reason']} | evidence: {judge['evidence']!r}",
        failure_mode=None if judge["passed"] else "PROMPT_GAP",
    )


def check_dob_correct(mock: dict, agent: Optional[dict]) -> CheckResult:
    if not agent:
        return CheckResult(
            id="09_gave_correct_dob",
            description="Agent gave correct DOB",
            passed=False,
            hard_fail=True,
            detail="no agent trace",
            failure_mode="TIMING",
        )
    spoken = agent_spoken_text(agent)
    expected_dob = mock.get("params", {}).get("expected_dob_spoken", "March 14 1985")
    judge = _llm_judge(
        question=(
            f"Did the agent state the patient's date of birth as {expected_dob!r} "
            "(or any equivalent representation: '03/14/1985', 'three fourteen "
            "eighty five', 'March 14th 1985', etc.)?"
        ),
        context=f"AGENT SAID:\n{spoken[:2000]}",
    )
    return CheckResult(
        id="09_gave_correct_dob",
        description=f"Agent gave correct DOB ({expected_dob})",
        passed=judge["passed"],
        hard_fail=True,
        detail=f"{judge['reason']} | evidence: {judge['evidence']!r}",
        failure_mode=None if judge["passed"] else "LLM_HALLUCINATED",
    )


def check_auth_ref_correct(mock: dict, agent: Optional[dict]) -> CheckResult:
    if not agent:
        return CheckResult(
            id="10_gave_correct_auth_ref",
            description="Agent gave correct auth reference",
            passed=False,
            hard_fail=True,
            detail="no agent trace",
            failure_mode="TIMING",
        )
    spoken = agent_spoken_text(agent)
    expected_ref = mock.get("params", {}).get("expected_auth_ref", "AUTH-2026-0042")
    judge = _llm_judge(
        question=(
            f"Did the agent state the authorization reference number {expected_ref!r} "
            "(or any equivalent: 'A-U-T-H-2026-0042', 'auth twenty-twenty-six oh oh "
            "four two', etc.)?"
        ),
        context=f"AGENT SAID:\n{spoken[:2000]}",
    )
    return CheckResult(
        id="10_gave_correct_auth_ref",
        description=f"Agent gave correct auth ref ({expected_ref})",
        passed=judge["passed"],
        hard_fail=True,
        detail=f"{judge['reason']} | evidence: {judge['evidence']!r}",
        failure_mode=None if judge["passed"] else "LLM_HALLUCINATED",
    )


def check_read_back_status(mock: dict, agent: Optional[dict]) -> CheckResult:
    if not agent:
        return CheckResult(
            id="11_read_back_status",
            description="Agent read auth status back to confirm before ending",
            passed=False,
            hard_fail=True,
            detail="no agent trace",
            failure_mode="TIMING",
        )
    spoken = agent_spoken_text(agent)
    expected_status = mock.get("params", {}).get("auth_status", "approved")
    judge = _llm_judge(
        question=(
            f"After the representative said the authorization was {expected_status!r}, "
            "did the agent EXPLICITLY repeat/confirm the status back (e.g. "
            f"'so this is {expected_status}, correct?') BEFORE ending the call? "
            "Just receiving the info isn't enough — the agent must have read it back "
            "for verification."
        ),
        context=f"AGENT SAID:\n{spoken[:2000]}",
    )
    return CheckResult(
        id="11_read_back_status",
        description="Agent read auth status back before ending",
        passed=judge["passed"],
        hard_fail=True,
        detail=f"{judge['reason']} | evidence: {judge['evidence']!r}",
        failure_mode=None if judge["passed"] else "PROMPT_GAP",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level scoring
# ─────────────────────────────────────────────────────────────────────────────


ALL_CHECKS = [
    check_picked_up,
    check_chose_english,
    check_chose_prior_auth,
    check_refused_self_service,
    check_npi_correct,
    check_npi_retry_calm,
    check_silent_during_hold,
    # Human-representative phase is intentionally disabled from the benchmark
    # for now. We need a human-in-the-loop process design before optimizing
    # agent behavior against reps from many insurance companies; otherwise we
    # risk overfitting to this mock rep script. Keep these checks implemented
    # above so they can be re-enabled once the process is defined:
    #   check_announced_to_rep,
    #   check_dob_correct,
    #   check_auth_ref_correct,
    #   check_read_back_status,
    check_under_5min,
]


def score_episode(mock: dict, agent: Optional[dict]) -> EpisodeScore:
    score = EpisodeScore(
        episode_id=mock.get("episode_id", "?"),
        room=mock.get("room", "?"),
        job_id=agent.get("job_id") if agent else None,
        model=(agent.get("model", "unknown") if agent else "unknown"),
        rep_persona=mock.get("rep_persona", "?"),
        duration_s=(mock.get("ended_at") or mock.get("started_at", 0))
        - mock.get("started_at", 0),
    )
    for check_fn in ALL_CHECKS:
        try:
            score.checks.append(check_fn(mock, agent))
        except Exception as e:
            logger.exception("Check %s crashed", check_fn.__name__)
            score.checks.append(
                CheckResult(
                    id=check_fn.__name__,
                    description=f"(check crashed: {e!r})",
                    passed=False,
                    hard_fail=True,
                    detail=repr(e),
                    failure_mode="IVR_SIGNAL",
                )
            )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Output rendering
# ─────────────────────────────────────────────────────────────────────────────


def _color(text: str, code: int) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _color(t, 32)


def _red(t: str) -> str:
    return _color(t, 31)


def _yellow(t: str) -> str:
    return _color(t, 33)


def _dim(t: str) -> str:
    return _color(t, 2)


def render_episode(score: EpisodeScore, verbose: bool = False) -> None:
    status = _green("PASS") if score.hard_pass else _red("FAIL")
    hp, ht = score.hard_score
    ap, at = score.all_score
    print()
    print(
        f"{status}  {score.episode_id}  model={score.model}  rep={score.rep_persona}  "
        f"hard={hp}/{ht}  all={ap}/{at}  {score.duration_s:.0f}s"
    )
    print(f"  room={score.room}  job={score.job_id or '-'}")
    for check in score.checks:
        mark = _green("✓") if check.passed else (_red("✗") if check.hard_fail else _yellow("!"))
        kind = "" if check.hard_fail else _dim(" (warn)")
        line = f"  {mark} {check.id}: {check.description}{kind}"
        print(line)
        if verbose or not check.passed:
            print(_dim(f"      → {check.detail}"))
            if check.failure_mode and not check.passed:
                print(_dim(f"      → mode: {check.failure_mode}"))


def render_leaderboard(scores: list[EpisodeScore]) -> None:
    """Per-model summary table — the headline number for A/B comparisons."""
    if not scores:
        return
    by_model: dict[str, list[EpisodeScore]] = {}
    for s in scores:
        by_model.setdefault(s.model, []).append(s)

    print()
    print("═" * 76)
    print("  LEADERBOARD — pass rate by model")
    print("═" * 76)
    print(f"  {'model':<18} {'episodes':<10} {'pass-rate':<12} {'avg-dur':<10} "
          f"{'avg-checks':<12} {'top-failure-mode':<22}")
    print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*22}")
    # Sort by pass-rate descending so the winner is on top.
    ordered = sorted(
        by_model.items(),
        key=lambda kv: (-sum(1 for s in kv[1] if s.hard_pass) / max(1, len(kv[1])),),
    )
    for model, group in ordered:
        n = len(group)
        pass_count = sum(1 for s in group if s.hard_pass)
        pass_rate = 100 * pass_count / n
        avg_dur = sum(s.duration_s for s in group) / n
        avg_checks = sum(s.hard_score[0] for s in group) / n
        hard_total = group[0].hard_score[1] if group else 0
        # Find the most common failure mode across this group.
        mode_counts: dict[str, int] = {}
        for s in group:
            for c in s.checks:
                if not c.passed and c.failure_mode:
                    mode_counts[c.failure_mode] = mode_counts.get(c.failure_mode, 0) + 1
        top_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "(none)"
        print(
            f"  {model:<18} {n:<10} "
            f"{pass_count}/{n} ({pass_rate:>3.0f}%)  "
            f"{avg_dur:>5.0f}s    "
            f"{avg_checks:>4.1f}/{hard_total:<2}    "
            f"{top_mode:<22}"
        )
    print()


def render_aggregate(scores: list[EpisodeScore]) -> None:
    if not scores:
        print("No episodes to score.")
        return
    n = len(scores)
    passed = sum(1 for s in scores if s.hard_pass)
    print()
    print("═" * 76)
    print(f"  AGGREGATE  {passed}/{n} episodes passed all hard checks ({100*passed/n:.0f}%)")
    print("═" * 76)

    # Per-check pass rate.
    check_stats: dict[str, dict] = {}
    for s in scores:
        for c in s.checks:
            stat = check_stats.setdefault(
                c.id, {"passed": 0, "total": 0, "hard": c.hard_fail, "modes": {}}
            )
            stat["total"] += 1
            if c.passed:
                stat["passed"] += 1
            elif c.failure_mode:
                stat["modes"][c.failure_mode] = stat["modes"].get(c.failure_mode, 0) + 1

    print()
    print("  Per-check pass rate:")
    for cid in sorted(check_stats.keys()):
        s = check_stats[cid]
        rate = 100 * s["passed"] / s["total"]
        kind = "    " if s["hard"] else "warn"
        bar = ("█" * int(rate / 5)).ljust(20)
        modes = ""
        if s["modes"]:
            mode_str = " ".join(f"{m}×{n}" for m, n in s["modes"].items())
            modes = _dim(f"  [{mode_str}]")
        print(f"  {kind}  {cid:<28} {bar} {rate:5.0f}%  ({s['passed']}/{s['total']}){modes}")

    # Failure-mode summary.
    print()
    print("  Failure modes (across all failed checks):")
    mode_totals: dict[str, int] = {}
    for s in scores:
        for c in s.checks:
            if not c.passed and c.failure_mode:
                mode_totals[c.failure_mode] = mode_totals.get(c.failure_mode, 0) + 1
    if not mode_totals:
        print(_dim("    (none — everything passing)"))
    else:
        for mode, count in sorted(mode_totals.items(), key=lambda x: -x[1]):
            print(f"    {mode:<22} ×{count}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_since(arg: str) -> float:
    """'1h' → 3600.0 ; '15m' → 900.0 ; '30s' → 30.0 ; bare number → seconds."""
    if not arg:
        return 0
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if arg[-1] in mult:
        return float(arg[:-1]) * mult[arg[-1]]
    return float(arg)


def main() -> None:
    p = argparse.ArgumentParser(description="Score voice-agent episodes against rubric")
    p.add_argument("--since", default="", help="only score episodes from the last N (e.g. 1h, 30m)")
    p.add_argument("--episode", default="", help="score one specific episode_id")
    p.add_argument("--model", default="", help="only score episodes from a specific model (grok-voice/gpt-rt2/cascaded)")
    p.add_argument("--verbose", action="store_true", help="show details for passing checks too")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--leaderboard", action="store_true", help="show per-model comparison table")
    p.add_argument(
        "--judge",
        default="",
        help="LLM judge override, e.g. 'anthropic/claude-opus-4-7', 'openai/gpt-4o-mini', "
        "'xai', or 'groq/llama-3.3-70b-versatile'. Same effect as setting JUDGE_MODEL. "
        "Default: anthropic/claude-opus-4-7.",
    )
    args = p.parse_args()

    if args.judge:
        # Push the override into the env so the judge factory picks it up,
        # and reset the lazy-init globals in case score_episodes was already
        # imported in this process (matters for dashboard.py which imports us).
        global _judge_client, _judge_initialized, _judge_provider, _judge_model, _judge_disabled_reason
        os.environ["JUDGE_MODEL"] = args.judge
        _judge_client = None
        _judge_initialized = False
        _judge_provider = None
        _judge_model = None
        _judge_disabled_reason = None

    since = parse_since(args.since) if args.since else None
    pairs = load_episodes(EPISODES_DIR, since_seconds=since)

    if args.episode:
        pairs = [p for p in pairs if p[0].get("episode_id") == args.episode]

    if args.model:
        pairs = [p for p in pairs if (p[1] and p[1].get("model") == args.model)]

    if not pairs:
        print("No episodes found.", file=sys.stderr)
        sys.exit(1)

    scores: list[EpisodeScore] = []
    for mock, agent in pairs:
        score = score_episode(mock, agent)
        scores.append(score)
        if not args.json:
            render_episode(score, verbose=args.verbose)

    if args.json:
        out = [
            {
                "episode_id": s.episode_id,
                "room": s.room,
                "job_id": s.job_id,
                "model": s.model,
                "rep_persona": s.rep_persona,
                "duration_s": s.duration_s,
                "hard_pass": s.hard_pass,
                "hard_score": list(s.hard_score),
                "all_score": list(s.all_score),
                "checks": [
                    {
                        "id": c.id,
                        "passed": c.passed,
                        "hard_fail": c.hard_fail,
                        "detail": c.detail,
                        "failure_mode": c.failure_mode,
                    }
                    for c in s.checks
                ],
            }
            for s in scores
        ]
        print(json.dumps(out, indent=2))
    else:
        render_aggregate(scores)
        if args.leaderboard or len({s.model for s in scores}) > 1:
            # Auto-show the leaderboard if there's more than one model in the
            # results — that's the whole point of the multi-model harness.
            render_leaderboard(scores)


if __name__ == "__main__":
    main()
