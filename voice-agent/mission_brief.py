"""
Mission brief schema — the structured context dispatched with every agent job.

Why a schema (vs. just a dict):
  - Validation: catch bad NPIs / missing CPTs at dispatch time, not mid-call
  - Documentation: the source of truth for what the agent needs
  - Prompt rendering: a single .render_prompt() method keeps system-prompt
    construction in one place, so prompt-engineering iterations don't drift
    from the underlying brief shape

Used by:
  - agent.py entrypoint() — parses ctx.job.metadata into MissionBrief
  - MissionAgent.__init__ — receives the brief and renders the prompt
  - dispatch metadata — JSON-serialized MissionBrief sent in `lk dispatch create`

Wire format (what goes in --metadata):
  {
    "mock": false,                              # true for mock IVR episodes
    "phone_number": "+1XXXXXXXXXX",             # required for telephony jobs
    "supervisor_number": "+1XXXXXXXXXX",        # for SIP REFER warm transfers (Phase 2.6)
    "payer_name": "MockHealth",
    "patient": {
      "name": "Jane Doe",
      "dob": "1985-03-14",                      # ISO date
      "member_id": "W123456789"
    },
    "provider": {
      "name": "Dr. Smith",
      "npi": "1234567890",                      # 10 digits, validated
      "tax_id": "12-3456789"
    },
    "auth": {
      "reference_number": "AUTH-2026-0042",
      "service_date": "2026-05-20",             # ISO date
      "cpt_codes": ["99213"]
    }
  }
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Patient(BaseModel):
    name: str
    dob: date
    member_id: Optional[str] = None

    def dob_spoken(self) -> str:
        """Return DOB in human-friendly spoken form: 'March fourteenth nineteen eighty five'."""
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        ordinals_low = {
            1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
            19: "nineteenth", 20: "twentieth", 21: "twenty-first",
            22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth",
            25: "twenty-fifth", 26: "twenty-sixth", 27: "twenty-seventh",
            28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
            31: "thirty-first",
        }
        # Year: split into "nineteen eighty five" form for years 1900-1999, "two thousand X" for 2000s
        y = self.dob.year
        if 1900 <= y <= 1999:
            tens = (y // 10) % 10
            ones = y % 10
            tens_words = ["", "", "twenty", "thirty", "forty", "fifty",
                          "sixty", "seventy", "eighty", "ninety"]
            ones_words = ["", "one", "two", "three", "four", "five",
                          "six", "seven", "eight", "nine"]
            year_str = f"nineteen {tens_words[tens]} {ones_words[ones]}".strip().replace("  ", " ")
        else:
            year_str = str(y)  # fallback for unusual years
        return f"{months[self.dob.month - 1]} {ordinals_low[self.dob.day]} {year_str}".strip()

    def dob_digits(self) -> str:
        """Return DOB as 8-digit MMDDYYYY for DTMF entry."""
        return self.dob.strftime("%m%d%Y")


class Provider(BaseModel):
    name: str
    npi: str = Field(..., description="10-digit National Provider Identifier")
    tax_id: Optional[str] = None

    @field_validator("npi")
    @classmethod
    def _check_npi(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if len(digits) != 10:
            raise ValueError(f"NPI must be 10 digits, got {len(digits)}: {v!r}")
        return digits  # normalize: strip any dashes/spaces


class Auth(BaseModel):
    reference_number: str
    service_date: Optional[date] = None
    cpt_codes: list[str] = Field(default_factory=list)

    def reference_spoken(self) -> str:
        """Spell out the auth reference for clarity over a noisy phone line.

        'AUTH-2026-0042' → 'A U T H dash 2 0 2 6 dash 0 0 4 2'
        """
        out = []
        for ch in self.reference_number:
            if ch == "-":
                out.append("dash")
            elif ch.isdigit():
                out.append(ch)
            else:
                out.append(ch.upper())
        return " ".join(out)


class MissionBrief(BaseModel):
    """Top-level brief dispatched with every job."""

    mock: bool = False  # True for mock-IVR episodes (no SIP dial)
    phone_number: Optional[str] = None
    supervisor_number: Optional[str] = None
    payer_name: str = "the payer"  # fallback if not provided

    # Model selection — controls which speech backend agent.py uses.
    # Values: "gpt-rt2"    (OpenAI gpt-realtime-2; primary/default)
    #         "grok-voice" (xAI grok-voice-think-fast-1.0; opt-in research path)
    #         "cascaded"   (Deepgram-STT + Groq-Llama + Deepgram-TTS; fallback)
    # Defaults to GPT-Realtime-2 because the current harness shows it reliably
    # passes IVR navigation (L1-L6) and only needs human-rep phase hardening.
    model: str = "gpt-rt2"

    patient: Patient
    provider: Provider
    auth: Auth

    @model_validator(mode="after")
    def _check_phone_or_mock(self) -> "MissionBrief":
        # A real telephony job needs a phone_number. A mock job needs mock=True.
        # We allow both to be unset for fully-local smoke tests.
        if self.phone_number and self.mock:
            raise ValueError(
                "Mission brief has both phone_number and mock=true; pick one."
            )
        return self

    @classmethod
    def from_metadata(cls, metadata_json: str) -> "MissionBrief":
        """Parse a JSON string from job.metadata into a validated MissionBrief.

        Tolerates the legacy minimal form `{"phone_number": "+1..."}` and
        `{"mock": true}` by filling in defaults — useful while we transition
        from Phase 1's bare metadata to the full brief.
        """
        raw = json.loads(metadata_json or "{}")

        # Legacy/test compatibility: if patient/provider/auth aren't provided,
        # fall back to the test fixture so old dispatch commands still work.
        # Phase 2.3+ should always send full briefs.
        if "patient" not in raw or "provider" not in raw or "auth" not in raw:
            raw.setdefault("patient", {
                "name": "Jane Doe",
                "dob": "1985-03-14",
                "member_id": "W123456789",
            })
            raw.setdefault("provider", {
                "name": "Dr. Smith",
                "npi": "1234567890",
                "tax_id": "12-3456789",
            })
            raw.setdefault("auth", {
                "reference_number": "AUTH-2026-0042",
                "service_date": "2026-05-20",
                "cpt_codes": ["99213"],
            })
            raw.setdefault("payer_name", "MockHealth")

        return cls.model_validate(raw)
