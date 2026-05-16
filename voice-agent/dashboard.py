"""
HTML dashboard for voice-agent eval runs.

Renders a single self-contained HTML file with:
  - Leaderboard: per-model pass rate, avg checks, top failure mode
  - Per-check matrix: rows = checks 01..N, cols = models, cells = pass-rate %
  - Per-episode cards: collapsible per-episode details with
      * check-by-check results
      * collapsible agent trace (STT transcripts, tool calls, state changes)
      * collapsible mock-IVR trace (prompts, DTMF received, agent-speech detect)
      * raw JSON file links

Reuses score_episodes.load_episodes() + score_episode() so the scoring logic
stays in one place. If the Groq judge is rate-limited (which we've hit), the
LLM-judged checks will appear with reason="judge error: ..." — the dashboard
shows that exactly as the scorer recorded it so we can tell the difference
between agent failure and infrastructure failure at a glance.

Usage:
    python dashboard.py                      # writes dashboard.html
    python dashboard.py --since 1h           # only recent runs
    python dashboard.py --out report.html    # custom path
    python dashboard.py --open               # also open in default browser
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

from score_episodes import (
    EPISODES_DIR,
    EpisodeScore,
    load_episodes,
    parse_since,
    score_episode,
)


# ─────────────────────────────────────────────────────────────────────────────
# Styling — kept inline so the HTML file is portable (no CDN, no fonts)
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body {
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fafafa;
  color: #1a1a1a;
  margin: 0;
  padding: 32px;
  max-width: 1200px;
  margin: 0 auto;
}
h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px 0; }
h2 { font-size: 16px; font-weight: 600; margin: 32px 0 12px 0; color: #333; }
.meta { color: #888; font-size: 12px; margin-bottom: 24px; }
.section { background: white; border: 1px solid #e5e5e5; border-radius: 8px; padding: 20px; margin-bottom: 16px; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }
th { font-weight: 600; color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; background: #fafafa; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.model { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 500; }

/* Pass-rate cells */
.rate { display: inline-block; min-width: 50px; padding: 2px 8px; border-radius: 4px; font-variant-numeric: tabular-nums; font-size: 12px; font-weight: 500; }
.rate-100 { background: #d1f5d3; color: #0a5c0e; }
.rate-good { background: #e8f7ea; color: #1d6624; }
.rate-mid  { background: #fff4d6; color: #7a5800; }
.rate-poor { background: #fde0e0; color: #8a1818; }
.rate-zero { background: #f3d3d3; color: #6b0e0e; }
.rate-na   { background: #f0f0f0; color: #888; }

/* Matrix table — checks × models */
.matrix th, .matrix td { text-align: center; }
.matrix th:first-child, .matrix td:first-child { text-align: left; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }

/* Episode cards */
details.episode { background: white; border: 1px solid #e5e5e5; border-radius: 8px; margin-bottom: 8px; }
details.episode > summary {
  cursor: pointer;
  padding: 12px 16px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 12px;
  list-style: none;
}
details.episode > summary::-webkit-details-marker { display: none; }
details.episode > summary::before { content: "▸"; color: #999; transition: transform 0.1s; display: inline-block; width: 10px; }
details.episode[open] > summary::before { transform: rotate(90deg); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.badge-pass { background: #d1f5d3; color: #0a5c0e; }
.badge-fail { background: #fde0e0; color: #8a1818; }
.summary-meta { color: #888; font-size: 12px; margin-left: auto; }

.episode-body { padding: 0 16px 16px 30px; }
.check-list { list-style: none; padding: 0; margin: 16px 0; }
.check-list li { padding: 6px 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.check-pass { color: #0a5c0e; }
.check-fail { color: #8a1818; }
.check-warn { color: #7a5800; }
.check-detail { color: #666; font-size: 11px; margin-left: 18px; }
.failure-mode { display: inline-block; background: #f0f0f0; padding: 1px 6px; border-radius: 3px; font-size: 10px; color: #666; margin-left: 6px; }

/* Trace viewer — collapsible nested details */
details.trace { margin: 8px 0; }
details.trace > summary {
  cursor: pointer;
  font-size: 12px;
  color: #555;
  font-weight: 500;
  padding: 6px 8px;
  background: #f5f5f5;
  border-radius: 4px;
  list-style: none;
}
details.trace > summary::-webkit-details-marker { display: none; }
details.trace > summary::before { content: "▸"; color: #999; margin-right: 6px; display: inline-block; width: 8px; }
details.trace[open] > summary::before { transform: rotate(90deg); }

.trace-timeline {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  line-height: 1.5;
  max-height: 400px;
  overflow-y: auto;
  background: #fafafa;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 8px 10px;
  margin-top: 6px;
}
.event { display: grid; grid-template-columns: 50px 110px 1fr; gap: 8px; padding: 2px 0; }
.event-ts { color: #999; text-align: right; }
.event-type { color: #555; font-weight: 500; }
.event-payload { color: #333; word-break: break-word; }

.event-tool { background: #fff8e1; }
.event-dtmf { background: #e3f2fd; }
.event-ivr  { background: #f3e5f5; }
.event-heard { background: #e8f5e9; }
.event-error { background: #ffebee; color: #b71c1c; }
.event-end { background: #e0f2f1; }

.raw-link { font-size: 11px; color: #1976d2; text-decoration: none; margin-right: 12px; }
.raw-link:hover { text-decoration: underline; }

/* Transcript viewer */
.transcript {
  background: #fafafa;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 10px 12px;
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.6;
  max-height: 300px;
  overflow-y: auto;
}
.turn { margin-bottom: 8px; }
.turn-role { font-weight: 600; color: #555; font-size: 11px; text-transform: uppercase; }
.turn-content { color: #1a1a1a; }
.turn-dtmf .turn-role { color: #b8860b; }
.turn-dtmf .turn-content { font-family: monospace; background: #fffbe6; padding: 2px 6px; border-radius: 3px; display: inline-block; border: 1px solid #f0e0a0; }
.turn-dtmf .turn-content.dtmf-error { background: #ffe6e6; border-color: #f0a0a0; color: #a00; }
.turn-ts { display: inline-block; width: 50px; font-family: monospace; font-size: 10px; color: #999; }

/* Replay viewer */
.replay {
  background: #fbfbfb;
  border: 1px solid #e8e8e8;
  border-radius: 6px;
  margin-top: 6px;
  overflow: hidden;
}
.replay-row {
  display: grid;
  grid-template-columns: 62px 110px 1fr;
  gap: 10px;
  padding: 8px 10px;
  border-bottom: 1px solid #eee;
}
.replay-row:last-child { border-bottom: none; }
.replay-ts {
  color: #999;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  text-align: right;
}
.replay-label {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.replay-text { color: #222; word-break: break-word; }
.replay-ivr .replay-label { color: #6a1b9a; }
.replay-heard .replay-label { color: #2e7d32; }
.replay-agent .replay-label { color: #1565c0; }
.replay-dtmf .replay-label { color: #b8860b; }
.replay-tool .replay-label { color: #795548; }
.replay-error .replay-label { color: #b71c1c; }
.replay-wait .replay-label, .replay-wait .replay-text { color: #888; }
.replay-digits {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #fffbe6;
  border: 1px solid #f0e0a0;
  border-radius: 3px;
  padding: 1px 6px;
}

/* Audio recording */
.recording-player {
  background: #fafafa;
  border: 1px solid #eee;
  border-radius: 6px;
  padding: 10px 12px;
  margin-top: 6px;
}
.recording-player audio { width: 100%; margin: 6px 0; }
.recording-meta { color: #777; font-size: 11px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Rate-color helper — gives the cell its background
# ─────────────────────────────────────────────────────────────────────────────


def rate_class(passed: int, total: int) -> str:
    if total == 0:
        return "rate-na"
    pct = 100 * passed / total
    if pct == 100:
        return "rate-100"
    if pct >= 75:
        return "rate-good"
    if pct >= 50:
        return "rate-mid"
    if pct > 0:
        return "rate-poor"
    return "rate-zero"


def fmt_rate(passed: int, total: int) -> str:
    if total == 0:
        return '<span class="rate rate-na">n/a</span>'
    pct = 100 * passed / total
    return f'<span class="rate {rate_class(passed, total)}">{passed}/{total} ({pct:.0f}%)</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────────────────────


def render_leaderboard(scores: list[EpisodeScore]) -> str:
    """Top section: one row per model, sorted by pass rate desc."""
    by_model: dict[str, list[EpisodeScore]] = {}
    for s in scores:
        by_model.setdefault(s.model, []).append(s)

    rows = []
    for model, group in sorted(
        by_model.items(),
        key=lambda kv: -(sum(1 for s in kv[1] if s.hard_pass) / max(1, len(kv[1]))),
    ):
        n = len(group)
        passed = sum(1 for s in group if s.hard_pass)
        avg_dur = sum(s.duration_s for s in group) / n
        avg_checks = sum(s.hard_score[0] for s in group) / n
        hard_total = group[0].hard_score[1] if group else 0
        # Top failure mode for this model.
        mode_counts: dict[str, int] = {}
        for s in group:
            for c in s.checks:
                if not c.passed and c.failure_mode:
                    mode_counts[c.failure_mode] = mode_counts.get(c.failure_mode, 0) + 1
        top_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "—"
        rows.append(
            f"<tr>"
            f"<td class='model'>{html.escape(model)}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{fmt_rate(passed, n)}</td>"
            f"<td class='num'>{avg_checks:.1f}/{hard_total}</td>"
            f"<td class='num'>{avg_dur:.0f}s</td>"
            f"<td><span class='failure-mode'>{html.escape(top_mode)}</span></td>"
            f"</tr>"
        )

    return (
        "<div class='section'>"
        "<h2>Leaderboard</h2>"
        "<table>"
        "<thead><tr>"
        "<th>Model</th><th class='num'>N</th><th class='num'>Pass rate</th>"
        "<th class='num'>Avg checks</th><th class='num'>Avg dur</th>"
        "<th>Top failure</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_check_matrix(scores: list[EpisodeScore]) -> str:
    """Per-check pass rate, models as columns."""
    by_model: dict[str, list[EpisodeScore]] = {}
    for s in scores:
        by_model.setdefault(s.model, []).append(s)
    models = sorted(by_model.keys())

    # Collect every check id in deterministic order (by id prefix number).
    all_check_ids: list[tuple[str, str, bool]] = []
    seen = set()
    for s in scores:
        for c in s.checks:
            if c.id in seen:
                continue
            seen.add(c.id)
            all_check_ids.append((c.id, c.description, c.hard_fail))
    all_check_ids.sort(key=lambda x: x[0])

    header = "<tr><th>Check</th>" + "".join(
        f"<th>{html.escape(m)}</th>" for m in models
    ) + "</tr>"

    rows = []
    for cid, desc, hard in all_check_ids:
        cells = []
        for m in models:
            group = by_model[m]
            p = sum(1 for s in group for c in s.checks if c.id == cid and c.passed)
            t = sum(1 for s in group for c in s.checks if c.id == cid)
            cells.append(f"<td>{fmt_rate(p, t)}</td>")
        kind = "" if hard else " <span class='failure-mode'>warn</span>"
        rows.append(
            f"<tr><td>{html.escape(cid)} · {html.escape(desc)}{kind}</td>"
            f"{''.join(cells)}</tr>"
        )

    return (
        "<div class='section'>"
        "<h2>Per-check pass rate</h2>"
        "<table class='matrix'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode card with collapsible traces
# ─────────────────────────────────────────────────────────────────────────────


def _event_class(ev_type: str) -> str:
    """Color-code event rows in the trace timeline."""
    if ev_type in ("tools_executed", "dtmf_sent"):
        return "event-tool"
    if ev_type in ("dtmf_received",):
        return "event-dtmf"
    if ev_type in ("ivr_spoke", "ivr_aborted"):
        return "event-ivr"
    if ev_type == "agent_heard":
        return "event-heard"
    if ev_type in ("session_error", "episode_aborted"):
        return "event-error"
    if ev_type in ("session_closed", "ended"):
        return "event-end"
    return ""


def render_trace_timeline(events: list[dict]) -> str:
    """Render a list of trace events as a colored, scrollable timeline."""
    if not events:
        return "<div class='trace-timeline'><i style='color:#999'>(no events)</i></div>"
    rows = []
    for ev in events:
        ts = ev.get("ts", 0)
        et = ev.get("type", "?")
        payload = ev.get("payload", {})
        # Try to summarize the payload in one line.
        if et == "agent_heard":
            text = payload.get("transcript", "")
            summary = html.escape(text[:200])
        elif et == "ivr_spoke":
            text = payload.get("text", "")
            summary = html.escape(text[:200])
        elif et == "dtmf_received":
            summary = f"<b>{html.escape(payload.get('digit', '?'))}</b>"
        elif et == "dtmf_sent":
            digits = payload.get("digits", "?")
            ok = payload.get("success")
            err = payload.get("error")
            summary = f"<b>{html.escape(digits)}</b>"
            if not ok:
                summary += f" <span style='color:#c00'>(failed: {html.escape(str(err))})</span>"
        elif et == "tools_executed":
            calls = payload.get("calls", [])
            summary_parts = []
            for c in calls:
                name = c.get("name", "?")
                args = c.get("arguments", "")
                out = c.get("output", "")
                summary_parts.append(
                    f"<b>{html.escape(name)}</b>({html.escape(str(args)[:80])}) → {html.escape(str(out)[:80])}"
                )
            summary = " · ".join(summary_parts)
        elif et == "conversation_item":
            role = payload.get("role", "?")
            content = payload.get("text_content") or payload.get("content") or ""
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            summary = f"<b>{html.escape(role)}:</b> {html.escape(str(content)[:200])}"
        elif et == "session_error":
            err = payload.get("error", "")
            # Strip the giant LLMError wrapping for readability.
            short = str(err)
            if "RateLimitError" in short:
                short = "RateLimitError (Groq tokens-per-day exhausted)"
            elif len(short) > 200:
                short = short[:200] + "…"
            summary = html.escape(short)
        elif et == "episode_aborted":
            summary = html.escape(payload.get("reason", "?"))
        else:
            # Generic payload truncation.
            summary = html.escape(json.dumps(payload, default=str)[:200])
        cls = _event_class(et)
        rows.append(
            f"<div class='event {cls}'>"
            f"<span class='event-ts'>{ts:.2f}s</span>"
            f"<span class='event-type'>{html.escape(et)}</span>"
            f"<span class='event-payload'>{summary}</span>"
            f"</div>"
        )
    return f"<div class='trace-timeline'>{''.join(rows)}</div>"


def render_transcript(events: list[dict]) -> str:
    """Conversational turns interleaved with DTMF presses, sorted by ts."""
    turns = []
    for ev in events:
        t = ev.get("type")
        p = ev.get("payload", {})
        ts = ev.get("ts", 0)
        if t == "dtmf_sent":
            digits = str(p.get("digits", ""))
            err = p.get("error") if not p.get("success", True) else None
            label = "DTMF (failed)" if err else "DTMF"
            turns.append((ts, "dtmf", label, digits, err))
            continue
        if t != "conversation_item":
            continue
        role = p.get("role")
        if role not in ("user", "assistant"):
            continue
        content = p.get("text_content") or p.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if not content:
            continue
        turns.append((ts, "speech", role, str(content), None))
    turns.sort(key=lambda x: x[0])
    rendered = []
    for ts, kind, label, content, err in turns:
        ts_str = f"<span class='turn-ts'>{ts:6.2f}s</span>"
        if kind == "dtmf":
            err_cls = " dtmf-error" if err else ""
            err_str = f" ({html.escape(str(err))})" if err else ""
            inner = f"press {html.escape(content)}{err_str}"
            rendered.append(
                f"<div class='turn turn-dtmf'>{ts_str}"
                f"<span class='turn-role'>{html.escape(label)}</span> "
                f"<span class='turn-content{err_cls}'>{inner}</span></div>"
            )
        else:
            rendered.append(
                f"<div class='turn'>{ts_str}"
                f"<span class='turn-role'>{html.escape(label)}</span> "
                f"<span class='turn-content'>{html.escape(content)}</span></div>"
            )
    if not rendered:
        return "<div class='transcript'><i style='color:#999'>(no conversation items)</i></div>"
    return f"<div class='transcript'>{''.join(rendered)}</div>"


def _text_from_content(content: object) -> str:
    """Normalize LiveKit trace content fields into displayable text."""
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c is not None)
    return str(content or "")


def _shorten(text: str, limit: int = 420) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _event_abs_ts(trace: dict, ev: dict) -> float:
    return float(trace.get("started_at") or 0) + float(ev.get("ts") or 0)


def _collect_replay_events(mock_trace: dict, agent_trace: Optional[dict]) -> list[dict]:
    """Merge mock and agent trace events into a human-readable call replay.

    The mock and agent traces each use their own relative clock. Convert both
    back to absolute epoch-ish time first, then display relative to the first
    observed event so the order matches what happened in the LiveKit room.
    """
    raw: list[dict] = []

    for ev in mock_trace.get("events", []):
        et = ev.get("type")
        payload = ev.get("payload", {})
        if et == "ivr_spoke":
            raw.append(
                {
                    "abs_ts": _event_abs_ts(mock_trace, ev),
                    "kind": "ivr",
                    "label": "IVR",
                    "text": payload.get("text", ""),
                }
            )
        elif et == "dtmf_received":
            raw.append(
                {
                    "abs_ts": _event_abs_ts(mock_trace, ev),
                    "kind": "dtmf",
                    "label": "DTMF GOT",
                    "text": str(payload.get("digit", "?")),
                    "digits": True,
                }
            )
        elif et in ("ivr_aborted", "episode_aborted"):
            raw.append(
                {
                    "abs_ts": _event_abs_ts(mock_trace, ev),
                    "kind": "error",
                    "label": "MOCK ERROR",
                    "text": payload.get("reason") or json.dumps(payload, default=str),
                }
            )

    if agent_trace:
        for ev in agent_trace.get("events", []):
            et = ev.get("type")
            payload = ev.get("payload", {})
            if et == "agent_heard":
                raw.append(
                    {
                        "abs_ts": _event_abs_ts(agent_trace, ev),
                        "kind": "heard",
                        "label": "AGENT HEARD",
                        "text": payload.get("transcript", ""),
                    }
                )
            elif et == "conversation_item" and payload.get("role") == "assistant":
                text = payload.get("text_content") or payload.get("content") or ""
                raw.append(
                    {
                        "abs_ts": _event_abs_ts(agent_trace, ev),
                        "kind": "agent",
                        "label": "AGENT SAID",
                        "text": _text_from_content(text),
                    }
                )
            elif et == "dtmf_sent":
                raw.append(
                    {
                        "abs_ts": _event_abs_ts(agent_trace, ev),
                        "kind": "dtmf",
                        "label": "DTMF SENT",
                        "text": str(payload.get("digits", "?")),
                        "digits": True,
                        "error": payload.get("error") if not payload.get("success", True) else None,
                    }
                )
            elif et == "tools_executed":
                for call in payload.get("calls", []):
                    # DTMF calls are already rendered from dtmf_sent, which is
                    # easier to scan and includes success/error metadata.
                    if call.get("name") == "send_dtmf":
                        continue
                    raw.append(
                        {
                            "abs_ts": _event_abs_ts(agent_trace, ev),
                            "kind": "tool",
                            "label": "TOOL",
                            "text": (
                                f"{call.get('name', '?')}({call.get('arguments', '')})"
                                f" -> {call.get('output', '')}"
                            ),
                            "error": call.get("is_error"),
                        }
                    )
            elif et in ("session_error", "episode_aborted"):
                raw.append(
                    {
                        "abs_ts": _event_abs_ts(agent_trace, ev),
                        "kind": "error",
                        "label": "AGENT ERROR",
                        "text": payload.get("error") or payload.get("reason") or json.dumps(payload, default=str),
                    }
                )

    raw = [ev for ev in raw if ev.get("text")]
    raw.sort(key=lambda ev: ev["abs_ts"])
    if not raw:
        return []

    base_ts = raw[0]["abs_ts"]
    replay: list[dict] = []
    prev_rel = 0.0
    for ev in raw:
        rel = ev["abs_ts"] - base_ts
        if replay and rel - prev_rel >= 12:
            replay.append(
                {
                    "ts": prev_rel,
                    "kind": "wait",
                    "label": "WAIT",
                    "text": f"{rel - prev_rel:.0f}s of silence/processing/hold",
                }
            )
        item = dict(ev)
        item["ts"] = rel
        replay.append(item)
        prev_rel = rel
    return replay


def render_replay(mock_trace: dict, agent_trace: Optional[dict]) -> str:
    """Render a merged call replay for fast visual debugging."""
    events = _collect_replay_events(mock_trace, agent_trace)
    if not events:
        return "<div class='replay'><div class='replay-row'><span></span><span></span><span class='replay-text' style='color:#999'>(no replayable events)</span></div></div>"

    rows = []
    for ev in events:
        kind = html.escape(str(ev.get("kind", "")))
        label = html.escape(str(ev.get("label", "?")))
        text = _shorten(str(ev.get("text", "")))
        if ev.get("error"):
            text = f"{text} (error: {ev.get('error')})"
        content = (
            f"<span class='replay-digits'>{html.escape(text)}</span>"
            if ev.get("digits")
            else html.escape(text)
        )
        rows.append(
            f"<div class='replay-row replay-{kind}'>"
            f"<span class='replay-ts'>{float(ev.get('ts', 0)):.2f}s</span>"
            f"<span class='replay-label'>{label}</span>"
            f"<span class='replay-text'>{content}</span>"
            f"</div>"
        )
    return f"<div class='replay'>{''.join(rows)}</div>"


def render_recording_player(mock_trace: dict) -> str:
    """Render an audio player when a mock episode has a local recording."""
    recording = mock_trace.get("recording") or {}
    path = str(recording.get("path") or "")
    if not path:
        return (
            "<div class='recording-player'>"
            "<div class='recording-meta'>No audio recording for this episode.</div>"
            "</div>"
        )
    channels = recording.get("channels") or {}
    left = channels.get("left", "left")
    right = channels.get("right", "right")
    return (
        "<div class='recording-player'>"
        f"<audio controls preload='metadata' src='{html.escape(path)}'></audio>"
        f"<div class='recording-meta'>"
        f"Stereo WAV · left: {html.escape(str(left))} · right: {html.escape(str(right))} · "
        f"<code>{html.escape(path)}</code>"
        f"</div>"
        "</div>"
    )


def render_episode_card(
    score: EpisodeScore,
    mock_trace: dict,
    agent_trace: Optional[dict],
) -> str:
    badge = (
        "<span class='badge badge-pass'>PASS</span>"
        if score.hard_pass
        else "<span class='badge badge-fail'>FAIL</span>"
    )
    hp, ht = score.hard_score
    summary = (
        f"<summary>"
        f"{badge}"
        f"<span>{html.escape(score.model)}</span>"
        f"<span style='color:#666'>· {html.escape(score.rep_persona)}</span>"
        f"<span style='color:#999'>· {html.escape(score.episode_id)}</span>"
        f"<span class='summary-meta'>{hp}/{ht} hard checks · {score.duration_s:.0f}s</span>"
        f"</summary>"
    )

    # Check list.
    check_items = []
    for c in score.checks:
        if c.passed:
            mark = "✓"
            cls = "check-pass"
        elif c.hard_fail:
            mark = "✗"
            cls = "check-fail"
        else:
            mark = "!"
            cls = "check-warn"
        mode = (
            f"<span class='failure-mode'>{html.escape(c.failure_mode)}</span>"
            if c.failure_mode and not c.passed
            else ""
        )
        detail = (
            f"<div class='check-detail'>{html.escape(c.detail)}</div>"
            if c.detail and (not c.passed or c.hard_fail)
            else ""
        )
        check_items.append(
            f"<li class='{cls}'>{mark} {html.escape(c.id)} · {html.escape(c.description)} {mode}{detail}</li>"
        )
    check_list = f"<ul class='check-list'>{''.join(check_items)}</ul>"

    # Raw file links — point to local paths (open in browser if served, else show path).
    raw_links = []
    raw_links.append(
        f"<a class='raw-link' href='{html.escape('episodes/' + (mock_trace.get('episode_id') or '') + '.json' if False else '')}'>"
        f"(mock & agent JSON paths shown below)</a>"
    )
    mock_path = f"episodes/ep_{mock_trace.get('episode_id', '?')}.json"
    if mock_path:
        raw_links_html = (
            f"<div style='margin-top:12px;font-size:11px;color:#888'>"
            f"<b>raw traces:</b><br>"
            f"&nbsp;&nbsp;mock&nbsp; <code>{html.escape(mock_path)}</code><br>"
        )
        if agent_trace:
            agent_path = f"episodes/agent_{agent_trace.get('job_id', '?')}.json"
            raw_links_html += f"&nbsp;&nbsp;agent <code>{html.escape(agent_path)}</code>"
        raw_links_html += "</div>"
    else:
        raw_links_html = ""

    # Collapsible trace sections.
    agent_events = (agent_trace or {}).get("events", [])
    mock_events = mock_trace.get("events", [])

    transcript_section = (
        "<details class='trace'>"
        "<summary>Conversation transcript</summary>"
        f"{render_transcript(agent_events)}"
        "</details>"
        if agent_events
        else ""
    )

    replay_section = (
        "<details class='trace' open>"
        "<summary>Replay</summary>"
        f"{render_replay(mock_trace, agent_trace)}"
        "</details>"
    )

    recording_section = (
        "<details class='trace' open>"
        "<summary>Recording</summary>"
        f"{render_recording_player(mock_trace)}"
        "</details>"
    )

    agent_trace_section = (
        "<details class='trace'>"
        f"<summary>Agent event timeline ({len(agent_events)} events)</summary>"
        f"{render_trace_timeline(agent_events)}"
        "</details>"
        if agent_events
        else "<div style='color:#999;font-size:12px'>(no agent trace correlated to this episode)</div>"
    )

    mock_trace_section = (
        "<details class='trace'>"
        f"<summary>Mock IVR event timeline ({len(mock_events)} events)</summary>"
        f"{render_trace_timeline(mock_events)}"
        "</details>"
    )

    return (
        "<details class='episode'>"
        f"{summary}"
        "<div class='episode-body'>"
        f"{check_list}"
        f"{recording_section}"
        f"{replay_section}"
        f"{transcript_section}"
        f"{agent_trace_section}"
        f"{mock_trace_section}"
        f"{raw_links_html}"
        "</div>"
        "</details>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def render_dashboard(scores: list[EpisodeScore], pairs: list[tuple[dict, Optional[dict]]]) -> str:
    """Glue all sections together into one HTML doc."""
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    n = len(scores)
    n_pass = sum(1 for s in scores if s.hard_pass)
    pct = (100 * n_pass / n) if n else 0
    n_models = len({s.model for s in scores})

    leaderboard = render_leaderboard(scores) if scores else ""
    matrix = render_check_matrix(scores) if scores else ""

    # Pair scores with their original traces. Sort newest-first so the most
    # recent runs are at the top.
    score_to_traces: dict[str, tuple[dict, Optional[dict]]] = {}
    for mock, agent in pairs:
        ep_id = mock.get("episode_id", "?")
        score_to_traces[ep_id] = (mock, agent)
    scores_sorted = sorted(
        scores,
        key=lambda s: -(score_to_traces.get(s.episode_id, ({}, None))[0].get("started_at") or 0),
    )
    episode_cards = "".join(
        render_episode_card(
            s,
            mock_trace=score_to_traces.get(s.episode_id, ({}, None))[0],
            agent_trace=score_to_traces.get(s.episode_id, ({}, None))[1],
        )
        for s in scores_sorted
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RumaCare Voice Agent — Eval Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<h1>Voice Agent Eval Dashboard</h1>
<div class='meta'>
  {n} episodes &middot; {n_models} model(s) &middot; {n_pass}/{n} passed all hard checks ({pct:.0f}%) &middot;
  generated {when}
</div>
{leaderboard}
{matrix}
<h2>Episodes (newest first)</h2>
<p style="color:#888; font-size:12px; margin: -8px 0 12px 0">
  Click any row to expand. Each card shows the check breakdown plus collapsible
  replay, conversation transcript, agent event timeline, and mock IVR timeline.
</p>
{episode_cards}
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Render an HTML eval dashboard from episode traces")
    p.add_argument("--since", default="", help="only include episodes from the last N (1h, 30m, 24h)")
    p.add_argument("--out", default="dashboard.html", help="output path (default: dashboard.html in cwd)")
    p.add_argument("--open", dest="open_after", action="store_true", help="open the dashboard in default browser")
    p.add_argument(
        "--judge",
        default="",
        help="LLM judge override, e.g. 'anthropic/claude-opus-4-7', "
        "'openai/gpt-4o-mini', 'xai', or 'groq'. "
        "Default: anthropic/claude-opus-4-7.",
    )
    args = p.parse_args()
    if args.judge:
        import os as _os
        _os.environ["JUDGE_MODEL"] = args.judge

    since = parse_since(args.since) if args.since else None
    pairs = load_episodes(EPISODES_DIR, since_seconds=since)
    if not pairs:
        print("No episodes found.", file=sys.stderr)
        sys.exit(1)

    scores: list[EpisodeScore] = []
    for i, (mock, agent) in enumerate(pairs, 1):
        print(f"\rScoring {i}/{len(pairs)}…", end="", flush=True, file=sys.stderr)
        scores.append(score_episode(mock, agent))
    print(file=sys.stderr)
    # Orphan failed startup attempts have no agent trace and score as "unknown".
    # Keep their raw JSON/audio on disk, but do not include them in the review UI.
    scores = [s for s in scores if s.model != "unknown"]

    html_doc = render_dashboard(scores, pairs)
    out_path = Path(args.out).resolve()
    out_path.write_text(html_doc)
    print(f"Wrote {out_path} ({len(html_doc):,} chars, {len(scores)} episodes)")

    if args.open_after:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
