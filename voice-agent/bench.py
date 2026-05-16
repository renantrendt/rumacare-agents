"""
Bench runner — sweeps (models × personas × runs) episodes for head-to-head eval.

Why this script:
  When evaluating two realtime models against each other, doing it by hand is
  error-prone: it's easy to forget to alternate models, dispatch the wrong
  metadata, or get inconsistent persona coverage. This script makes a single
  invocation produce a balanced, reproducible eval set.

Outputs:
  - One mock IVR trace + one agent trace per episode (existing format).
  - Optional local WAV recording per episode with --record-audio.
  - After all runs complete, runs score_episodes.py and prints the leaderboard.

Usage:
    python bench.py --personas all --runs 3          # default GPT-RT2 eval
    python bench.py --models gpt-rt2 --runs 1        # smoke check primary model
    python bench.py --models grok-voice,gpt-rt2 --runs 2 --hold-seconds 4  # opt-in comparison

Notes:
  - The LiveKit worker (`python agent.py dev`) must already be running.
  - Episodes are dispatched sequentially so we don't slam rate limits or the
    worker (which only handles one job at a time in dev mode).
  - Default 5s settle delay between episodes — give the previous job's
    shutdown callbacks time to dump traces before the next room is created.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env.local", override=True)

logger = logging.getLogger("bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


ALL_PERSONAS = ["polite_sarah", "rushed_john", "confused_maria"]
ALL_MODELS = ["gpt-rt2", "grok-voice", "cascaded"]


def dispatch_episode(model: str, room: str) -> str:
    """Use the `lk` CLI to dispatch one job to the running worker. Returns the
    dispatch ID. Surfaces the CLI's error output if dispatch fails."""
    metadata = json.dumps({"mock": True, "model": model})
    cmd = [
        "lk", "dispatch", "create",
        "--agent-name", "rumacare",
        "--metadata", metadata,
        "--room", room,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("lk dispatch failed: %s", result.stderr)
        raise RuntimeError(f"lk dispatch returned {result.returncode}")
    # Extract dispatch ID from output like `id:"AD_..."`.
    for tok in result.stdout.split():
        if tok.startswith('id:"AD_'):
            return tok[4:-1]  # strip `id:"` and trailing `"`
    return "?"


def run_mock(room: str, persona: str, hold_seconds: float, record_audio: bool = False) -> int:
    """Run mock_ivr.py against the given room. Returns subprocess exit code."""
    cmd = [
        sys.executable, "mock_ivr.py",
        "--room", room,
        # mock_ivr.py's --hold-seconds is type=int, so coerce here. We keep
        # bench's CLI as float for nicer ergonomics, but the wire format must
        # match what the mock argparser expects.
        "--hold-seconds", str(int(hold_seconds)),
        "--persona", persona,
    ]
    if record_audio:
        cmd.append("--record-audio")
    # Capture output but stream the interesting lines to our log so the user
    # can see progress without drowning in deepgram chatter.
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip()
            # Forward only signal-bearing lines; mute the audio-encoding spam.
            if any(k in line for k in ("DTMF", "Episode", "aborted", "spoke", "connected")):
                print(f"    {line}")
    return proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description="Bench voice-agent models against the mock IVR")
    p.add_argument(
        "--models",
        default="gpt-rt2",
        help="comma-separated: gpt-rt2,grok-voice,cascaded. Default: gpt-rt2",
    )
    p.add_argument("--personas", default="all", help="comma-separated persona names, or 'all'")
    p.add_argument("--runs", type=int, default=1, help="episodes per (model × persona) cell")
    p.add_argument("--hold-seconds", type=float, default=3.0, help="mock IVR hold queue length")
    p.add_argument("--settle-seconds", type=float, default=6.0, help="wait between episodes")
    p.add_argument(
        "--record-audio",
        action="store_true",
        help="record local stereo WAVs for dashboard playback",
    )
    p.add_argument("--score-after", action="store_true", default=True, help="run scorer at end (default on)")
    p.add_argument("--no-score", dest="score_after", action="store_false", help="skip scoring")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        sys.exit(f"unknown model(s): {bad}. Valid: {ALL_MODELS}")
    if args.personas == "all":
        personas = list(ALL_PERSONAS)
    else:
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]

    total = len(models) * len(personas) * args.runs
    logger.info(
        "Bench plan: %d models × %d personas × %d runs = %d episodes",
        len(models), len(personas), args.runs, total,
    )
    started_at = time.time()
    episode_specs: list[tuple[str, str, int]] = []  # (model, persona, run_index)
    # Round-robin order: model × persona alternated so a model isn't blamed
    # for picking up "late-in-eval" weirdness.
    for run_idx in range(args.runs):
        for model in models:
            for persona in personas:
                episode_specs.append((model, persona, run_idx))

    for i, (model, persona, run_idx) in enumerate(episode_specs, start=1):
        room = f"bench-{model}-{persona}-r{run_idx}-{int(time.time())}"
        print()
        print("─" * 76)
        print(f"  [{i}/{total}]  model={model}  persona={persona}  run={run_idx}")
        print(f"  room={room}")
        try:
            disp_id = dispatch_episode(model, room)
            print(f"  dispatched: {disp_id}")
        except Exception as e:
            logger.error("Skipping episode (dispatch failed): %s", e)
            continue
        # Give the worker a moment to pick up the dispatch + connect.
        time.sleep(4)
        rc = run_mock(room, persona, args.hold_seconds, record_audio=args.record_audio)
        if rc != 0:
            logger.warning("mock_ivr exited with code %d for %s", rc, room)
        time.sleep(args.settle_seconds)

    elapsed = time.time() - started_at
    print()
    print("═" * 76)
    print(f"  Bench complete: {total} episodes in {elapsed:.0f}s")
    print("═" * 76)

    if args.score_after:
        # Score everything created in the last `elapsed + 30s` window so we
        # catch all the runs we just made (and only them).
        since_arg = f"{int(elapsed + 30)}s"
        print()
        print(f"  Scoring last {since_arg}…")
        subprocess.run(
            [sys.executable, "score_episodes.py", "--since", since_arg, "--leaderboard"],
            cwd=Path(__file__).parent,
        )


if __name__ == "__main__":
    main()
