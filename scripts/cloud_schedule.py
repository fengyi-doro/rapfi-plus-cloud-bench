"""Validate dispatch inputs before allocating the match runners."""
import json
import os
import re
from pathlib import Path

from datasets import DATA


def schedule(pairs, candidate, clock):
    book = json.loads((DATA / "v3/cloud_openings.json").read_bytes())
    if not 1 <= pairs <= len(book):
        raise ValueError(f"pairs must be between 1 and {len(book)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", candidate) or candidate.lower() in {"orig", "draw"}:
        raise ValueError("use a distinct candidate label of 1–64 letters, digits, dots, underscores or hyphens")
    match = re.fullmatch(r"([0-9]+)\+([0-9]+)", clock)
    if not match:
        raise ValueError("time_control must be initial_ms+increment_ms, e.g. 1500+40")
    initial, increment = map(int, match.groups())
    if initial <= 0 or initial + 225 * increment > 2**31 - 1:
        raise ValueError("the initial clock must be positive and fit the engine's millisecond counter through move 225")
    count = min(20, pairs)
    return dict(matrix=json.dumps(list(range(1, count + 1))), shards=count,
                initial_ms=initial, increment_ms=increment)


if __name__ == "__main__":
    plan = schedule(int(os.environ["PAIRS"]), os.environ["RAPFI_CANDIDATE_LABEL"], os.environ["TIME_CONTROL"])
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for key, value in plan.items():
            print(f"{key}={value}", file=output)
