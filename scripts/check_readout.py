"""Verify fitted logits against the independently decoded training artifact on a cloud CPU."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from datasets import DATA, ROOT, dump, sha
from engine import Engine, filename


def header_array(header, name):
    line = next(line for line in header.splitlines() if " " + name + "[" in line)
    literal = line.split(" = ", 1)[1].rstrip(";")
    return json.loads(literal.replace("{", "[").replace("}", "]"))


def trace(engine, row):
    engine.select(row["rule"])
    engine.clean_position(row["moves"])
    lines = [s for s in engine.sync("TRACEVALUEHEAD") if s.startswith("MESSAGE VALUE_HEAD ")]
    if len(lines) != 1:
        raise ValueError("expected one native value feature trace")
    _, _, rule, side, w, l, d, *features = lines[0].split()
    if (int(rule), int(side)) != (row["rule"], len(row["moves"]) % 2) or len(features) != 64:
        raise ValueError("native value trace describes a different position")
    return tuple(map(float, (w, l, d))), tuple(map(int, features))


def main():
    manifest = json.loads((DATA/"v4/fitted_readout.json").read_bytes())
    header = (ROOT/"src_plus/eval/mix9_readout_weights.h").read_text(encoding="ascii")
    weights, bias = (header_array(header, name) for name in ("Delta", "Bias"))
    flat = [c for head in weights for row in head for c in row]
    data = struct.pack("576b", *flat) + struct.pack("<9i", *(c for head in bias for c in head))
    digest = hashlib.sha256(data).hexdigest()
    if (digest != manifest["weights_sha256"] or f'"{digest}"' not in header
            or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("embedded readout disagrees with the cloud fit")
    count, changed = Counter(), Counter()
    max_error = 0.
    cpu = min(os.sched_getaffinity(0))
    with ExitStack() as stack:
        os.environ["RAPFI_ROOT_MODE"] = "control"
        native = stack.enter_context(Engine("v4", cpu))
        os.environ["RAPFI_ROOT_MODE"] = "readout"
        fitted = stack.enter_context(Engine("v4", cpu))
        for row in json.loads((DATA/"positions.json").read_bytes()):
            if row["rule"] not in (1, 2):
                continue
            head = 0 if row["rule"] == 1 else 1 + len(row["moves"]) % 2
            if count[head] == 6:
                continue
            original, features = trace(native, row)
            actual, check_features = trace(fitted, row)
            if features != check_features:
                raise ValueError("fitted model changed the frozen trunk")
            expected = [v + (b + sum(a * f for a, f in zip(w, features))) / 16384
                        for v, b, w in zip(original, bias[head], weights[head])]
            error = max(abs(v - pred) for v, pred in zip(actual, expected))
            if not math.isfinite(error) or error > 2e-5:
                raise ValueError(f"playing and reference WDL logits disagree in head {head}")
            count[head] += 1
            changed[head] += actual != original
            max_error = max(max_error, error)
            if sum(count.values()) == 18:
                break
    if count != {0:6, 1:6, 2:6} or any(h["selected"] and not changed[h["head"]] for h in manifest["heads"]):
        raise ValueError("native trace did not exercise all fitted outcomes")
    manifest.update(binary_sha256=sha(filename("v4")), probe_count=sum(count.values()),
                    changed_probes_by_head=dict(changed), max_logit_error=max_error)
    dump(Path(os.environ["RAPFI_ARENA_BIN"])/"readout.json", manifest)
    print(f"{sum(count.values())} playing positions verified, maximum logit error {max_error:.8g}")


if __name__ == "__main__":
    main()
