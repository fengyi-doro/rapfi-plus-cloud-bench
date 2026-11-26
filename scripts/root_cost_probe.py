"""Measure enabled/disabled end-to-end search cost on identical roots using archived match binaries."""
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import sys

import psutil

sys.path.insert(0, str(Path.cwd() / "scripts"))
from datasets import DATA, dump, sha
from engine import Engine, filename, topology


def probe(summary, output):
    protocol = summary["protocol"]
    name = summary["players"][0]
    mode = protocol["root_mode"]
    binary = sha(filename(name))
    if binary != summary["binary_hashes"]["Linux"][name] or sha(filename("v4")) != binary:
        raise ValueError("probe must toggle features on the exact same match executable")
    info = topology()
    cpu = min(psutil.Process().cpu_affinity())
    psutil.Process().cpu_affinity([cpu])
    positions = json.loads((DATA / "positions.json").read_bytes())
    samples = []
    for rule in (1, 2):
        rows = [r for r in positions if r["rule"] == rule and 10 <= len(r["moves"]) <= 80][:24]
        if len(rows) != 24:
            raise ValueError("diagnostic position sample incomplete")
        with ExitStack() as stack:
            os.environ.update(RAPFI_ROOT_MODE="control", RAPFI_PROOF_FIRST="false", RAPFI_CLOCK_POLICY="standard")
            control = stack.enter_context(Engine("v4", cpu))
            os.environ.update(RAPFI_ROOT_MODE=mode, RAPFI_CLOCK_POLICY=protocol.get("clock_policy") or "standard",
                              RAPFI_PROOF_FIRST=str(bool(protocol.get("proof_first"))).lower())
            candidate = stack.enter_context(Engine(name, cpu))
            for player in (control, candidate):
                player.repin(cpu)
                player.select(rule)
                player.sync("INFO thread_num 1\nINFO pondering 0\nINFO show_detail 0\nINFO max_depth 5\n"
                            "INFO timeout_match 60000\nINFO timeout_turn 5000\nINFO time_increment 40")
                player.affinity_masks()
            for i, pos in enumerate(rows):
                samples_for_root = []
                for rep in (0, 1):
                    sample = {}
                    order = (control, candidate) if (i + rep) % 2 == 0 else (candidate, control)
                    for player in order:
                        player.clean_position(pos["moves"])
                        player.sync("INFO time_left 60000")
                        turn = player.think("YXNBEST 1", timeout=6)
                        sample["on" if player is candidate else "off"] = turn
                    samples_for_root.append(sample)
                diff = statistics.mean((s["on"]["wall_ns"]-s["off"]["wall_ns"])/1e6 for s in samples_for_root)
                samples.append(dict(position=pos["id"], rule=rule, ply=len(pos["moves"]), mean_delta_ms=diff,
                                    observations=samples_for_root))
    by_rule = [[s["mean_delta_ms"] for s in samples if s["rule"] == rule] for rule in (1, 2)]
    avg = statistics.mean(s["mean_delta_ms"] for s in samples)
    se = math.sqrt(sum(statistics.variance(g) / len(g) for g in by_rule)) / 2
    observations = [o for r in samples for o in r["observations"]]
    means = {k:statistics.mean(o[k]["wall_ns"] / 1e6 for o in observations) for k in ("on", "off")}
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, dict(match_run=int(os.environ["MATCH_RUN"]), match_commit=protocol["commit"], binary_sha256=binary,
                      run=f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                      root_mode=mode, model_sha256=protocol.get("oracle_sha256"), cpu=info["cpu"], assigned_vcpu=cpu,
                      positions=len(samples), observations=len(observations), depth_cap=5, mean_ms=means,
                      mean_delta_ms=avg, ci95_delta_ms=[avg-1.96*se, avg+1.96*se],
                      ci_method="paired position differences, stratified by rule, normal approximation",
                      prior_calls=sum("root_prior_ns" in o["on"] for o in observations), samples=samples))
    print(f"{name}: {means['off']:.3f} -> {means['on']:.3f} ms; paired change {avg:+.3f} ms")


if __name__ == "__main__":
    probe(json.loads(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2]))
