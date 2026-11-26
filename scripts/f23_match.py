"""Run the fixed paired schedule, replay each ledger and compare certified records."""
from dataclasses import dataclass
import json
import os
import shutil
import tarfile
from pathlib import Path
import subprocess
import sys

import numpy as np

from datasets import DATA, SEED, dump, sha
from distributed_sprt import aggregate
from f23_build import ARMS, COUNTERS, arm_name, filename, load
from replay import verify
from report import elo

BOOK = DATA / "v3/cloud_openings.json"
DIST = Path("dist")
REPORT = Path("report")


@dataclass(frozen=True)
class Duel:
    label: str
    mask: int
    opponent: str
    layout: str

    @property
    def slug(self):
        return f"{self.label}-vs-{self.opponent}"

    @property
    def binary(self):
        return arm_name(self.mask)


def schedule(phase, mask):
    if phase == "screen":
        return [Duel(arm_name(bit), bit, opp, "packed-1t") for i, bit in enumerate(ARMS.values())
                for opp in (("v4", "orig") if i % 2 == 0 else ("orig", "v4"))]
    if phase == "bundles":
        return [Duel(arm_name(bit), bit, opp, "packed-1t")
                for bit, opponents in ((18, ("v4", "orig")), (30, ("orig", "v4")))
                for opp in opponents]
    bits = int(mask)
    if not 0 <= bits <= 31:
        raise ValueError("unknown module mask")
    if phase == "combine":
        return [Duel("f23-combination", bits, "v4", "packed-1t")]
    if phase not in ("cert1", "cert4"):
        raise ValueError("unknown tournament phase")
    threads = 1 if phase == "cert1" else 4
    return [Duel(f"rapfi-plus-v5-{threads}t", bits, "orig", f"isolated-{threads}t")]


def manifest(duel, build):
    if duel.mask and build["masks"].get(duel.binary) != duel.mask:
        raise ValueError(f"the build has no such arm: {duel.binary}")
    for name in (duel.binary, duel.opponent):
        if sha(DIST / filename(name)) != build["binaries"][name]:
            raise ValueError(f"binary mismatch: {name}")
    return dict(**build, mask=duel.mask, profile=duel.binary, build_run=os.environ["BUILD_RUN"],
                build_sha=os.environ.get("BUILD_SHA"))


def play(duels, build):
    shard = int(os.environ["SHARD"])
    for duel in duels:
        manifest(duel, build)
        out = Path("results") / duel.slug
        env = dict(os.environ, RAPFI_CANDIDATE_LABEL=duel.label, RAPFI_BUILD_PROFILE=duel.binary,
                   RAPFI_LANE_LAYOUT=duel.layout, RAPFI_ARENA_BIN=str(DIST), RAPFI_ROOT_MODE="control")
        args = [sys.executable, "scripts/arena_fast.py", "--book", str(BOOK), "--pairs", "500",
                "--shards", "20", "--shard", str(shard), "--lanes", "4", "--vcpu-lanes",
                "--challenger", duel.label, "--opponent", duel.opponent, "--initial-ms", "1500",
                "--increment-ms", "40", "--out", str(out)]
        subprocess.run(args, env=env, check=True)
        _, audit = verify(out, BOOK)
        if audit["games"] != 50 or audit["pairs"] != 25 or audit["referee_mismatches"] or audit["json_psq_mismatches"]:
            raise ValueError(f"shard replay failed: {duel.slug} {shard}")


def paired_record(new, old):
    old_points = {r["opening"]: r["half_points"] for r in old["pair_ledger"]}
    new_points = {r["opening"]: r["half_points"] for r in new["pair_ledger"]}
    book = load(BOOK)
    if set(old_points) != set(new_points) or set(old_points) != {r["id"] for r in book}:
        raise ValueError("record openings do not match")
    if old["protocol"]["lane_layout"] != new["protocol"]["lane_layout"] or old["protocol"]["book_sha256"] != new["protocol"]["book_sha256"]:
        raise ValueError("record layout or book differs")
    rng = np.random.default_rng(SEED)
    boot = np.zeros((30000, 2))
    for rule in (1, 2):
        ids = [r["id"] for r in book if r["rule"] == rule]
        pairs = np.array([[new_points[k], old_points[k]] for k in ids]) / 4
        assert len(ids) == 250
        for start in range(0, len(boot), 500):
            sample = rng.integers(len(ids), size=(min(500, len(boot)-start), len(ids)))
            boot[start:start+len(sample)] += pairs[sample].mean(axis=1) / 2
    lo, hi = np.quantile(400 * np.log10(boot[:, 0] / (1-boot[:, 0]))
                         - 400 * np.log10(boot[:, 1] / (1-boot[:, 1])), [.025, .975])
    return dict(delta_elo=new["scores"]["All"]["elo"]-old["scores"]["All"]["elo"],
                ci95_delta_elo=[float(lo), float(hi)], historical_elo=old["scores"]["All"]["elo"],
                delta_win_percentage_points=(new["scores"]["All"]["W"]-old["scores"]["All"]["W"])/10)


def publish(duels, build, phase):
    REPORT.mkdir(exist_ok=True)
    results = {}
    for duel in duels:
        out = REPORT / duel.slug
        out.mkdir(exist_ok=True)
        info = out / "build.json"
        dump(info, manifest(duel, build))
        paths = [Path("shards") / f"shard-{i}" / duel.slug for i in range(1, 21)]
        r = aggregate(paths, BOOK, 500, 20, out, 0, 5, build_info=info)
        assert [r["scores"][rule]["games"] for rule in ("All", "Standard", "Renju")] == [1000,500,500]
        audit = {key:sum(s["audit"][key] for s in r["shards"]) for key in
                 ("pairs","games","board_moves_replayed","timed_turns","json_psq_mismatches",
                  "referee_mismatches","hard_flags","illegal_moves","overlapping_vcpu_assignments")}
        assert audit["pairs"] == 500 and all(audit[k] == 0 for k in
                   ("referee_mismatches","json_psq_mismatches","overlapping_vcpu_assignments"))
        audit["no_legal_black_reply"] = sum(len(s["audit"]["no_legal_black_reply"]) for s in r["shards"])
        totals = r["f23_telemetry"][duel.label]["counts"]
        results[duel.slug] = dict(players=r["players"], mask=duel.mask, layout=duel.layout,
                                  scores=r["scores"], audit=audit, span_seconds=r["playing_span_seconds"],
                                  cpu_capacity=sum(min(4, s["vcpus"]) for s in r["shards"]),
                                  peak_simultaneous_games=r["peak_simultaneous_games"],
                                  binaries=r["binary_hashes"]["Linux"], flags=r["flags_by_player"],
                                  illegal=r["illegal_by_player"], move_timing=r["move_timing"],
                                  mother_ci95=r.get("mother_cluster_ci95_elo"),
                                  counts={k:dict(zip(COUNTERS,v,strict=True)) for k,v in totals.items()})
        if phase.startswith("cert"):
            results[duel.slug]["record_comparison"] = paired_record(r, load(Path("historical-summary/summary.json")))
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write((out / "summary.md").read_text() + "\n")
    decision = {}
    if phase in ("screen", "bundles"):
        hypotheses = ARMS if phase == "screen" else {"b1-b4": 18, "b1-b2-b3-b4": 30}
        for name, bit in hypotheses.items():
            challenger = arm_name(bit)
            versus_v4 = results[challenger + "-vs-v4"]
            versus_orig = results[challenger + "-vs-orig"]
            endpoint = "Renju" if name == "a" else "All"
            low = versus_v4["scores"][endpoint]["ci95_elo"][0]
            keep = (low > 0 and versus_v4["scores"]["All"]["elo"] > 0
                    and versus_orig["scores"]["All"]["elo"] > 0)
            decision[name] = dict(primary=endpoint, lower95=low, promoted=keep)
    dump(REPORT / "suite.json", dict(phase=phase, run=os.environ["GITHUB_RUN_ID"],
                                     sha=os.environ["GITHUB_SHA"], build=os.environ["BUILD_RUN"],
                                     results=results, decision=decision))

def package(duel, build):
    # Package only the fingerprint that passed this exact certification round.
    result = load(REPORT / "suite.json")["results"][duel.slug]
    digest = result["binaries"][duel.label]
    release = Path("release") / "rapfi-plus-v5"
    release.mkdir(parents=True, exist_ok=True)
    binary = release / "pbrain-rapfi-plus-v5"
    shutil.copy(DIST / filename(duel.binary), binary)
    binary.chmod(0o755)
    if sha(binary) != digest:
        raise ValueError("release executable differs from the certified engine")
    files = {binary.name: digest}
    for name in [*build["runtime"], "Copying.txt", "Networks-LICENSE.txt"]:
        shutil.copy(name, release / name)
        files[name] = sha(release / name)
    if any(files[k] != v for k, v in build["runtime"].items()):
        raise ValueError("release runtime differs from certified weights")
    suite = load(REPORT / "suite.json")
    certified = {suite["phase"]: dict(run=suite["run"], scores=result["scores"], audit=result["audit"])}
    other_path = os.environ.get("OTHER_CERT_SUITE")
    if other_path:
        other = load(Path(other_path))
        opposite = "cert4" if suite["phase"] == "cert1" else "cert1"
        if other["phase"] != opposite or len(other["results"]) != 1:
            raise ValueError("the release requires the other thread allocation")
        additional, = other["results"].values()
        label, opp = additional["players"]
        if (opp != "orig" or additional["mask"] != duel.mask or additional["binaries"][label] != digest
                or additional["binaries"][opp] != result["binaries"][duel.opponent]
                or additional["audit"]["games"] != 1000
                or additional["audit"]["referee_mismatches"] or additional["audit"]["json_psq_mismatches"]):
            raise ValueError("the second certification does not match the release")
        certified[opposite] = dict(run=other["run"], scores=additional["scores"], audit=additional["audit"])
    dump(release / "certification.json", dict(run=suite["run"], mask=duel.mask,
         source=build["baseline_commit"] if not duel.mask else load(REPORT / duel.slug / "build.json")["build_sha"],
         build_run=os.environ["BUILD_RUN"], package_run=os.environ["GITHUB_RUN_ID"],
         executable=digest, scores=result["scores"], audit=result["audit"], certifications=certified))
    files["certification.json"] = sha(release / "certification.json")
    with (release / "SHA256SUMS").open("w", encoding="ascii") as checksums:
        for name, file_hash in sorted(files.items()):
            print(f"{file_hash}  {name}", file=checksums)
    with tarfile.open("rapfi-plus-v5-linux-avx2.tar.gz", "w:gz", compresslevel=3) as archive:
        archive.add(release, arcname=release.name)


if __name__ == "__main__":
    task = sys.argv[1]
    phase, mask = os.environ["PHASE"], os.environ["MASK"]
    duels = schedule(phase, mask)
    build = load(DIST / "f23-build.json")
    if task == "play":
        play(duels, build)
    elif task == "publish":
        publish(duels, build, phase)
    elif task == "package":
        if not phase.startswith("cert") or len(duels) != 1:
            raise ValueError("a completed final is required")
        package(duels[0], build)
    elif task == "check":
        for duel in duels:
            manifest(duel, build)
        print(duels)
    else:
        raise ValueError(task)
