"""Fit bounded integer residuals to the frozen Mix9SVQ WDL head on independent games."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from datasets import DATA, canonical, dump
from cloud_distill import ARCHIVE, TEACHER, digest

ARCHIVE_SHA = "f6b45f816fd9fe3f94ea751a48b1c3a7db0a52ea3f806b4def1f12464a43eab6"
UTILITY = np.array([1., 0., .5])
NODE_LIMIT = 12000


def score(text):
    if text and text.startswith(("+M","-M")):
        return float(text[0] == "+")
    if text and re.fullmatch(r"[+-]?\d+", text):
        return 1 / (1 + math.exp(-max(-40., min(40., int(text) / 200))))
    return float("nan")


def extract_lane(task):
    from engine import Engine
    cpu, rows = task
    os.sched_setaffinity(0, {cpu})
    extracted = []
    with Engine("v4", cpu) as engine:
        engine.sync(f"INFO thread_num 1\nINFO pondering 0\nINFO show_detail 2\nINFO timeout_match 60000"
                    f"\nINFO timeout_turn 5000\nINFO max_depth 18\nINFO max_node {NODE_LIMIT}")
        for ix, rule, moves in rows:
            engine.select(rule)
            engine.clean_position(moves)
            lines = [s for s in engine.sync("INFO time_left 60000\nTRACEVALUEHEAD") if s.startswith("MESSAGE VALUE_HEAD ")]
            if len(lines) != 1:
                raise ValueError(f"invalid feature extraction at position {ix}")
            _, _, rr, side, win, loss, draw, *features = lines[0].split()
            channels = np.array(list(map(int, features)))
            if int(rr) != rule or int(side) != len(moves)%2 or len(channels) != 64 or min(channels) < 0 or max(channels) > 127:
                raise ValueError(f"wrong native feature schema at {ix}")
            turn = engine.think("YXNBEST 1", timeout=9)
            extracted.append((ix, channels.astype(np.uint8), [float(win), float(loss), float(draw)],
                              score(turn["eval"]), turn["depth"], turn.get("nodes") or 0))
    return extracted


def extract(shard, out):
    from engine import filename
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("training-input/dataset.json").read_bytes())
    path = Path("training-input/positions.npz")
    if digest(path) != manifest["dataset_sha256"] or manifest["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("stale input games")
    with np.load(path) as positions:
        packets, sources = positions["packets"], positions["source"]
    with np.load(f"teacher-labels/labels-{shard}.npz") as labels:
        ids, targets = labels["ids"], labels["wdl"].astype(np.float64)
        if labels["teacher"].item() != TEACHER or labels["dataset"].item() != manifest["dataset_sha256"]:
            raise ValueError("frozen B24 predictions do not match the source positions")
    expected = np.array_split(np.arange(len(packets)), 20)[shard]
    if not np.array_equal(expected, ids):
        raise ValueError("shard ledger differs from teacher")
    book = defaultdict(lambda:defaultdict(set))
    for opening in json.loads((DATA/"v3/cloud_openings.json").read_bytes()):
        book[opening["rule"]][len(opening["moves"])].add(canonical(opening["moves"]))
    rows, omitted = [], []
    with TemporaryDirectory() as tmp:
        archive = Path(tmp)/"source.zip"
        subprocess.run(["curl","-fLsS","--retry","3","-o",str(archive),ARCHIVE],check=True)
        if digest(archive) != ARCHIVE_SHA:
            raise ValueError("public source archive changed")
        cache = {}
        with zipfile.ZipFile(archive) as games:
            for ix in ids:
                packet, name = packets[ix], str(sources[ix])
                rule, side, last = map(int, packet[:3])
                if name not in cache:
                    moves = []
                    for line in games.read(name).splitlines()[1:]:
                        m = re.fullmatch(rb"(\d+),(\d+),(\d+)", line.strip())
                        if not m:
                            break
                        moves.append(15*(int(m[2])-1) + int(m[1])-1)
                    overlap = any(len(moves) >= ply and canonical(moves[:ply]) in banned
                                  for ply, banned in book[rule].items())
                    cache[name] = (moves, overlap)
                moves, overlap = cache[name]
                if overlap:
                    omitted.append(ix)
                    continue
                ply = int(np.count_nonzero(packet[3:]))
                root = moves[:ply]
                check = np.zeros(225, dtype=np.uint8)
                for j, p in enumerate(root):
                    check[p] = 1 if j%2 == side else 2
                if not root or root[-1] != last or side != ply%2 or not np.array_equal(check, packet[3:]):
                    raise ValueError(f"unmatched history at {ix}")
                rows.append((int(ix), rule, root))
    cpus = sorted(os.sched_getaffinity(0))[:4]
    with ProcessPoolExecutor(len(cpus)) as pool:
        chunks = list(pool.map(extract_lane, [(cpu, rows[i::len(cpus)]) for i,cpu in enumerate(cpus)]))
    results = sorted((r for chunk in chunks for r in chunk), key=lambda r:r[0])
    if not results:
        raise ValueError("development shard has no independent positions")
    ix, hidden, base, search, depths, nodes = zip(*results)
    ix = np.array(ix)
    file = out/f"readout-{shard}.npz"
    np.savez_compressed(file, ids=ix, omitted=np.asarray(omitted,dtype=np.int32), h=np.stack(hidden), base=base,
                        deep=search, depths=depths, nodes=nodes, teacher=targets[ix-ids[0]], engine=digest(filename("v4")),
                        dataset=manifest["dataset_sha256"], source_teacher=TEACHER)
    print(f"shard {shard}: {len(ix)} positions, {len(omitted)} excluded, {np.isfinite(search).sum()} search scores")


def softmax(logits):
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def fit(out):
    from scipy.optimize import minimize
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("training-input/dataset.json").read_bytes())
    source = Path("training-input/positions.npz")
    if digest(source) != manifest["dataset_sha256"]:
        raise ValueError("training set changed")
    with np.load(source) as f:
        packets, split = f["packets"], f["split"]
    fields = defaultdict(list)
    engines = set()
    for path in sorted(Path("features").glob("readout-*.npz")):
        with np.load(path) as f:
            if f["dataset"].item() != manifest["dataset_sha256"] or f["source_teacher"].item() != TEACHER:
                raise ValueError("feature provenance mismatch")
            engines.add(f["engine"].item())
            for k in ("ids","omitted","h","base","deep","depths","nodes","teacher"):
                fields[k].append(f[k])
    all_data = {k:np.concatenate(v) for k,v in fields.items()}
    ids, omitted = all_data["ids"], all_data["omitted"]
    if (len(engines) != 1 or len(ids)+len(omitted) != len(packets)
            or set(ids) & set(omitted) or set(np.r_[ids,omitted]) != set(range(len(packets)))):
        raise ValueError("incomplete training ledger")
    h, base, deep, teacher = (all_data[k] for k in ("h","base","deep","teacher"))
    h = h.astype(np.float64)
    teacher /= teacher.sum(1, keepdims=True)
    if not np.isfinite(base).all() or not np.isfinite(teacher).all():
        raise ValueError("invalid WDL logits")
    # A deep score contributes a scalar target. B24 supplies all three outcomes.
    # Mask uncovered engine scores in the fit and measure coverage separately.
    groups = np.where(packets[ids,0] == 1, 0, 1 + packets[ids,1])
    splits = split[ids]
    metrics, coefficients, biases = [], [], []
    for group in (0,1,2):
        ix = [np.flatnonzero((groups==group) & (splits==s)) for s in (0,1,2)]
        if any(len(g) < 40 for g in ix):
            raise ValueError("insufficient independent games in a head split")
        train, val, test = ix
        features = np.column_stack((h[train]/128, np.ones(len(train))))
        original = softmax(base[train])
        target_wdl = .60*teacher[train] + .40*original
        covered = np.isfinite(deep[train])
        target_score = np.where(covered, deep[train], original @ UTILITY)

        def objective(theta):
            weight = theta.reshape(65, 3)
            logit = base[train] + features @ weight
            p = np.maximum(softmax(logit), 1e-12)
            q = np.clip(p @ UTILITY, 1e-9, 1-1e-9)
            wdl_loss = -(target_wdl * np.log(p)).sum(1).mean()
            score_loss = -np.mean(target_score*np.log(q) + (1-target_score)*np.log1p(-q))
            dq = (q-target_score) / (q*(1-q))
            grad_logit = .50*(p-target_wdl) + .50*dq[:,None]*p*(UTILITY-q[:,None])
            loss = .50*(wdl_loss+score_loss) + .002*np.square(weight[:-1]).sum()
            grad = features.T @ grad_logit / len(train)
            grad[:-1] += .004*weight[:-1]
            return loss, grad.ravel()

        result = minimize(objective, np.zeros(65*3), method="L-BFGS-B", jac=True,
                          bounds=[(-.5,.5)]*192 + [(-.8,.8)]*3, options={"maxiter":160,"ftol":1e-11})
        weights = result.x.reshape(65,3)
        def measure(rows, scale):
            coef = np.rint(128*scale*weights[:-1]).astype(np.int8)
            bias = np.rint(16384*scale*weights[-1]).astype(np.int32)
            p = softmax(base[rows] + (h[rows] @ coef + bias) / 16384.)
            q = p @ UTILITY
            missing = ~np.isfinite(deep[rows])
            deep_mse = np.mean((q[~missing]-deep[rows][~missing])**2)
            b24_mse = np.mean((q-teacher[rows]@UTILITY)**2)
            return dict(deep_mse=float(deep_mse), b24_mse=float(b24_mse), combined=float(.7*deep_mse+.3*b24_mse))
        scales = (0.,.25,.5,.75,1.)
        trials = {s:measure(val,s) for s in scales}
        selected = min(scales, key=lambda s:trials[s]["combined"])
        coefficients.append(np.rint(128*selected*weights[:-1]).astype(np.int8).T)
        biases.append(np.rint(16384*selected*weights[-1]).astype("<i4"))
        metrics.append(dict(head=group, split_sizes=list(map(len,ix)), search_coverage=int(np.isfinite(deep[test]).sum()),
                            validation=trials, selected=selected, test_base=measure(test,0.), test_selected=measure(test,selected),
                            fit_status=result.message))
    coeff, bias = np.stack(coefficients), np.stack(biases)
    sha = hashlib.sha256(coeff.tobytes() + bias.tobytes()).hexdigest()
    def braces(value):
        return "{" + ", ".join(braces(v) if isinstance(v,list) else str(v) for v in value) + "}"
    (out/"mix9_readout_weights.h").write_text(
        "// Derived from public 2024 Gomocup positions. Final-layer logits use scale 1/16384.\n"
        "#pragma once\n#include <cstdint>\nnamespace Evaluation::mix9svq::readout {\n"
        f'inline constexpr char Digest[] = "{sha}";\n'
        f"inline constexpr int8_t Delta[3][3][64] = {braces(coeff.tolist())};\n"
        f"inline constexpr int32_t Bias[3][3] = {braces(bias.tolist())};\n}}\n",encoding="ascii")
    dump(out/"readout.json", dict(native_engine_sha256=engines.pop(), input_sha256=manifest["dataset_sha256"],
                                  b24_teacher_sha256=TEACHER, positions_used=len(ids), excluded_game_prefixes=len(omitted),
                                  weights_sha256=sha, max_nodes=NODE_LIMIT,
                                  search_score_coverage=int(np.isfinite(deep).sum()), heads=metrics,
                                  training_run=f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"))
    print(f"readout {sha}: validation scales {[r['selected'] for r in metrics]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract","fit"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.shard, args.out)
    else:
        fit(args.out)
