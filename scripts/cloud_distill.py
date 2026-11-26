"""Build a small global root student from a frozen KataGomo teacher on independent 2024 games."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
import zipfile

import numpy as np

from datasets import DATA, canonical, dump
from rules import valid_history

ARCHIVE = "https://gomocup.org/static/tournaments/2024/results/gomocup2024results.zip"
TEACHER = "0cf38a64b3cd3dd9dd39927e4acd29e07872f27af7bfaca6d48c3ecbf4d86fa9"
TEACHER_SOURCE = "92ec1e7cf101e41c128ec8b0093f6eec893ee5fa757730edeb619e5334c8147c"
TEACHER_RUN = 45829849427
SEED = 240615


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(out):
    from tempfile import TemporaryDirectory
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    held = json.loads((DATA/"v3/cloud_openings.json").read_bytes())
    excluded = {(r["rule"], canonical(r["moves"])) for r in held}
    seen = set(excluded)
    packets, splits, sources = [], [], []
    counts = Counter()
    blocked = 0
    quotas = [7000, 500, 500]
    with TemporaryDirectory() as tmp:
        archive = Path(tmp)/"gomocup2024results.zip"
        subprocess.run(["curl", "-fLsS", "--retry", "3", "-o", str(archive), ARCHIVE], check=True)
        source_hash = digest(archive)
        with zipfile.ZipFile(archive) as games:
            for rule, prefix in ((1, "standard"), (2, "renju")):
                files = sorted(n for n in games.namelist() if n.lower().startswith(prefix) and n.lower().endswith(".psq"))
                rng.shuffle(files)
                for name in files:
                    if all(counts[rule, s] >= q for s, q in enumerate(quotas)):
                        break
                    moves = []
                    for line in games.read(name).splitlines()[1:]:
                        m = re.fullmatch(rb"(\d+),(\d+),(\d+)", line.strip())
                        if not m:
                            break
                        x, y = int(m[1])-1, int(m[2])-1
                        if not (0 <= x < 15 and 0 <= y < 15):
                            break
                        moves.append(15*y+x)
                    game_key = repr((rule, canonical(moves))).encode("ascii")
                    bucket = int.from_bytes(hashlib.sha256(game_key).digest()[:4], "little") % 10
                    split = 2 if bucket == 0 else 1 if bucket == 1 else 0
                    if counts[rule, split] >= quotas[split]:
                        continue
                    last = min(len(moves)-1, 80)
                    if last < 16 or not valid_history(moves[:last], rule):
                        continue
                    plies = list(range(16, last+1))
                    rng.shuffle(plies)
                    for ply in plies[:16]:
                        history = moves[:ply]
                        key = rule, canonical(history)
                        if key in seen:
                            blocked += key in excluded
                            continue
                        seen.add(key)
                        side = ply % 2
                        cells = bytearray(225)
                        for i, p in enumerate(history):
                            cells[p] = 1 if i % 2 == side else 2
                        packets.append(bytes((rule, side, history[-1])) + cells)
                        sources.append(name)
                        splits.append(split)
                        counts[rule, split] += 1
                        if counts[rule, split] == quotas[split]:
                            break
    if any(counts[r, s] == 0 for r in (1, 2) for s in range(3)):
        raise ValueError(f"missing rule or game split in public archive: {counts}")
    game_sets = [{g for g, s in zip(sources, splits) if s == i} for i in range(3)]
    if any(game_sets[i] & game_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("source game leaked between development splits")
    path = out/"positions.npz"
    np.savez_compressed(path, packets=np.frombuffer(b"".join(packets), np.uint8).reshape(-1, 228),
                        split=np.array(splits, np.uint8), source=np.asarray(sources))
    manifest = dict(archive=ARCHIVE, archive_sha256=source_hash, dataset_sha256=digest(path), seed=SEED,
                    split_method="SHA-256 of rule and D4-canonical final stone sets, grouped by source game",
                    sampled_plies=[16, 80], temperature=1.25,
                    labels="frozen B24 dynamic INT8 policy and WDL", teacher_run=TEACHER_RUN, teacher_sha256=TEACHER,
                    source_game_counts=list(map(len, game_sets)), held_out_tournament_roots=len(excluded),
                    excluded_exact_or_d4_book_positions=blocked, positions=len(packets),
                    counts={str(r):[counts[r, s] for s in range(3)] for r in (1, 2)})
    dump(out/"dataset.json", manifest)
    print(manifest, flush=True)


def label_chunk(task):
    from kata_oracle import features, session
    cpu, model, packets = task
    os.sched_setaffinity(0, {cpu})
    net = session(model)
    policies, values = [], []
    for packet in packets:
        b = packet.tobytes()
        policy, value = net.run(["out_policy", "out_value"], features(b, 39))
        logits = np.where(packet[3:] == 0, policy[0, 0, :225] / 1.25, -1e9)
        p = np.exp(logits - logits.max())
        v = np.exp(value[0] - value[0].max())
        policies.append((p / p.sum()).astype(np.float16))
        values.append((v / v.sum()).astype(np.float16))
    return np.stack(policies), np.stack(values)


def label(out, shard):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("training-input/dataset.json").read_bytes())
    data = Path("training-input/positions.npz")
    teacher = Path("teacher/katagomo-pv-int8.onnx")
    if digest(teacher) != TEACHER or digest(data) != manifest["dataset_sha256"]:
        raise ValueError("unexpected teacher or source positions")
    with np.load(data) as samples:
        packets = samples["packets"]
    ids = np.array_split(np.arange(len(packets)), 20)[shard]
    cpus = sorted(os.sched_getaffinity(0))[:4]
    chunks = np.array_split(ids, len(cpus))
    tasks = [(cpu, teacher, packets[ix]) for cpu, ix in zip(cpus, chunks) if len(ix)]
    with ProcessPoolExecutor(len(tasks)) as pool:
        predictions = list(pool.map(label_chunk, tasks))
    policy, wdl = (np.concatenate(a) for a in zip(*predictions))
    if len(ids) != len(wdl) or not np.isfinite(policy).all() or not np.isfinite(wdl).all():
        raise ValueError("invalid teacher batch")
    np.savez_compressed(out/f"labels-{shard}.npz", ids=ids, policy=policy, wdl=wdl,
                        teacher=TEACHER, dataset=manifest["dataset_sha256"])
    print(f"shard {shard}: {len(ids)} positions, {len(tasks)} one-CPU teachers")


def train(out):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    import onnx
    import onnxruntime as ort
    from kata_oracle import features, predict, session

    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("training-input/dataset.json").read_bytes())
    input_path = Path("training-input/positions.npz")
    if digest(input_path) != manifest["dataset_sha256"]:
        raise ValueError("training input changed")
    with np.load(input_path) as samples:
        packets = torch.from_numpy(samples["packets"])
        splits = samples["split"]
    ids = [np.flatnonzero(splits == i) for i in range(3)]
    expected = set(range(len(packets)))
    policies = np.empty((len(packets), 225), np.float32)
    wdls = np.empty((len(packets), 3), np.float32)
    for path in sorted(Path("labels").glob("*.npz")):
        with np.load(path) as labels:
            ix = labels["ids"]
            if (labels["teacher"].item() != TEACHER or labels["dataset"].item() != manifest["dataset_sha256"]
                    or len(ix) != len(set(ix)) or not set(ix) <= expected):
                raise ValueError(f"incompatible or duplicate predictions: {path}")
            expected.difference_update(ix)
            policies[ix], wdls[ix] = labels["policy"], labels["wdl"]
    if expected:
        raise ValueError(f"{len(expected)} missing teacher predictions")
    pt = torch.from_numpy(policies)
    vt = torch.from_numpy(wdls)
    pt /= pt.sum(1, keepdim=True)
    vt /= vt.sum(1, keepdim=True)

    class RootStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(7, 24, 5, padding=2)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(24, 24, 3, padding=d, dilation=d),
                                                       nn.ReLU(), nn.Conv2d(24, 24, 1)) for d in (1, 2, 4)])
            self.context = nn.Linear(24, 24)
            self.policy = nn.Conv2d(24, 1, 1)
            self.value = nn.Sequential(nn.Linear(24*9, 48), nn.ReLU(), nn.Linear(48, 3))

        def forward(self, spatial, glob):
            rules = glob[:, 3:6, None, None].expand(-1, -1, 15, 15)
            x = F.relu(self.stem(torch.cat((spatial[:, :3], spatial[:, 6:7], rules), 1)))
            for block in self.blocks:
                x = F.relu(x + block(x))
            x = x * torch.sigmoid(self.context(x.mean((2, 3))))[:, :, None, None]
            logits = self.policy(x).flatten(1)
            policy = torch.cat((logits, torch.zeros_like(logits[:, :1])), 1).unsqueeze(1)
            return policy, self.value(F.avg_pool2d(x, 5).flatten(1))

    def batch(ix, augment=False):
        packet = packets[ix]
        n = len(ix)
        cells = packet[:, 3:].reshape(n, 15, 15)
        spatial = torch.zeros((n, 22, 15, 15))
        spatial[:, 0] = 1
        spatial[:, 1] = cells == 1
        spatial[:, 2] = cells == 2
        spatial[:, 6].flatten(1).scatter_(1, packet[:, 2:3].long(), 1.)
        glob = torch.zeros((n, 39))
        glob[:, 1] = 1
        glob[:, 3] = packet[:, 0] == 1
        glob[:, 4] = packet[:, 0] == 2
        glob[:, 5] = glob[:, 4] * (2 * packet[:, 1].float() - 1)
        target = pt[ix].reshape(n, 15, 15)
        if augment:
            rotation = int(rng.integers(4))
            spatial = torch.rot90(spatial, rotation, (2, 3))
            target = torch.rot90(target, rotation, (1, 2))
            if rng.integers(2):
                spatial, target = spatial.flip((3,)), target.flip((2,))
        return spatial, glob, target.flatten(1), vt[ix]

    student = RootStudent()
    optim = torch.optim.AdamW(student.parameters(), lr=.002, weight_decay=.0001)

    def epoch(ix, learning):
        student.train(learning)
        total = np.zeros(3)
        if learning:
            ix = rng.permutation(ix)
        with torch.set_grad_enabled(learning):
            for start in range(0, len(ix), 96):
                rows = ix[start:start+96]
                spatial, glob, policy, wdl = batch(rows, learning)
                pred, value = student(spatial, glob)
                occupied = spatial[:, 1:3].sum(1).flatten(1) > 0
                logits = pred[:, 0, :225].masked_fill(occupied, -1e4)
                loss = (-(policy*F.log_softmax(logits/1.25, 1)).sum(1).mean()
                        -.8*(wdl*F.log_softmax(value, 1)).sum(1).mean())
                if learning:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
                side_score = (value.softmax(1)-wdl) @ torch.tensor([1., 0., .5])
                total += (loss.item()*len(rows), (logits.argmax(1)==policy.argmax(1)).sum().item(),
                          side_score.abs().sum().item())
        avg = total/len(ix)
        return dict(loss=float(avg[0]), teacher_top1=float(avg[1]), mean_abs_score_error=float(avg[2]))

    history = []
    best_loss = float("inf")
    for e in range(16):
        if e in (9, 13):
            for group in optim.param_groups:
                group["lr"] *= .4
        training = epoch(ids[0], True)
        validation = epoch(ids[1], False)
        history.append(dict(epoch=e+1, train=training, validation=validation))
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best = {k:v.detach().clone() for k,v in student.state_dict().items()}
        print(f"epoch {e+1} validation {validation}", flush=True)
    student.load_state_dict(best)
    test = epoch(ids[2], False)
    x, g, _, _ = batch(ids[0][:1])
    fp_path = Path("student-fp32.onnx")
    torch.onnx.export(student.eval(), (x, g), str(fp_path), input_names=["input_spatial", "input_global"],
                      output_names=["out_policy", "out_value"], opset_version=17, dynamo=False,
                      dynamic_axes={k:{0:"batch"} for k in ("input_spatial", "input_global", "out_policy", "out_value")})

    class Calibration(CalibrationDataReader):
        def __init__(self):
            self.samples = iter(rng.choice(ids[0], min(80,len(ids[0])), replace=False))
        def get_next(self):
            index = next(self.samples, None)
            return None if index is None else features(packets[index].numpy().tobytes(), 39)

    model = out/"student-int8.onnx"
    quantize_static(fp_path, model, Calibration(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    fp, quant = session(fp_path), session(model)
    probes = []
    selected = []
    for rule in (1, 2):
        candidates = [i for i in ids[2] if packets[i, 0].item() == rule]
        selected.extend(rng.choice(candidates, min(64, len(candidates)), replace=False))
    for i in selected:
        packet = packets[i].numpy().tobytes()
        original = predict(fp, packet)
        start = time.perf_counter_ns()
        q = predict(quant, packet)
        ms = (time.perf_counter_ns()-start)/1e6
        empty = np.flatnonzero(np.frombuffer(packet, np.uint8, offset=3) == 0)
        probes.append(dict(rule=packet[0], ms=ms, eval_abs_delta=abs(float(original[-1]-q[-1])),
                           top1_agree=bool(original[empty].argmax()==q[empty].argmax()),
                           teacher_top1=bool(empty[q[empty].argmax()]==pt[i].argmax().item()),
                           teacher_score_error=abs(float(q[-1] - (vt[i, 0]+.5*vt[i, 2]).item()))))
    op = dict(Counter(n.op_type for n in onnx.load(model).graph.node))
    if not op.get("DequantizeLinear"):
        raise ValueError("exported student does not contain quantized weights")
    sha = digest(model)
    run = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    info = dict(family="distilled", source="KataGomo B24 CPU teacher",
                source_sha256=TEACHER_SOURCE, teacher_sha256=TEACHER, teacher_run=TEACHER_RUN,
                model_sha256=sha, models={"both":dict(file=model.name, sha256=sha)}, bytes=model.stat().st_size,
                ort=ort.__version__, onnx=onnx.__version__, providers=quant.get_providers(),
                intra_threads=1, inter_threads=1, operators=op, probes=probes,
                median_probe_ms=statistics.median(p["ms"] for p in probes),
                max_wdl_score_delta=max(p["eval_abs_delta"] for p in probes), training_run=run,
                training_data=manifest, parameters=sum(p.numel() for p in student.parameters()),
                epochs=history, test=test, precision="static per-channel S8S8 QDQ")
    dump(out/"oracle.json", info)
    print(f"student {sha}, {info['bytes']} bytes, {info['median_probe_ms']:.3f} ms, test {test}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "label", "train"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "label":
        label(args.out, args.shard)
    else:
        train(args.out)
