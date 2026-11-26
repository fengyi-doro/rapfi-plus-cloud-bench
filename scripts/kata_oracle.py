"""Fixed 15x15 KataGomo policy and WDL on the search lane's single CPU."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time
import urllib.request
import zipfile

import numpy as np
import onnxruntime as ort

SOURCE = "https://github.com/hzyhhzy/KataGomo/releases/download/gomoku20261004/katagomo_b24c256h8tflrs.onnx"
DIGEST = "92ec1e7cf101e41c128ec8b0093f6eec893ee5fa757730edeb619e5334c8147c"
MODEL = "katagomo-pv-int8.onnx"
NANO_URL = "https://download.gomocup.com/ai/KATAGOMO26.zip"
NANO_MEMBERS = {1:("standard15x.onnx", 5448739, 0x4b259ad5), 2:("renju15x.onnx", 5448742, 0x1c862737)}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=8)
def global_width(net):
    widths = {i.name:i.shape for i in net.get_inputs()}
    width = widths["input_global"][1]
    if width not in (19, 39) or widths["input_spatial"][1:] != [22, 15, 15]:
        raise ValueError(f"unsupported KataGomo feature schema: {widths}")
    return width


def features(packet, global_size):
    rule, side, last = packet[:3]
    cells = np.frombuffer(packet, dtype=np.uint8, offset=3).reshape(15, 15)
    if rule not in (1, 2) or side not in (0, 1) or (last != 255 and last >= 225) or cells.max() > 2:
        raise ValueError("malformed Rapfi root position")
    spatial = np.zeros((1, 22, 15, 15), np.float32)
    glob = np.zeros((1, global_size), np.float32)
    spatial[0, 0] = 1
    spatial[0, 1] = cells == 1
    spatial[0, 2] = cells == 2
    if last != 255 and global_size == 39:
        spatial[0, 6, last // 15, last % 15] = 1
        glob[0, 1] = 1
    glob[0, 3] = rule == 1
    glob[0, 4] = rule == 2
    glob[0, 5] = 2 * side - 1 if rule == 2 else 0
    return {"input_spatial": spatial, "input_global": glob}


def predict(net, packet):
    policy, wdl = net.run(["out_policy", "out_value"], features(packet, global_width(net)))
    if policy.shape[0] != 1 or policy.size < 226 or policy.size % 226 or wdl.shape != (1, 3):
        raise ValueError(f"unexpected KataGomo heads: {policy.shape}, {wdl.shape}")
    prob = np.exp(wdl[0] - wdl[0].max())
    prob /= prob.sum()
    result = np.empty(226, dtype="<f4")
    result[:225] = policy.reshape(-1, 226)[0, :225]
    result[-1] = prob[0] + .5 * prob[2]
    if not np.isfinite(result).all():
        raise ValueError("non-finite model output")
    return result


@lru_cache(maxsize=1)
def playing_session(directory):
    manifest = json.loads((directory / "oracle.json").read_bytes())
    family = os.environ["RAPFI_ORACLE_FAMILY"]
    if (manifest["family"] != family or manifest["ort"] != ort.__version__
            or family == "b24" and manifest["source_sha256"] != DIGEST):
        raise ValueError("oracle model or runtime mismatch")
    rule_sessions = {}
    hashes = {}
    for key, model in manifest["models"].items():
        path = directory / model["file"]
        if sha256(path) != model["sha256"]:
            raise ValueError(f"playing model digest mismatch for {path.name}")
        net = session(path)
        for rule in ((1, 2) if key == "both" else (int(key),)):
            rule_sessions[rule] = net
            hashes[rule] = model["sha256"]
    if set(rule_sessions) != {1, 2}:
        raise ValueError("incomplete model rules")
    return rule_sessions, hashes


def serve(connection, net, lane_cpu):
    # Linux thread affinity is inherited from the referee. Explicitly verify
    # which vCPU performs each model call even in a four-vCPU search lane.
    os.sched_setaffinity(0, {lane_cpu})
    with connection, connection.makefile("rb") as stream:
        while packet := stream.read(228):
            if len(packet) != 228 or os.sched_getaffinity(0) != {lane_cpu}:
                raise ValueError("truncated oracle request or CPU affinity mismatch")
            connection.sendall(predict(net[packet[0]], packet).tobytes())


def position_packets(rows):
    for row in rows:
        moves, rule = row["moves"], row["rule"]
        side = len(moves) % 2
        cells = bytearray(225)
        for ply, move in enumerate(moves):
            cells[move] = 1 if ply % 2 == side else 2
        yield bytes((rule, side, moves[-1] if moves else 255)) + cells


def prepare_one(source, pv, model, rows):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

    graph = onnx.load(source)
    needed = {"out_policy", "out_value"}
    outputs = [o for o in graph.graph.output if o.name in needed]
    nodes = []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            nodes.append(node)
            needed.update(node.input)
    initializers = [w for w in graph.graph.initializer if w.name in needed]
    infos = [v for v in graph.graph.value_info if v.name in needed]
    for key, values in (("node", reversed(nodes)), ("output", outputs),
                        ("initializer", initializers), ("value_info", infos)):
        field = getattr(graph.graph, key)
        del field[:]
        field.extend(values)
    onnx.checker.check_model(graph)
    onnx.save(graph, pv)
    reference = session(pv)
    global_size = global_width(reference)
    print(f"{source.name} inputs {[(i.name,i.shape,i.type) for i in reference.get_inputs()]}", flush=True)
    op = Counter(n.op_type for n in graph.graph.node)
    probes, calibration = [], []
    for rule in (1, 2):
        group = [r for r in rows if r["rule"] == rule]
        if group:
            probes.extend(position_packets(group[:6]))
            calibration.extend(position_packets(group[6:30]))
    if op["MatMulInteger"] or op["QLinearConv"] or op["DequantizeLinear"]:
        # Tournament releases may already contain quantized weights.
        shutil.copyfile(pv, model)
        precision = "upstream quantization, policy/value subgraph"
    elif op["MatMul"] > op["Conv"]:
        quantize_dynamic(pv, model, weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"], extra_options={"MatMulConstBOnly": True})
        precision = "dynamic QInt8 constant MatMul/Gemm, other ops FP32"
    else:
        class Calibration(CalibrationDataReader):
            def __init__(self):
                self.packets = iter(calibration)

            def get_next(self):
                packet = next(self.packets, None)
                return features(packet, global_size) if packet is not None else None

        quantize_static(pv, model, Calibration(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm"])
        precision = "static per-channel INT8 QDQ Conv/MatMul/Gemm, other ops FP32"
    operators = Counter(n.op_type for n in onnx.load(model).graph.node)
    if not any(operators[op] for op in ("MatMulInteger", "QLinearConv", "DequantizeLinear")):
        raise ValueError(f"no quantized operators: {operators}")
    quantized = session(model)
    if quantized.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("oracle must use only the CPU provider")
    measurements = []
    for packet in probes:
        upstream = predict(reference, packet)
        start = time.perf_counter_ns()
        q = predict(quantized, packet)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        cells = np.frombuffer(packet, dtype=np.uint8, offset=3)
        available = np.flatnonzero(cells == 0)
        measurements.append(dict(rule=packet[0], ply=int(np.count_nonzero(cells)),
                                 eval_abs_delta=abs(float(upstream[-1] - q[-1])),
                                 top1_agree=bool(upstream[available].argmax() == q[available].argmax()), ms=elapsed))
    return dict(file=model.name, source_member=source.name, source_sha256=sha256(source),
                sha256=sha256(model), bytes=model.stat().st_size, precision=precision, input_globals=global_size,
                operators=dict(operators), probes=measurements)


def prepare(directory, book, family):
    import onnx
    directory.mkdir(parents=True, exist_ok=True)
    rows = json.loads(book.read_bytes())
    rows = [r for r in rows if r["rule"] in (1, 2)]
    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        if family == "b24":
            source = work / "upstream.onnx"
            urllib.request.urlretrieve(SOURCE, source)
            if sha256(source) != DIGEST:
                raise ValueError("upstream model digest mismatch")
            url, source_digest = SOURCE, DIGEST
            models = {"both":prepare_one(source, work/"pv.onnx", directory/MODEL, rows)}
        else:
            package = work / "KATAGOMO26.zip"
            subprocess.run(["curl", "-fLsS", "--retry", "3", "-o", str(package), NANO_URL], check=True)
            url, source_digest = NANO_URL, sha256(package)
            if package.stat().st_size != 34770229:
                raise ValueError("tournament package size changed")
            models = {}
            with zipfile.ZipFile(package) as archive:
                for rule, (member, size, crc) in NANO_MEMBERS.items():
                    meta = archive.getinfo(member)
                    if (meta.file_size, meta.CRC) != (size, crc):
                        raise ValueError(f"unexpected tournament member: {member}")
                    source = work / member
                    source.write_bytes(archive.read(meta))
                    models[str(rule)] = prepare_one(source, work/"pv.onnx", directory/("pv-int8-"+member),
                                                    [r for r in rows if r["rule"] == rule])
        probes = [p for m in models.values() for p in m["probes"]]
        if {p["rule"] for p in probes} != {1, 2}:
            raise ValueError("missing rule coverage in model probes")
        identity = {k:m["sha256"] for k,m in models.items()}
        digest = (identity["both"] if "both" in identity else
                  hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
        operators = sum((Counter(m["operators"]) for m in models.values()), Counter())
        manifest = dict(family=family, source=url, source_sha256=source_digest, model_sha256=digest,
                        models=models, bytes=sum(m["bytes"] for m in models.values()), ort=ort.__version__,
                        onnx=onnx.__version__, providers=["CPUExecutionProvider"], intra_threads=1, inter_threads=1,
                        operators=dict(operators), probes=probes, median_probe_ms=statistics.median(p["ms"] for p in probes),
                        max_wdl_score_delta=max(p["eval_abs_delta"] for p in probes))
        (directory / "oracle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{family} source {source_digest}, files {identity}, CPU median {manifest['median_probe_ms']:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--family", choices=("b24", "nano"), default="b24")
    args = parser.parse_args()
    prepare(args.out, args.positions, args.family)
