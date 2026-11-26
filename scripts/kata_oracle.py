"""Fixed 15x15 KataGomo policy and WDL on the search lane's single CPU."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

import numpy as np
import onnxruntime as ort

SOURCE = "https://github.com/hzyhhzy/KataGomo/releases/download/gomoku20261004/katagomo_b24c256h8tflrs.onnx"
DIGEST = "92ec1e7cf101e41c128ec8b0093f6eec893ee5fa757730edeb619e5334c8147c"
MODEL = "katagomo-pv-int8.onnx"


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


def features(packet):
    rule, side, last = packet[:3]
    cells = np.frombuffer(packet, dtype=np.uint8, offset=3).reshape(15, 15)
    if rule not in (1, 2) or side not in (0, 1) or (last != 255 and last >= 225) or cells.max() > 2:
        raise ValueError("malformed Rapfi root position")
    spatial = np.zeros((1, 22, 15, 15), np.float32)
    glob = np.zeros((1, 39), np.float32)
    spatial[0, 0] = 1
    spatial[0, 1] = cells == 1
    spatial[0, 2] = cells == 2
    if last != 255:
        spatial[0, 6, last // 15, last % 15] = 1
        glob[0, 1] = 1
    glob[0, 3] = rule == 1
    glob[0, 4] = rule == 2
    glob[0, 5] = 2 * side - 1 if rule == 2 else 0
    return {"input_spatial": spatial, "input_global": glob}


def predict(net, packet):
    policy, wdl = net.run(["out_policy", "out_value"], features(packet))
    if policy.shape != (1, 4, 226) or wdl.shape != (1, 3):
        raise ValueError(f"unexpected KataGomo heads: {policy.shape}, {wdl.shape}")
    prob = np.exp(wdl[0] - wdl[0].max())
    prob /= prob.sum()
    result = np.empty(226, dtype="<f4")
    result[:225] = policy[0, 0, :225]
    result[-1] = prob[0] + .5 * prob[2]
    if not np.isfinite(result).all():
        raise ValueError("non-finite model output")
    return result


@lru_cache(maxsize=1)
def playing_session(directory):
    manifest = json.loads((directory / "oracle.json").read_bytes())
    if (manifest["source_sha256"] != DIGEST or sha256(directory / MODEL) != manifest["model_sha256"]
            or manifest["ort"] != ort.__version__):
        raise ValueError("oracle model or runtime mismatch")
    return session(directory / MODEL)


def serve(connection, net, lane_cpu):
    # Linux thread affinity is inherited from the referee. Explicitly verify
    # which vCPU performs each model call even in a four-vCPU search lane.
    os.sched_setaffinity(0, {lane_cpu})
    with connection, connection.makefile("rb") as stream:
        while packet := stream.read(228):
            if len(packet) != 228 or os.sched_getaffinity(0) != {lane_cpu}:
                raise ValueError("truncated oracle request or CPU affinity mismatch")
            connection.sendall(predict(net, packet).tobytes())


def prepare(directory, book):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from tempfile import TemporaryDirectory

    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        source, pv = (Path(temporary) / f for f in ("upstream.onnx", "pv.onnx"))
        urllib.request.urlretrieve(SOURCE, source)
        if sha256(source) != DIGEST:
            raise ValueError("upstream model digest mismatch")
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
        model = directory / MODEL
        # Quantize constant dense projection weights. Attention between two
        # activations, convolutions, normalization and WDL softmax stay floating point.
        quantize_dynamic(pv, model, weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"], extra_options={"MatMulConstBOnly": True})
        operators = Counter(n.op_type for n in onnx.load(model).graph.node)
        if not operators["MatMulInteger"]:
            raise ValueError("no INT8 projection in derived model")
        reference = session(pv)
        quantized = session(model)
        if quantized.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("oracle must use only the CPU provider")
        # Use independent frozen diagnostic positions, not the tournament opening book.
        rows = json.loads(book.read_bytes())
        probes = []
        for rule in (1, 2):
            for row in [r for r in rows if r["rule"] == rule][:6]:
                moves = row["moves"]
                side = len(moves) % 2
                cells = bytearray(225)
                for ply, move in enumerate(moves):
                    cells[move] = 1 if ply % 2 == side else 2
                packet = bytes((rule, side, moves[-1] if moves else 255)) + cells
                fp = predict(reference, packet)
                start = time.perf_counter_ns()
                q = predict(quantized, packet)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                available = np.flatnonzero(np.frombuffer(cells, dtype=np.uint8) == 0)
                probes.append(dict(rule=rule, ply=len(moves), eval_abs_delta=abs(float(fp[-1] - q[-1])),
                                   top1_agree=bool(fp[available].argmax() == q[available].argmax()), ms=elapsed))
        if {p["rule"] for p in probes} != {1, 2}:
            raise ValueError("missing rule coverage in model probes")
        manifest = dict(source=SOURCE, source_sha256=DIGEST, model_sha256=sha256(model), bytes=model.stat().st_size,
                        precision="per-channel dynamic QInt8 constant MatMul/Gemm, other ops FP32",
                        ort=ort.__version__, onnx=onnx.__version__, providers=quantized.get_providers(),
                        intra_threads=1, inter_threads=1, operators=dict(operators), probes=probes,
                        median_probe_ms=statistics.median(p["ms"] for p in probes),
                        max_wdl_score_delta=max(p["eval_abs_delta"] for p in probes))
        (directory / "oracle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{operators['MatMulInteger']} INT8 matrices, {len(probes)} FP32 comparison probes, "
              f"median {manifest['median_probe_ms']:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--positions", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out, args.positions)
