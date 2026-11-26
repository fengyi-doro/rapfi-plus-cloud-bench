"""Verify build parity and exercise the two Linux engines on an AVX2-only CPU model."""
import argparse
import json
from pathlib import Path
import subprocess

from bit_gate import board_command
from datasets import DATA, ROOT, dump, sha


def check(directory, profile):
    manifests = [json.loads((directory / f"{variant}.json").read_bytes()) for variant in ("orig", profile)]
    shared = {key: manifests[0][key] for key in ("target", "profile", "zig", "flags", "cpp")}
    if any(shared[key] != manifests[1][key] for key in shared):
        raise ValueError("toolchain, profile or compilation flags differ between engines")
    if shared["target"] != "linux" or shared["profile"] != profile:
        raise ValueError("cached engines are for a different target or build profile")
    required = {"-march=x86_64_v3", "-mavx2", "-mfma", "-mbmi2", "-DRAPFI_AVX2_ONLY"}
    flags = shared["flags"]
    if not required.issubset(flags) or any("vnni" in f.lower() or "avx512" in f.lower() or "native" in f for f in flags):
        raise ValueError(f"unexpected ISA flags: {flags}")
    runtime = json.loads((DATA / "runtime_files.json").read_bytes())
    if any(sha(ROOT / name) != digest for name, digest in runtime.items()):
        raise ValueError("model or config SHA-256 differs from the frozen release")
    # TRACESEARCH synchronously exercises the value and policy evaluation on both rule sets.
    board = board_command([112, 113, 100, 126, 98])
    commands = ["START 15", "INFO thread_num 1"]
    for rule in (1, 4):
        commands.extend([f"INFO rule {rule}", "RESTART", board, "TRACESEARCH"])
    commands.extend(["END", ""])
    binaries = {}
    for manifest in manifests:
        binary = directory / Path(manifest["exe"]).name
        digest = sha(binary)
        if digest != manifest["sha256"]:
            raise ValueError(f"executable hash mismatch: {binary}")
        result = subprocess.run(["qemu-x86_64", "-cpu", "Haswell", str(binary), "--config", str(ROOT / "config.toml")],
                                input="\n".join(commands), text=True, capture_output=True, timeout=120)
        required_weights = ("standard_bs15", "renju_bs15_black", "renju_bs15_white")
        if result.returncode or result.stdout.count("Static Eval[Black]") != 2 or not all(w in result.stdout for w in required_weights):
            raise RuntimeError(f"{binary}: {result.returncode}\n{result.stdout[-5000:]}\n{result.stderr[-1500:]}")
        binaries[manifest["variant"]] = digest
        print(f"{binary.name}: Haswell Standard and Renju evaluation passed; SHA-256 {digest}")
    dump(directory / "build-info.json", dict(**shared, binaries=binaries, runtime=runtime, smoke_cpu="QEMU Haswell"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("profile", choices=["v3", "v4"])
    args = parser.parse_args()
    check(args.directory, args.profile)
