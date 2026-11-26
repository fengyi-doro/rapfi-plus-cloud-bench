"""Link CPU-only F23 arms from one set of ABI-identical objects and pinned controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_v3 import build, compile_unit
from datasets import ROOT, DATA, dump, sha

ORIG = "622200dd31b39e4120cccec3ea3a47eaadda8c7fc231e7763b514474fea26ab9"
V4 = "299e3ba3daaed9d3d317285e0c349b4543938b707829729120ecc651841b60d4"
BASE_SHA = "9f4858626bffebc1d9e47de243c304bb69180108"
ARMS = {"a": 1, "b1": 2, "b2": 4, "b3": 8, "b4": 16}
COUNTERS = ("root_calls", "main_calls", "vcf_calls", "root_proofs", "main_proofs",
            "vcf_proofs", "exhausted", "singular_extensions", "quiet_malus",
            "residual_updates", "residual_uses", "lmr_adjustments", "proof_white_nodes", "proof_ns")


def arm_name(mask):
    return "f23-" + "-".join(name for name, bit in ARMS.items() if mask & bit) if mask else "v4"


def filename(name):
    return "pbrain-rapfi-" + ("plus-v4" if name == "v4" else name)


def load(path):
    return json.loads(path.read_bytes())


def controls(out):
    source = ROOT / "historical"
    if not (source / "build-info.json").exists():
        frozen = ROOT / "frozen"
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=frozen, text=True).strip() != BASE_SHA:
            raise ValueError("wrong baseline checkout")
        source.mkdir(exist_ok=True)
        for variant in ("orig", "v4"):
            subprocess.run([sys.executable, "scripts/build_v3.py", variant, "--target", "linux",
                            "--profile", "v4", "--out", str(source), "-j", "4"], cwd=frozen, check=True)
            shutil.copyfile(frozen / "build" / f"linux-{variant}" / "manifest.json", source / f"{variant}.json")
    for name, digest in (("orig", ORIG), ("v4", V4)):
        src = source / filename(name)
        if sha(src) != digest or load(source / f"{name}.json")["sha256"] != digest:
            raise ValueError(f"historical binary hash mismatch for {name}")
        shutil.copy(src, out / src.name)
    return load(source / "v4.json")


def link_arm(mask, shared, out):
    label = arm_name(mask)
    tree = Path("src_plus")
    dest = Path("build") / label
    dest.mkdir(parents=True, exist_ok=True)
    objects = []
    for name, old in zip(shared["sources"], shared["objects"], strict=True):
        if name in ("search/ab/search.cpp", "search/ab/history.cpp"):
            obj = dest / Path(old).name
            extra = shared["flags"][len(shared["flags"]) - 6:]  # target, ISA fence and three v4 switches
            if extra[:2] != ["-target", "x86_64-linux-musl"]:
                raise ValueError(f"unexpected compile flags {extra}")
            subprocess.run(compile_unit(tree / name, tree, obj, [*extra, f"-DRAPFI_F23_MASK={mask}"]), check=True)
            objects.append(str(obj))
        else:
            objects.append(old)
    exe = out / filename(label)
    start = shared["link"].index(shared["objects"][0])
    subprocess.run([*shared["link"][:start], *objects, "-pthread", "-o", str(exe)], check=True)
    return label, sha(exe)


def main(masks):
    if any(mask < 0 or mask > 31 for mask in masks):
        raise ValueError("invalid F23 module mask")
    runtime = load(DATA / "runtime_files.json")
    if any(sha(ROOT / name) != digest for name, digest in runtime.items()):
        raise ValueError("runtime differs from the frozen match")
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    historical = controls(out)
    objdir = Path("build/linux-v4")
    if not (objdir / "manifest.json").exists():
        build("v4", "linux", 4, Path("build/shared"), objdir, "v4")
    shared = load(objdir / "manifest.json")
    for key in ("profile", "cpp", "target", "zig", "flags"):
        if shared[key] != historical[key]:
            raise ValueError(f"historical toolchain parity failed: {key}")
    binaries = {"orig": ORIG, "v4": V4}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for entry in pool.map(lambda mask: link_arm(mask, shared, out), sorted(set(masks) - {0})):
            binaries[entry[0]] = entry[1]
    dump(out / "f23-build.json", dict(binaries=binaries, runtime=runtime, baseline_commit=BASE_SHA,
                                     masks={arm_name(m): m for m in set(masks)}, flags=shared["flags"],
                                     zig=shared["zig"], cpp=shared["cpp"]))
    for binary in out.glob("pbrain-rapfi-*"):
        binary.chmod(0o755)
    print(json.dumps(binaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("masks", nargs="+", type=int)
    main(parser.parse_args().masks)
