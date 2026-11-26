"""Build an F9-only trial or a complete playing binary with the release's Zig flags."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from datasets import ROOT, dump, sha
sys.path.insert(0, str(ROOT))
from build import CPP, FLAGS, ZIG, compile_unit


def build(variant, target, jobs, out, build_dir=None, profile=None):
    os.chdir(ROOT)
    tree = Path("src_orig" if variant == "orig" else "src_plus")
    windows = target == "windows" or (target == "native" and os.name == "nt")
    target_args = [] if target == "native" else [
        "-target", "x86_64-windows-gnu" if windows else "x86_64-linux-musl"]
    dest = build_dir or Path("build") / (variant if target == "native" else f"{target}-{variant}")
    dest.mkdir(parents=True, exist_ok=True)
    names = re.findall(r"^\s*([\w/.]+\.cpp)\s*$",
                       (tree / "CMakeLists.txt").read_text().split("main.cpp", 1)[0], re.M)
    names += ["main.cpp", "external/lz4/src/lz4_all.c", "external/lz4/src/xxhash.c"]
    partial = (variant == "f9" and target == "native" and windows
               and Path("build/plus/manifest.json").is_file() and Path("data/v2/release.json").is_file())
    if partial:
        base = json.loads(Path("build/plus/manifest.json").read_bytes())
        release = json.loads(Path("data/v2/release.json").read_bytes())
        partial = (base["sha256"] == release["exe"]["plus"] and base["sources"] == names
                   and all(Path(obj).is_file() for obj in base["objects"]))
    flags = target_args + (["-DRAPFI_LINE_QUOTIENT=1", "-DRAPFI_DISABLE_PAIRING"] if variant == "f9" else [])
    profile = profile or variant
    if variant != "orig" and profile != variant:
        raise ValueError("the candidate and its build profile must match")
    if not windows:
        flags.append("-DRAPFI_AVX2_ONLY")
    if profile == "v4":
        flags += ["-DRAPFI_F10", "-DRAPFI_V4_SCATTER", "-DRAPFI_V4_POLICY_TILE"]
        if windows:
            flags += ["-mavxvnni", "-DUSE_VNNI"]

    def compile_one(item):
        index, name = item
        if partial and name not in ("game/pattern.cpp", "game/board.cpp", "game/movegen.cpp"):
            return base["objects"][index]
        obj = dest / (name.replace("/", "_") + ".o")
        cmd = compile_unit(tree / name, tree, obj, flags)
        result = subprocess.run(cmd, capture_output=True)
        obj.with_suffix(".log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace") + name)
        return str(obj)

    with ThreadPoolExecutor(jobs) as pool:
        objects = list(pool.map(compile_one, enumerate(names)))
    out = out or (dest if variant == "f9" else ROOT)
    out.mkdir(parents=True, exist_ok=True)
    name = "plus-" + variant if variant in ("v3", "v4") else variant
    exe = out / ("pbrain-rapfi-" + name + (".exe" if windows else ""))
    libs = ["-ladvapi32"] if windows else ["-pthread"]
    # The native Windows gate reads its own PDB. Hosted match jobs only need
    # the protocol and can download a stripped executable.
    strip = [] if target == "native" else ["-s"]
    cmd = [*ZIG, *CPP[:2], *FLAGS, *target_args, *strip, "-static", *objects, *libs, "-o", str(exe)]
    subprocess.run(cmd, check=True)
    dump(dest / "manifest.json", dict(variant=variant, profile=profile, target=target, exe=str(exe), sha256=sha(exe),
        zig=subprocess.check_output(ZIG+["version"], text=True).strip(), flags=FLAGS+flags,
        cpp=CPP, sources=names, objects=objects, link=cmd))
    print(exe, sha(exe), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["orig", "v3", "v4", "f9"])
    p.add_argument("--target", choices=["native", "linux", "windows"], default="native")
    p.add_argument("--profile", choices=["v3", "v4"], help="apply the candidate's complete build flags to both trees")
    p.add_argument("--out", type=Path)
    p.add_argument("--build-dir", type=Path, help="object and manifest directory for an isolated trial")
    p.add_argument("-j", type=int, default=4)
    a = p.parse_args()
    build(a.variant, a.target, a.j, a.out, a.build_dir, a.profile)

