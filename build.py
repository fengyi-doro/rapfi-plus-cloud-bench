"""Build the upstream playing target and its CPU variants with one Zig/Clang toolchain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
ZIG = [sys.executable, "-m", "ziglang"]
FLAGS = ["-O3", "-march=x86_64_v3", "-mavx2", "-mfma", "-mbmi2", "-ffp-contract=off", "-DNDEBUG",
         "-DUSE_AVX2", "-DUSE_SSE", "-DMULTI_THREADING", "-Wno-deprecated-declarations"]
CPP = ["c++", "-std=c++20", "-fno-char8_t", "-include", "bit"]
INCLUDES = ["external/cpptoml/include", "external/cxxopts/include", "external/lz4/include",
            "external/simde/include"]
MODULES = {"orig": (), "p1": (1,), "p2": (2,), "p3": (3,), "p4": (4,),
           "p123": (1, 2, 3), "plus": (1, 3, 4)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_unit(source, tree, destination, extra=()):
    cxx = source.suffix == ".cpp"
    return ZIG + (CPP if cxx else ["cc", "-std=c11"]) + FLAGS + [
        "-I" + str(tree / include) for include in INCLUDES
    ] + list(extra) + ["-MMD", "-MF", str(destination.with_suffix(".d")),
                       "-c", str(source), "-o", str(destination)]


def link(objects, output):
    library = "-ladvapi32" if os.name == "nt" else "-pthread"
    return ZIG + CPP[:2] + FLAGS + ["-static", *map(str, objects), library, "-o", str(output)]


def mix9_ablation(tree, dest, enabled):
    """Restore complete upstream functions for inactive modules, including the original loop shape."""
    path = Path("eval/mix9svqnnue.cpp")
    code = (tree / path).read_text(encoding="utf-8")
    original = (Path("src_orig") / path).read_text(encoding="utf-8")
    sections = [(3, "void Accumulator::move(", "std::tuple<float, float, float> Accumulator::evaluateValue("),
                (1, "std::tuple<float, float, float> Accumulator::evaluateValue(", "void Accumulator::evaluatePolicy("),
                (2, "void Accumulator::evaluatePolicy(", "Evaluator::Evaluator(")]
    for module, first, following in sections:
        if module not in enabled:
            a, b = code.index(first), code.index(following)
            code = code[:a] + original[original.index(first):original.index(following)] + code[b:]
    source = dest / "mix9svqnnue.cpp"
    source.write_text(code, encoding="utf-8")
    return source


def build(variant, jobs):
    os.chdir(ROOT)
    tree = Path("src_orig" if variant == "orig" else "src_plus")
    cmake = (tree / "CMakeLists.txt").read_text()
    sources = re.findall(r"^\s*([\w/.]+\.cpp)\s*$", cmake.split("main.cpp", 1)[0], re.M)
    sources += ["main.cpp", "external/lz4/src/lz4_all.c", "external/lz4/src/xxhash.c"]
    dest = Path("build") / variant
    dest.mkdir(parents=True, exist_ok=True)
    # The two playing executables each compile every TU from their own source tree.
    # Ablations rebuild changed evaluator/search TUs and link the other playing objects.
    ablation = variant not in ("orig", "plus")
    def compile_source(name):
        obj = dest / (name.replace("/", "_") + ".o")
        source = tree / name
        extra = []
        if ablation and name == "eval/mix9svqnnue.cpp" and variant in ("p1", "p2", "p3"):
            source = mix9_ablation(tree, dest, MODULES[variant])
            extra = ["-iquote", str(tree/"eval")]
        elif ablation:
            use_plus = (variant == "p123" and name != "search/ab/search.cpp"
                        or name == "search/ab/rzone.cpp"
                        or variant == "p4" and name in ("search/ab/search.cpp", "command/gomocup.cpp")
                        or name == "eval/mix10nnue.cpp" and 1 in MODULES[variant])
            obj = Path("build/plus" if use_plus else "build/orig") / obj.name
            if not obj.is_file():
                raise RuntimeError("Build orig and plus before ablations")
            return str(obj), None
        if os.name != "nt":
            extra.append("-DRAPFI_AVX2_ONLY")
        cmd = compile_unit(source, tree, obj, extra)
        p = subprocess.run(cmd, capture_output=True)
        obj.with_suffix(".log").write_bytes(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError((p.stdout + p.stderr).decode(errors="replace") + "\n" + name)
        return str(obj), cmd
    with ThreadPoolExecutor(jobs) as pool:
        compiled = list(pool.map(compile_source, sources))
    objects = [obj for obj, _ in compiled]
    exe = (dest if ablation else Path(".")) / (f"pbrain-rapfi-{variant}" + (".exe" if os.name == "nt" else ""))
    command = link(objects, exe)
    subprocess.run(command, check=True)
    manifest = dict(variant=variant, zig=subprocess.check_output(ZIG + ["version"], text=True).strip(),
                    flags=FLAGS + ([] if os.name == "nt" else ["-DRAPFI_AVX2_ONLY"]), cpp=CPP, sources=sources, objects=objects,
                    commands=[cmd for _, cmd in compiled if cmd] + [command],
                    exe=str(exe), sha256=digest(exe))
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(variant, manifest["sha256"], exe, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="*", metavar="VARIANT", help="one or more of " + ", ".join(MODULES))
    parser.add_argument("-j", type=int, default=6)
    args = parser.parse_args()
    if any(name not in MODULES for name in args.variant):
        parser.error("unknown variant")
    for name in args.variant or ("orig", "plus"):
        build(name, args.j)
