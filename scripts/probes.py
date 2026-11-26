"""Build diagnostic programs using the matching playing object manifest."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import compile_unit, link


def run(variants=("orig","plus")):
    os.chdir(ROOT)
    for variant in variants:
        tree = Path("src_orig" if variant == "orig" else "src_plus")
        dest = Path("build") / variant
        manifest = json.loads((dest / "manifest.json").read_bytes())
        extra = [*manifest["flags"], "-I" + str(tree)]
        obj = dest / "vcf_probe.o"
        subprocess.run(compile_unit(Path("scripts/vcf_probe.cpp"), tree, obj, extra), check=True)
        objects = [o for o in manifest["objects"] if Path(o).name not in ("main.cpp.o", "search_ab_search.cpp.o")]
        subprocess.run(link([*objects, obj], dest / "vcf_probe.exe"), check=True)
        if variant in ("plus", "v3", "v4"):
            obj = dest / "defense_probe.o"
            subprocess.run(compile_unit(Path("scripts/defense_probe.cpp"), tree, obj, extra), check=True)
            objects = [o for o in manifest["objects"] if Path(o).name != "main.cpp.o"]
            subprocess.run(link([*objects, obj], dest / "defense_probe.exe"), check=True)
            if variant in ("v3", "v4"):
                continue
            obj = dest / "kernel_check.o"
            subprocess.run(compile_unit(Path("scripts/kernel_check.cpp"), tree, obj, ["-I" + str(tree)]), check=True)
            exe = dest / "kernel_check.exe"
            subprocess.run(link([obj], exe), check=True)
            (ROOT/"validation/kernels.json").write_bytes(subprocess.check_output([str(exe)]))


if __name__ == "__main__":
    run(sys.argv[1:] or ("orig", "plus"))
