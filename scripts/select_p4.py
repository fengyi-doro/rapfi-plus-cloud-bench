"""Archive the screened binaries and apply the predeclared admission result to the playing source."""
import difflib
import json
from pathlib import Path
import shutil
from datasets import ROOT, dump, sha
from engine import filename


def apply():
    screen = ROOT/"bench/screen"
    result = json.loads((screen/"summary.json").read_bytes())
    archive = screen/"binaries"
    archive.mkdir(exist_ok=True)
    for v, expected in result["exe"].items():
        binary = filename(v)
        target = archive / binary.name
        if not target.exists():
            assert sha(binary)==expected, f"screening binary changed: {v}"
            shutil.copyfile(binary,target)
        assert sha(target)==expected
    old = ROOT/"src_orig/search/ab/search.cpp"
    new = ROOT/"src_plus/search/ab/search.cpp"
    patch = ROOT/"patches/p4_dual_five_trial.patch"
    if not patch.exists():
        patch.write_bytes("".join(difflib.unified_diff(old.read_text().splitlines(True),
             new.read_text().splitlines(True),"a/search/ab/search.cpp","b/search/ab/search.cpp")).encode())
    assert patch.stat().st_size
    if not result["admit_playing_shortcut"]:
        new.write_bytes(old.read_bytes())
    dump(screen/"decision.json",dict(admitted=result["admit_playing_shortcut"],source_sha256=sha(new),
                                     trial_patch_sha256=sha(patch)))
    print("Playing shortest-VCF specialization admitted:", result["admit_playing_shortcut"])


if __name__ == "__main__":
    apply()
