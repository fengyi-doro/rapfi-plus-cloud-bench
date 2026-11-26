"""Produce an auditable patch and check that both playing targets used identical build settings."""
import difflib
import json
from datasets import DATA, ROOT, dump, sha


def run():
    old,new=ROOT/"src_orig",ROOT/"src_plus"
    names=sorted({p.relative_to(tree).as_posix() for tree in (old,new) for p in tree.rglob("*") if p.is_file()})
    diff,changed=[],[]
    for name in names:
        a,b=old/name,new/name
        left=a.read_bytes() if a.exists() else b""
        right=b.read_bytes() if b.exists() else b""
        if left==right:
            continue
        changed.append(name)
        diff.extend(difflib.unified_diff(left.decode().splitlines(True),right.decode().splitlines(True),
                    "a/"+name if a.exists() else "/dev/null","b/"+name if b.exists() else "/dev/null"))
    patch=ROOT/"patches/rapfi-plus.patch"
    patch.write_bytes("".join(diff).encode())
    manifests=[json.loads((ROOT/"build"/v/"manifest.json").read_bytes()) for v in ("orig","plus")]
    assert all(manifests[0][k]==manifests[1][k] for k in ("zig","flags","cpp"))
    for manifest in manifests:
        assert sha(ROOT/manifest["exe"])==manifest["sha256"]
    def normalize(cmd, v):
        return [s.replace("src_"+v,"src_engine").replace("build\\"+v+"\\","build\\engine\\")
                .replace("build/"+v+"/","build/engine/").replace("pbrain-rapfi-"+v,"pbrain-rapfi-engine") for s in cmd]
    original,plus=manifests
    commands={n:normalize(cmd,"plus") for n,cmd in zip(plus["sources"],plus["commands"][:-1],strict=True)}
    assert all(commands[n]==normalize(cmd,"orig") for n,cmd in zip(original["sources"],original["commands"][:-1],strict=True))
    assert normalize(original["commands"][-1],"orig")==[
        s for s in normalize(plus["commands"][-1],"plus") if not s.endswith("search_ab_rzone.cpp.o")]
    dump(DATA/"build_audit.json",dict(compiler=original["zig"],identical_flags=True,
         common_translation_units=len(original["sources"]),changed_or_added=changed,source_patch_sha256=sha(patch)))


if __name__=="__main__":
    run()
