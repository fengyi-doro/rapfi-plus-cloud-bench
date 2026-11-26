"""Exercise the complete production Mix10 large head with a reproducible, untrained fixture."""
import hashlib
import json
import struct
from tempfile import TemporaryDirectory
from pathlib import Path
import lz4.frame
import numpy as np
from bit_gate import capture
from datasets import DATA, ROOT, dump, sha


def network(path):
    # Each 64-channel codebook row is exactly 80 packed bytes. Expanding the
    # matching indices directly preserves the original ten-bit feature stream.
    source = ROOT/"mix9svqfreestyle_bsmix.bin.lz4"
    data = lz4.frame.decompress(source.read_bytes())
    magic,_,_,_,description = struct.unpack_from("<5I",data)
    offset = 20+description
    def take(n):
        nonlocal offset
        result = data[offset:offset+n]
        assert len(result) == n
        offset += n
        return result
    codebook = np.frombuffer(take(2*65536*80),np.uint8).reshape(2,65536,80)
    indices = np.frombuffer(take(2*442503*2),"<u2").reshape(2,442503)
    conv = take(640)
    def fc(out, ins):
        return take(out*(ins+4))
    pw1,pw2=fc(64,64),fc(528,64)
    stars=[[fc(128,64),fc(128,64),fc(64,64)] for _ in range(4)]
    l1,l2,l3=fc(64,320),fc(64,64),fc(4,64)
    output,bias=take(64),take(4)
    take(44)
    assert offset==len(data)
    small=pw2[:272*64]+pw2[528*64:528*64+272*4]
    arch=0xb53b0258 ^ (8<<16 | 4<<8 | 8)
    note=b"Rapfi Plus full-head equivalence fixture. Not jointly trained."
    with lz4.frame.open(path,mode="wb") as out:
        out.write(struct.pack("<5I",magic,arch,7,1<<14,len(note))+note)
        for side in range(2):
            out.write(codebook[side,indices[side]].tobytes())
        for component in (conv,pw1,small,pw1,pw2,small,pw1,l2,l3,bytes(48),stars[0][0],
                          *(s[2] for s in stars),l1,l2,l3,bytes(48),output,output,bias,bias,bytes(56)):
            out.write(component)
    return sha(path)


def run():
    rows=json.loads((DATA/"positions.json").read_bytes())[::5]
    with TemporaryDirectory(dir=ROOT/"build") as tmp:
        model=Path(tmp)/"mix10_fixture.bin.lz4"
        model_sha=network(model)
        original=(ROOT/"config.toml").read_text(encoding="utf-8")
        first,following=original.split("[[model.evaluator.weights]]",1)
        config=Path(tmp)/"config.toml"
        first=first.replace('type = "mix9svq"','type = "mix10"').replace(
            '"model210901.bin"',f'"{(ROOT/"model210901.bin").as_posix()}"')
        config.write_text(first+'[[model.evaluator.weights]]\nweight_file = "'+
                          model.as_posix()+'"\n\n[search]'+following.split("[search]",1)[1],encoding="utf-8")
        values={v:capture(v,rows,"mix10","mix10",config) for v in ("orig","plus")}
        assert values["orig"]==values["plus"]
        distinct=len({tuple(p["wdl"]) for p in values["orig"]})
        assert distinct>=10,distinct
        summary=dict(positions=len(rows),distinct_wdl=distinct,policy_words=sum(r["empty"]*2 for r in values["orig"]),
            mismatches=0,fixture_sha256=model_sha,exe={v:sha(ROOT/f"pbrain-rapfi-{v}.exe") for v in values},
            raw_sha256={v:hashlib.sha256(json.dumps(p,separators=(",",":")).encode()).hexdigest()
                        for v,p in values.items()},training="compatible tensor layout fixture, not jointly trained")
        dump(ROOT/"validation/mix10.json",summary)
        print("Mix10 full head",summary,flush=True)


if __name__=="__main__":
    run()
