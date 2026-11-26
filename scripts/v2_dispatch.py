"""Observe the native four-lane dispatch directly in the finished playing PE."""
import json
import re
import subprocess
import time
from bit_gate import CDB,ADDR,board_command
from datasets import ROOT,DATA,sha,dump

def run():
    release=json.loads((DATA/"v2/release.json").read_bytes())
    out=ROOT/"validation/v2_dispatch.json"
    if "orbit" not in release["selected"]:
        dump(out,dict(enabled=False))
        return
    exe=ROOT/"pbrain-rapfi-plus.exe"
    module="pbrain_rapfi_plus"
    args=[CDB,"-sins","-y",str(exe.parent)]
    listing=subprocess.check_output([*args,"-c",f"lm m {module}; x {module}!*orbitDot4*; q",str(exe)],
           encoding="utf8",errors="replace",timeout=25)
    number=lambda s:int(s.replace("\x60",""),16)
    base=number(re.search(rf"(?m)^({ADDR}) {ADDR}\s+{module}\s+",listing)[1])
    funcs=[(number(p)-base,name) for p,name in re.findall(rf"(?m)^({ADDR}) ({module}!\S*orbitDot4\S*)",listing)]
    assert len(funcs)>=2,listing[-1500:]
    commands=[f'bp {module}+0x{delta:x} ".echo ORBIT_ENTER_{i}; gc"' for i,(delta,name) in enumerate(funcs)]
    commands+=[".echo DISPATCH_READY","g"]
    log=ROOT/"validation/v2_dispatch.log"
    e=subprocess.Popen([exe,"--config",ROOT/"config.toml"],cwd=ROOT,stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    d=None
    try:
        with log.open("wb") as f:
            d=subprocess.Popen([*args,"-G","-c","; ".join(commands),"-p",str(e.pid)],stdout=f,stderr=subprocess.STDOUT,
                 cwd=ROOT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            until=time.monotonic()+20
            while time.monotonic()<until:
                if re.search(rb"(?m)^DISPATCH_READY\r?$",log.read_bytes()):break
                if d.poll() is not None:raise RuntimeError(log.read_text(errors="replace")[-1600:])
                time.sleep(.025)
            else:raise TimeoutError("CDB dispatch observer")
            row=json.loads((DATA/"positions.json").read_bytes())[0]
            protocol="START 15\nINFO rule 0\n"+board_command(row["moves"])+"\nTRACESEARCH\nEND\n"
            msg=e.communicate(protocol.encode(),timeout=40)[0]
            assert e.returncode==0 and b"ERROR " not in msg
            d.wait(timeout=15)
    finally:
        for p in (d,e):
            if p and p.poll() is None:
                p.kill();p.wait()
    data=log.read_text(errors="replace")
    hits={name:len(re.findall(rf"(?m)^ORBIT_ENTER_{i}\s*$",data)) for i,(_,name) in enumerate(funcs)}
    assert all(hits.values()),hits
    dump(out,dict(enabled=True,exe=sha(exe),native_routine_entries=hits))
    print("Retail dot-product dispatch",hits,flush=True)

if __name__=="__main__":run()

