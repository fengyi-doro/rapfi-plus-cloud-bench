"""Exercise both models, the shield, and root/leaf inference through Gomocup on Linux."""
import json
from pathlib import Path
import subprocess

p = subprocess.Popen(["dist/pbrain-omega-cloud"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)

def exchange(text, done):
    print(text.splitlines()[0], flush=True)
    p.stdin.write(text+"\n")
    p.stdin.flush()
    output = []
    for line in p.stdout:
        line = line.rstrip()
        print(line, flush=True)
        if line.startswith(("ERROR", "UNKNOWN")):
            raise RuntimeError(line)
        output.append(line)
        if done(line):
            return output
    raise RuntimeError(f"engine closed with {p.poll()}")

try:
    exchange("START 15", lambda s: s == "OK")
    book = json.loads(Path("data/v3/cloud_openings.json").read_bytes())
    for rule in (1, 2):
        exchange(f"INFO rule {1 if rule == 1 else 4}\nRESTART", lambda s: s == "OK")
        moves = [112, 113, 97, 128, 81, 96]
        diagram = "\n".join(f"{m%15},{m//15},{1 if i%2==len(moves)%2 else 2}" for i,m in enumerate(moves))
        exchange(f"YXBOARD\n{diagram}\nDONE\nINFO timeout_match 5000\nINFO time_left 5000\nABOUT", lambda s: "author=" in s)
        lines = exchange("YXNBEST 1", lambda s: s and s[0].isdigit() and "," in s)
        t = json.loads(next(line[14:] for line in lines if line.startswith("MESSAGE OMEGA ")))
        if t["eval_rows"] <= t["discarded"] or t["visits"] <= 1 or t["cpu_move"] < 0 or t["shield_lost"]:
            raise ValueError(f"missing tree evaluations or v4 advice under rule {rule}: {t}")
finally:
    if p.poll() is None:
        p.stdin.write("END\n")
        p.stdin.flush()
        p.wait(timeout=10)
