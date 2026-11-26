"""Windows Gomocup transport, core topology, monotonic clocks and cycle counters."""
import ctypes as ct
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import re
import socket
import struct
import subprocess
import threading
import time
import platform
if os.name == "nt":
    import winreg
import psutil
from bit_gate import board_command
from datasets import DATA, ROOT, sha

COORD = re.compile(r"(-?\d+),(-?\d+)")


class DeadlineExpired(TimeoutError):
    def __init__(self, command, elapsed_ns, lines=()):
        self.command = command
        self.elapsed_ns = elapsed_ns
        self.lines = lines
        super().__init__(f"response deadline for {command[:40]} at {elapsed_ns} ns")

    def __reduce__(self):
        return type(self), (self.command, self.elapsed_ns, self.lines)


def root_metrics(lines):
    metrics = {}
    for line in lines:
        if line.startswith("MESSAGE F23 "):
            values = list(map(int, line.split()[2:]))
            if len(values) != 13 or "f23_mask" in metrics:
                raise ValueError(f"malformed F23 counters: {line}")
            metrics.update(f23_mask=values[0], f23_counts=values[1:])
        if line.startswith("MESSAGE ROOT_THREATS "):
            metrics["root_threat_extensions"] = int(line.split()[-1])
        if line.startswith("MESSAGE ROOT_PROOF "):
            _, _, win, excluded, nodes = line.split()
            metrics.update(root_proven_win=int(win), root_proven_losses=int(excluded), root_proof_nodes=int(nodes))
        if line.startswith("MESSAGE ROOT_PRIOR "):
            _, _, mode, ns = line.split()
            if "root_prior_ns" in metrics or mode != os.environ.get("RAPFI_ROOT_MODE") or int(ns) <= 0:
                raise ValueError(f"invalid root timing: {line}")
            metrics.update(root_prior=mode, root_prior_ns=int(ns))
    return metrics


if os.name == "nt":
    K = ct.WinDLL("kernel32", use_last_error=True)
    K.OpenThread.argtypes = [ct.c_ulong, ct.c_int, ct.c_ulong]
    K.OpenThread.restype = ct.c_void_p
    K.CloseHandle.argtypes = [ct.c_void_p]
    K.QueryThreadCycleTime.argtypes = K.QueryProcessCycleTime.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64)]
    K.K32EnumProcessModules.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_ulong, ct.POINTER(ct.c_ulong)]
    K.ReadProcessMemory.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]


    class GroupAffinity(ct.Structure):
        _fields_ = [("mask",ct.c_size_t),("group",ct.c_ushort),("reserved",ct.c_ushort*3)]


    K.GetThreadGroupAffinity.argtypes = [ct.c_void_p,ct.POINTER(GroupAffinity)]



def utc():
    return datetime.now(timezone.utc).isoformat()


def topology(core_domains=False):
    if os.name != "nt":
        eligible = sorted(os.sched_getaffinity(0))
        cores = []
        for cpu in eligible:
            folder = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            pkg = int((folder/"topology/physical_package_id").read_text())
            physical = int((folder/"topology/core_id").read_text())
            capacity = folder/"cpu_capacity"
            cores.append(dict(group=pkg, physical=physical, logical=cpu,
                              efficiency=int(capacity.read_text()) if capacity.exists() else 0))
        unique = {}
        for row in cores:
            unique.setdefault((row["group"],row["physical"]), row["logical"])
        processors = [dict(line.split(":", 1) for line in section.splitlines() if ":" in line)
                      for section in Path("/proc/cpuinfo").read_text().strip().split("\n\n")]
        processors = [{k.strip(): v.strip() for k, v in row.items()} for row in processors]
        active = [row for row in processors if int(row["processor"]) in eligible]
        flags = sorted(set.intersection(*(set(row["flags"].split()) for row in active))) if active and all("flags" in row for row in active) else []
        return dict(cpu=active[0].get("model name", platform.machine()) if active else platform.machine(),
                    os=platform.platform(), cores=cores, cpu_flags=flags,
                    performance=list(unique.values()), helper=[], sampled_utc=utc())
    size = ct.c_ulong()
    K.GetSystemCpuSetInformation(None, 0, ct.byref(size), None, 0)
    buf = ct.create_string_buffer(size.value)
    if not K.GetSystemCpuSetInformation(buf, size, ct.byref(size), None, 0):
        raise ct.WinError(ct.get_last_error())
    cores = []
    at = 0
    while at < size.value:
        length, kind = struct.unpack_from("<II", buf, at)
        if kind == 0:
            group, logical, physical, _, _, efficiency = struct.unpack_from("<H5B", buf, at+12)
            entry=dict(group=group, logical=logical, physical=physical, efficiency=efficiency)
            if core_domains:
                entry["last_level_cache"]=buf.raw[at+16]
            cores.append(entry)
        assert length > 0
        at += length
    allowed = set(psutil.Process().cpu_affinity())
    eligible = [c for c in cores if c["group"] == 0 and c["logical"] in allowed]
    highest = max(c["efficiency"] for c in eligible)
    unique = {}
    for c in eligible:
        if c["efficiency"] == highest:
            unique.setdefault(c["physical"], c["logical"])
    loads = [psutil.cpu_percent(interval=.4, percpu=True) for _ in range(4)]
    performance = sorted(unique.values(), key=lambda i: sum(sample[i] for sample in loads))
    assistants = sorted(c["logical"] for c in eligible if c["efficiency"] < highest)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    return dict(cpu=cpu, os=platform.platform(), cores=cores, performance=performance, helper=assistants, load_pct_before=loads,
                sampled_utc=utc())


def pin_handle(proc, core, cpu_pool=None):
    child = psutil.Process(proc.pid)
    cpus = cpu_pool or [core]
    child.cpu_affinity(cpus)
    assert sorted(child.cpu_affinity()) == sorted(cpus)
    if os.name == "nt":
        child.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    return child


def filename(variant):
    distributed = os.environ.get("RAPFI_ARENA_BIN")
    if distributed:
        if variant == os.environ.get("RAPFI_CANDIDATE_LABEL"):
            variant = os.environ["RAPFI_BUILD_PROFILE"]
        name = {"v2":"plus-v2", "v3":"plus-v3", "v4":"plus-v4"}.get(variant, variant)
        return Path(distributed) / ("pbrain-rapfi-" + name + (".exe" if os.name == "nt" else ""))
    archived = os.environ.get("RAPFI_SCREEN_BINARIES")
    if archived:
        return Path(archived)/f"pbrain-rapfi-{variant}.exe"
    suffix = ".exe" if os.name == "nt" else ""
    if variant in ("v1", "v2", "v3", "v4"):
        return ROOT / f"pbrain-rapfi-plus-{variant}{suffix}"
    folder = ROOT if variant in ("orig", "plus") else ROOT/"build"/variant
    return folder / f"pbrain-rapfi-{variant}{suffix}"


def cycles(handle, query):
    value = ct.c_uint64()
    if not query(handle, ct.byref(value)):
        raise ct.WinError(ct.get_last_error())
    return value.value


@lru_cache
def node_layout(variant):
    if os.name != "nt" or os.environ.get("RAPFI_ARENA_BIN") or os.environ.get("RAPFI_SCREEN_BINARIES"):
        return None
    individual = DATA/"v2"/("nodes_"+variant+".json")
    path = DATA/"node_symbols.json"
    if not path.exists():
        return None
    if variant == "v4":
        layout = json.loads((DATA/"v4/nodes_v4.json").read_bytes())
    elif variant == "v2":
        layout = json.loads((DATA/"v3/nodes_v2.json").read_bytes())
    elif variant in ("f9", "v3"):
        layout = json.loads((DATA/"v3"/("nodes_"+variant+".json")).read_bytes())
    elif variant=="v1":
        layout=json.loads((ROOT/"archive/v1/data/node_symbols.json").read_bytes())["plus"]
    elif (ROOT/"build"/variant/"nodes.json").is_file():
        layout=json.loads((ROOT/"build"/variant/"nodes.json").read_bytes())
    else:
        layout = (json.loads(individual.read_bytes()) if variant not in ("orig","plus") and individual.exists()
                  else json.loads(path.read_bytes())[variant])
    if layout["sha256"] != sha(filename(variant)):
        raise ValueError("Regenerate the exact node-counter layout with scripts/node_symbols.py")
    return layout


class Engine:
    def __init__(self, variant, core, cpu_pool=None, search_threads=1):
        self.variant, self.core = variant, core
        self.cpu_pool = sorted(cpu_pool) if cpu_pool else None
        self.search_threads = search_threads
        self.oracle_reader = None
        self.oracle_models = {}
        child_options = {}
        oracle_connection = child_connection = None
        if variant == os.environ.get("RAPFI_CANDIDATE_LABEL") and os.environ.get("RAPFI_ROOT_MODE") == "kata":
            from kata_oracle import playing_session, serve
            net, self.oracle_models = playing_session(Path(os.environ["RAPFI_ARENA_BIN"]))
            oracle_connection, child_connection = socket.socketpair()
            fd = child_connection.fileno()
            child_options = dict(pass_fds=(fd,), env=dict(os.environ, RAPFI_ORACLE_FD=str(fd)))
        try:
            self.proc = subprocess.Popen([str(filename(variant)), "--config", str(ROOT/"config.toml")],
                 cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **child_options)
        except BaseException:
            if oracle_connection is not None:
                oracle_connection.close()
            raise
        finally:
            if child_connection is not None:
                child_connection.close()
        if oracle_connection is not None:
            self.oracle_reader = threading.Thread(target=serve, args=(oracle_connection, net, core), daemon=True)
            self.oracle_reader.start()
        self.handles = {}
        try:
            self.ps = pin_handle(self.proc, core, self.cpu_pool)
            self.inbox = queue.Queue()
            def drain():
                for raw in self.proc.stdout:
                    stamp = time.perf_counter_ns()
                    self.inbox.put((stamp, raw.decode("utf-8", errors="replace").rstrip()))
                self.inbox.put((time.perf_counter_ns(), None))
            self.reader = threading.Thread(target=drain, daemon=True)
            self.reader.start()
            self.rule = None
            self.weights = []
            first, _ = self.exchange("START 15", lambda s: s == "OK")
            assert any("Evaluator set to mix9svq" in s for s in first), first
            self.sync(f"INFO thread_num {search_threads}\nINFO hash_size 8192\nINFO pondering 0\n"
                      "INFO max_node 0\nINFO show_detail 2")
            layout = node_layout(variant)
            self.counter = None
            if layout:
                modules = (ct.c_void_p * 256)()
                needed = ct.c_ulong()
                if not K.K32EnumProcessModules(int(self.proc._handle), modules, ct.sizeof(modules), ct.byref(needed)):
                    raise ct.WinError(ct.get_last_error())
                vector = modules[0] + layout["vector_rva"]
                first = self.read_word(vector)
                assert self.read_word(vector+8) == first+8, "expected exactly one search worker"
                self.counter = self.read_word(first) + layout["nodes_offset"]
        except BaseException:
            self.close()
            raise

    def submit(self, command):
        start = time.perf_counter_ns()
        self.proc.stdin.write((command + "\n").encode())
        self.proc.stdin.flush()
        return start

    def collect(self, start, command, complete, timeout=60):
        lines = []
        deadline = start + int(timeout * 1e9)
        while True:
            try:
                stamp, line = self.inbox.get(timeout=max(.0001, (deadline - time.perf_counter_ns())/1e9))
            except queue.Empty as exc:
                stamp = time.perf_counter_ns()
                if stamp < deadline:
                    continue
                raise DeadlineExpired(command,stamp-start,lines) from exc
            if line is None or line.startswith(("ERROR", "UNKNOWN")) or "Unknown Info Parameter" in line:
                raise RuntimeError((self.variant, command[:100], line, lines[-4:]))
            lines.append(line)
            if "load weight from " in line or "weight loaded in " in line:
                self.weights.append(line)
            if complete(line):
                return lines, stamp - start

    def exchange(self, command, complete, timeout=60):
        return self.collect(self.submit(command), command, complete, timeout)

    def sync(self, commands):
        return self.exchange(commands + "\nABOUT", lambda s: "author=" in s)[0]

    def select(self, rule):
        if self.rule == rule:
            return
        self.exchange(f"STOP\nINFO rule {4 if rule == 2 else rule}\nRESTART", lambda s: s == "OK")
        self.sync(board_command([112, 113, 100, 126, 98]) + "\nTRACESEARCH")
        expected = (("freestyle_bsmix",), ("standard_bs15",), ("renju_bs15_black", "renju_bs15_white"))[rule]
        assert all(any("mix9svq" + part in s for s in self.weights) for part in expected), self.weights
        assert any("weight loaded in " in s for s in self.weights)
        self.rule = rule

    def clean_position(self, moves):
        self.sync("STOP\nRESTART\nYXHASHCLEAR\n" + board_command(moves))

    def cycle_snapshot(self):
        for t in self.ps.threads():
            if t.id not in self.handles:
                h = K.OpenThread(0x0800, False, t.id)
                if not h:
                    if ct.get_last_error() == 87 and t.id not in {s.id for s in self.ps.threads()}:
                        continue
                    raise ct.WinError(ct.get_last_error())
                self.handles[t.id] = h
        return {tid: cycles(h, K.QueryThreadCycleTime) for tid, h in self.handles.items()}

    def affinity_masks(self):
        if os.name != "nt":
            tids = self.ps.threads()
            result = {t.id:sorted(os.sched_getaffinity(t.id)) for t in tids}
            if any(cpus != (self.cpu_pool or [self.core]) for cpus in result.values()):
                raise RuntimeError(("worker affinity mismatch", result, self.core))
            return result
        self.cycle_snapshot()
        process_mask = sum(1 << k for k in self.ps.cpu_affinity())
        assert process_mask == 1 << self.core
        observed = {}
        for tid in [t.id for t in self.ps.threads() if t.id in self.handles]:
            group = GroupAffinity()
            if not K.GetThreadGroupAffinity(self.handles[tid],ct.byref(group)):
                raise ct.WinError(ct.get_last_error())
            assert group.group == 0 and group.mask & process_mask == process_mask
            observed[tid] = dict(group=group.group,group_mask=group.mask,effective_mask=group.mask&process_mask)
        assert observed
        return observed

    def repin(self, core):
        cpus = self.cpu_pool or [core]
        self.ps.cpu_affinity(cpus)
        if os.name != "nt":
            for thread in self.ps.threads():
                os.sched_setaffinity(thread.id, cpus)
        self.core = core

    def read_word(self, address):
        val, copied = ct.c_uint64(), ct.c_size_t()
        if not K.ReadProcessMemory(int(self.proc._handle), address, ct.byref(val), 8, ct.byref(copied)):
            raise ct.WinError(ct.get_last_error())
        assert copied.value == 8
        return val.value

    def think(self, command="YXNBEST 1", measure_cycles=False, timeout=90):
        if measure_cycles:
            before = self.cycle_snapshot()
            process_before = cycles(int(self.proc._handle), K.QueryProcessCycleTime)
        lines, ns = self.exchange(command, COORD.fullmatch, timeout)
        fields = {}
        for line in lines:
            m = re.fullmatch(r"INFO (DEPTH|SELDEPTH|EVAL|TOTALNODES|TOTALTIME) (\S+)", line)
            if m:
                fields[m[1]] = m[2]
        final = next((m for line in reversed(lines) if (m := re.search(
            r"\| Depth (\d+)-(\d+) \| Eval (\S+) \| Node ([\dKMG]+) \| Time ", line))), None)
        nodes = int(fields.get("TOTALNODES", 0))
        if self.counter is not None and final:
            nodes = self.read_word(self.counter)
            multiplier = {"K":1000, "M":1000000, "G":1000000000}.get(final[4][-1],1)
            encoded = int(final[4] if multiplier == 1 else final[4][:-1])
            assert nodes//multiplier == encoded, (self.variant, nodes, final[4])
        elif final and final[4].isdigit():
            nodes = int(final[4])
        elif final and (os.name != "nt" or os.environ.get("RAPFI_ARENA_BIN")):
            nodes = nodes if "TOTALNODES" in fields else None
        elif final and not fields.get("TOTALNODES"):
            raise RuntimeError(("missing exact node telemetry", lines[-12:]))
        x, y = map(int, lines[-1].split(","))
        result = dict(x=x, y=y, wall_ns=ns, nodes=nodes, depth=int(final[1] if final else fields.get("DEPTH", 0)),
                      eval=final[3] if final else fields.get("EVAL"),
                      draw_offer=any(s.strip().upper() == "MESSAGE DRAW_OFFER" for s in lines),
                      engine_ms=int(fields["TOTALTIME"]) if "TOTALTIME" in fields else None)
        result.update(root_metrics(lines))
        if measure_cycles:
            after = self.cycle_snapshot()
            proc_delta = cycles(int(self.proc._handle), K.QueryProcessCycleTime) - process_before
            by_thread = {tid: after[tid] - val for tid, val in before.items()}
            tid = max(by_thread, key=by_thread.get)
            result.update(worker_tid=tid, worker_cycles=by_thread[tid], thread_cycles=by_thread,
                          process_cycles=proc_delta)
        return result

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"STOP\nEND\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=4)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        for h in self.handles.values():
            K.CloseHandle(h)
        if hasattr(self, "reader"):
            self.reader.join(timeout=1)
        if self.oracle_reader is not None:
            self.oracle_reader.join(timeout=2)
        self.proc.stdin.close()
        self.proc.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
