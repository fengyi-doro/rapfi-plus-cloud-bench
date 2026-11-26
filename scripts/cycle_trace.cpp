// Diagnostic link only. Time the optimized, out-of-line call tree with Windows worker cycles.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
struct Frame { uintptr_t fn; uint64_t last, self; };
struct Bucket { uintptr_t fn; uint64_t cycles, calls; };
thread_local Frame frames[2048];
thread_local int depth=0;
thread_local uint64_t startCycle=0;
thread_local Bucket slots[8192];
unsigned serial=0;
uint64_t now() {
    ULONG64 value;
    if (!QueryThreadCycleTime(GetCurrentThread(), &value)) std::abort();
    return value;
}
void account(Frame f) {
    size_t slot=(f.fn>>4) % 8192;
    for (unsigned i=0;i<8192;++i,slot=(slot+1)%8192) {
        if (slots[slot].fn && slots[slot].fn!=f.fn) continue;
        slots[slot].fn=f.fn;
        slots[slot].cycles+=f.self;
        ++slots[slot].calls;
        return;
    }
    std::abort();
}
}
extern "C" void rapfiProfileBegin() {
    if (depth) std::abort();
    startCycle=now();
    frames[0]={1,startCycle,0};
    depth=1;
}
extern "C" void rapfiProfileEnd() {
    const auto end=now();
    if (depth!=1) std::abort();
    frames[0].self+=end-frames[0].last;
    depth=0;
    const char *path=std::getenv("RAPFI_CYCLES_FILE");
    if (!path) return;
    FILE *out=std::fopen(path,"a");
    if (!out) std::abort();
    ++serial;
    std::fprintf(out,"SEARCH %u %llu %llu\n",serial,(unsigned long long)(end-startCycle),
                 (unsigned long long)frames[0].self);
    auto base=(uintptr_t)GetModuleHandleW(nullptr);
    for (auto &s:slots) {
        if (!s.fn) continue;
        std::fprintf(out,"CALL %llx %llu %llu\n",(unsigned long long)(s.fn-base),
                     (unsigned long long)s.cycles,(unsigned long long)s.calls);
        s={};
    }
    std::fclose(out);
}
extern "C" void __cyg_profile_func_enter(void *fn,void *) {
    if (!depth) return;
    const auto stamp=now();
    Frame &parent=frames[depth-1];
    parent.self+=stamp-parent.last;
    if (depth>=2048) std::abort();
    Frame &child=frames[depth++];
    child.fn=(uintptr_t)fn;
    child.self=0;
    child.last=now();
}
extern "C" void __cyg_profile_func_exit(void *fn,void *) {
    if (!depth) return;
    const auto stamp=now();
    if (depth<2 || frames[depth-1].fn!=(uintptr_t)fn) std::abort();
    Frame &child=frames[--depth];
    child.self+=stamp-child.last;
    account(child);
    frames[depth-1].last=now();
}

