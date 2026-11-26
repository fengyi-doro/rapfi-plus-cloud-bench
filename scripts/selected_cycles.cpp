// Narrow diagnostic probe for the four hottest engine functions in the unmodified IP trace.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
struct Frame { unsigned id; uint64_t prior, self; };
thread_local Frame stack[128];
thread_local unsigned depth=0;
thread_local uint64_t totals[5],calls[5],start=0,timerFloor=0;
uint64_t tick() {
    ULONG64 c;
    if (!QueryThreadCycleTime(GetCurrentThread(),&c)) std::abort();
    return c;
}
}
extern "C" void enterSelected(unsigned id) {
    if (id==0) {
        if (depth) std::abort();
        if (!timerFloor) {
            uint64_t floor[511];
            for (auto &v:floor) {auto a=tick(); v=tick()-a;}
            std::sort(floor,floor+511);
            timerFloor=floor[255];
        }
        start=tick();
        stack[depth++]={0,start,0};
    } else if (depth) {
        auto c=tick();
        stack[depth-1].self+=c-stack[depth-1].prior;
        if (depth==128) std::abort();
        stack[depth++]={id,c,0};
    }
}
extern "C" void exitSelected(unsigned id) {
    if (!depth) return;
    const auto c=tick();
    const auto f=stack[--depth];
    if (id!=f.id) std::abort();
    totals[id]+=f.self+c-f.prior;
    ++calls[id];
    if (depth) stack[depth-1].prior=c;
    else {
        FILE *file=std::fopen(std::getenv("RAPFI_CYCLES_FILE"),"a");
        if (!file) std::abort();
        std::fprintf(file,"%llu %llu",(unsigned long long)(c-start),(unsigned long long)timerFloor);
        for (int i=0;i<5;++i) {
            std::fprintf(file," %llu %llu",(unsigned long long)totals[i],(unsigned long long)calls[i]);
            totals[i]=calls[i]=0;
        }
        std::fprintf(file,"\n");
        std::fclose(file);
    }
}

