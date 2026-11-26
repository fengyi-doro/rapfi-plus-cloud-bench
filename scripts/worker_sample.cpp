// A profiling-only observer. Sample an isolated, native search thread outside the engine.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>

struct Counts { uint64_t samples = 0, cycles = 0; };
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    HANDLE thread = OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME,
                               FALSE, std::strtoul(argv[1], nullptr, 10));
    HANDLE mem = OpenFileMappingA(FILE_MAP_READ, FALSE, argv[2]);
    if (!thread || !mem) return 3;
    auto *control = static_cast<volatile LONG *>(MapViewOfFile(mem, FILE_MAP_READ, 0, 0, 8));
    if (!control) return 4;
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << std::atoi(argv[3]));
    FILE *out = std::fopen(argv[4], "w");
    if (!out) return 5;
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    uint32_t rng = 0x459b1c73u;
    std::map<uint64_t, Counts> rows;
    LONG previousWindow = -1;
    uint64_t previousCycles = 0;
    std::puts("READY");
    std::fflush(stdout);
    while (!control[1]) {
        const LONG window = control[0];
        if (!(window & 1)) { previousWindow=-1; SwitchToThread(); continue; }
        QueryPerformanceCounter(&counter);
        rng = 1664525*rng + 1013904223;
        const int64_t stop = counter.QuadPart + freq.QuadPart * (95 + rng%121) / 1000000;
        do { YieldProcessor(); QueryPerformanceCounter(&counter); } while (counter.QuadPart < stop);
        if (window != control[0]) continue;
        CONTEXT ctx{};
        ctx.ContextFlags=CONTEXT_CONTROL;
        if (SuspendThread(thread)==DWORD(-1)) return 6;
        uint64_t cycles=0;
        const bool ok=GetThreadContext(thread,&ctx) && QueryThreadCycleTime(thread,&cycles);
        const DWORD resumed=ResumeThread(thread);
        if (resumed==DWORD(-1) || !ok) return 7;
        if (window==control[0] && window==previousWindow && cycles>previousCycles) {
            auto &r=rows[ctx.Rip];
            ++r.samples;
            r.cycles+=cycles-previousCycles;
        }
        previousWindow=window;
        previousCycles=cycles;
    }
    std::fprintf(out,"ip,samples,cycles\n");
    for (const auto &[ip,c]:rows)
        std::fprintf(out,"%llx,%llu,%llu\n",(unsigned long long)ip,
                     (unsigned long long)c.samples,(unsigned long long)c.cycles);
    std::fclose(out);
    UnmapViewOfFile((const void*)control);
    CloseHandle(mem);
    CloseHandle(thread);
}

