// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "inference.h"
#include "shield.h"
#include <atomic>
#include <sstream>
namespace omega {
struct Options {
    int batch=8,rootWork=1200,leafWork=72,rootDepth=15,leafDepth=7,cacheEntries=16384,visitLimit=0,inputVcf=240;
    float cpuct=1.35f,fpu=.20f,priorTemp=1.f,cpuPrior=.23f;
    double budgetScale=1.0; int shieldCore=-1;
    bool f15=true,f17=true,shield=true,reuse=true,frontload=true;
    static Options environment();
};
struct Turn {
    int f15Calls=0,f15Decisive=0,f17Calls=0,f17Wins=0,f17Nodes=0;
    int move=-1,lo=-1,hi=1,visits=0,reused=0,evalRows=0,slots=0,hits=0,probes=0,
        vcfWork=0,cpuMove=-1,cpuDepth=0,cpuScore=0,cpuMate=0,refutations=0,
        veto=0,takeover=0,correction=0,extended=0,early=0,draw=0,shieldCore=-1,shieldLost=0,discarded=0,
        vctWork=0,vctWins=0,poolSearch=0,inputNodes=0,inputWins=0,inputThreats=0,proofChecks=0,proofVeto=0,
        cpuChoiceChecks=0,cpuChoiceRefuted=0,shieldDeferred=0,shieldRecovered=0;
    double value=0,budget=0,elapsed=0,inferWaitMs=0,inferSubmitMs=0,cpuStopMs=0;
    const char* reason="mcts";
    std::vector<std::array<double,4>> candidates;
    std::vector<std::array<int,5>> replies;
    std::vector<std::array<int,4>> pool;
    std::string json() const;
};
class Search {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    Search(const fs::path& home,const Options& options);
    ~Search();
    void reset(int rule);
    void set(const std::vector<int>& history,bool knownLast=true);
    void advance(int move);
    void clearCache();
    Turn think(double softMs,double hardMs,const std::atomic<uint64_t>& stop,uint64_t seq);
    const std::vector<int>& history() const;
    const std::string& device() const;
};
}
