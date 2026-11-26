// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "f15/host.hpp"
#include "f17/vct_algebra.hpp"
#include "network.h"

namespace omega {
struct TacticalResult {int value=0, move=-1, calls=0, wins=0, nodes=0;};
inline TacticalResult foulTrap(const Features& f,Clock::time_point until) {
    if(f.rule!=2 || f.ply<8 || Clock::now()>=until) return {};
    auto board=f15::fromOmegaDiagram(f.cells);
    f15::VcfLimits budget{180,[](void* end) {
        return Clock::now()>=*static_cast<Clock::time_point*>(end);
    },&until};
    const auto result=f15::probe(board,f.rule,f.ply%2,budget);
    // F15's positional nonloss bounds do not return a playable draw witness.
    // Only its decisive, replayed white VCF/foul certificates drive this hook.
    if(result.lo==result.hi && result.lo!=0)
        return {result.lo,result.winningMove,1,1,0};
    return {0,-1,1,0,0};
}
inline TacticalResult continuousThreats(const Features& f,Clock::time_point until) {
    if(f.ply<6 || Clock::now()>=until) return {};
    f17::Board board;
    if(!f17::fromCells(f.cells.data(),board)) throw std::logic_error("invalid F17 diagram");
    f17::Limits budget;
    budget.plies=11;budget.nodes=500;budget.deadline=until;
    auto result=f17::solve(board,f17::Side(f.ply%2),f17::Rule(f.rule),budget);
    if(result.status==f17::Status::Proven && result.proof)
        return {1,result.proof->gain,1,1,int(result.stats.nodes())};
    return {0,-1,1,0,int(result.stats.nodes())};
}
} // namespace omega
