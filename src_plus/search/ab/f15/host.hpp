// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "certificate.hpp"
#include "potential.hpp"
#include <span>
namespace f15 {
inline Board fromOmegaDiagram(std::span<const uint8_t,Cells> cells) {
    Board b;
    for(int p=0;p<Cells;++p) {
        if(cells[p]==1)b.black.add(p);
        else if(cells[p]==2)b.white.add(p);
        else if(cells[p]!=0)throw std::invalid_argument("Omega diagram cell");
    }
    return b;
}
struct NodeBound {int lo=-1,hi=1,winningMove=-1;bool exhausted=false;};
// WDL from the actual side to move. The host handles terminal positions,
// pass history, opening choices, and any artificial move restriction first.
inline NodeBound probe(const Board& b,int rule,int sideToMove,VcfLimits work={},
                       const Pairs* againstBlack=nullptr,const Pairs* againstWhite=nullptr) {
    if((rule!=1 && rule!=2) || unsigned(sideToMove)>1) throw std::invalid_argument("unsupported game");
    NodeBound result;
    if(rule==2) {
        auto proof=Vcf(true,work).solve(b,sideToMove);
        result.exhausted=proof.exhausted;
        if(proof.win && replayCertificate(b,sideToMove,proof.line)) {
            int value=sideToMove?1:-1;
            return {value,value,sideToMove?proof.line.front().gain:-1,false};
        }
    }
    for(int maker:{0,1}) {
        const Pairs* pairs=maker?againstWhite:againstBlack;
        bool boundsMaker=esBound(b,rule,maker,sideToMove) || (pairs && mixedBound(b,rule,maker,*pairs));
        if(boundsMaker) {
            if(maker==sideToMove) result.hi=0;
            else result.lo=0;
        }
    }
    return result;
}
} // namespace f15
