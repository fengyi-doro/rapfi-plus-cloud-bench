// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "search/ab/rzone.h"
namespace omega::vcfinput {
namespace z=Search::RZone;
struct Result {int status=3;uint64_t nodes=0;z::Proof proof;z::Mask zone;};
// KataGomo's three VCF input states distinguish a win, a complete negative
// search, and a search that exceeded its node budget. Only a verified win
// supplies a minimax bound or a must-play zone.
inline Result inspect(z::Position& p,Color side,int remaining,unsigned budget,bool defenceZone){
    Result r;z::Work w{0,budget};
    r.proof=z::findWin(p,side,remaining,w);r.nodes=w.nodes;
    if(r.proof.empty())r.status=w.nodes<budget?2:3;
    else if(z::verify(p,side,r.proof,defenceZone?&r.zone:nullptr))r.status=1;
    else r.proof.clear();
    return r;
}
}
