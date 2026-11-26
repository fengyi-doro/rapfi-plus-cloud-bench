// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "renju_vcf.hpp"
namespace f15 {
// Recompute geometric support, every actual recursive Black legality query,
// and the full-board counter-five guard. Coordinates in localZone alone are
// never a transposition key. Input must be a nonterminal Renju root.
inline bool replayCertificate(Board b,int side,const std::vector<Step>& line,Mask* localZone=nullptr) {
    if(line.empty() || unsigned(side)>1) return false;
    Mask zone;
    for(size_t i=0;i<line.size();++i) {
        auto s=line[i];
        bool blackPhase=i==0 && side==0;
        if((blackPhase && s.gain!=-1) || (!blackPhase && (unsigned(s.gain)>=Cells || b.occupied().has(s.gain)))) return false;
        bool madeFive=!blackPhase && completions(b,1,false).has(s.gain);
        if(!blackPhase) {b=b.with(s.gain,1);zone.add(s.gain);}
        if(madeFive!=(s.kind==Five)) return false;
        bool last=i+1==line.size();
        if(last==(s.kind==ForcedBlock)) return false;
        auto recordSupport=[&](int target) {
            Mask filled=b.white;filled.add(target);bool ok=false;
            for(const auto& seg:geometry.through[target][1]) if(filled.contains(seg.body)) {
                zone=zone|seg.body;ok=true;
            }
            return ok;
        };
        if(s.kind==Five) {
            if(!recordSupport(s.gain)) return false;
        } else {
            if(completions(b,0,true).any() || unsigned(s.block)>=Cells) return false;
            auto wins=completions(b,1,false);
            if(!wins.has(s.block) || !recordSupport(s.block)) return false;
            if(s.kind==Fork) {
                if(unsigned(s.other)>=Cells || s.other==s.block || !wins.has(s.other) || !recordSupport(s.other)) return false;
                if(localZone) while(wins.any()) if(!recordSupport(wins.pop())) return false;
            } else {
                if(wins.count()!=1) return false;
                auto status=blackMove(b,s.block,localZone?&zone:nullptr);
                if(s.kind==Foul) {if(status<Overline || status!=s.status) return false;}
                else if(s.kind==ForcedBlock) {if(status!=Legal)return false;b=b.with(s.block,0);}
                else return false;
            }
        }
    }
    if(localZone) *localZone=zone;
    return true;
}
} // namespace f15
