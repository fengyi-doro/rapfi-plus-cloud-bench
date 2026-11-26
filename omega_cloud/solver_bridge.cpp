// SPDX-License-Identifier: GPL-3.0-or-later
// Compile the existing, tested solver ABI into the playing executable.
#include "solver.cpp"
#include "search/ab/pairing/f10/search.h"
#include "threat_search.h"
#include "vcf_inputs.h"

extern "C" State* omega_diagram(int rule,const uint8_t* cells) {
    auto s=std::make_unique<State>(rule);
    int black=0, white=0;
    for(int p=0;p<225;p++) if(cells[p]) {
        if(cells[p]>2) return nullptr;
        Color color=Color(cells[p]-1);
        s->board.put(p,color);
        s->pairs.put(p,color);
        if(color==BLACK) black++; else white++;
    }
    if(black!=white && black!=white+1) return nullptr;
    for(int p=0;p<225;p++) if(cells[p] && s->board.winsOn(p,Color(cells[p]-1))) {
        s->terminal = Color(cells[p]-1)==s->next()?1:-1; break;
    }
    if(s->terminal==2 && black+white==225) s->terminal=0;
    s->pairs.finished=s->terminal!=2;
    return s.release();
}

namespace {
bool forbiddenBlack(State* s,int p) {
    static const auto aligned=[] {
        std::array<z::Mask,225> result{};
        for(int p=0;p<225;p++) for(int q=0;q<225;q++) {
            int dx=std::abs(p%15-q%15),dy=std::abs(p/15-q/15);
            if(p!=q && std::max(dx,dy)<=5 && (!dx || !dy || dx==dy)) result[p].add(q);
        }
        return result;
    }();
    int near=0;
    for(int k=0;k<4;k++)near+=std::popcount(aligned[p].words[k]&~(s->pairs.empty.word[k]|s->pairs.white.word[k]));
    return near>=3 && s->board.forbidden(p,BLACK);
}
}
extern "C" int omega_legal(State* s,const uint64_t* mask,uint8_t* out) {
    if(s->terminal!=2) return 0;
    int count=0;
    for(int i=0;i<4;i++) for(auto word=s->pairs.empty.word[i]&mask[i];word;word&=word-1) {
        int p=64*i+std::countr_zero(word);
        if(s->rule==RENJU && s->next()==BLACK) {
            // Each possible foul needs at least three distinct pre-existing black
            // stones on these rays. The remaining points use the recursive referee.
            if(forbiddenBlack(s,p))continue;
        }
        out[count++]=uint8_t(p);
    }
    return count;
}
extern "C" void omega_fouls(State* s,uint64_t* mask) {
    std::fill_n(mask,4,0);
    if(s->rule!=RENJU || s->terminal!=2)return;
    for(int k=0;k<4;k++)for(auto word=s->pairs.empty.word[k];word;word&=word-1){
        int p=64*k+std::countr_zero(word);
        if(forbiddenBlack(s,p))mask[k]|=1ULL<<(p%64);
    }
}
extern "C" int omega_threat(State* s,const uint8_t* candidates,int count,int root,int* nodes) {
    using namespace omega::threat;
    if(s->terminal!=2 || s->pairs.ply<6 || s->pairs.ply>105)return -1;
    Metrics stats;
    bool shortlist=root==2;
    auto until=Timer::now()+std::chrono::microseconds(shortlist?1450:root?950:180);
    if(shortlist){
        auto direct=omega::vcfinput::inspect(s->board,s->next(),225-s->pairs.ply,640,false);
        stats.nodes=direct.nodes;
        if(direct.status==1){*nodes=int(stats.nodes);return direct.proof.front().attack;}
    }
    auto proof=prove(s->board,s->next(),std::span(candidates,size_t(count)),shortlist?32:root?16:6,shortlist?17:root?15:11,
                     shortlist?2300:root?1400:220,until,stats,root?2:1);
    *nodes=int(stats.nodes);
    return proof?proof->attack:-1;
}
struct NeuralVCF {int mine,theirs,winningMove,nodes;uint64_t zone[4];};
extern "C" void omega_vcf_inputs(State* s,unsigned budget,NeuralVCF* out) {
    auto me=s->next();int remaining=z::Cells-s->pairs.ply;
    auto own=omega::vcfinput::inspect(s->board,me,remaining,budget,false);
    *out={own.status,0,own.status==1?own.proof.front().attack:-1,int(own.nodes),{}};
    if(own.status==1)return;
    auto opp=omega::vcfinput::inspect(s->board,~me,remaining,budget,true);
    out->theirs=opp.status;out->nodes+=int(opp.nodes);
    std::copy(opp.zone.words.begin(),opp.zone.words.end(),out->zone);
}
extern "C" int omega_draw(State* s,const uint8_t* cells) {
    if(s->pairs.ply<f10::FirstSearchPly || s->terminal!=2 ||
       std::max(s->pairs.count(0),s->pairs.count(1))>f10::RootLiveCap) return -1;
    f10::Board b{f8::Rule(s->rule)};
    for(int p=0;p<225;p++) if(cells[p]) b.put(p,cells[p]-1);
    // Transient shapes while loading a diagram have no bearing on the final game.
    b.finished=false;
    f10::Proof first;
    if(!f10::rootDraw(b,first)) return -1;
    int p=first.pre>=0?first.pre:s->pairs.empty.first();
    return p>=0 && !s->board.forbidden(p,s->next())?p:-1;
}
