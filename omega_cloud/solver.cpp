// SPDX-License-Identifier: GPL-3.0-or-later
// C ABI over the same geometry, recursive Renju rules and F8 code used by Rapfi.
#include "search/ab/rzone.h"
#include "search/ab/pairing/lazy.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>

namespace z = Search::RZone;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
struct State {
    Rule rule;
    z::Position board;
    f8::Position pairs;
    int terminal = 2; // 2 ongoing, otherwise W/D/L for the next player
    explicit State(int r) : rule(Rule(r)), board(rule), pairs(f8::Rule(r)) {}
    Color next() const { return Color(pairs.ply % 2); }
    bool play(int m) {
        Color c = next();
        if (terminal != 2 || m < 0 || m >= z::Cells || board.at(m) != EMPTY
            || board.forbidden(m,c)) return false;
        board.put(m,c);
        pairs.put(m,c);
        if (board.winsOn(m,c)) terminal = -1;
        else if (pairs.ply == z::Cells) terminal = 0;
        return true;
    }
};
struct Verdict {
    int lo, hi, move, reason, steps, nodes, outside, inside, lower, upper;
    uint64_t mask[4];
    int proof[4*32];
};
enum Source { Unknown, Terminal, Five, VCF, DoubleThreat, ForcedBlock, RZone, PairDraw, NoLegal };

API State* gh_root(int rule, const uint8_t* history, int n) {
    if (rule < 0 || rule > 2 || n < 0 || n > z::Cells) return nullptr;
    auto s = std::make_unique<State>(rule);
    for (int i=0; i<n; ++i) if (!s->play(history[i])) return nullptr;
    return s.release();
}
API State* gh_child(const State* parent, int m) {
    auto s = std::make_unique<State>(*parent);
    if (!s->play(m)) return nullptr;
    return s.release();
}
API void gh_free(State* s) { delete s; }
API int gh_terminal(const State* s) { return s->terminal; }
API int gh_legal(State* s, uint8_t* out) {
    int n=0;
    if (s->terminal != 2) return n;
    for (int p=0; p<z::Cells; ++p)
        if (s->board.at(p)==EMPTY && !s->board.forbidden(p,s->next())) out[n++]=uint8_t(p);
    return n;
}

static f8::Bounds bounds(const State& s) {
    if (s.rule==FREESTYLE || s.pairs.ply<f8::MinPly || s.terminal!=2
        || (s.rule==RENJU && s.next()!=WHITE)
        || std::min(s.pairs.count(0),s.pairs.count(1))>f8::MaxLive) return {};
    return f8::lazy(s.pairs,true,s.rule==STANDARD);
}

// Obtain a playable witness even when the root itself never uses the GPU.
// A covered family with no preplay permits any first defence. For a rank
// sandwich, enumerate a first move in the very same pessimistic rank-2 game.
API int gh_nonloss_move(State* s) {
    if (bounds(*s).lo<0) return -1;
    int me=s->next(), opp=1-me;
    f8::Witness w;
    if (s->pairs.count(opp)<=f8::MaxLive && f8::pairing(f8::Family(s->pairs,opp),true,w)) {
        return w.preplay!=f8::None ? w.preplay : s->pairs.empty.first();
    }
    auto a=f8::stableGraph(f8::Family(s->pairs,me));
    auto b=f8::easierGraph(f8::Family(s->pairs,opp));
    for (int p=0; p<z::Cells; ++p) {
        if (!s->pairs.empty.has(p) || s->board.forbidden(p,s->next())) continue;
        if (a.singles.has(p)) return p;
        auto after=a;
        for (int j=0; j<4; ++j) after.singles.word[j] |= a.adj[p].word[j];
        auto free=s->pairs.empty;
        free.reset(p);
        if (f8::outcome(free,b,after)<=0) return p;
    }
    return -1;
}
static void setMask(Verdict& v, const z::Mask& m) {
    std::copy(m.words.begin(),m.words.end(),v.mask);
}
static void known(Verdict& v, int score, Source reason, int move=-1) {
    v.lo=v.hi=score;
    v.reason=reason;
    v.move=move;
}
API void gh_probe(State* s, int plies, unsigned budget, Verdict* out) {
    *out = {};
    auto &v=*out;
    v.lo=-1; v.hi=1; v.move=-1;
    std::fill_n(v.mask,4,UINT64_MAX);
    if (s->terminal!=2) { known(v,s->terminal,Terminal); return; }
    auto cert=bounds(*s);
    // Keep the production rule policy. A rank-only decisive lower bound
    // still implies a nonloss. Retain the upper bound only for Standard.
    if (cert.lo>=0) { v.lo=0; v.lower=1; }
    if (cert.hi<=0) { v.hi=0; v.upper=1; }
    if (v.lo==v.hi) { known(v,0,PairDraw); return; }
    Color me=s->next(), opp=~me;
    auto &pos=s->board;
    int win=pos.targets(me,4).pop();
    if (win!=z::NoCell) { known(v,1,Five,win); return; }
    auto danger=pos.targets(opp,4);
    int mustBlock=danger.pop();
    if (mustBlock!=z::NoCell) {
        if (danger.any() || pos.forbidden(mustBlock,me)) { known(v,-1,DoubleThreat); return; }
        z::Mask only;
        only.add(mustBlock); setMask(v,only); v.reason=ForcedBlock;
    }
    z::Work work{0,budget};
    auto attack=z::findWin(pos,me,plies,work);
    if (!attack.empty() && z::verify(pos,me,attack)) {
        known(v,1,VCF,attack.front().attack);
        v.steps=int(attack.size());
        for (int i=0; i<v.steps; ++i) {
            const auto &p=attack[i];
            v.proof[4*i]=p.attack; v.proof[4*i+1]=p.block;
            v.proof[4*i+2]=p.other; v.proof[4*i+3]=int(p.kind);
        }
        v.nodes=int(work.nodes); return;
    }
    z::Mask viable;
    if (mustBlock!=z::NoCell) viable.add(mustBlock);
    else {
        auto threat=z::findWin(pos,opp,plies,work);
        z::Mask zone;
        if (threat.empty() || !z::verify(pos,opp,threat,&zone)) { v.nodes=int(work.nodes); return; }
        z::Mustplay must;
        must.add({std::move(threat),zone});
        viable=must.intersection;
        v.reason=RZone;
        v.outside=z::Cells-s->pairs.ply-viable.size(); // empty cells, before legality intersection
    }
    auto todo=viable;
    for (int p; work.nodes<work.limit && (p=todo.pop())!=z::NoCell;) {
        bool ruledOut=pos.forbidden(p,me);
        if (!ruledOut) {
            pos.put(p,me);
            auto reply=z::findWin(pos,opp,plies,work);
            ruledOut=!reply.empty() && z::verify(pos,opp,reply);
            pos.take(p);
        }
        if (ruledOut) { viable.words[p/64] &= ~(uint64_t(1)<<(p%64)); ++v.inside; }
    }
    v.nodes=int(work.nodes);
    setMask(v,viable);
    if (!viable.any()) known(v,-1,RZone);
}

// Resident classifier microbenchmark. A compiler barrier forces each query.
API double gh_time_f8(const State* s, int repeats, int* checksum) {
    auto begin=std::chrono::steady_clock::now();
    int total=0;
    for (int i=0; i<repeats; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        auto r=bounds(*s); total+=(r.lo+1)*3+r.hi+1;
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    *checksum=total;
    return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/repeats;
}
