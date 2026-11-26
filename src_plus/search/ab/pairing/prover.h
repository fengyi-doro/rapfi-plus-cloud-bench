// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace f8 {
constexpr int N = 15, Cells = 225, Windows = 572, None = 255;
enum Rule { Free, Standard, Renju };
inline bool exactFive(Rule rule, int color) { return rule == Standard || (rule == Renju && color == 0); }
inline bool onBoard(int x, int y) { return unsigned(x) < N && unsigned(y) < N; }
constexpr int dirs[4][2] = {{1,0},{0,1},{1,1},{-1,1}};

struct CellsMask {
    std::array<uint64_t,4> word{};
    bool has(int p) const { return (word[p/64] >> (p%64)) & 1; }
    void set(int p) { word[p/64] |= uint64_t(1) << (p%64); }
    void reset(int p) { word[p/64] &= ~(uint64_t(1) << (p%64)); }
    int first() const {
        for (int i = 0; i < 4; ++i) if (word[i]) return 64*i + std::countr_zero(word[i]);
        return None;
    }
    int count() const {
        int n = 0;
        for (auto w : word) n += std::popcount(w);
        return n;
    }
    CellsMask operator&(const CellsMask &b) const {
        CellsMask result;
        for (int i = 0; i < 4; ++i) result.word[i] = word[i] & b.word[i];
        return result;
    }
};
using Live = std::array<uint64_t,9>;
struct Window { std::array<uint8_t,5> p; std::array<uint8_t,2> flank; };
struct Geometry {
    std::array<Window,Windows> lines{};
    std::array<Live,Cells> interior{}, flank{};
    std::array<std::array<uint16_t,Cells>,Cells> pairId{};
    int pairs = 0;
    Geometry() {
        int w = 0;
        for (auto &d : dirs) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
            const int dx=d[0], dy=d[1];
            if (!onBoard(x+4*dx,y+4*dy)) continue;
            auto cell = [&](int t) -> uint8_t {
                return onBoard(x+t*dx,y+t*dy) ? (y+t*dy)*N+x+t*dx : None;
            };
            uint64_t bit = uint64_t(1) << (w%64);
            auto &line = lines[w];
            for (int i=0; i<5; ++i) {
                int p = line.p[i] = cell(i);
                interior[p][w/64] |= bit;
                for (int j=0; j<i; ++j) {
                    int q = line.p[j];
                    if (!pairId[p][q]) pairId[p][q] = pairId[q][p] = ++pairs;
                }
            }
            line.flank = {cell(-1),cell(5)};
            for (int p : line.flank) if (p != None) flank[p][w/64] |= bit;
            ++w;
        }
        if (w != Windows || pairs >= 3601) throw std::logic_error("invalid geometry");
    }
};
inline const Geometry geometry;
struct Position {
    Rule rule;
    int ply = 0;
    bool finished = false;
    CellsMask empty, white;
    std::array<Live,2> live;
    explicit Position(Rule r = Standard) : rule(r) {
        empty.word = {~0ULL,~0ULL,~0ULL,(1ULL << (Cells-192))-1};
        for (auto &side : live) {
            side.fill(~0ULL);
            side.back() = (1ULL << (Windows-512))-1;
        }
    }
    void put(int p, int color) {
        empty.reset(p);
        if (color) white.set(p);
        for (int i=0; i<9; ++i) live[1-color][i] &= ~geometry.interior[p][i];
        if (exactFive(rule,color)) for (int i=0; i<9; ++i) live[color][i] &= ~geometry.flank[p][i];
        // Terminal input can occur in analysis of an externally supplied board.
        // A live exact-five window already excludes a same-colour flank.
        for (int i=0; i<9 && !finished; ++i) {
            auto hits = live[color][i] & geometry.interior[p][i];
            while (hits) {
                const auto &line = geometry.lines[64*i+std::countr_zero(hits)];
                if (std::none_of(line.p.begin(),line.p.end(),[&](int q){return empty.has(q);})) {
                    finished=true;
                    break;
                }
                hits &= hits-1;
            }
        }
        ++ply;
    }
    int count(int color) const {
        int n=0;
        for (auto w : live[color]) n += std::popcount(w);
        return n;
    }
    // Existing white stones are permanent barriers. Test every square black may
    // ever have to play, including arbitrary tempo moves outside the pairing.
    bool allFutureBlackMovesLegal() const {
        for (int p=0; p<Cells; ++p) if (empty.has(p)) {
            int possibleFours=0;
            for (auto &d : dirs) {
                int length=1;
                for (int sign : {-1,1}) {
                    int x=p%N + sign*d[0], y=p/N + sign*d[1];
                    while (onBoard(x,y) && !white.has(y*N+x)) {
                        if (++length >= 6) return false;
                        x += sign*d[0]; y += sign*d[1];
                    }
                }
                if (length == 5 && ++possibleFours >= 2) return false;
            }
        }
        return true;
    }
};
struct Edge { uint8_t n=0; std::array<uint8_t,5> p{}; bool stable=false; };
struct Family {
    std::array<Edge,Windows> edges;
    std::array<int,7> start{};
    int size() const { return start[6]; }
    explicit Family(const Position &b, int attacker) {
        // Counting sort by edge size. Short edges constrain the matching first.
        std::array<Edge,Windows> raw;
        int len=0;
        for (int i=0; i<9; ++i) for (uint64_t bits=b.live[attacker][i]; bits; bits &= bits-1) {
            const auto &line = geometry.lines[64*i+std::countr_zero(bits)];
            auto &e=raw[len++];
            for (auto p : line.p) if (b.empty.has(p)) e.p[e.n++] = p;
            if (!e.n) throw std::logic_error("terminal position submitted to prover");
            e.stable = !exactFive(b.rule,attacker) ||
                std::none_of(line.flank.begin(),line.flank.end(),[&](int p){ return p!=None && b.empty.has(p); });
            ++start[e.n+1];
        }
        for (int i=1; i<=6; ++i) start[i]+=start[i-1];
        auto next=start;
        for (int i=0; i<len; ++i) edges[next[raw[i].n]++] = raw[i];
    }
};
struct Witness {
    int preplay=None;
    std::array<uint8_t,Cells> mate;
    Witness() { mate.fill(None); }
};
struct Preferences {
    std::array<uint8_t,3601> freq{};
    std::array<uint8_t,Cells> degree{};
    explicit Preferences(const Family &f) {
        for (int k=0; k<f.size(); ++k) {
            auto &e=f.edges[k];
            for (int i=0; i<e.n; ++i) {
                ++degree[e.p[i]];
                for (int j=0; j<i; ++j) ++freq[geometry.pairId[e.p[i]][e.p[j]]];
            }
        }
    }
    int score(int p, int q) const {
        return 256*freq[geometry.pairId[p][q]] - 2*std::max(degree[p],degree[q]) - degree[p] - degree[q];
    }
};
inline int cover(const Family &f, const Preferences &pref, Witness &w, int pre) {
    w.preplay=pre;
    w.mate.fill(None);
    for (int k=0; k<f.size(); ++k) {
        const auto &e=f.edges[k];
        int best=-10000, a=None, b=None;
        bool done=false;
        for (int i=0; i<e.n && !done; ++i) {
            int p=e.p[i];
            if (p==pre) { done=true; break; }
            for (int j=0; j<i; ++j) {
                int q=e.p[j];
                if (w.mate[p]==q) { done=true; break; }
                if (w.mate[p]!=None || w.mate[q]!=None) continue;
                int score=pref.score(p,q);
                if (score>best) { best=score; a=p; b=q; }
            }
        }
        if (done) continue;
        if (a==None) return k;
        w.mate[a]=b; w.mate[b]=a;
    }
    return -1;
}
inline bool pairing(const Family &f, bool defenderFirst, Witness &w) {
    int forced=None;
    for (int k=f.start[1]; k<f.start[2]; ++k) {
        int p=f.edges[k].p[0];
        if (!defenderFirst || (forced!=None && forced!=p)) return false;
        forced=p;
    }
    if (f.size()==0) { w=Witness(); return true; }
    Preferences pref(f);
    int fail=cover(f,pref,w,forced);
    if (fail < 0) return true;
    if (!defenderFirst || forced!=None) return false;
    // Try a first defence on a failed edge or on a currently conflicting mate.
    // Exhausting this list gives unknown, never an impossibility claim.
    std::array<int,10> trials{};
    int count=0;
    const auto &e=f.edges[fail];
    for (int i=0; i<e.n; ++i) {
        trials[count++]=e.p[i];
        if (int q=w.mate[e.p[i]]; q!=None) trials[count++]=q;
    }
    for (int i=0; i<count; ++i) if (cover(f,pref,w,trials[i])<0) return true;
    return false;
}

// Exact solver on the two artificial rank-two graphs. Only the sandwich of
// scores transfers to Gomoku. Every long possible target gets an easier pair.
struct Graph {
    CellsMask singles;
    std::array<CellsMask,Cells> adj{};
    void pair(int a,int b) { adj[a].set(b); adj[b].set(a); }
};
inline Graph stableGraph(const Family &f) {
    Graph g;
    for (int k=0; k<f.start[3]; ++k) {
        const auto &e=f.edges[k];
        if (!e.stable) continue;
        if (e.n==1) g.singles.set(e.p[0]); else g.pair(e.p[0],e.p[1]);
    }
    return g;
}
inline Graph easierGraph(const Family &f) {
    Graph g;
    std::array<int,Cells> degree{};
    for (int k=0; k<f.size(); ++k) {
        const auto &e=f.edges[k];
        if (e.n==1) { g.singles.set(e.p[0]); continue; }
        int best=100000, a=None,b=None;
        bool covered=false;
        for (int i=0; i<e.n && !covered; ++i) {
            int p=e.p[i];
            if (g.singles.has(p)) { covered=true; break; }
            for (int j=0; j<i; ++j) {
                int q=e.p[j];
                if (g.adj[p].has(q)) { covered=true; break; }
                int cost=64*std::max(degree[p],degree[q])+degree[p]+degree[q];
                if (cost<best) { best=cost; a=p; b=q; }
            }
        }
        if (!covered) { g.pair(a,b); ++degree[a]; ++degree[b]; }
    }
    return g;
}
inline bool fork(const Graph &g, CellsMask free) {
    for (int i=0; i<4; ++i) for (auto bits=free.word[i]; bits; bits&=bits-1)
        if ((g.adj[64*i+std::countr_zero(bits)] & free).count()>=2) return true;
    return false;
}
inline bool matchingSideSurvives(CellsMask free, const Graph &mine,const Graph &other) {
    while (fork(other,free)) {
        bool kept=false;
        for (int p=0; p<Cells && !kept; ++p) if (free.has(p)) {
            CellsMask left=free;
            int v=p;
            bool mineNext=true;
            for (;;) {
                left.reset(v);
                CellsMask next=(mineNext?mine:other).adj[v] & left;
                int n=next.count();
                if (n>=2) break;
                if (!n) {
                    if (!mineNext) { free=left; kept=true; }
                    else if (!fork(other,left)) return true;
                    break;
                }
                v=next.first(); mineNext=!mineNext;
            }
        }
        if (!kept) return false;
    }
    return true;
}
inline int outcome(CellsMask free,const Graph &first,const Graph &second) {
    const Graph *g[2]={&first,&second};
    CellsMask threats[2]={first.singles,second.singles};
    int turn=0;
    for (;;) {
        int sign=turn ? -1 : 1;
        if (threats[turn].first()!=None) return sign;
        int n=threats[1-turn].count();
        if (n>=2) return -sign;
        if (!n) {
            if (fork(*g[turn],free)) return sign;
            return matchingSideSurvives(free,*g[turn],*g[1-turn]) ? 0 : -sign;
        }
        int block=threats[1-turn].first();
        free.reset(block);
        threats[1-turn]={};
        threats[turn]=g[turn]->adj[block] & free;
        turn=1-turn;
    }
}
struct Bounds {
    int lo=-1, hi=1;
    bool pairLo=false, pairHi=false, blackFree=false;
};
// Requested bounds are in the side-to-move frame. K is per attacking family.
// Rank inference only needs stable targets on the other side, even if their
// family exceeds K. A bound for black requires the Renju legality certificate.
inline Bounds prove(const Position &b, bool lower, bool upper, int cap, bool useRank,
                    Witness *lowerWitness=nullptr, Witness *upperWitness=nullptr) {
    int self=b.ply%2, opp=1-self;
    Bounds r;
    lower=lower && b.count(opp)<=cap;
    upper=upper && b.count(self)<=cap;
    if (b.rule==Renju && ((lower && !self) || (upper && !opp))) {
        r.blackFree=b.allFutureBlackMovesLegal();
        lower=lower && (self || r.blackFree);
        upper=upper && (opp || r.blackFree);
    }
    if (!lower && !upper) return r;
    Witness temp;
    // Construct families on demand, keeping the cheaper pairing-only path short.
    if (!useRank) {
        if (lower) r.pairLo=pairing(Family(b,opp),true,lowerWitness?*lowerWitness:temp);
        if (upper) r.pairHi=pairing(Family(b,self),false,upperWitness?*upperWitness:temp);
        if (r.pairLo) r.lo=0;
        if (r.pairHi) r.hi=0;
        return r;
    }
    Family own(b,self), other(b,opp);
    if (lower) {
        r.pairLo=pairing(other,true,lowerWitness?*lowerWitness:temp);
        r.lo=std::max(r.pairLo?0:-1,outcome(b.empty,stableGraph(own),easierGraph(other)));
    }
    if (upper) {
        r.pairHi=pairing(own,false,upperWitness?*upperWitness:temp);
        r.hi=std::min(r.pairHi?0:1,outcome(b.empty,easierGraph(own),stableGraph(other)));
    }
    if (r.lo>r.hi) throw std::logic_error("inconsistent certified bounds");
    return r;
}
} // namespace f8
