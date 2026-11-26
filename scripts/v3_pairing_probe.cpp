// SPDX-License-Identifier: GPL-3.0-or-later
// Independent coordinate and literal rank-two oracles from the F8 experiment.
#include "search/ab/pairing/search_hook.h"
#include <iostream>
#include <random>
#include <atomic>
#include <chrono>
#include <string_view>
using namespace f8;
static void require(bool b,const char *message) { if (!b) throw std::logic_error(message); }

// Coordinate-based verifier has its own stone array, direction enumeration and
// flank test. It neither reads live window bits nor runs the pairing algorithm.
struct Reference {
    std::array<int,Cells> stone;
    Reference() { stone.fill(-1); }
    int at(int x,int y) const { return onBoard(x,y)?stone[N*y+x]:-2; }
    int check(int attacker, f8::Rule rule, bool defenderFirst,const Witness *w) const {
        if (w) {
            require(w->preplay==None || (defenderFirst && stone.at(w->preplay)==-1),"preplay");
            for (int p=0;p<Cells;++p) if (int q=w->mate[p]; q!=None)
                require(p!=q && stone[p]==-1 && stone.at(q)==-1 && p!=w->preplay &&
                        w->mate[q]==p,"disjoint free pair");
        }
        int live=0;
        constexpr int step[4][2]={{0,1},{1,0},{1,1},{1,-1}};
        for (int y=0;y<N;++y) for (int x=0;x<N;++x) for (auto &d : step) {
            int dx=d[0],dy=d[1];
            if (!onBoard(x+4*dx,y+4*dy)) continue;
            bool exact=(rule==Standard || (rule==Renju && attacker==0));
            if (exact && (at(x-dx,y-dy)==attacker || at(x+5*dx,y+5*dy)==attacker)) continue;
            int holes[5],n=0; bool blocked=false;
            for (int j=0;j<5;++j) {
                int p=(y+j*dy)*N+x+j*dx;
                if (stone[p]==1-attacker) { blocked=true; break; }
                if (stone[p]<0) holes[n++]=p;
            }
            if (blocked) continue;
            ++live;
            require(n>0,"terminal");
            if (w) {
                bool hit=false;
                for (int i=0;i<n;++i) {
                    if (holes[i]==w->preplay) hit=true;
                    for (int j=0;j<i;++j) if (w->mate[holes[i]]==holes[j]) hit=true;
                }
                require(hit,"uncovered coordinate window");
            }
        }
        return live;
    }
};
int bruteGraph(int n, const Graph &a, const Graph &b) {
    int singles[2]={int(a.singles.word[0]),int(b.singles.word[0])};
    int adj[2][8]{};
    for (int p=0;p<n;++p) { adj[0][p]=a.adj[p].word[0]; adj[1][p]=b.adj[p].word[0]; }
    auto solve = [&](auto &visit,int my,int other,int color)->int {
        int empty=((1<<n)-1) & ~(my|other);
        if (!empty) return 0;
        int best=-1;
        for (int bits=empty;bits;bits&=bits-1) {
            int p=std::countr_zero(unsigned(bits)),bit=1<<p;
            if ((singles[color]&bit) || (adj[color][p]&my)) return 1;
            best=std::max(best,-visit(visit,other,my|bit,1-color));
            if (best==1) break;
        }
        return best;
    };
    return solve(solve,0,0,0);
}
static int graphTests() {
    std::mt19937 rng(250318163);
    int games=4096+512;
    for (int i=0;i<games;++i) {
        int n=i<4096?3:4+rng()%5;
        Graph gs[2]; CellsMask free;
        for (int p=0;p<n;++p) free.set(p);
        for (int color=0;color<2;++color) {
            int mask=color?i%64:i/64, k=n;
            for (int p=0;p<n;++p) {
                if (i<4096 ? (mask>>p)&1 : rng()%25==0) gs[color].singles.set(p);
                for (int q=0;q<p;++q,++k)
                    if (i<4096 ? (mask>>k)&1 : rng()%5==0) gs[color].pair(p,q);
            }
        }
        require(outcome(free,gs[0],gs[1])==bruteGraph(n,gs[0],gs[1]),"rank-two differential");
    }
    return games;
}
[[gnu::noinline]] PairingSearch::Cut timedClassify(const Position &p, Value beta)
{
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto result=PairingSearch::classify(p,beta);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return result;
}

int main(int argc,char **argv) try {
    if (argc==2 && std::string_view(argv[1])=="--timing") {
        int id,rule,size;
        while (std::cin >> id >> rule >> size) {
            Position pos(rule==1?Standard:Renju);
            for (int i=0,p; i<size; ++i) { std::cin >> p; pos.put(p,i%2); }
            if (pos.finished || std::min(pos.count(0),pos.count(1))>MaxLive) continue;
            if (rule==2 && size%2==0) continue;
            for (int beta=0; beta<=(rule==1); ++beta) {
                int checksum=0;
                constexpr int repeats=64;
                auto start=std::chrono::steady_clock::now();
                for (int j=0; j<repeats; ++j) checksum+=int(timedClassify(pos,Value(beta)));
                auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-start).count();
                std::cout << id << ' ' << rule << ' ' << size << ' ' << beta << ' '
                          << pos.count(0) << ' ' << pos.count(1) << ' '
                          << double(ns)/repeats << ' ' << checksum << '\n';
            }
        }
        return 0;
    }
    std::cout << "graphs " << graphTests() << '\n';
    int id, rule, size;
    while (std::cin >> id >> rule >> size) {
        Position position(rule == 1 ? Standard : Renju);
        Reference original;
        for (int i=0; i<size; ++i) {
            int p;
            std::cin >> p;
            require(original.stone.at(p) == -1, "duplicate play");
            original.stone[p]=i%2;
            position.put(p,i%2);
            const auto exact=PairingSearch::classify(position, Value(1));
            const auto lower=PairingSearch::classify(position, VALUE_DRAW);
            if (position.ply < MinPly || position.finished) {
                require(exact==PairingSearch::Cut::None && lower==PairingSearch::Cut::None, "inactive cut");
                continue;
            }
            if (position.ply == f8::Cells) continue;
            require(exact != PairingSearch::Cut::DrawLowerBound && lower != PairingSearch::Cut::ExactDraw,
                    "search window polarity");
            require(rule == 1 || (exact == PairingSearch::Cut::None
                    && (position.ply%2 || lower == PairingSearch::Cut::None)), "Renju policy");
            const int stm=position.ply%2;
            Witness lo,hi;
            Bounds proof=prove(position,rule==1 || stm==1,rule==1,MaxLive,false,&lo,&hi);
            int lc=original.check(1-stm,position.rule,true,proof.pairLo?&lo:nullptr);
            int hc=original.check(stm,position.rule,false,proof.pairHi?&hi:nullptr);
            require(lc==position.count(1-stm) && hc==position.count(stm),"live windows differ");
            std::cout << id << ' ' << position.ply << ' ' << int(exact) << ' ' << int(lower)
                      << ' ' << proof.pairLo << ' ' << proof.pairHi << '\n';
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
}
