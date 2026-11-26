// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
struct State;
struct Verdict {
    int lo,hi,move,reason,steps,nodes,outside,inside,lower,upper;
    uint64_t mask[4];
    int proof[4*32];
};
struct NeuralVCF {
    int mine,theirs,winningMove,nodes;
    uint64_t zone[4];
};
extern "C" {
State* gh_root(int,const uint8_t*,int);
State* gh_child(const State*,int);
void gh_free(State*);
int gh_terminal(const State*);
int gh_nonloss_move(State*);
void gh_probe(State*,int,unsigned,Verdict*);
State* omega_diagram(int,const uint8_t*);
int omega_legal(State*,const uint64_t*,uint8_t*);
void omega_fouls(State*,uint64_t*);
int omega_threat(State*,const uint8_t*,int,int,int*);
void omega_vcf_inputs(State*,unsigned,NeuralVCF*);
int omega_draw(State*,const uint8_t*);
}
