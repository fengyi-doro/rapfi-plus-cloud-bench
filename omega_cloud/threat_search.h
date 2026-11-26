// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "search/ab/rzone.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
namespace omega::threat {
namespace z=Search::RZone;
using Timer=std::chrono::steady_clock;
struct Tree;
struct Reply {int move;z::Proof continuation;std::unique_ptr<Tree> next;};
struct Tree { int attack; z::Proof outside;std::vector<Reply> replacements; };
struct Metrics {uint64_t nodes=0;int attacks=0;};
struct Place {
    z::Position& position; int move;
    Place(z::Position& p,int m,Color c):position(p),move(m){p.put(m,c);}
    ~Place(){position.take(move);}
};
// A quiet attack needs enough nearby material to threaten a subsequent VCF.
// Ordering restricts the OR node only. Every legal defender reply remains covered.
inline const auto& windows(){
    static const auto value=[]{
        std::array<std::vector<std::array<uint8_t,5>>,225> windows;
        for(auto [dx,dy]:std::array<std::pair<int,int>,4>{{{1,0},{0,1},{1,1},{-1,1}}})
            for(int y=0;y<15;y++)for(int x=0;x<15;x++){
                int ex=x+4*dx,ey=y+4*dy;if(ex<0 || ex>=15 || ey>=15)continue;
                std::array<uint8_t,5> line{};
                for(int i=0;i<5;i++)line[i]=uint8_t(x+i*dx+15*(y+i*dy));
                for(int p:line)windows[p].push_back(line);
            }
        return windows;
    }();
    return value;
}
inline std::vector<int> plausible(const z::Position& pos,Color side,std::span<const uint8_t> hints,int limit){
    std::array<int,225> priority{};std::vector<int> result;
    if(!hints.empty()){
        for(int m:hints)if(pos.at(m)==EMPTY && !priority[m]){priority[m]=1;result.push_back(m);}
        if(int(result.size())>limit)result.resize(limit);
        return result;
    }
    for(int p=0;p<225;p++)if(pos.at(p)==EMPTY){
        for(const auto& line:windows()[p]){
            int n=0;
            for(int q:line){
                auto stone=pos.at(q);
                if(stone==~side){n=0;break;}
                n+=stone==side;
            }
            if(n>=2)priority[p]+=n>=3?4:1;
        }
        if(priority[p])result.push_back(p);
    }
    std::stable_sort(result.begin(),result.end(),[&](int a,int b){return priority[a]>priority[b];});
    if(int(result.size())>limit)result.resize(limit);
    return result;
}
inline std::optional<Tree> prove(z::Position& pos,Color side,std::span<const uint8_t> hints,
                                  int limit,int plies,uint64_t budget,Timer::time_point deadline,Metrics& stats,
                                  int quietDepth=1){
    if(stats.nodes>=budget || Timer::now()>=deadline || pos.targets(~side,4).any())return {};
    auto moves=plausible(pos,side,hints,limit);
    auto search=[&](uint64_t perCall){
        z::Work work{0,std::min(budget-stats.nodes,perCall)};
        auto proof=z::findWin(pos,side,plies,work);stats.nodes+=work.nodes;
        return proof;
    };
    for(int move:moves){
        if(stats.nodes>=budget || Timer::now()>=deadline)break;
        if(pos.forbidden(move,side))continue;
        ++stats.attacks;
        Place attack(pos,move,side);
        if(pos.winsOn(move,side) || pos.targets(~side,4).any())continue;
        auto original=search(std::min<uint64_t>(384,std::max<uint64_t>(64,budget/5)));
        z::Mask required;
        if(original.empty() || !z::verify(pos,side,original,&required))continue;
        Tree tree{move,std::move(original),{}};
        bool coversAll=true;
        for(int reply;(reply=required.pop())!=z::NoCell;){
            if(Timer::now()>=deadline){coversAll=false;break;}
            if(pos.forbidden(reply,~side))continue;
            Place defend(pos,reply,~side);
            if(pos.winsOn(reply,~side)){coversAll=false;break;}
            if(z::verify(pos,side,tree.outside))continue;
            if(stats.nodes>=budget){coversAll=false;break;}
            auto alternative=search(std::min<uint64_t>(512,budget-stats.nodes));
            if(!alternative.empty() && z::verify(pos,side,alternative)){
                tree.replacements.push_back({reply,std::move(alternative),{}});continue;
            }
            if(quietDepth<=1){coversAll=false;break;}
            auto child=prove(pos,side,{},std::min(limit,14),plies,budget,deadline,stats,quietDepth-1);
            if(!child){coversAll=false;break;}
            tree.replacements.push_back({reply,{},std::make_unique<Tree>(std::move(*child))});
        }
        if(coversAll)return tree;
    }
    return {};
}
}
