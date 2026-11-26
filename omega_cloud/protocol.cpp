// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.h"
#include <condition_variable>
#include <cctype>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <csignal>
using namespace omega;
namespace {
struct Item {uint64_t seq;std::string text;};
struct Commands {
    std::mutex mutex;std::condition_variable changed;std::deque<Item> queue;
    std::atomic<uint64_t> interrupt{0};bool closed=false;
    void read() {
        uint64_t seq=0;std::string line;
        while(std::getline(std::cin,line)) {
            if(!line.empty() && line.back()=='\r')line.pop_back();
            std::string name=line.substr(0,line.find_first_of(" \t"));
            for(auto& c:name)c=char(std::toupper((unsigned char)c));
            ++seq;
            if(name=="END" || name=="STOP")interrupt.store(seq,std::memory_order_relaxed);
            {std::lock_guard lock(mutex);queue.push_back({seq,line});}
            changed.notify_one();
            if(name=="END")break;
        }
        {std::lock_guard lock(mutex);closed=true;}
        changed.notify_one();
    }
    bool pop(Item& result) {
        std::unique_lock lock(mutex);changed.wait(lock,[&]{return closed || !queue.empty();});
        if(queue.empty())return false;
        result=std::move(queue.front());queue.pop_front();return true;
    }
};
void output(const std::string& s){std::cout<<s<<'\n'<<std::flush;}
struct ProtocolClock {
    int turn=1000,match=0,left=-1;
    double increment=-1,priorBank=-1,priorWall=0;
    void reset(){left=-1;increment=-1;priorBank=-1;priorWall=0;}
    void timeLeft(int ms) {
        if(priorBank>=0 && ms>0) {
            double observed=ms-priorBank+priorWall;
            if(observed>=-4 && observed<=2000) {
                observed=std::max(0.,observed);
                increment=increment<0?observed:.65*increment+.35*observed;
            }
            priorBank=-1;
        }
        left=ms;
    }
    std::pair<double,double> limits(int ply,double scale,bool frontload) const {
        double bank=left>=0?left:match>0?match:0;
        double cap=turn>0?turn:bank>0?bank:1000;
        if(left>=0 || bank>0)cap=std::min(cap,bank);
        double hard=std::max(1.,cap-std::clamp(cap*(cap<200?.14:.05),8.,85.));
        double inc=increment>=0?increment:std::clamp(bank*.021,0.,100.);
        double divisor=ply<12?18:ply<65?12:ply<110?17:std::max(5.,(225-ply)*.32);
        if(frontload && bank>std::max(220.,5.0*inc) && ply<45)divisor=ply<=24?7.5:9.5;
        double requested=bank>0?inc*.85+bank/divisor:cap*.85;
        if(bank>0 && bank<200)requested=std::min({requested,bank*.40,std::max(12.,inc*.75)});
        double soft=std::min(requested*scale,std::max(1.,hard-std::clamp(cap*.025,8.,40.)));
        return {std::max(1.,soft),hard};
    }
};
int point(std::string_view text,int* who=nullptr) {
    std::istringstream stream{std::string(text)};int x,y;char comma;
    if(!(stream>>x>>comma>>y) || comma!=',' || x<0 || x>=15 || y<0 || y>=15)
        throw std::invalid_argument("invalid coordinate");
    if(who && (!(stream>>comma>>*who) || comma!=',' || (*who!=1 && *who!=2)))
        throw std::invalid_argument("BOARD requires player 1 or 2");
    stream>>std::ws;if(!stream.eof())throw std::invalid_argument("extra coordinate data");
    return x+15*y;
}
void position(Search& search,Commands& source) {
    Item record;std::array<std::vector<int>,2> by;std::vector<std::pair<int,int>> raw;
    std::array<bool,225> occupied{};bool done=false;
    while(source.pop(record)) {
        if(record.text=="DONE" || record.text=="done"){done=true;break;}
        int who,p=point(record.text,&who);
        if(occupied[p])throw std::invalid_argument("duplicate BOARD coordinate");
        occupied[p]=true;raw.emplace_back(p,who);by[who-1].push_back(p);
    }
    if(!done)throw std::invalid_argument("incomplete BOARD");
    // BOARD describes the engine (1) and its opponent (2). Side to move
    // follows the total count, even when managers list stones by color.
    if(by[1].size()!=by[0].size()+raw.size()%2)throw std::invalid_argument("invalid stone counts");
    int first=raw.size()%2;bool ordered=true;std::vector<int> hist;
    for(size_t i=0;i<raw.size();i++) {
        int color=(first+i)%2;ordered&=raw[i].second==color+1;
        hist.push_back(by[color][i/2]);
    }
    if(!ordered) {
        const auto& old=search.history();
        if(old.size()<=hist.size() && hist.size()-old.size()<=2) {
            std::array<int,225> colors{};
            for(size_t i=0;i<hist.size();i++)colors[hist[i]]=1+i%2;
            bool extends=true;
            for(size_t i=0;i<old.size();i++) {
                if(colors[old[i]]!=int(1+i%2))extends=false;
                colors[old[i]]=0;
            }
            if(extends) {
                auto next=old;
                while(next.size()<hist.size()) {
                    auto it=std::find(colors.begin(),colors.end(),1+int(next.size()%2));
                    if(it==colors.end())throw std::invalid_argument("inconsistent diagram");
                    next.push_back(int(it-colors.begin()));*it=0;
                }
                search.set(next);return;
            }
        }
    }
    search.set(hist,ordered);
}
}
int main(int argc,char** argv) {
    std::signal(SIGPIPE,SIG_IGN);
    pin(layout().controller);
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    Commands commands;std::thread reader([&]{commands.read();});
    try {
        Options opts=Options::environment();
        const auto folder=fs::canonical("/proc/self/exe").parent_path();
        std::unique_ptr<Search> search;ProtocolClock timing;int rule=1;
        auto ready=[&]()->Search&{if(!search)throw std::logic_error("START 15 is required");return *search;};
        auto move=[&](uint64_t seq) {
            auto started=Clock::now();auto& engine=ready();
            auto [soft,hard]=timing.limits(int(engine.history().size()),opts.budgetScale,opts.frontload);
            Turn result=engine.think(soft,hard,commands.interrupt,seq);
            if(result.move<0 && std::string_view(result.reason)=="no_legal") {
                std::array<bool,225> occupied{};
                for(int p:engine.history()) occupied[p]=true;
                auto empty=std::find(occupied.begin(),occupied.end(),false);
                if(empty==occupied.end()) throw std::logic_error("called on full board");
                result.move=int(empty-occupied.begin());
            } else {
                if(result.move<0)throw std::logic_error("no legal move in position");
                engine.advance(result.move);
            }
            result.elapsed=milliseconds(Clock::now()-started);
            output("MESSAGE OMEGA "+result.json());
            output(std::to_string(result.move%15)+","+std::to_string(result.move/15));
            timing.priorBank=timing.left;timing.priorWall=milliseconds(Clock::now()-started);
        };
        Item item;
        while(commands.pop(item)) try {
            std::istringstream stream(item.text);std::string cmd;stream>>cmd;
            for(auto& c:cmd)c=char(std::toupper((unsigned char)c));
            if(cmd=="END")break;
            if(cmd=="START" || cmd=="RECTSTART") {
                std::string size;stream>>size;
                if(size!=(cmd=="START"?"15":"15,15"))throw std::invalid_argument("supported board size is 15x15");
                if(!search){search=std::make_unique<Search>(folder,opts);if(rule!=1)search->reset(rule);}
                else search->reset(rule);
                timing.reset();output("MESSAGE Omega-Cloud-CPU "+search->device());output("OK");
            } else if(cmd=="RESTART") {ready().reset(rule);timing.reset();output("OK");}
            else if(cmd=="INFO") {
                std::string key;int val;stream>>key;
                for(auto& c:key)c=char(std::tolower((unsigned char)c));
                if(!(stream>>val))continue;
                if(key=="rule") {
                    if(val==4)val=2;
                    if(val!=1 && val!=2)throw std::invalid_argument("unsupported rule");
                    if(rule!=val){rule=val;if(search)search->reset(rule);timing.reset();}
                } else if(key=="timeout_turn")timing.turn=std::max(0,val);
                else if(key=="timeout_match")timing.match=std::max(0,val);
                else if(key=="time_left")timing.timeLeft(val);
                else if(key=="time_increment")timing.increment=std::max(0,val);
            } else if(cmd=="BOARD" || cmd=="YXBOARD") {position(ready(),commands);if(cmd=="BOARD")move(item.seq);}
            else if(cmd=="TURN") {std::string xy;stream>>xy;ready().advance(point(xy));move(item.seq);}
            else if(cmd=="BEGIN" || cmd=="YXNBEST") {
                if(cmd=="BEGIN" && !ready().history().empty())throw std::logic_error("BEGIN requires empty board");
                move(item.seq);
            } else if(cmd=="TAKEBACK") {
                std::string xy;stream>>xy;auto h=ready().history();
                if(h.empty() || h.back()!=point(xy))throw std::invalid_argument("TAKEBACK requires last move");
                h.pop_back();ready().set(h);output("OK");
            } else if(cmd=="YXHASHCLEAR")ready().clearCache();
            else if(cmd=="ABOUT")output("name=\"Omega-Cloud-CPU\", version=\"1.0\", author=\"Omega project\", country=\"\"");
            else if(cmd=="STOP" || cmd.empty())continue;
            else output("UNKNOWN "+cmd);
        } catch(const std::exception& e){output(std::string("ERROR ")+e.what());}
    } catch(const std::exception& e){output(std::string("ERROR ")+e.what());}
    reader.join();return 0;
}
