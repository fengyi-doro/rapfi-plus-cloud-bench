// SPDX-License-Identifier: GPL-3.0-or-later
#include "shield.h"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace omega {
using Time=std::chrono::steady_clock;
namespace {
struct Handle {
    int fd=-1;
    void close(){if(fd>=0) ::close(fd); fd=-1;}
    ~Handle(){close();}
};
void require(bool success,const char* operation) {
    if(!success) throw std::runtime_error(std::string(operation)+": "+std::strerror(errno));
}
int coordinate(std::string_view s) {
    int x=-1,y=-1;
    auto a=std::from_chars(s.data(),s.data()+s.size(),x);
    if(a.ec!=std::errc{} || a.ptr==s.data()+s.size() || *a.ptr!=',') return -1;
    auto b=std::from_chars(a.ptr+1,s.data()+s.size(),y);
    if(b.ec!=std::errc{} || x<0 || x>14 || y<0 || y>14) return -1;
    return x+15*y;
}
}
struct Shield::Impl {
    Handle input,output;
    pid_t pid=-1;
    std::string partial;
    Advice latest,detail,reported;
    std::vector<Advice> scores;
    std::vector<int> requested,board;
    int selectedCore, rule=-1;
    bool active=false,acked=false,started=false,settled=false,hashFence=false,positionKnown=false,stopRequested=false;
    bool checking=false,restricted=false;
    bool deferred=false;
    bool paused=false;
    std::filesystem::path directory;
    int pvIndex=0;
    Time::time_point retry{};
    std::ofstream trace;
    explicit Impl(int core):selectedCore(core){
        if(const char* file=std::getenv("OMEGA_CPU_TRACE")) trace.open(file,std::ios::app);
    }
    void record(const char* dir,std::string_view msg){
        if(trace.is_open())trace<<Time::now().time_since_epoch().count()<<' '<<dir<<' '<<msg<<std::endl;
    }
    void shut() {
        if(pid>0) {
            ::kill(pid,SIGKILL);
            while(::waitpid(pid,nullptr,0)<0 && errno==EINTR) {}
            pid=-1;
        }
        input.close();output.close();
        paused=false;deferred=false;active=false;started=false;
    }
    ~Impl(){shut();}
    void boot(const std::filesystem::path& home) {
        directory=home;
        const auto exe=(home/"pbrain-rapfi-plus-v4").string();
        const auto config=std::filesystem::absolute("config.toml").string();
        int in[2],out[2];
        require(::pipe2(in,O_CLOEXEC)==0,"shield stdin");
        Handle read,write;
        read.fd=in[0];input.fd=in[1];
        require(::pipe2(out,O_CLOEXEC)==0,"shield stdout");
        output.fd=out[0];write.fd=out[1];
        cpu_set_t affinity;CPU_ZERO(&affinity);CPU_SET(selectedCore,&affinity);
        auto parent=::getpid();
        pid=::fork();require(pid>=0,"fork shield");
        if(pid==0) {
            if(::prctl(PR_SET_PDEATHSIG,SIGKILL) || ::getppid()!=parent ||
               ::sched_setaffinity(0,sizeof(affinity),&affinity) ||
               ::dup2(read.fd,STDIN_FILENO)<0 || ::dup2(write.fd,STDOUT_FILENO)<0 ||
               ::dup2(write.fd,STDERR_FILENO)<0) ::_exit(126);
            ::execl(exe.c_str(),exe.c_str(),"--config",config.c_str(),static_cast<char*>(nullptr));
            ::_exit(127);
        }
        read.close();write.close();
        require(::fcntl(output.fd,F_SETFL,O_NONBLOCK)==0,"nonblocking shield output");
        sync("START 15\nINFO thread_num 1\nINFO hash_size 8192\nINFO pondering 0\nINFO max_node 0\nINFO show_detail 2\n");
        started=true;
    }
    void send(const std::string& cmd) {
        record(">",cmd);
        for(size_t offset=0;offset<cmd.size();) {
            auto n=::write(input.fd,cmd.data()+offset,cmd.size()-offset);
            if(n<0 && errno==EINTR) continue;
            require(n>0,"write shield");
            offset+=size_t(n);
        }
    }
    static void parseScore(std::string_view s,Advice& a) {
        if(s.size()>2 && (s[0]=='+' || s[0]=='-') && s[1]=='M') {
            a.hasScore=true; a.mate=s[0]=='+'?1:-1; a.score=a.mate*30000;
        } else {
            int score=0; auto r=std::from_chars(s.data(),s.data()+s.size(),score);
            if(r.ec==std::errc{}) {a.hasScore=true;a.mate=0;a.score=score;}
        }
    }
    void line(const std::string& s) {
        record("<",s);
        if(hashFence?s.starts_with("MESSAGE Transposition table full:"):s.find("author=")!=std::string::npos) acked=true;
        if(s.starts_with("ERROR")) throw std::runtime_error("Rapfi v4: "+s);
        if(!active) return;
        if(s.starts_with("INFO PV ") && s!="INFO PV DONE"){detail={};pvIndex=std::atoi(s.c_str()+8);}
        else if(s.starts_with("INFO EVAL ")) parseScore(std::string_view(s).substr(10),detail);
        else if(s.starts_with("INFO DEPTH ")) detail.depth=std::atoi(s.c_str()+11);
        else if(s.starts_with("INFO BESTLINE ")) {
            auto rest=std::string_view(s).substr(14);detail.pvLength=0;
            while(!rest.empty() && detail.pvLength<int(detail.pv.size())) {
                auto end=rest.find(' ');int move=coordinate(rest.substr(0,end));
                if(move<0)break;
                detail.pv[detail.pvLength++]=move;
                if(end==std::string_view::npos)break;
                rest.remove_prefix(end+1);
            }
            detail.move=detail.pvLength?detail.pv[0]:-1;
        }
        else if(s=="INFO PV DONE" && detail.move>=0 && detail.hasScore) {
            if(pvIndex==0)latest=detail;
            if(restricted){
                auto it=std::find_if(scores.begin(),scores.end(),[&](const Advice& a){return a.move==detail.move;});
                if(it!=scores.end())*it=detail;else scores.push_back(detail);
            }
        }
        else if(s.starts_with("MESSAGE Speed ")) {
            auto eval=s.find(" | Eval "),depth=s.find(" | Depth ");
            if(eval!=std::string::npos) parseScore(std::string_view(s).substr(eval+8),reported);
            if(depth!=std::string::npos) reported.depth=std::atoi(s.c_str()+depth+9);
        }
        else if(int move=coordinate(s); move>=0) {
            auto it=std::find_if(scores.begin(),scores.end(),[&](const Advice& a){return a.move==move;});
            if(it!=scores.end())latest=*it;
            else if(detail.hasScore && detail.move==move)latest=detail;
            if(reported.hasScore){
                latest.score=reported.score;latest.mate=reported.mate;latest.depth=reported.depth;latest.hasScore=true;
            }
            if(!latest.pvLength || move!=latest.pv[0]){latest.pv[0]=move;latest.pvLength=1;}
            latest.move=move; latest.complete=true; active=false;
            board=requested;board.push_back(move);positionKnown=true;
        }
    }
    void drain() {
        for(;;) {
            std::array<char,8192> buf{};
            auto n=::read(output.fd,buf.data(),buf.size());
            if(n<0 && errno==EINTR) continue;
            if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK)) break;
            require(n>0,"shield output closed");
            partial.append(buf.data(),n);
            size_t i=0,j;
            while((j=partial.find('\n',i))!=std::string::npos) {
                auto end=j; if(end>i && partial[end-1]=='\r') end--;
                line(partial.substr(i,end-i)); i=j+1;
            }
            partial.erase(0,i);
        }
    }
    bool idle(bool acceptCoordinate=false) {
        if(settled)return true;
        if(!checking){hashFence=!hashFence;acked=false;checking=true;retry=Time::now();}
        drain();
        if(acked){settled=true;checking=false;return true;}
        // The coordinate is sufficient to end this turn. Rapfi clears its
        // thinking flag just after it writes the coordinate. Require an actual
        // idle acknowledgement before sending the following board or search.
        if(acceptCoordinate && !active && latest.complete)return true;
        if(Time::now()>=retry){
            send(hashFence?"YXSHOWHASHUSAGE\n":"ABOUT\n");retry=Time::now()+std::chrono::milliseconds(4);
        }
        return false;
    }
    void pauseLateSearch() {
        if(paused) return;
        require(::kill(pid,SIGSTOP)==0,"pause late shield");
        int status;
        pid_t stopped;
        do {stopped=::waitpid(pid,&status,WUNTRACED);} while(stopped<0 && errno==EINTR);
        require(stopped==pid && WIFSTOPPED(status),"confirm shield pause");
        paused=true;deferred=true;
    }
    bool finishDeferred() {
        if(!deferred)return true;
        if(paused) {require(::kill(pid,SIGCONT)==0,"resume shield");paused=false;}
        if(!idle())return false;
        active=false;deferred=false;return true;
    }
    void sync(std::string command) {
        hashFence=false;acked=false;settled=false;checking=false;
        send(command+"ABOUT\n");
        auto deadline=Time::now()+std::chrono::seconds(20);
        while(!acked) {
            drain();
            if(Time::now()>=deadline) throw std::runtime_error("Rapfi initialization timeout");
            if(!acked)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        settled=true;
    }
};
Shield::Shield(const std::filesystem::path& home,int core):p(std::make_unique<Impl>(core)) {p->boot(home);}
Shield::~Shield()=default;
bool Shield::available() const{return p->started;}
bool Shield::deferred() const{return p->deferred;}
bool Shield::recovering() const{return p->deferred && !p->paused;}
int Shield::core() const{return p->selectedCore;}
void Shield::reset(int rule) {
    if(!available()) return;
    if(p->deferred){
        auto limit=Time::now()+std::chrono::seconds(1);
        while(p->started && !p->finishDeferred() && Time::now()<limit)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(!p->started || p->deferred){
            auto folder=p->directory;int core=p->selectedCore;p->shut();
            p=std::make_unique<Impl>(core);p->boot(folder);
        }
    }
    std::string cmd="STOP\nINFO rule "+std::to_string(rule==2?4:rule)+"\nRESTART\n";
    if(rule!=p->rule) {
        cmd+="YXBOARD\n7,7,2\n8,7,1\n10,6,2\n6,8,1\n8,6,2\nDONE\nTRACESEARCH\nRESTART\n";
        p->rule=rule;
    }
    p->sync(cmd+"YXBLOCKRESET\nYXHASHCLEAR\n");
    p->positionKnown=false;p->restricted=false;
}
bool Shield::start(const std::vector<int>& hist,int ms,const std::vector<int>& choices) {
    if(!available() || !p->finishDeferred() || p->active || !p->idle())return false;
    p->latest={};p->detail={};p->reported={};p->scores.clear();p->pvIndex=0;
    std::ostringstream cmd;
    cmd<<"INFO timeout_match 0\nINFO timeout_turn "<<std::max(35,ms+25)<<'\n';
    if(p->restricted)cmd<<"YXBLOCKRESET\n";
    p->restricted=!choices.empty();
    bool same=p->positionKnown && p->board==hist;
    bool turn=!p->restricted && p->positionKnown && hist.size()==p->board.size()+1 &&
              std::equal(p->board.begin(),p->board.end(),hist.begin());
    bool undo=p->positionKnown && hist.size()+1==p->board.size() &&
              std::equal(hist.begin(),hist.end(),p->board.begin());
    if(undo)cmd<<"TAKEBACK "<<p->board.back()%15<<','<<p->board.back()/15<<'\n';
    else if(!same && !turn){
        cmd<<"YXBOARD\n";
        for(size_t i=0;i<hist.size();i++) cmd<<hist[i]%15<<','<<hist[i]/15<<','<<(i%2==hist.size()%2?1:2)<<'\n';
        cmd<<"DONE\n";
    }
    if(p->restricted){
        std::array<bool,225> keep{};
        for(int m:hist)keep[m]=true;
        for(int m:choices)keep[m]=true;
        cmd<<"YXBLOCK\n";
        for(int m=0;m<225;m++)if(!keep[m])cmd<<m%15<<','<<m/15<<'\n';
        cmd<<"DONE\n";
    }
    if(turn)cmd<<"TURN "<<hist.back()%15<<','<<hist.back()/15<<'\n';
    else cmd<<"YXNBEST "<<std::max<size_t>(choices.size(),1)<<'\n';
    p->requested=hist;p->active=true;p->settled=false;p->checking=false;p->positionKnown=false;p->stopRequested=false;p->send(cmd.str());return true;
}
Advice Shield::poll(){if(available()) p->drain(); return p->latest;}
const std::vector<Advice>& Shield::alternatives() const{return p->scores;}
void Shield::requestStop(){
    if(available() && p->active && !p->stopRequested){p->send("STOP\n");p->stopRequested=true;}
}
Advice Shield::stop(Time::time_point deadline) {
    if(available() && !p->settled) {
        requestStop();
        while(!p->idle(true)){
            if(Time::now()>=deadline){p->pauseLateSearch();return p->latest;}
            std::this_thread::yield();
        }
        p->active=false;p->deferred=false;
    }
    return p->latest;
}
}
