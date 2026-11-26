// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.h"
#include "solver_api.h"
#include "cloud_tactics.h"
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
namespace omega {
namespace {
int integer(const char* key,int fallback,int lo,int hi) {
    if(auto v=std::getenv(key)) return std::clamp(std::stoi(v),lo,hi);
    return fallback;
}
double number(const char* key,double fallback,double lo,double hi) {
    if(auto v=std::getenv(key)) {
        double n=std::stod(v);
        if(!std::isfinite(n)) throw std::invalid_argument(key);
        return std::clamp(n,lo,hi);
    }
    return fallback;
}
uint64_t mix(uint64_t x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
uint64_t stone(int p,int c){return mix(uint64_t(4*p+c)+0x9e3779b97f4a7c15ULL);}
struct StateDeleter{void operator()(State* s) const{gh_free(s);}};
struct Node;
struct Edge {
    int move;
    float prior;
    std::unique_ptr<Node> child;
};
struct Node {
    std::unique_ptr<State,StateDeleter> state;
    Features f;
    std::array<uint64_t,8> hashes{};
    std::array<uint64_t,4> allowed{~0ULL,~0ULL,~0ULL,~0ULL};
    std::vector<Edge> edges;
    std::array<uint8_t,225> legal{};
    int count=0,lo=-1,hi=1,witness=-1,source=0,visits=0,pending=0,sym=0;
    uint64_t hash=0;
    double sum=0;
    float estimate=0;
    Node* parent=nullptr;
    bool probed=false,expanded=false,localLower=false,rootProbed=false,shortlistChecked=false;
    Node(State* s,Features features,Node* father):state(s),f(std::move(features)),parent(father) {
        if(!s) throw std::invalid_argument("illegal position");
        if(parent) hashes=parent->hashes;
        for(int d=0;d<8;d++) {
            const auto& map=symmetries()[d];
            if(parent) hashes[d]^=stone(map[f.last],f.cells[f.last]);
            else for(int p=0;p<225;p++) if(f.cells[p]) hashes[d]^=stone(map[p],f.cells[p]);
            uint64_t tag=mix(10000+(f.last<0?225:map[f.last])+256*(f.rule*2+f.ply%2));
            auto key=hashes[d]^tag;
            if(!d || key<hash){sym=d;hash=key;}
        }
    }
    std::unique_ptr<Node> child(int move) {
        Features next=f; next.cells[move]=uint8_t(1+f.ply%2); next.ply++; next.last=move;next.myVcf=next.oppVcf=0;
        return std::make_unique<Node>(gh_child(state.get(),move),next,this);
    }
    bool permits(int move) const {
        return std::find(legal.begin(),legal.begin()+count,move)!=legal.begin()+count;
    }
    void expand(const Evaluation& e,float temperature) {
        estimate=e.value; expanded=true;edges.reserve(count);
        const auto& map=symmetries()[sym]; float peak=-std::numeric_limits<float>::infinity();
        for(int i=0;i<count;i++) peak=std::max(peak,e.logits[map[legal[i]]]);
        double total=0;
        for(int i=0;i<count;i++) {
            float prior=std::exp((e.logits[map[legal[i]]]-peak)/temperature);
            edges.push_back({legal[i],prior,{}}); total+=prior;
        }
        for(auto& edge:edges) edge.prior=float(edge.prior/total);
        std::stable_sort(edges.begin(),edges.end(),[](auto& a,auto& b){return a.prior>b.prior;});
    }
    void propagate() {
        if(!expanded || edges.empty() || lo==hi) return;
        int lower=-1,upper=-1,move=-1;
        for(auto& edge:edges) {
            int l=edge.child?-edge.child->hi:-1, h=edge.child?-edge.child->lo:1;
            if(l>lower){lower=l;move=edge.move;} upper=std::max(upper,h);
        }
        if(lower>=lo && move>=0){lo=lower;witness=move;}
        hi=std::min(hi,upper);
        if(lo>hi) throw std::logic_error("inconsistent minimax certificate");
        if(lo==hi) source=10;
    }
};
struct Cached {
    uint64_t key=0;
    Features feature;
    bool occupied=false;
    Evaluation eval;
    bool matches(const Node& n) const {
        if(!occupied || key!=n.hash || feature.rule!=n.f.rule || feature.ply!=n.f.ply ||
           feature.myVcf!=n.f.myVcf || feature.oppVcf!=n.f.oppVcf) return false;
        const auto& map=symmetries()[n.sym];
        if(feature.last!=(n.f.last<0?-1:map[n.f.last])) return false;
        for(int p=0;p<225;p++) if(feature.cells[map[p]]!=n.f.cells[p]) return false;
        return true;
    }
    void save(const Node& n, const Evaluation& value) {
        const auto& map=symmetries()[n.sym];
        for(int p=0;p<225;p++) feature.cells[map[p]]=n.f.cells[p];
        feature.rule=n.f.rule; feature.ply=n.f.ply;feature.last=n.f.last<0?-1:map[n.f.last];
        feature.myVcf=n.f.myVcf;feature.oppVcf=n.f.oppVcf;
        key=n.hash; eval=value; occupied=true;
    }
};
void reserve(Node* n,int delta){for(;n;n=n->parent)n->pending+=delta;}
void backup(Node* n,float value) {
    for(;n;n=n->parent) {
        n->propagate();
        value=std::clamp(value,float(n->lo),float(n->hi));
        n->visits++; n->sum+=value; value=-value;
    }
}
const char* proofName(int id) {
    static constexpr const char* names[]={"mcts","terminal","five","vcf","double_threat","forced_block",
        "rzone","f8_draw","no_legal","f10_draw","minimax","vct","f15","f17"};
    return id>=0 && id<14?names[id]:"mcts";
}
struct Flight{Slot* slot=nullptr;std::vector<Node*> leaves;uint64_t order=0;};
}
Options Options::environment(){
    Options o;
    o.batch=integer("OMEGA_BATCH",4,1,8);
    o.f15=integer("OMEGA_F15",1,0,1);o.f17=integer("OMEGA_F17",1,0,1);
    o.rootWork=integer("OMEGA_ROOT_WORK",1200,0,16000);
    o.leafWork=integer("OMEGA_LEAF_WORK",72,0,4096);
    o.rootDepth=integer("OMEGA_ROOT_DEPTH",15,1,31);
    o.leafDepth=integer("OMEGA_LEAF_DEPTH",7,1,31);
    o.cpuct=number("OMEGA_CPUCT",1.35,.05,8);
    o.fpu=number("OMEGA_FPU",.20,0,2);
    o.priorTemp=number("OMEGA_PRIOR_TEMP",1.,.3,4);
    o.cpuPrior=number("OMEGA_CPU_PRIOR",.23,0,.9);
    o.budgetScale=number("OMEGA_TIME_SCALE",1.,.2,4);
    o.shield=integer("OMEGA_SHIELD",1,0,1);
    o.shieldCore=layout().shield;
    o.cacheEntries=integer("OMEGA_CACHE",16384,0,131072);
    o.visitLimit=integer("OMEGA_VISIT_LIMIT",0,0,1000000);
    o.inputVcf=integer("OMEGA_INPUT_VCF",240,0,4096);
    o.reuse=integer("OMEGA_REUSE",1,0,1);o.frontload=integer("OMEGA_FRONTLOAD",1,0,1);
    return o;
}
std::string Turn::json() const {
    std::ostringstream s;
    s<<"{\"move\":"<<move<<",\"reason\":\""<<reason<<"\",\"lo\":"<<lo<<",\"hi\":"<<hi
     <<",\"value\":"<<value<<",\"visits\":"<<visits<<",\"reused\":"<<reused<<",\"eval_rows\":"<<evalRows
     <<",\"slots\":"<<slots<<",\"discarded\":"<<discarded<<",\"hits\":"<<hits<<",\"probes\":"<<probes<<",\"vcf_work\":"<<vcfWork
     <<",\"f15_calls\":"<<f15Calls<<",\"f15_decisive\":"<<f15Decisive
     <<",\"f17_calls\":"<<f17Calls<<",\"f17_wins\":"<<f17Wins<<",\"f17_nodes\":"<<f17Nodes
     <<",\"vct_work\":"<<vctWork<<",\"vct_win\":"<<vctWins<<",\"proof_checks\":"<<proofChecks<<",\"proof_veto\":"<<proofVeto<<",\"input_vcf_nodes\":"<<inputNodes<<",\"input_vcf_wins\":"<<inputWins<<",\"input_vcf_threats\":"<<inputThreats<<",\"cpu_move\":"<<cpuMove<<",\"cpu_depth\":"<<cpuDepth<<",\"cpu_score\":"<<cpuScore
     <<",\"cpu_mate\":"<<cpuMate<<",\"shield_core\":"<<shieldCore<<",\"shield_lost\":"<<shieldLost<<",\"refutations\":"<<refutations
     <<",\"veto\":"<<veto<<",\"takeover\":"<<takeover<<",\"correction\":"<<correction<<",\"extended\":"<<extended<<",\"early\":"<<early<<",\"draw\":"<<draw
     <<",\"cpu_choice_checks\":"<<cpuChoiceChecks<<",\"cpu_choice_refuted\":"<<cpuChoiceRefuted
     <<",\"shield_deferred\":"<<shieldDeferred<<",\"shield_recovered\":"<<shieldRecovered
     <<",\"budget_ms\":"<<budget<<",\"elapsed_ms\":"<<elapsed<<",\"infer_wait_ms\":"<<inferWaitMs
     <<",\"infer_submit_ms\":"<<inferSubmitMs<<",\"cpu_stop_ms\":"<<cpuStopMs<<",\"top\":[";
    for(size_t i=0;i<candidates.size();i++) {
        if(i) s<<',';
        const auto& a=candidates[i];s<<'['<<a[0]<<','<<a[1]<<','<<a[2]<<','<<a[3]<<']';
    }
    s<<"],\"replies\":[";
    for(size_t i=0;i<replies.size();i++){
        if(i)s<<',';
        const auto& r=replies[i];s<<'['<<r[0]<<','<<r[1]<<','<<r[2]<<','<<r[3]<<','<<r[4]<<']';
    }
    s<<"],\"multipv\":"<<poolSearch<<",\"cpu_pool\":[";
    for(size_t i=0;i<pool.size();i++){
        if(i)s<<',';
        const auto& r=pool[i];s<<'['<<r[0]<<','<<r[1]<<','<<r[2]<<','<<r[3]<<']';
    }
    s<<"]}"; return s.str();
}
struct Search::Impl {
    fs::path home;
    Options cfg;
    Device device;
    std::unique_ptr<Kernel> single,many;
    std::unique_ptr<Shield> shield;
    bool shieldNeedsScore=false;
    int rule=1;
    std::vector<int> hist;
    std::unique_ptr<Node> root;
    std::vector<Cached> cache;
    explicit Impl(const fs::path& folder,const Options& opts):home(folder),cfg(opts),
      cache(std::max(1,cfg.cacheEntries)) {
        if(cfg.shield) shield=std::make_unique<Shield>(home,cfg.shieldCore);
    }
    void probe(Node& n,Turn& t,bool asRoot) {
        if(n.probed && (!asRoot || n.rootProbed)) return;
        Verdict v;
        gh_probe(n.state.get(),asRoot?cfg.rootDepth:cfg.leafDepth,asRoot?cfg.rootWork:cfg.leafWork,&v);
        t.probes++;t.vcfWork+=v.nodes;n.probed=true;n.rootProbed|=asRoot;
        n.lo=std::max(n.lo,v.lo);n.hi=std::min(n.hi,v.hi);n.localLower|=bool(v.lower);
        if(n.lo>n.hi) throw std::logic_error("conflicting local bounds");
        if(v.move>=0)n.witness=v.move;
        if(v.reason)n.source=v.reason;
        if(n.lo!=n.hi)for(int i=0;i<4;i++)n.allowed[i]&=v.mask[i];
        if(n.lo<n.hi && (asRoot || n.parent==root.get()) && n.f.ply<=105) {
            const auto start=Clock::now();
            auto accept=[&](const TacticalResult& r,int source) {
                if(!r.value) return;
                if(r.value<n.lo || r.value>n.hi) throw std::logic_error("incompatible tactical proof");
                n.lo=n.hi=r.value; n.witness=r.move; n.source=source;
            };
            if(cfg.f15) {
                auto r=foulTrap(n.f,start+std::chrono::microseconds(asRoot?850:240));
                t.f15Calls+=r.calls;t.f15Decisive+=r.wins;accept(r,12);
            }
            if(cfg.f17 && n.lo<n.hi) {
                auto r=continuousThreats(n.f,Clock::now()+std::chrono::microseconds(asRoot?1600:320));
                t.f17Calls+=r.calls;t.f17Wins+=r.wins;t.f17Nodes+=r.nodes;accept(r,13);
            }
        }
        if(cfg.inputVcf && n.lo<n.hi && v.reason!=5 && !n.f.myVcf){
            NeuralVCF result;omega_vcf_inputs(n.state.get(),cfg.inputVcf,&result);
            n.f.myVcf=uint8_t(result.mine);n.f.oppVcf=uint8_t(result.theirs);t.inputNodes+=result.nodes;
            if(result.mine==1){
                if(n.hi<1)throw std::logic_error("VCF contradicts upper bound");
                n.witness=result.winningMove;n.lo=n.hi=1;n.source=3;t.inputWins++;
            }else if(result.theirs==1){
                for(int i=0;i<4;i++)n.allowed[i]&=result.zone[i];
                n.source=6;t.inputThreats++;
            }
        }
        if(n.lo<n.hi && n.hi==1 && v.reason==0 && (asRoot || n.parent==root.get())) {
            std::array<uint8_t,16> hints{};int count=0,work=0;
            if(n.expanded)for(const auto& edge:n.edges){
                if(count==int(hints.size()))break;
                hints[count++]=uint8_t(edge.move);
            }
            int win=omega_threat(n.state.get(),hints.data(),count,asRoot,&work);
            t.vctWork+=work;
            if(win>=0){n.lo=n.hi=1;n.witness=win;n.source=11;t.vctWins++;}
        }
        if(n.lo!=n.hi && asRoot && n.f.ply>=90) {
            if(int witness=omega_draw(n.state.get(),n.f.cells.data());witness>=0) {
                n.witness=witness;n.lo=n.hi=0;n.source=9;
            }
        }
        if(asRoot && v.lower && n.witness<0) n.witness=gh_nonloss_move(n.state.get());
        constexpr uint64_t all[]={~0ULL,~0ULL,~0ULL,~0ULL};
        n.count=omega_legal(n.state.get(),n.lo==n.hi?all:n.allowed.data(),n.legal.data());
        if(!n.count && gh_terminal(n.state.get())==2) {
            n.lo=n.hi=-1;n.count=omega_legal(n.state.get(),all,n.legal.data());n.source=n.count?6:8;
        }
        if(n.expanded && n.lo!=n.hi) {
            std::erase_if(n.edges,[&](const Edge& e){return !n.permits(e.move);});
            double sum=0;for(const auto& e:n.edges)sum+=e.prior;
            if(sum>0) for(auto& e:n.edges)e.prior/=sum;
            n.propagate();
        }
    }
    Node* select(Node& n,const Advice& cpu,int pvIndex) {
        if(n.lo==n.hi) return nullptr;
        if(!n.expanded) return n.pending?nullptr:&n;
        float parent=float(n.visits?n.sum/n.visits:n.estimate);
        double exploration=cfg.cpuct*std::sqrt(std::max(1,n.visits+n.pending));
        int guide=pvIndex>=0 && pvIndex<cpu.pvLength && cpu.depth>=6 && cpu.mate>=0 ? cpu.pv[pvIndex]:-1;
        double weight=pvIndex==0?cfg.cpuPrior:cfg.cpuPrior*.55;
        std::array<bool,225> tried{};
        for(;;) {
            Edge* best=nullptr;double peak=-1e30;
            for(auto& e:n.edges) {
                Node* c=e.child.get();
                if(tried[e.move] || (c && (c->lo==c->hi || -c->lo<=n.lo || (!c->expanded&&c->pending)))) continue;
                int done=c?c->visits:0, queued=c?c->pending:0;
                double q=done+queued?(-c->sum+queued*std::max(-1.f,parent-cfg.fpu-.12f))/(done+queued):parent-cfg.fpu;
                double prior=e.prior;
                if(guide>=0) prior=(1-weight)*prior+(e.move==guide?weight:0);
                double score=q+exploration*prior/(1+done+queued);
                if(score>peak){peak=score;best=&e;}
            }
            if(!best)return nullptr;
            if(!best->child) best->child=n.child(best->move);
            Node* found=select(*best->child,cpu,guide==best->move?pvIndex+1:-1);
            if(found) return found;
            tried[best->move]=true;
        }
    }
    std::vector<Edge*> ranking() {
        std::vector<Edge*> choices;
        for(auto& e:root->edges) {
            if(root->lo!=root->hi && e.child && -e.child->lo<root->lo) continue;
            if(root->lo<0 && e.child && e.child->lo==1 && root->hi>-1) continue;
            choices.push_back(&e);
        }
        std::stable_sort(choices.begin(),choices.end(),[](Edge* a,Edge* b){
            int an=a->child?a->child->visits:0,bn=b->child?b->child->visits:0;
            return an!=bn?an>bn:a->prior>b->prior;
        });
        return choices;
    }
    bool eligibleCpuMove(int move) const {
        if(!root->permits(move))return false;
        if(root->lo==0 && root->witness==move)return true;
        for(const auto& e:root->edges)if(e.move==move && e.child) {
            return root->lo==0?e.child->hi<=0:e.child->lo<1;
        }
        return root->lo<0;
    }
    int verifyShortlist(const Advice& cpu,Clock::time_point limit,Turn& turn){
        if(root->lo==root->hi || root->f.ply>=105 || Clock::now()>=limit)return -1;
        auto rank=ranking();if(rank.empty())return -1;
        int first=rank.front()->move,rejectedFirst=-1;
        for(size_t i=0;i<rank.size() && root->lo<root->hi && Clock::now()<limit;i++){
            Edge& edge=*rank[i];
            if(i>=2 && edge.move!=cpu.move)continue;
            if(!edge.child)edge.child=root->child(edge.move);
            auto& child=*edge.child;
            if(child.lo==child.hi || child.hi<1 || child.shortlistChecked)continue;
            child.shortlistChecked=true;int work=0;turn.proofChecks++;
            int win=omega_threat(child.state.get(),nullptr,0,2,&work);turn.vctWork+=work;
            if(win<0)continue;
            child.lo=child.hi=1;child.witness=win;child.source=11;
            constexpr uint64_t all[]={~0ULL,~0ULL,~0ULL,~0ULL};
            child.count=omega_legal(child.state.get(),all,child.legal.data());
            child.probed=child.rootProbed=true;turn.vctWins++;
            backup(&child,1);
            if(edge.move==first)rejectedFirst=first;
        }
        return rejectedFirst;
    }
    int fastMove() {
        if(root->lo!=root->hi && root->count!=1) return -1;
        if(root->witness>=0 && root->permits(root->witness)) return root->witness;
        if(root->lo==0 && root->localLower) {
            int witness=gh_nonloss_move(root->state.get());
            if(root->permits(witness)) return witness;
        }
        return root->count?root->legal[0]:-1;
    }
    Turn run(double soft,double hard,const std::atomic<uint64_t>& stop,uint64_t seq) {
        auto begin=Clock::now();
        auto after=[&](double ms){return begin+std::chrono::microseconds(int64_t(ms*1000));};
        Turn t;t.budget=soft;t.reused=root->visits;t.shieldCore=shield?shield->core():-1;
        auto elapsed=[&]{return milliseconds(Clock::now()-begin);};
        probe(*root,t,true);
        if(root->lo==root->hi || root->count<=1) {
            t.move=fastMove();t.lo=root->lo;t.hi=root->hi;t.value=t.lo==t.hi?t.lo:root->estimate;
            t.reason=root->count==1 && t.lo!=t.hi?"forced_block":proofName(root->source);
            t.draw=t.lo==0 && t.hi==0;t.elapsed=elapsed();return t;
        }
        bool emergency=hard<94;
        bool cpuEligible=shield && shield->available() && !emergency;
        auto startRoot=[&]{return shield->start(hist,int(soft*(soft>=65?.40:.70)));};
        bool useCpu=cpuEligible && startRoot();
        Advice cpu; int examining=-1;
        enum class Phase { Root, Pool, Reply };
        Phase phase=Phase::Root;
        int checkedCpuChoice=-1;
        std::unordered_map<int,Advice> tested,pooled;
        auto recordCpu=[&](const Advice& a){
            if(phase==Phase::Root)cpu=a;
            else if(phase==Phase::Reply && a.hasScore && (a.complete || a.mate>0))tested[examining]=a;
            else if(phase==Phase::Pool){
                for(const auto& item:shield->alternatives()){
                    pooled[item.move]=item;
                    if(item.mate>0 && eligibleCpuMove(item.move))cpu=item;
                }
                if(a.mate>0 && eligibleCpuMove(a.move))cpu=a;
            }
        };
        std::array<Flight,2> flights{};
        uint64_t order=0;
        auto consume=[&](Flight& f) {
            double allowance=std::clamp(soft*.14,16.,34.);
            auto beganWait=Clock::now();
            bool ready=f.slot->finishBy(after(std::min(soft+allowance,hard-(useCpu?16.:5.))));
            t.inferWaitMs+=milliseconds(Clock::now()-beganWait);
            if(!ready) {
                for(Node* n:f.leaves)reserve(n,-1);
                t.discarded+=int(f.leaves.size());f.leaves.clear();f.slot=nullptr;return;
            }
            for(size_t i=0;i<f.leaves.size();i++){
                Node* n=f.leaves[i];auto e=f.slot->get(i);reserve(n,-1);
                if(cfg.cacheEntries)cache[n->hash%cache.size()].save(*n,e);
                n->expand(e,cfg.priorTemp); backup(n,e.value);
            }
            f.leaves.clear();f.slot=nullptr;
        };
        int stable=-1,stableCount=0,checkpoint=0; bool takeover=false;
        while(root->lo!=root->hi && (!cfg.visitLimit || root->visits+root->pending<cfg.visitLimit) &&
              stop.load(std::memory_order_relaxed)<=seq) {
            double now=elapsed();
            if(!useCpu && cpuEligible && now<std::min(35.,.35*soft))useCpu=startRoot();
            if(useCpu) {
                auto a=shield->poll();
                recordCpu(a);
                if(cpu.mate>0 && root->hi==1 && eligibleCpuMove(cpu.move)) {takeover=true;break;}
                if(phase==Phase::Reply && examining==checkedCpuChoice && a.mate>0 && !t.cpuChoiceRefuted){
                    t.cpuChoiceRefuted=1;
                    // Leave clock for an opponent search at an alternative before changing move.
                    if(pooled.empty() && hard>=280 && soft-now<45){
                        double limit=std::min(hard-25.,now+std::clamp(hard*.065,26.,85.));
                        if(limit>soft){soft=limit;t.budget=soft;t.extended=1;}
                    }
                    if(!a.complete)shield->requestStop();
                }
                if(!t.extended && now>soft*.48 && hard>=480 && root->f.ply<85 &&
                   root->visits>=30 && cpu.depth>=10 && cpu.mate==0 && root->permits(cpu.move)) {
                    auto rank=ranking();
                    double value=root->sum/std::max(1,root->visits);
                    bool openChoice=cfg.frontload && root->f.ply<42 && rank.size()>=2 && rank[0]->child && rank[1]->child &&
                                    rank[0]->child->visits<1.7*std::max(1,rank[1]->child->visits);
                    if(!rank.empty() && rank.front()->move!=cpu.move &&
                       (openChoice || value<-.40 || cpu.score<=-160 || (value>.48 && cpu.score>200))) {
                        double limit=std::min(soft*(cfg.frontload?1.75:1.55),hard*(cfg.frontload && root->f.ply<44?.43:.34));
                        if(limit>soft+15){soft=limit;t.budget=soft;t.extended=1;}
                    }
                }
                if(a.complete && now+12<soft) {
                    auto rank=ranking();
                    bool poolWanted=phase==Phase::Root && !rank.empty() && cpu.depth>=5 && eligibleCpuMove(cpu.move) &&
                                    rank.front()->move!=cpu.move && soft-now>=21;
                    double value=root->sum/std::max(1,root->visits);
                    bool checkCpuChoice=checkedCpuChoice<0 && rank.size()>=2 && cpu.hasScore && cpu.mate==0 &&
                        eligibleCpuMove(cpu.move) && (value<-.18 || cpu.score<=-120) && soft-now>=20 &&
                        ((phase==Phase::Root && rank.front()->move==cpu.move) || phase==Phase::Pool);
                    if(checkCpuChoice){
                        auto h=hist;h.push_back(cpu.move);
                        if(shield->start(h,std::max(9,int((soft-now)*.62)))){
                            examining=checkedCpuChoice=cpu.move;phase=Phase::Reply;t.refutations++;t.cpuChoiceChecks++;
                        }
                    } else if(poolWanted) {
                        std::vector<int> choices{cpu.move};
                        for(const auto* e:rank){
                            if(e->move!=cpu.move)choices.push_back(e->move);
                            if(choices.size()==3)break;
                        }
                        if(choices.size()>1 && shield->start(hist,std::max(10,int((soft-now)*.78)),choices)){
                            phase=Phase::Pool;t.poolSearch++;
                        }
                    } else {
                    // Give the opponent a separate root search at each promising
                    // candidate. A root mate on another move cannot refute this one.
                    for(size_t i=0;i<std::min<size_t>(3,rank.size());i++) {
                        int move=rank[i]->move;
                        if(tested.contains(move) || move==cpu.move) continue;
                        auto h=hist;h.push_back(move);
                        if(shield->start(h,std::max(7,int((soft-now)*.73)))){
                            examining=move;phase=Phase::Reply;t.refutations++;
                        }
                        break;
                    }
                    }
                }
            }
            if(now>=soft || now+(root->expanded?std::max(16.,cfg.batch*2.8):11.)>=hard || (emergency && root->expanded)) break;
            if(!cfg.visitLimit && root->visits-t.reused>=checkpoint+16 && now>soft*(cfg.frontload?.29:.40)) {
                checkpoint=root->visits-t.reused;auto rank=ranking();
                if(!rank.empty() && rank[0]->child && rank[0]->child->visits>.75*std::max(1,root->visits-1)) {
                    stableCount=rank[0]->move==stable?stableCount+1:1;stable=rank[0]->move;
                    bool waitingForReply=phase==Phase::Reply && examining==checkedCpuChoice && !tested.contains(examining);
                    if(stableCount>=2 && !waitingForReply){t.early=1;break;}
                } else stableCount=0;
            }
            int free=flights[0].slot?(flights[1].slot?-1:1):0;
            if(free<0) {consume(flights[flights[0].order<flights[1].order?0:1]);continue;}
            if(soft<75 && flights[1-free].slot){consume(flights[1-free]);continue;}
            if(root->expanded && many->slots[free]->busy && !flights[1-free].slot && !many->slots[1-free]->busy)free=1-free;
            auto& f=flights[free];
            Slot* slot=root->expanded?many->slots[free].get():single->slots[0].get();
            int progress=0;
            if(!slot->busy) while(int(f.leaves.size())<slot->batch && root->lo!=root->hi) {
                if(cfg.visitLimit && root->visits+root->pending>=cfg.visitLimit)break;
                if(elapsed()>=soft || stop.load(std::memory_order_relaxed)>seq) break;
                Node* n=select(*root,cpu,t.cpuChoiceRefuted && cpu.move==checkedCpuChoice?-1:0);
                if(!n)break;
                probe(*n,t,false);progress++;
                if(n->lo==n->hi) {backup(n,float(n->lo));continue;}
                if(n->count==1 && n!=root.get()) {
                    Evaluation forced{};forced.value=n->parent?-n->parent->estimate:0;
                    n->expand(forced,cfg.priorTemp);continue;
                }
                auto& entry=cache[n->hash%cache.size()];
                if(entry.matches(*n)) {
                    n->expand(entry.eval,cfg.priorTemp);backup(n,entry.eval.value);t.hits++;continue;
                }
                std::array<uint64_t,4> blackFouls{};
                if(n->f.rule==2)omega_fouls(n->state.get(),blackFouls.data());
                reserve(n,1);slot->fill(int(f.leaves.size()),n->f,n->sym,blackFouls.data());f.leaves.push_back(n);
            }
            if(!f.leaves.empty()){
                f.slot=slot;f.order=order++;
                auto beganSubmit=Clock::now();slot->start();t.inferSubmitMs+=milliseconds(Clock::now()-beganSubmit);
                t.evalRows+=int(f.leaves.size());t.slots+=slot->batch;
            } else if(flights[1-free].slot) consume(flights[1-free]);
            else if(!progress) break;
        }
        bool finishCpu=useCpu || (cpuEligible && shield->recovering());
        if(finishCpu)shield->requestStop();
        for(auto& f:flights) if(f.slot) consume(f);
        if(finishCpu) {
            auto beganStop=Clock::now();
            double grace=std::clamp(soft*.11,10.,25.);
            auto a=shield->stop(after(std::min(elapsed()+grace,hard-8.)));
            t.cpuStopMs=milliseconds(Clock::now()-beganStop);
            if(useCpu){
                recordCpu(a);
                if(phase==Phase::Reply && examining==checkedCpuChoice && a.mate>0)t.cpuChoiceRefuted=1;
                takeover=(takeover || cpu.mate>0) && root->hi==1 && eligibleCpuMove(cpu.move);
                t.cpuMove=cpu.move;t.cpuDepth=cpu.depth;t.cpuScore=cpu.score;t.cpuMate=cpu.mate;
            }
            t.shieldLost=!shield->available();
            t.shieldDeferred=shield->deferred();
            if(shieldNeedsScore && useCpu && cpu.move>=0){t.shieldRecovered=1;shieldNeedsScore=false;}
            if(t.shieldDeferred)shieldNeedsScore=true;
            if(t.shieldLost)shieldNeedsScore=false;
        }
        int beforeProof=-1;
        if(!emergency && stop.load(std::memory_order_relaxed)<=seq)
            beforeProof=verifyShortlist(cpu,after(std::min(hard-8.,elapsed()+3.8)),t);
        takeover=takeover && root->hi==1 && eligibleCpuMove(cpu.move);
        auto rank=ranking();
        for(const auto& [move,a]:tested)t.replies.push_back({move,a.depth,a.score,a.mate,int(a.complete)});
        std::sort(t.replies.begin(),t.replies.end());
        for(const auto& [move,a]:pooled)t.pool.push_back({move,a.depth,a.score,a.mate});
        std::sort(t.pool.begin(),t.pool.end());
        for(size_t i=0;i<std::min<size_t>(5,rank.size());i++){
            const auto& edge=*rank[i];auto* c=edge.child.get();
            t.candidates.push_back({double(edge.move),double(c?c->visits:0),c&&c->visits?-c->sum/c->visits:root->estimate,edge.prior});
        }
        t.lo=root->lo;t.hi=root->hi;t.value=root->visits?root->sum/root->visits:root->estimate;
        t.visits=root->visits-t.reused;
        if(root->lo==root->hi) {t.move=fastMove();t.reason=proofName(root->source);t.value=t.lo;t.draw=t.lo==0;}
        else if(takeover && root->lo<1) {t.move=cpu.move;t.reason="cpu_mate";t.takeover=1;}
        else if(root->lo==0 && root->witness>=0) {t.move=root->witness;t.reason="nonloss";}
        else {
            int preferred=rank.empty()?(root->permits(cpu.move)?cpu.move:root->legal[0]):rank[0]->move;
            if(rank.empty() && !root->permits(cpu.move)) {
                const auto& cached=cache[root->hash%cache.size()];bool hit=cached.matches(*root);
                float best=-1e30f;
                for(int i=0;i<root->count;i++){
                    int m=root->legal[i],last=root->f.last<0?112:root->f.last;
                    float value=hit?cached.eval.logits[symmetries()[root->sym][m]]:
                        -float(std::abs(m%15-last%15)+std::abs(m/15-last/15))
                        -.05f*(std::abs(m%15-7)+std::abs(m/15-7));
                    if(value>best){preferred=m;best=value;}
                }
            }
            t.move=preferred;
            auto rejected=[&](int move){
                auto it=tested.find(move);auto pv=pooled.find(move);
                return (it!=tested.end() && it->second.mate>0) || (pv!=pooled.end() && pv->second.mate<0);
            };
            if(rejected(preferred)) {
                for(auto edge:rank) {
                    auto it=tested.find(edge->move);
                    auto pv=pooled.find(edge->move);
                    if(!rejected(edge->move) && ((it!=tested.end() && it->second.complete && it->second.hasScore) ||
                           (pv!=pooled.end() && pv->second.depth>=7 && pv->second.hasScore))){
                        t.move=edge->move;break;
                    }
                }
                if(t.move==preferred && cpu.hasScore && cpu.mate>=0 && eligibleCpuMove(cpu.move) && !rejected(cpu.move))
                    t.move=cpu.move;
                if(t.move!=preferred){t.veto=1;t.reason="cpu_veto";}
            }
            if(t.move==preferred && !rejected(preferred) && pooled.contains(preferred) && !rank.empty()){
                const auto& incumbent=pooled.at(preferred);
                auto mix=[&](const Edge& e,const Advice& a){
                    double net=e.child && e.child->visits ? -e.child->sum/e.child->visits : t.value;
                    double weight=a.depth>=11?.65:.48;
                    return weight*std::tanh(a.score/410.)+(1-weight)*net;
                };
                double best=mix(*rank.front(),incumbent)+.065;
                for(const auto* edge:rank){
                    auto it=pooled.find(edge->move);
                    if(it==pooled.end() || rejected(edge->move) || !eligibleCpuMove(edge->move))continue;
                    const auto& alternative=it->second;
                    if(incumbent.depth<7 || alternative.depth<std::max(7,incumbent.depth-2) ||
                       alternative.score<incumbent.score+70)continue;
                    double evidence=mix(*edge,alternative);
                    if(evidence>best){t.move=edge->move;best=evidence;}
                }
                if(t.move!=preferred){t.correction=1;t.reason="cpu_multipv";}
            }
            if(t.move==preferred && cpu.mate==0 && cpu.hasScore && cpu.depth>=10 &&
               cpu.move!=preferred && eligibleCpuMove(cpu.move) && !rejected(cpu.move)) {
                auto testedBest=tested.find(preferred);
                const Advice* reply=testedBest==tested.end()?nullptr:&testedBest->second;
                // Scores at the child have the opponent's sign. A genuine
                // mate veto above stays distinct from finite-depth advice.
                bool scoreGap=reply && reply->complete && reply->depth>=8 && reply->mate==0 && cpu.score+reply->score>=200;
                bool defensive=(t.value<-.55 && cpu.score<=-180) &&
                    (!reply || (!reply->mate && cpu.score+reply->score>=80));
                if(scoreGap || defensive){t.move=cpu.move;t.correction=1;t.reason="cpu_defence";}
            }
        }
        if(!root->permits(t.move)) throw std::logic_error("chosen move violates solver filter");
        if(beforeProof>=0 && t.move!=beforeProof && !eligibleCpuMove(beforeProof) && t.lo<t.hi){
            t.proofVeto=1;
            if(std::string_view(t.reason)=="mcts")t.reason="vct_veto";
        }
        t.elapsed=elapsed();return t;
    }
};
Search::Search(const fs::path& home,const Options& o):p(std::make_unique<Impl>(home,o)){reset(1);}
Search::~Search()=default;
void Search::reset(int rule) {
    if(!p->single || rule!=p->rule) {
        auto& net=p->device.network(p->home,rule);
        p->single=std::make_unique<Kernel>(p->device,net,1,1);
        p->many=std::make_unique<Kernel>(p->device,net,p->cfg.batch,2);
    }
    p->root.reset();p->hist.clear();p->rule=rule;p->shieldNeedsScore=false;clearCache();set({});
    if(p->shield && !p->shield->available())
        p->shield=std::make_unique<Shield>(p->home,p->cfg.shieldCore);
    if(p->shield) p->shield->reset(rule);
}
void Search::set(const std::vector<int>& history,bool knownLast) {
    if(p->root && p->hist==history && (knownLast || p->root->f.last<0))return;
    if(p->root && knownLast && p->hist.size()<history.size() && history.size()-p->hist.size()<=2
       && std::equal(p->hist.begin(),p->hist.end(),history.begin())){
        // Validate a whole BOARD replacement before changing the resident tree.
        const State* prior=p->root->state.get();
        std::unique_ptr<State,StateDeleter> trial;
        for(size_t i=p->hist.size();i<history.size();i++){
            trial.reset(gh_child(prior,history[i]));prior=trial.get();if(!prior)break;
        }
        if(prior){for(size_t i=p->hist.size();i<history.size();i++)advance(history[i]);return;}
    }
    Features f;f.rule=p->rule;f.ply=int(history.size());f.last=history.empty()||!knownLast?-1:history.back();
    for(size_t i=0;i<history.size();i++){
        int m=history[i];if(m<0||m>=225||f.cells[m])throw std::invalid_argument("invalid BOARD");
        f.cells[m]=uint8_t(1+i%2);
    }
    State* state=nullptr;
    if(knownLast){std::vector<uint8_t> moves(history.begin(),history.end());state=gh_root(f.rule,moves.data(),f.ply);}
    if(!state){f.last=-1;state=omega_diagram(f.rule,f.cells.data());}
    auto tree=std::make_unique<Node>(state,f,nullptr);p->hist=history;p->root=std::move(tree);
}
void Search::advance(int move) {
    if(move<0 || move>=225 || p->root->f.cells[move]) throw std::invalid_argument("invalid move");
    std::unique_ptr<Node> child;
    if(p->cfg.reuse)for(auto& edge:p->root->edges) if(edge.move==move && edge.child){child=std::move(edge.child);break;}
    if(!child) child=p->root->child(move);
    child->parent=nullptr;p->root=std::move(child);p->hist.push_back(move);
}
void Search::clearCache(){for(auto& c:p->cache)c.occupied=false;}
const std::vector<int>& Search::history() const{return p->hist;}
const std::string& Search::device() const{return p->device.name;}
Turn Search::think(double soft,double hard,const std::atomic<uint64_t>& stop,uint64_t seq){return p->run(soft,hard,stop,seq);}
}
