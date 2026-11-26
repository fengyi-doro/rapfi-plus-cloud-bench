// SPDX-License-Identifier: GPL-3.0-or-later
#include "network.h"
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace omega;

int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("usage: omega-bench model_directory");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"omega-bench");
    constexpr std::array<int,16> moves{112,113,97,127,81,111,129,99,98,80,114,96,143,128,142,115};
    std::array<Features,8> boards;
    std::cout<<std::setprecision(8)<<"[\n";
    bool first=true;
    for(int rule:{1,2}) for(int threads:{2,4}) {
        Network net(env,fs::path(argv[1])/(rule==1?"standard15x.onnx":"renju15x.onnx"),threads);
        Batch one(net,1);
        std::array<Evaluation,8> reference;
        for(int i=0;i<8;i++) {
            Features f; f.rule=rule; f.ply=6+i; f.last=moves[f.ply-1];
            for(int j=0;j<f.ply;j++) f.cells[moves[j]]=uint8_t(1+j%2);
            boards[i]=f; one.fill(0,f,i); one.run(); reference[i]=one.get(0);
        }
        for(int size:{1,4,8}) {
            Batch b(net,size);
            double delta=0;
            for(int j=0;j<8;j+=size) {
                for(int i=0;i<size;++i) b.fill(i,boards[j+i],j+i);
                b.run();
                for(int i=0;i<size;++i) {
                    auto out=b.get(i);
                    delta=std::max(delta,double(std::abs(out.value-reference[j+i].value)));
                    for(int p=0;p<225;++p) delta=std::max(delta,double(std::abs(out.logits[p]-reference[j+i].logits[p])));
                }
            }
            if(delta>0.005) throw std::runtime_error("batch disagrees with single-row evaluation");
            for(int i=0;i<3;i++) b.run();
            std::vector<double> latency;
            auto start=Clock::now();
            while(latency.size()<30 || milliseconds(Clock::now()-start)<1500) {
                auto t=Clock::now(); b.run(); latency.push_back(milliseconds(Clock::now()-t));
            }
            double elapsed=milliseconds(Clock::now()-start);
            std::sort(latency.begin(),latency.end());
            if(!first) std::cout<<",\n";
            first=false;
            std::cout<<"{\"rule\":"<<rule<<",\"threads\":"<<threads<<",\"batch\":"<<size
                     <<",\"global_width\":"<<net.globals<<",\"policy_heads\":"<<net.policies
                     <<",\"calls\":"<<latency.size()<<",\"evals_s\":"<<1000*size*latency.size()/elapsed
                     <<",\"p50_ms\":"<<latency[latency.size()/2]<<",\"p95_ms\":"<<latency[(latency.size()*95-1)/100]
                     <<",\"max_row_error\":"<<delta<<"}"<<std::flush;
        }
    }
    std::cout<<"\n]\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
