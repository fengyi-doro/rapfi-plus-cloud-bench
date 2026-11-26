// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace omega {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
struct Features {
    std::array<uint8_t,225> cells{};
    int rule=1, ply=0, last=-1;
    uint8_t myVcf=0, oppVcf=0;
};
using Map = std::array<uint8_t,225>;
inline const std::array<Map,8>& symmetries() {
    static const auto maps=[] {
        std::array<Map,8> result{};
        for(int d=0; d<8; ++d) for(int p=0; p<225; ++p) {
            int x=p%15, y=p/15;
            if(d&4) std::swap(x,y);
            if(d&1) x=14-x;
            if(d&2) y=14-y;
            result[d][p]=uint8_t(x+15*y);
        }
        return result;
    }();
    return maps;
}
struct Evaluation {
    std::array<float,225> logits{};
    float value=0;
};
class Network {
    Ort::Session session;
    static Ort::SessionOptions options(int threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.AddConfigEntry("session.intra_op.allow_spinning","0");
        return opts;
    }
    std::vector<int64_t> shape(const char* name,bool input) const {
        Ort::AllocatorWithDefaultOptions alloc;
        size_t count=input?session.GetInputCount():session.GetOutputCount();
        for(size_t i=0;i<count;++i) {
            auto key=input?session.GetInputNameAllocated(i,alloc):session.GetOutputNameAllocated(i,alloc);
            if(std::string(key.get())==name) {
                auto type=input?session.GetInputTypeInfo(i):session.GetOutputTypeInfo(i);
                auto info=type.GetTensorTypeAndShapeInfo();
                if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    throw std::runtime_error(std::string(name)+" requires float32");
                return info.GetShape();
            }
        }
        throw std::runtime_error(std::string("missing tensor ")+name);
    }
public:
    int globals, policies;
    Network(Ort::Env& env,const fs::path& path,int threads)
        : session(env,path.c_str(),options(threads)) {
        const auto spatial=shape("input_spatial",true), global=shape("input_global",true);
        const auto policy=shape("out_policy",false), value=shape("out_value",false);
        if(spatial.size()!=4 || spatial[1]!=22 || spatial[2]!=15 || spatial[3]!=15 ||
           global.size()!=2 || (global[1]!=19 && global[1]!=39) ||
           policy.size()!=3 || policy[1]<=0 || policy[2]!=226 || value.size()!=2 || value[1]!=3)
            throw std::runtime_error("unsupported KataGomo input or output dimensions");
        globals=int(global[1]); policies=int(policy[1]);
    }
    void infer(Ort::RunOptions& opts,const Ort::Value* inputs,Ort::Value* outputs) {
        static constexpr const char* ins[]={"input_spatial","input_global"};
        static constexpr const char* outs[]={"out_policy","out_value"};
        session.Run(opts,ins,inputs,2,outs,outputs,2);
    }
};
class Batch {
    Network& net;
    Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::array<std::vector<float>,4> data;
    std::array<Ort::Value,4> tensor{Ort::Value(nullptr),Ort::Value(nullptr),Ort::Value(nullptr),Ort::Value(nullptr)};
public:
    const int batch;
    Ort::RunOptions options;
    Batch(Network& network,int size):net(network),batch(size) {
        const std::array<std::vector<int64_t>,4> shapes={{{size,22,15,15},{size,net.globals},{size,net.policies,226},{size,3}}};
        for(size_t i=0;i<shapes.size();++i) {
            size_t length=1;
            for(auto dim:shapes[i]) length*=size_t(dim);
            data[i].resize(length);
            tensor[i]=Ort::Value::CreateTensor<float>(memory,data[i].data(),length,shapes[i].data(),shapes[i].size());
        }
        for(int i=0;i<size;++i) fill(i,Features{},0);
    }
    void fill(int row,const Features& f,int sym,const uint64_t* fouls=nullptr) {
        const auto& map=symmetries()[sym];
        float* s=data[0].data()+row*22*225;
        float* g=data[1].data()+row*net.globals;
        std::fill_n(s,22*225,0.f); std::fill_n(s,225,1.f); std::fill_n(g,net.globals,0.f);
        const int color=1+f.ply%2;
        for(int p=0;p<225;++p) if(f.cells[p]) s[(f.cells[p]==color?225:450)+map[p]]=1;
        if(net.globals==39 && f.last>=0) {g[1]=1; s[6*225+map[f.last]]=1;}
        g[3]=float(f.rule==1); g[4]=float(f.rule==2); g[5]=f.rule==2?(color==1?-1.f:1.f):0.f;
        if(f.myVcf) g[(net.globals==19 && f.myVcf==3)?11:6+f.myVcf]=1;
        if(f.oppVcf) g[net.globals==19?(f.oppVcf==3?12:8+f.oppVcf):9+f.oppVcf]=1;
        if(fouls && f.rule==2) {
            g[6]=1;
            for(int k=0;k<4;++k) for(auto w=fouls[k];w;w&=w-1) {
                const int p=64*k+std::countr_zero(w);
                if(p<225) s[(3+f.ply%2)*225+map[p]]=1;
            }
        }
    }
    void run() {net.infer(options,tensor.data(),tensor.data()+2);}
    Evaluation get(int row) const {
        Evaluation result;
        std::copy_n(data[2].data()+row*net.policies*226,225,result.logits.begin());
        const float* v=data[3].data()+3*row;
        const float m=std::max({v[0],v[1],v[2]});
        const float w=std::exp(v[0]-m),l=std::exp(v[1]-m),d=std::exp(v[2]-m);
        result.value=(w-l)/(w+l+d);
        if(!std::isfinite(result.value) || !std::all_of(result.logits.begin(),result.logits.end(),
                [](float x){return std::isfinite(x);})) throw std::runtime_error("non-finite network output");
        return result;
    }
};
} // namespace omega
