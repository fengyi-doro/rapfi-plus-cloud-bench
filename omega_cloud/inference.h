// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "network.h"
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <sched.h>
#include <thread>

namespace omega {
struct CpuLayout { int controller,shield; cpu_set_t inference; };
inline const CpuLayout& layout() {
    static const auto result=[] {
        cpu_set_t allowed;
        if(sched_getaffinity(0,sizeof(allowed),&allowed)) throw std::runtime_error("read CPU affinity");
        std::vector<int> cpus;
        for(int n=0;n<CPU_SETSIZE;++n) if(CPU_ISSET(n,&allowed)) cpus.push_back(n);
        if(cpus.size()!=4) throw std::runtime_error("start Omega with affinity to the four playing vCPUs");
        CpuLayout result{cpus[0],cpus[3],{}};
        CPU_ZERO(&result.inference);
        CPU_SET(cpus[1],&result.inference); CPU_SET(cpus[2],&result.inference);
        return result;
    }();
    return result;
}
inline void pin(const cpu_set_t& cpus) {
    if(sched_setaffinity(0,sizeof(cpus),&cpus)) throw std::runtime_error("set CPU affinity");
}
inline void pin(int cpu) {cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set); pin(set);}

class InferenceQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::packaged_task<void()>> pending;
    bool closing=false;
    std::thread worker;
    void loop() {
        pin(layout().inference);
        for(;;) {
            std::unique_lock lock(mutex);
            changed.wait(lock,[&]{return closing || !pending.empty();});
            if(pending.empty()) return;
            auto task=std::move(pending.front());
            pending.pop_front();
            lock.unlock();
            task();
        }
    }
public:
    InferenceQueue():worker([this]{loop();}) {}
    ~InferenceQueue() {
        {std::lock_guard lock(mutex); closing=true;}
        changed.notify_one(); worker.join();
    }
    std::future<void> submit(std::packaged_task<void()> task) {
        auto future=task.get_future();
        {std::lock_guard lock(mutex); pending.push_back(std::move(task));}
        changed.notify_one();
        return future;
    }
};
struct Slot : Batch {
    InferenceQueue& queue;
    std::future<void> pending;
    bool busy=false;
    Slot(Network& net,int b,InferenceQueue& worker):Batch(net,b),queue(worker) {}
    void start() {
        if(busy) throw std::logic_error("inference slot is busy");
        options.UnsetTerminate();
        pending=queue.submit(std::packaged_task<void()>([this]{run();}));
        busy=true;
    }
    void wait() {if(busy) {busy=false;pending.get();}}
    bool finishBy(Clock::time_point end) {
        if(pending.wait_until(end)==std::future_status::ready){wait();return true;}
        options.SetTerminate();
        try { wait(); }
        catch(const Ort::Exception& e) {
            std::string msg=e.what();
            if(msg.find("terminate")==std::string::npos && msg.find("Terminate")==std::string::npos) throw;
        }
        return false;
    }
    ~Slot(){if(pending.valid()) pending.wait();}
};
struct Device {
    std::string name=std::string("ONNX Runtime ")+OrtGetApiBase()->GetVersionString()+" CPUExecutionProvider";
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"omega-cloud"};
    InferenceQueue worker;
    std::array<std::unique_ptr<Network>,2> nets;
    Network& network(const fs::path& home,int rule) {
        if(rule!=1 && rule!=2) throw std::invalid_argument("supported rules are Standard and Renju");
        auto& net=nets[rule-1];
        if(!net) {
            // ORT intra-op workers inherit this affinity at creation. The MCTS controller
            // and the one-thread Rapfi child each have their own remaining vCPU.
            pin(layout().inference);
            try { net=std::make_unique<Network>(env,home/(rule==1?"standard15x.onnx":"renju15x.onnx"),2); }
            catch(...) {pin(layout().controller);throw;}
            pin(layout().controller);
        }
        return *net;
    }
};
struct Kernel {
    std::vector<std::unique_ptr<Slot>> slots;
    Kernel(Device& device,Network& net,int size,int count) {
        for(int i=0;i<count;++i) {
            auto slot=std::make_unique<Slot>(net,size,device.worker);
            slot->start();slot->wait();slot->get(0);
            slots.push_back(std::move(slot));
        }
    }
};
} // namespace omega
