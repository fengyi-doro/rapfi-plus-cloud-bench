// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace omega {
struct Advice {
    int move=-1, depth=0, score=0, mate=0;
    std::array<int,12> pv{};
    int pvLength=0;
    bool hasScore=false, complete=false;
};
class Shield {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    Shield(const std::filesystem::path& home,int core);
    ~Shield();
    void reset(int rule);
    bool available() const;
    bool deferred() const;
    bool recovering() const;
    bool start(const std::vector<int>& history,int ms,const std::vector<int>& choices={});
    Advice poll();
    const std::vector<Advice>& alternatives() const;
    void requestStop();
    Advice stop(std::chrono::steady_clock::time_point deadline);
    int core() const;
};
}
