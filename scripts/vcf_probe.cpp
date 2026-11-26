// SPDX-License-Identifier: GPL-3.0-or-later
// This test TU exposes the three private alpha-beta entries for direct comparison.
#include "search/ab/search.cpp"
#include "command/command.h"
#include "config.h"
#include <bit>
#include <iostream>

template <Rule R>
void inspect(Board &b)
{
    const auto hash = b.zobristKey();
    const int ply = b.ply();
    const Value initial = Evaluation::evaluate<R>(b);
    Value v[3];
    for (int i = 0; i < 3; ++i) {
        TT.clear();
        StackArray stack(MAX_PLY, initial);
        auto *ss = stack.rootStack();
        if (i == 0)
            v[i] = vcfsearch<R, PV>(b, ss, -VALUE_INFINITE, VALUE_INFINITE);
        else if (i == 1)
            v[i] = vcfdefend<R, PV>(b, ss, -VALUE_INFINITE, VALUE_INFINITE);
        else
            v[i] = search<R, PV>(b, ss, -VALUE_INFINITE, VALUE_INFINITE, 0, false);
        if (b.zobristKey() != hash || b.ply() != ply)
            throw std::logic_error("unbalanced move/undo");
    }
    // Exercise sparse policy inputs and a flushed sibling move on the same
    // accumulator stack. Include a sentinel for every unrequested output cell.
    auto sample = [&]() {
        auto w = b.evaluator()->evaluateValue(b);
        std::array<uint32_t, 3> wdl {std::bit_cast<uint32_t>(w.win()),
                                     std::bit_cast<uint32_t>(w.loss()), std::bit_cast<uint32_t>(w.draw())};
        uint64_t checksum = 14695981039346656037ULL;
        for (auto word : wdl)
            checksum = (checksum ^ word) * 1099511628211ULL;
        Evaluation::PolicyBuffer policy(15);
        for (int p = 0; p < 225; ++p) {
            Pos xy(p % 15, p / 15);
            policy(p) = -9876.25f;
            policy.setComputeFlag(xy, b.isEmpty(xy) && (p * 7 + ply) % 11 < 3);
        }
        b.evaluator()->evaluatePolicy(b, policy);
        for (int p = 0; p < 225; ++p) {
            if (!policy.getComputeFlag(p) && policy(p) != -9876.25f)
                throw std::logic_error("wrote an unrequested policy");
            checksum = (checksum ^ std::bit_cast<uint32_t>(policy(p))) * 1099511628211ULL;
        }
        return checksum;
    };
    uint64_t parent = sample(), siblings = 0;
    int children = 0;
    for (int p = 0; p < 225 && children < 7; ++p) {
        Pos xy(p % 15, p / 15);
        if (!b.isEmpty(xy) || (R == RENJU && b.sideToMove() == BLACK && b.checkForbiddenPoint(xy)))
            continue;
        b.move<R>(xy);
        siblings = siblings * 1099511628211ULL ^ sample();
        b.undo<R>();
        if (sample() != parent)
            throw std::logic_error("accumulator failed to restore after undo");
        ++children;
    }
    std::cout << ' ' << int(v[0]) << ' ' << int(v[1]) << ' ' << int(v[2]) << ' '
              << parent << ' ' << siblings << ' ' << children << '\n';
}

int main(int argc, char **argv) try
{
    Command::CommandLine::init(argc, argv);
    if (!Command::loadConfig())
        throw std::runtime_error("config");
    Config::GeneralCfg.messageMode = MsgMode::NONE;
    Engine.setNumThreads(1);
    Engine.searcher()->setMemoryLimit(8192);
    int id, ruleId, n;
    while (std::cin >> id >> ruleId >> n) {
        Rule rule = Rule(ruleId);
        Board b(15);
        b.newGame(rule);
        for (int j = 0, p; j < n; ++j) {
            std::cin >> p;
            b.move(rule, Pos(p % 15, p / 15));
        }
        SearchOptions opts;
        opts.rule.rule = rule;
        Engine.clear(true);
        Engine.ctx.reset(opts, false);
        auto &th = *Engine.main();
        th.clear();
        th.setBoardAndEvaluator(b);
        if (!th.board->evaluator())
            throw std::runtime_error("missing NNUE");
        std::cout << "CASE " << id;
        if (rule == FREESTYLE) inspect<FREESTYLE>(*th.board);
        else if (rule == STANDARD) inspect<STANDARD>(*th.board);
        else inspect<RENJU>(*th.board);
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
}
