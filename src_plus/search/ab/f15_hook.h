// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <optional>
#include "f15/certificate.hpp"
#include "searcher.h"
#include "searchstack.h"

namespace Search::AB::FoulTrap {

inline bool hasGain(const Board &board)
{
    return board.p4Count(WHITE, E_BLOCK4) || board.p4Count(WHITE, D_BLOCK4_PLUS)
           || board.p4Count(WHITE, C_BLOCK4_FLEX3) || board.p4Count(WHITE, B_FLEX4)
           || board.p4Count(WHITE, A_FIVE);
}

// Certify the played prefix once. A Renju foul or a five terminates a real
// game, even if a diagram submitted through the analysis protocol continues.
inline void prepare(SearchThread &th, bool handicap)
{
    const Board &board = *th.board;
    const auto &options = th.options();
    auto &data = *th.searchDataAs<ABSearchData>();
    data.f15Ready = false;
    if (options.rule != RENJU || board.size() != f15::Width || board.passMoveCount()
        || board.sideToMove() != Color(board.ply() % 2) || options.balanceMode
        || options.swapable || options.multiPV != 1 || !options.blockMoves.empty()
        || options.drawResult != SearchOptions::RES_DRAW || th.dbClient
        || th.engine.ctx.inPonder || handicap)
        return;

    f15::Board position;
    for (int i = 0; i < board.ply(); ++i) {
        const Pos move = board.getHistoryMove(i);
        if (!board.isInBoard(move) || board.get(move) != Color(i % 2))
            return;
        const int square = move.y() * f15::Width + move.x();
        if (position.occupied().has(square))
            return;
        if (i % 2 == BLACK && f15::blackMove(position, square) != f15::Legal)
            return;
        position = position.with(square, i % 2);
        if (i % 2 == WHITE)
            for (const auto &window : f15::geometry.through[square][1])
                if (position.white.contains(window.body))
                    return;
    }
    data.f15Baseline = {position.black.w, position.white.w};
    data.f15Ready = true;
}

struct Win {
    Value lowerBound;
    Pos gain;
};

inline bool cancelled(void *context)
{
    auto &thread = *static_cast<SearchThread *>(context);
    if (thread.isMainThread())
        thread.engine.ctx.checkExit();
    return thread.engine.isTerminating();
}

enum Site { RootProbe, MainProbe, VcfProbe };

inline std::optional<Win> probe(Board &board, SearchStack *ss, Site site)
{
    const int ply = ss ? ss->ply : 0;
    auto &thread = *board.thisThread();
    const auto &root = *thread.engine.main()->searchDataAs<ABSearchData>();
    if (board.sideToMove() != WHITE || !root.f15Ready || board.passMoveCount()
        || (ss && (ss->skipMove || ss->numNullMoves)) || !hasGain(board))
        return {};
    // Root is unrestricted. Internal probes spend scarce time on short
    // lines with a potential forbidden reply. A missed probe changes no bound.
    if (site != RootProbe && (ply > (site == MainProbe ? 8 : 6) || !board.p4Count(BLACK, FORBID)))
        return {};

    f15::Board position {f15::Mask{root.f15Baseline[BLACK]}, f15::Mask{root.f15Baseline[WHITE]}};
    for (int i = board.ply() - ply; i < board.ply(); ++i) {
        const Pos move = board.getHistoryMove(i);
        if (!board.isInBoard(move) || board.get(move) != Color(i % 2))
            return {};
        const int square = move.y() * f15::Width + move.x();
        position = position.with(square, i % 2);
        for (const auto &window : f15::geometry.through[square][1])
            if (position.stones(i % 2).contains(window.body))
                return {};
    }

    auto &counts = thread.searchDataAs<ABSearchData>()->f23Counts;
    ++counts[site];
    const uint64_t budget = site == RootProbe ? 1024 : (site == MainProbe ? 12 : 6);
    auto proof = f15::Vcf(true, {budget, cancelled, &thread}).solve(position, WHITE);
    if (proof.exhausted) ++counts[6];
    if (!proof.win || proof.exhausted || !f15::replayCertificate(position, WHITE, proof.line))
        return {};
    // A defender may legally ignore an unplayable block or a fork and
    // concede on White's next placement. The line bounds the longest defence.
    const int distance = 2 * int(proof.line.size()) - 1 + (proof.line.back().kind == f15::Five ? 0 : 2);
    if (distance > board.movesLeft() || board.nonPassMoveCount() + distance > thread.options().maxMoves
        || ply + distance >= MAX_PLY)
        return {};
    ++counts[3 + site];
    const int gain = proof.line.front().gain;
    return Win{mate_in(ply + distance), Pos(gain % f15::Width, gain / f15::Width)};
}

}  // namespace Search::AB::FoulTrap
