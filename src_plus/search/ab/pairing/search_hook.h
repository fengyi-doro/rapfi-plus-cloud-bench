// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "lazy.h"
#include "../../../game/board.h"
#include "../../searchthread.h"
#include "../../skill.h"
#include "../searchstack.h"
#ifdef RAPFI_F10
#include "f10/search.h"
#endif

namespace PairingSearch {

enum class Cut { None, ExactDraw, DrawLowerBound, DrawUpperBound };

#ifdef RAPFI_F10
using Position = f10::Board;
constexpr int MinRootPly = f10::FirstSearchPly, MaxRootLive = f10::RootLiveCap;

inline Cut classify(const Position &position, Value alpha, Value beta)
{
    const auto bounds = f10::fastBounds(position, alpha < VALUE_DRAW, beta > VALUE_DRAW);
    if (beta <= VALUE_DRAW && bounds.lo >= 0) return Cut::DrawLowerBound;
    if (alpha >= VALUE_DRAW && bounds.hi <= 0) return Cut::DrawUpperBound;
    return bounds.lo == 0 && bounds.hi == 0 ? Cut::ExactDraw : Cut::None;
}
#else
using Position = f8::Position;
constexpr int MinRootPly = f8::MinPly, MaxRootLive = f8::MaxLive;

// This interface is also used by the certificate audit. A lower bound of +1
// may safely fail high at zero. Only a two-sided zero can return an exact draw.
inline Cut classify(const f8::Position &position, Value beta)
{
    const bool failHigh = beta <= VALUE_DRAW;
    if (position.rule == f8::Renju && (!failHigh || position.ply % 2 == 0))
        return Cut::None;
    if (position.rule == f8::Free)
        return Cut::None;
    auto bounds = f8::lazy(position, true, position.rule == f8::Standard && !failHigh);
    if (failHigh && bounds.lo >= 0)
        return Cut::DrawLowerBound;
    if (bounds.lo == 0 && bounds.hi == 0)
        return Cut::ExactDraw;
    return Cut::None;
}
#endif

struct PathCache
{
    int rootPly, cachedDepth;
    std::array<Pos, f8::Cells> path;
    std::array<Position, f8::Cells+1> position;

    bool init(const Board &b, f8::Rule rule)
    {
        rootPly = b.ply();
        cachedDepth = 0;
        auto &root = position[0];
        root = Position(rule);
        for (int i = 0; i < rootPly; ++i) {
            Pos p = b.getHistoryMove(i);
            if (!b.isInBoard(p) || b.get(p) != Color(i % 2))
                return false;
            root.put(p.y()*15+p.x(), i%2);
        }
        return !root.finished && std::min(root.count(BLACK), root.count(WHITE)) <= MaxRootLive;
    }

    const Position *sync(const Board &b, int depth)
    {
        if (depth < 0 || b.ply() - depth != rootPly || b.passMoveCount())
            return nullptr;
        int i = 0;
        while (i < std::min(cachedDepth, depth) && path[i] == b.getHistoryMove(rootPly+i)) ++i;
        for (; i < depth; ++i) {
            Pos p = b.getHistoryMove(rootPly+i);
            Color color = Color((rootPly+i)%2);
            if (!b.isInBoard(p) || b.get(p) != color) {
                cachedDepth = i;
                return nullptr;
            }
            path[i] = p;
            position[i+1] = position[i];
            position[i+1].put(p.y()*15+p.x(), color);
        }
        cachedDepth = depth;
        return &position[depth];
    }
};

inline thread_local PathCache *active = nullptr;

inline void prepare(Search::SearchThread &thread)
{
    active = nullptr;
    const auto &b = *thread.board;
    const auto &o = thread.options();
    if (b.ply() < MinRootPly || b.ply() >= f8::Cells || b.size() != f8::N || b.passMoveCount()
        || (o.rule != STANDARD && o.rule != RENJU) || o.drawResult != Search::SearchOptions::RES_DRAW
        || o.maxMoves < f8::Cells || o.balanceMode || o.swapable || o.multiPV != 1
        || !o.blockMoves.empty() || thread.dbClient || b.sideToMove() != Color(b.ply()%2))
        return;
    static thread_local PathCache cache;
    if (cache.init(b, o.rule == STANDARD ? f8::Standard : f8::Renju)) active = &cache;
}

inline Pos rootDrawMove(Search::SearchThread &thread)
{
#if defined(RAPFI_F10) && !defined(RAPFI_DISABLE_PAIRING)
    const auto &opts = thread.options();
    if (opts.timeLimit && !thread.engine.ctx.inPonder && !Search::SkillMovePicker(opts.strengthLevel).enabled()
        && thread.board->ply() >= MinRootPly) {
        prepare(thread);
        f10::Proof defence;
        if (active && f10::rootDraw(active->position[0], defence)) {
            Pos move = defence.pre < 0 ? thread.rootMoves.front().pv[0]
                                       : Pos(defence.pre % f8::N, defence.pre / f8::N);
            if (std::find(thread.rootMoves.begin(), thread.rootMoves.end(), move) != thread.rootMoves.end())
                return move;
        }
    }
#endif
    return Pos::NONE;
}

template <Rule R>
bool cut(const Board &board, const Search::AB::SearchStack *ss, Value alpha, Value beta, Depth depth)
{
    if (depth < 3 || ss->skipMove || ss->numNullMoves) return false;
#ifdef RAPFI_F10
    const auto *position = active->sync(board, ss->ply);
    return position && classify(*position, alpha, beta) != Cut::None;
#else
    if constexpr (R == RENJU)
        if (board.sideToMove() != WHITE || beta > VALUE_DRAW) return false;
    const auto *position = active->sync(board, ss->ply);
    return position && classify(*position, beta) != Cut::None;
#endif
}

}  // namespace PairingSearch

