#include "rootprior.h"

#include "../../core/iohelper.h"
#include "../../eval/evaluator.h"
#include "../../eval/scoretables.h"
#include "../../game/board.h"
#include "../searchthread.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#ifdef __linux__
#include <cerrno>
#include <sys/socket.h>
#endif

namespace Search::AB {
namespace {

using Policy = Evaluation::PolicyBuffer;

// Replay through the normal evaluator hooks. The scratch board has no attached evaluator.
void replayOrientation(const Board &source, Board &board, Evaluation::Evaluator &ev,
                       TransformType transform)
{
    board.newGame(ev.rule);
    ev.initEmptyBoard();
    for (int i = 0; i < source.ply(); ++i) {
        const Pos move = source.getHistoryMove(i);
        if (move == Pos::PASS) {
            board.move(ev.rule, move);
            ev.afterPass(board);
        } else {
            const Pos p = applyTransform(move, source.size(), transform);
            ev.beforeMove(board, p);
            board.move(ev.rule, p);
            ev.afterMove(board, p);
        }
    }
}

void evaluateD4(SearchThread &thread, RootPrior &prior)
{
    const Board &root = *thread.board;
    auto &ev = *thread.evaluator;
    Board rotated(root.size());
    std::array<int, FULL_BOARD_CELL_COUNT> sums {};
    float winRate = 0;
    // Finish in the original orientation so the ordinary recursive search inherits
    // a synchronized accumulator, including the side that did not just evaluate.
    for (int i = 1; i <= TRANS_NB; ++i) {
        const auto transform = TransformType(i % TRANS_NB);
        replayOrientation(root, rotated, ev, transform);
        Policy policy(root.size());
        for (const auto &move : thread.rootMoves)
            policy.setComputeFlag(applyTransform(move.pv[0], root.size(), transform));
        ev.evaluatePolicy(rotated, policy);
        winRate += ev.evaluateValue(rotated).winningRate();
        for (const auto &move : thread.rootMoves) {
            const Pos p = move.pv[0];
            sums[p] += policy.score(applyTransform(p, root.size(), transform));
        }
    }
    for (const auto &move : thread.rootMoves)
        prior.policy[move.pv[0]] = Score(std::lround(double(sums[move.pv[0]]) / TRANS_NB));
    prior.value = Evaluation::winRateToValue(winRate / TRANS_NB);
}

#ifdef __linux__
void transfer(int fd, void *buffer, size_t length, bool sending)
{
    auto data = static_cast<unsigned char *>(buffer);
    while (length) {
        ssize_t n = sending ? send(fd, data, length, MSG_NOSIGNAL) : recv(fd, data, length, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("KataGomo CPU oracle disconnected");
        data += n;
        length -= size_t(n);
    }
}

void evaluateKata(SearchThread &thread, RootPrior &prior)
{
    const char *socketText = std::getenv("RAPFI_ORACLE_FD");
    if (!socketText)
        throw std::runtime_error("KataGomo CPU oracle socket is missing");
    const int fd = std::stoi(socketText);
    const Board &board = *thread.board;
    const Color side = board.sideToMove();
    const Pos last = board.getLastMove();
    std::array<uint8_t, 228> request {};
    request[0] = uint8_t(thread.options().rule.rule);
    request[1] = uint8_t(side);
    request[2] = board.isInBoard(last) ? uint8_t(last.y() * 15 + last.x()) : 255;
    for (int i = 0; i < 225; ++i) {
        Color piece = board.get(Pos(i % 15, i / 15));
        request[3 + i] = piece == side ? 1 : piece == ~side ? 2 : 0;
    }
    std::array<float, 226> response;
    transfer(fd, request.data(), request.size(), true);
    transfer(fd, response.data(), sizeof(response), false);
    const float globalWinRate = response.back();
    if (!std::isfinite(globalWinRate) || globalWinRate < 0 || globalWinRate > 1)
        throw std::runtime_error("invalid KataGomo WDL");
    auto &ev = *thread.evaluator;
    Policy local(15);
    for (const auto &move : thread.rootMoves)
        local.setComputeFlag(move.pv[0]);
    ev.evaluatePolicy(board, local);
    float bestGlobal = -std::numeric_limits<float>::infinity();
    Score bestLocal = std::numeric_limits<Score>::min();
    for (const auto &move : thread.rootMoves) {
        const Pos p = move.pv[0];
        const float logit = response[p.y() * 15 + p.x()];
        if (!std::isfinite(logit))
            throw std::runtime_error("invalid KataGomo policy");
        bestGlobal = std::max(bestGlobal, logit);
        bestLocal = std::max(bestLocal, local.score(p));
    }
    for (const auto &move : thread.rootMoves) {
        const Pos p = move.pv[0];
        float global = bestLocal + Policy::ScoreScale * (response[p.y() * 15 + p.x()] - bestGlobal);
        float score = .4f * local.score(p) + .6f * global;
        prior.policy[p] = Score(std::lround(std::clamp(score, -32000.f, 32000.f)));
    }
    prior.value = Evaluation::winRateToValue(.5f * (ev.evaluateValue(board).winningRate() + globalWinRate));
}
#endif

}  // namespace

void prepareRootPrior(SearchThread &thread, RootPrior &prior, std::string_view experiment)
{
    prior.value = VALUE_NONE;
    if (experiment != "d4" && experiment != "kata")
        return;
    const auto &board = *thread.board;
    const auto &opts = thread.options();
    if (!thread.evaluator || !opts.timeLimit || opts.balanceMode || board.size() != 15
        || board.ply() < 10 || board.ply() > 80 || opts.timeLeft < 400 || thread.rootMoves.size() < 4)
        return;
    for (Color side : {BLACK, WHITE})
        if (board.p4Count(side, A_FIVE) || board.p4Count(side, B_FLEX4)
            || board.p4Count(side, C_BLOCK4_FLEX3))
            return;
    auto start = std::chrono::steady_clock::now();
    prior.policy.fill(std::numeric_limits<Score>::min());
    if (experiment == "d4")
        evaluateD4(thread, prior);
    else {
#ifdef __linux__
        evaluateKata(thread, prior);
#else
        throw std::runtime_error("KataGomo CPU oracle requires a Linux socket");
#endif
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    MESSAGEL("ROOT_PRIOR " << experiment << ' ' << elapsed);
}

}  // namespace Search::AB
