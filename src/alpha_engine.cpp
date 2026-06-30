#include "alpha_engine.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace shogi {

AlphaEngineWrapper::AlphaEngineWrapper() : mcts_(nn_) {
    zobrist::init();
}

Move AlphaEngineWrapper::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move AlphaEngineWrapper::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

Move AlphaEngineWrapper::chooseMove(const Board& board, const SearchLimits& limits,
                                     const InfoCallback& infoCallback) {
    const auto searchStart = std::chrono::steady_clock::now();
    const int moveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int clampedTime = std::clamp(moveTime, 50, 600000);

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = SearchInfo{};
        return Move{};
    }

    if (legal.size() == 1) {
        SearchInfo info{};
        info.bestMove = legal[0];
        info.hasBestMove = true;
        info.pv.push_back(legal[0]);
        {
            std::lock_guard<std::mutex> lock(infoMutex_);
            lastSearchInfo_ = info;
        }
        if (infoCallback) infoCallback(info);
        return legal[0];
    }

    // Opening book
    if (bookEnabled_ && !book_.empty() && board.moveNumber <= 8) {
        const std::string bookUsi = book_.selectMove(board.hash, rng_);
        if (!bookUsi.empty()) {
            Move bookMove = parseUsiMove(board, bookUsi);
            bool isLegal = false;
            for (const Move& m : legal) {
                if (sameMove(m, bookMove)) { bookMove = m; isLegal = true; break; }
            }
            if (isLegal) {
                SearchInfo info{};
                info.bestMove = bookMove;
                info.hasBestMove = true;
                info.pv.push_back(bookMove);
                {
                    std::lock_guard<std::mutex> lock(infoMutex_);
                    lastSearchInfo_ = info;
                }
                if (infoCallback) infoCallback(info);
                return bookMove;
            }
        }
    }

    // Mate search (10% of budget, max 200ms)
    {
        const int mateBudget = std::min(clampedTime / 10, 200);
        MateResult mateResult = mateSolver_.searchMate(board, board.side, 31, mateBudget);
        if (mateResult.found) {
            SearchInfo info;
            info.depth = mateResult.moves;
            info.scoreCp = 30000;
            info.nodes = mateResult.nodes;
            info.timeMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - searchStart).count());
            info.bestMove = mateResult.bestMove;
            info.hasBestMove = true;
            info.pv = mateResult.pv;
            info.isMate = true;
            info.mateInMoves = mateResult.moves;
            {
                std::lock_guard<std::mutex> lock(infoMutex_);
                lastSearchInfo_ = info;
            }
            if (infoCallback) infoCallback(info);
            return mateResult.bestMove;
        }
    }

    // Tsumero detection: extend search if opponent has mate threat
    int effectiveSimulations = simulations_;
    {
        const Color opponent = (board.side == Black) ? White : Black;
        MateResult tsumero = mateSolver_.detectTsumero(board, opponent, 7);
        if (tsumero.found) {
            std::cout << "info string tsumero detected -- extending search" << std::endl;
            effectiveSimulations = simulations_ * 3 / 2;
        }
    }

    // MCTS search
    AlphaMCTSConfig config;
    config.simulations = effectiveSimulations;
    config.addNoise = true;
    auto result = mcts_.search(board, config, clampedTime, infoCallback);
    lastMCTSResult_ = result;

    SearchInfo info;
    info.pv = result.pv;
    info.depth = static_cast<int>(result.pv.size());
    info.scoreCp = mctsValueToCp(result.winRate);
    info.nodes = result.simulations;
    info.timeMs = result.timeMs;
    info.bestMove = result.bestMove;
    info.hasBestMove = result.bestMove.to >= 0;
    {
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = info;
    }
    return result.bestMove;
}

void AlphaEngineWrapper::clearGame() { mcts_.clearTree(); }

void AlphaEngineWrapper::setMaxMoveTimeMs(int ms) {
    maxMoveTimeMs_ = std::max(50, ms);
}

void AlphaEngineWrapper::setSimulations(int n) {
    simulations_ = std::max(1, n);
    mcts_.setSimulations(simulations_);
}

void AlphaEngineWrapper::setNNModel(const std::string& model) { nnModel_ = model; }
void AlphaEngineWrapper::setNNDevice(const std::string& device) { nnDevice_ = device; }
void AlphaEngineWrapper::setBatchSize(int n) { mcts_.setBatchSize(n); }
void AlphaEngineWrapper::setFPUReduction(double v) { mcts_.setFPUReduction(v); }
void AlphaEngineWrapper::setTemperatureDropMove(int m) { mcts_.setTemperatureDropMove(m); }
bool AlphaEngineWrapper::loadBook(const std::string& path) { return book_.load(path); }

bool AlphaEngineWrapper::ensureNN() {
    if (nn_.isLoaded()) return true;
    return nn_.loadModel(nnModel_, nnDevice_);
}

MateResult AlphaEngineWrapper::searchMate(const Board& board, int timeLimitMs) {
    return mateSolver_.searchMate(board, board.side, 31, timeLimitMs);
}

SearchInfo AlphaEngineWrapper::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(infoMutex_);
    return lastSearchInfo_;
}

} // namespace shogi
