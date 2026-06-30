#include "mcts_engine.h"

#include "mate_solver.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace shogi {

MCTSEngineWrapper::MCTSEngineWrapper() : mcts_(nn_) {
    zobrist::init();
    nn_.setEnabled(true);
}

Move MCTSEngineWrapper::chooseMove(const Board& board) {
    return chooseMove(board, SearchLimits{maxMoveTimeMs_});
}

Move MCTSEngineWrapper::chooseMove(const Board& board, const SearchLimits& limits) {
    return chooseMove(board, limits, InfoCallback{});
}

Move MCTSEngineWrapper::chooseMove(const Board& board, const SearchLimits& limits,
                                    const InfoCallback& infoCallback) {
    const auto searchStart = std::chrono::steady_clock::now();
    const int moveTime = limits.moveTimeMs > 0 ? limits.moveTimeMs : maxMoveTimeMs_;
    const int clampedTime = std::clamp(moveTime, 50, 600000);

    if (warnOnNoModel_ && !nn_.isReady()) {
        if (!fileExists(nn_.modelPath()))
            std::cout << "info string WARNING: " << nn_.modelPath() << " not found -- search quality degraded" << std::endl;
        else
            std::cout << "info string ERROR: " << nn_.modelPath() << " load failed -- search quality degraded" << std::endl;
    }

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info{};
        {
            std::lock_guard<std::mutex> lock(infoMutex_);
            lastSearchInfo_ = info;
        }
        return Move{};
    }

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

    int effectiveSimulations = simulations_;
    {
        const Color opponent = (board.side == Black) ? White : Black;
        MateResult tsumero = mateSolver_.detectTsumero(board, opponent, 7);
        if (tsumero.found) {
            std::cout << "info string tsumero detected -- extending search" << std::endl;
            effectiveSimulations = simulations_ * 3 / 2;
        }
    }

    MCTSConfig config;
    config.simulations = effectiveSimulations;
    auto result = mcts_.search(board, config, clampedTime, infoCallback);

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

void MCTSEngineWrapper::recordMove(const Board&, const Move&, bool) {}

void MCTSEngineWrapper::finishGame(int, Color) {}

void MCTSEngineWrapper::clearGame() { mcts_.clearTree(); }

void MCTSEngineWrapper::setMaxMoveTimeMs(int ms) {
    maxMoveTimeMs_ = std::max(50, ms);
}

void MCTSEngineWrapper::setSimulations(int n) {
    simulations_ = std::max(1, n);
    mcts_.setSimulations(simulations_);
}

void MCTSEngineWrapper::setNNModel(const std::string& model) { nn_.setModel(model); }
void MCTSEngineWrapper::setNNDevice(const std::string& device) { nn_.setDevice(device); }
void MCTSEngineWrapper::setBatchSize(int n) { mcts_.setBatchSize(n); }
bool MCTSEngineWrapper::loadBook(const std::string& path) { return book_.load(path); }
bool MCTSEngineWrapper::ensureNN() { return nn_.ensureReady(); }
bool MCTSEngineWrapper::nnReady() const { return nn_.isReady(); }
const std::string& MCTSEngineWrapper::nnLastError() const { return nn_.lastError(); }
const std::string& MCTSEngineWrapper::nnModelPath() const { return nn_.modelPath(); }
std::string MCTSEngineWrapper::nnDeviceUsed() const { return nn_.deviceUsed(); }
std::string MCTSEngineWrapper::nnCudaError() const { return nn_.cudaError(); }

MateResult MCTSEngineWrapper::searchMate(const Board& board, int timeLimitMs) {
    return mateSolver_.searchMate(board, board.side, 31, timeLimitMs);
}

SearchInfo MCTSEngineWrapper::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(infoMutex_);
    return lastSearchInfo_;
}

} // namespace shogi
