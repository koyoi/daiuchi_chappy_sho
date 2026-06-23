#include "mcts_engine.h"

#include "mate_solver.h"
#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <chrono>

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

    auto legal = generateLegalMoves(board, true);
    if (legal.empty()) {
        SearchInfo info{};
        {
            std::lock_guard<std::mutex> lock(infoMutex_);
            lastSearchInfo_ = info;
        }
        return Move{};
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

    MCTSConfig config;
    config.simulations = simulations_;
    auto result = mcts_.search(board, config, clampedTime, infoCallback);

    SearchInfo info;
    info.depth = result.simulations;
    info.scoreCp = static_cast<int>(result.winRate * 1000);
    info.nodes = result.simulations;
    info.timeMs = result.timeMs;
    info.bestMove = result.bestMove;
    info.hasBestMove = result.bestMove.to >= 0;
    info.pv = result.pv;
    {
        std::lock_guard<std::mutex> lock(infoMutex_);
        lastSearchInfo_ = info;
    }
    return result.bestMove;
}

void MCTSEngineWrapper::recordMove(const Board&, const Move&, bool) {}

void MCTSEngineWrapper::finishGame(int, Color) {}

void MCTSEngineWrapper::clearGame() {}

void MCTSEngineWrapper::setMaxMoveTimeMs(int ms) {
    maxMoveTimeMs_ = std::max(50, ms);
}

void MCTSEngineWrapper::setSimulations(int n) {
    simulations_ = std::max(1, n);
    mcts_.setSimulations(simulations_);
}

void MCTSEngineWrapper::setNNPython(const std::string& python) { nn_.setPython(python); }
void MCTSEngineWrapper::setNNScript(const std::string& script) { nn_.setScript(script); }
void MCTSEngineWrapper::setNNModel(const std::string& model) { nn_.setModel(model); }
void MCTSEngineWrapper::setNNDevice(const std::string& device) { nn_.setDevice(device); }
void MCTSEngineWrapper::setBatchSize(int n) { mcts_.setBatchSize(n); }
bool MCTSEngineWrapper::ensureNN() { return nn_.ensureProcess(); }
const std::string& MCTSEngineWrapper::nnLastError() const { return nn_.lastError(); }
const std::string& MCTSEngineWrapper::nnModelPath() const { return nn_.modelPath(); }

MateResult MCTSEngineWrapper::searchMate(const Board& board, int timeLimitMs) {
    return mateSolver_.searchMate(board, board.side, 31, timeLimitMs);
}

SearchInfo MCTSEngineWrapper::lastSearchInfo() const {
    std::lock_guard<std::mutex> lock(infoMutex_);
    return lastSearchInfo_;
}

} // namespace shogi
