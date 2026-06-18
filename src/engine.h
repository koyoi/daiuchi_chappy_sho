#pragma once

#include "evaluation.h"
#include "gpu_bridge.h"
#include "learning.h"
#include "shogi_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace shogi {

struct SearchLimits {
    int moveTimeMs = -1;
};

struct SearchInfo {
    int depth = 0;
    int scoreCp = 0;
    std::uint64_t nodes = 0;
    int timeMs = 0;
    Move bestMove{};
    bool hasBestMove = false;
};

using InfoCallback = std::function<void(const SearchInfo&)>;

class LearningEngine {
public:
    LearningEngine();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);
    void recordMove(const Board& before, const Move& move, bool engineMove);
    void finishGame(int engineResult, Color engineSide);
    void clearGame();

    void setLearningEnabled(bool enabled);
    void setSearchDepth(int depth);
    void setMaxMoveTimeMs(int milliseconds);
    void setHeavyEvaluation(bool enabled);
    void setThreads(int threads);
    int threadCount() const;
    void setWeightsPath(const std::string& path);
    void setTrainingDataPath(const std::string& path);
    void setGpuEnabled(bool enabled);
    void setGpuTrainOnGameEnd(bool enabled);
    void setGpuPython(const std::string& python);
    void setGpuScript(const std::string& script);
    void setGpuModel(const std::string& model);
    void setGpuDevice(const std::string& device);
    void loadWeights();
    SearchInfo lastSearchInfo() const;

private:
    int search(Board board, int depth, int alpha, int beta, Color rootSide) const;
    int quiescence(Board board, int depth, int alpha, int beta, Color rootSide) const;
    bool canForceMate(Board board, int depth, Color attacker) const;
    bool isTacticalMove(const Board& board, const Move& move) const;
    bool chooseMoveByGpu(const Board& board, const std::vector<Move>& legal, Move& selected);
    std::vector<int> scoreRootMovesParallel(
        const Board& board,
        const std::vector<Move>& orderedMoves,
        Color rootSide,
        int depth,
        const std::chrono::steady_clock::time_point& searchStart,
        const InfoCallback& infoCallback) const;
    std::vector<Move> orderMoves(const Board& board, const std::vector<Move>& moves, Color rootSide) const;
    int moveOrderScore(const Board& board, const Move& move, Color rootSide) const;
    bool shouldStop() const;
    std::uint64_t boardHash(const Board& board, Color rootSide) const;
    void setLastSearchInfo(const SearchInfo& info) const;
    int depthLimit() const;

    struct TranspositionEntry {
        int depth = -1;
        int score = 0;
        int flag = 0;
    };

    Evaluator evaluator_;
    OnlineLearner learner_;
    GpuBridge gpu_;
    mutable std::unordered_map<std::uint64_t, TranspositionEntry> transposition_;
    mutable std::mutex transpositionMutex_;
    mutable std::chrono::steady_clock::time_point deadline_{};
    mutable std::atomic_bool stopped_{false};
    mutable std::atomic_uint64_t nodes_{0};
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex lastSearchInfoMutex_;
    int maxMoveTimeMs_ = 1000;
    int threads_ = 1;
    std::mt19937 rng_;
    int searchDepth_ = 0;
};

} // namespace shogi
