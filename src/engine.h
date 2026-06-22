#pragma once

#include "evaluation.h"
#include "learning.h"
#include "search_types.h"
#include "shogi_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace shogi {

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
    void setOpeningSafety(bool enabled);
    void setThreads(int threads);
    int threadCount() const;
    void setWeightsPath(const std::string& path);
    void setTrainingDataPath(const std::string& path);
    void loadWeights();
    SearchInfo lastSearchInfo() const;

private:
    int search(Board& board, int depth, int alpha, int beta, Color rootSide) const;
    std::vector<Move> extractPV(Board board, Color rootSide, int maxDepth) const;
    int quiescence(Board& board, int depth, int alpha, int beta, Color rootSide) const;
    bool canForceMate(Board& board, int depth, Color attacker) const;
    bool isTacticalMove(const Board& board, const Move& move) const;
    std::vector<int> scoreRootMovesParallel(
        const Board& board,
        const MoveList& orderedMoves,
        const std::vector<int>& openingPenalties,
        Color rootSide,
        int depth,
        const std::chrono::steady_clock::time_point& searchStart,
        const InfoCallback& infoCallback) const;
    void orderMoves(const Board& board, MoveList& moves, Color rootSide) const;
    int moveOrderScore(const Board& board, const Move& move, Color rootSide) const;
    bool shouldStop() const;
    std::uint64_t boardHash(const Board& board, Color rootSide) const;
    void setLastSearchInfo(const SearchInfo& info) const;
    int depthLimit() const;

    struct TranspositionEntry {
        std::uint64_t key = 0;
        int depth = -1;
        int score = 0;
        std::uint8_t flag = 0;
        std::uint8_t generation = 0;
        Move bestMove{};
    };

    static constexpr int TTBits = 20;
    static constexpr int TTSize = 1 << TTBits;
    static constexpr int TTMask = TTSize - 1;
    static constexpr int LockCount = 64;

    Evaluator evaluator_;
    OnlineLearner learner_;
    mutable std::vector<TranspositionEntry> transposition_;
    mutable std::array<std::mutex, LockCount> transpositionMutex_;
    mutable std::uint8_t ttGeneration_{0};
    mutable std::chrono::steady_clock::time_point deadline_{};
    mutable std::atomic_bool stopped_{false};
    mutable std::atomic_uint64_t nodes_{0};
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex lastSearchInfoMutex_;
    int maxMoveTimeMs_ = 1000;
    int threads_ = 1;
    std::mt19937 rng_;
    int searchDepth_ = 0;
    bool openingSafety_ = true;
};

} // namespace shogi
