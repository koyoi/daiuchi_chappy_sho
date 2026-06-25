#pragma once

#include "mate_solver.h"
#include "nnue.h"
#include "opening_book.h"
#include "search_types.h"
#include "shogi_types.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace shogi {

class NNUEEngine {
public:
    NNUEEngine();

    Move chooseMove(const Board& board);
    Move chooseMove(const Board& board, const SearchLimits& limits);
    Move chooseMove(const Board& board, const SearchLimits& limits, const InfoCallback& infoCallback);

    void setSearchDepth(int depth);
    void setMaxMoveTimeMs(int milliseconds);
    void setThreads(int threads);
    int threadCount() const;
    void setBookEnabled(bool enabled);
    void setWarnOnNoWeights(bool enabled) { warnOnNoWeights_ = enabled; }
    void setReuseCache(bool enabled) { reuseCache_ = enabled; }
    bool loadBook(const std::string& path = "book.txt");
    bool loadNNUE(const std::string& path);
    const std::string& nnuePath() const { return nnuePath_; }
    SearchInfo lastSearchInfo() const;
    MateResult searchMate(const Board& board, int timeLimitMs = 500);

    void setParam(const std::string& name, int value);
    int getParam(const std::string& name) const;
    void setRootPruneWidth(int w) { rootPruneWidth_ = w; }

private:
    int search(Board& board, int depth, int ply, int alpha, int beta, Color rootSide, bool allowNullMove = true, const Move& prevMove = Move{}) const;
    int quiescence(Board& board, int depth, int ply, int alpha, int beta, Color rootSide) const;
    std::vector<Move> extractPV(Board board, Color rootSide, int maxDepth) const;
    void orderMoves(const Board& board, MoveList& moves, Color rootSide, int ply, const Move& prevMove = Move{}) const;
    int moveOrderScore(const Board& board, const Move& move, Color rootSide, int ply, const Move& ttMove, const Move& prevMove = Move{}) const;
    bool shouldStop() const;
    std::uint64_t boardHash(const Board& board, Color rootSide) const;
    void setLastSearchInfo(const SearchInfo& info) const;
    int depthLimit() const;
    void clearSearchTables() const;
    void storeKiller(int ply, const Move& move) const;
    void updateHistory(Color side, const Move& move, int depth, bool good) const;
    void storeCounterMove(Color side, const Move& prevMove, const Move& counterMove) const;
    void workerSearch(const Board& board, const MoveList& legal, Color rootSide, int threadId) const;

    int eval(const Board& board, Color rootSide, int ply) const;

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
    static constexpr int MaxPly = 128;
    static constexpr int KillerSlots = 2;
    int lmrFullDepthMoves_ = 4;
    int lmrMinDepth_ = 3;
    int nmpMinDepth_ = 3;
    int nmpReduction_ = 3;
    int futilityMargin1_ = 400;
    int futilityMargin2_ = 900;
    int aspirationWindow_ = 50;
    int iidMinDepth_ = 5;
    int deltaMargin_ = 1400;
    int qDepth_ = 6;
    int qCheckDepthMin_ = 4;

    NNUENetwork nnue_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    bool warnOnNoWeights_ = true;
    std::string nnuePath_ = "nnue.bin";
    mutable std::vector<TranspositionEntry> transposition_;
    mutable std::array<std::mutex, LockCount> transpositionMutex_;
    mutable std::uint8_t ttGeneration_{0};
    bool reuseCache_ = true;
    bool isRecentGeneration(std::uint8_t gen) const {
        return gen == ttGeneration_ || (reuseCache_ && gen == static_cast<std::uint8_t>(ttGeneration_ - 1));
    }
    mutable std::chrono::steady_clock::time_point deadline_{};
    mutable std::atomic_bool stopped_{false};
    mutable std::atomic_uint64_t nodes_{0};
    mutable SearchInfo lastSearchInfo_{};
    mutable std::mutex lastSearchInfoMutex_;
    mutable std::array<std::array<Move, KillerSlots>, MaxPly> killers_{};
    mutable std::array<std::array<std::array<std::int16_t, BoardSize>, BoardSize>, 2> history_{};
    mutable std::array<std::array<std::array<Move, BoardSize>, BoardSize>, 2> counterMoves_{};
    int maxMoveTimeMs_ = 1000;
    int threads_ = 1;
    std::mt19937 rng_;
    int searchDepth_ = 0;
    int rootPruneWidth_ = 15;
};

} // namespace shogi
