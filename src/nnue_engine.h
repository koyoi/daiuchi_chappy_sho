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
    bool loadBook(const std::string& path = "book.txt");
    bool loadNNUE(const std::string& path);
    SearchInfo lastSearchInfo() const;
    MateResult searchMate(const Board& board, int timeLimitMs = 500);

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

    int eval(const Board& board, Color rootSide) const;

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
    static constexpr int LMRFullDepthMoves = 4;
    static constexpr int LMRMinDepth = 3;
    static constexpr int NMPMinDepth = 3;
    static constexpr int NMPReduction = 3;
    static constexpr int FutilityMargin1 = 400;
    static constexpr int FutilityMargin2 = 900;
    static constexpr int AspirationWindow = 50;
    static constexpr int IIDMinDepth = 5;

    NNUENetwork nnue_;
    MateSolver mateSolver_;
    OpeningBook book_;
    bool bookEnabled_ = true;
    mutable std::vector<TranspositionEntry> transposition_;
    mutable std::array<std::mutex, LockCount> transpositionMutex_;
    mutable std::uint8_t ttGeneration_{0};
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
