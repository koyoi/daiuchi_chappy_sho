#pragma once

#include "shogi_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

namespace shogi {

struct MateResult {
    bool found = false;
    int moves = 0;
    Move bestMove{};
    std::vector<Move> pv;
    std::uint64_t nodes = 0;
};

class MateSolver {
public:
    MateSolver();

    MateResult searchMate(const Board& board, Color attacker, int maxDepth, int timeLimitMs = 0);
    MateResult detectTsumero(const Board& board, Color attacker, int maxDepth);
    void clearTable();

private:
    bool attackerSearch(Board& board, int depth, Color attacker, std::vector<Move>& pv);
    bool defenderSearch(Board& board, int depth, Color attacker, std::vector<Move>& pv);
    bool shouldStop() const;

    void orderAttackMoves(const Board& board, MoveList& moves, const Move& ttMove) const;
    void orderDefenseMoves(const Board& board, MoveList& moves) const;

    struct MateEntry {
        std::uint64_t key = 0;
        std::int16_t depth = -1;
        std::uint8_t result = 0;
        std::uint8_t generation = 0;
        Move bestMove{};
    };

    static constexpr std::uint8_t UNKNOWN = 0;
    static constexpr std::uint8_t PROVEN = 1;
    static constexpr std::uint8_t DISPROVEN = 2;

    static constexpr int TTBits = 18;
    static constexpr int TTSize = 1 << TTBits;
    static constexpr int TTMask = TTSize - 1;
    static constexpr int LockCount = 32;

    MateEntry probe(std::uint64_t key) const;
    void store(std::uint64_t key, int depth, std::uint8_t result, const Move& bestMove = Move{});

    std::vector<MateEntry> table_;
    mutable std::array<std::mutex, LockCount> mutex_;
    std::uint8_t generation_ = 0;
    std::atomic_bool stopped_{false};
    std::chrono::steady_clock::time_point deadline_{};
    std::atomic_uint64_t nodes_{0};
};

} // namespace shogi
