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
    static constexpr std::uint32_t INF_PN = 1000000000u;

    MateSolver();

    MateResult searchMate(const Board& board, Color attacker, int maxDepth, int timeLimitMs = 0);
    MateResult detectTsumero(const Board& board, Color attacker, int maxDepth);
    void clearTable();

private:

    struct DfpnEntry {
        std::uint64_t key = 0;
        std::uint32_t pn = 1;
        std::uint32_t dn = 1;
        Move bestMove{};
        std::uint8_t generation = 0;
    };

    static constexpr int TTBits = 20;
    static constexpr int TTSize = 1 << TTBits;
    static constexpr int TTMask = TTSize - 1;
    static constexpr int LockCount = 64;

    DfpnEntry probe(std::uint64_t key) const;
    void store(std::uint64_t key, std::uint32_t pn, std::uint32_t dn, const Move& bestMove = Move{});

    void dfpnSearch(Board& board, Color attacker, std::uint32_t thpn, std::uint32_t thdn,
                    std::vector<std::uint64_t>& path, bool attackerNode);

    void orderAttackMoves(const Board& board, MoveList& moves, const Move& ttMove) const;
    void orderDefenseMoves(const Board& board, MoveList& moves) const;

    std::vector<Move> restorePV(Board& board, Color attacker) const;

    bool shouldStop() const;

    std::vector<DfpnEntry> table_;
    mutable std::array<std::mutex, LockCount> mutex_;
    std::uint8_t generation_ = 0;
    std::atomic_bool stopped_{false};
    std::chrono::steady_clock::time_point deadline_{};
    std::atomic_uint64_t nodes_{0};
    std::uint64_t nodeLimit_ = 0;
};

} // namespace shogi
