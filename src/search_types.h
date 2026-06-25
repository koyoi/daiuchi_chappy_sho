#pragma once

#include "shogi_types.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace shogi {

struct SearchLimits {
    int moveTimeMs = -1;
    int remainingMs = -1;
    int incrementMs = 0;
    int byoyomiMs = 0;
};

struct SearchInfo {
    int depth = 0;
    int scoreCp = 0;
    std::uint64_t nodes = 0;
    int timeMs = 0;
    Move bestMove{};
    bool hasBestMove = false;
    std::vector<Move> pv;
    bool isMate = false;
    int mateInMoves = 0;
};

using InfoCallback = std::function<void(const SearchInfo&)>;

} // namespace shogi
