#pragma once

#include "shogi_types.h"

#include <cstdint>
#include <functional>
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
    std::vector<Move> pv;
};

using InfoCallback = std::function<void(const SearchInfo&)>;

} // namespace shogi
