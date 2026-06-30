#pragma once

#include "shogi_types.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace shogi {

struct SearchLimits {
    int moveTimeMs = -1;
    int remainingMs = -1;
    int incrementMs = 0;
    int byoyomiMs = 0;
    bool infinite = false;
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
    int multipv = 0;
};

struct RootMoveScore {
    Move move{};
    int score = 0;
};

inline int mctsValueToCp(double v) {
    double wp = (v + 1.0) / 2.0;
    wp = std::max(0.001, std::min(0.999, wp));
    return static_cast<int>(std::log(wp / (1.0 - wp)) * 300.0);
}

using InfoCallback = std::function<void(const SearchInfo&)>;

} // namespace shogi
