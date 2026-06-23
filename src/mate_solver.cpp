#include "mate_solver.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>

namespace shogi {

namespace {

int matePieceValue(PieceType type) {
    switch (type) {
    case Pawn: return 100;
    case Lance: return 300;
    case Knight: return 320;
    case Silver: return 520;
    case Gold: return 620;
    case Bishop: return 850;
    case Rook: return 1050;
    case ProPawn: case ProLance: case ProKnight: case ProSilver: return 560;
    case Horse: return 1150;
    case Dragon: return 1350;
    default: return 0;
    }
}

} // namespace

MateSolver::MateSolver()
    : table_(TTSize) {}

void MateSolver::clearTable() {
    for (auto& e : table_) {
        e = MateEntry{};
    }
    generation_ = 0;
}

MateSolver::MateEntry MateSolver::probe(std::uint64_t key) const {
    const int idx = static_cast<int>(key & TTMask);
    std::lock_guard<std::mutex> lock(mutex_[idx % LockCount]);
    return table_[idx];
}

void MateSolver::store(std::uint64_t key, int depth, std::uint8_t result, const Move& bestMove) {
    const int idx = static_cast<int>(key & TTMask);
    std::lock_guard<std::mutex> lock(mutex_[idx % LockCount]);
    auto& e = table_[idx];
    if (e.key != key || e.generation != generation_ || depth >= e.depth) {
        e.key = key;
        e.depth = static_cast<std::int16_t>(depth);
        e.result = result;
        e.generation = generation_;
        if (bestMove.to >= 0) {
            e.bestMove = bestMove;
        }
    }
}

bool MateSolver::shouldStop() const {
    return stopped_.load(std::memory_order_relaxed)
        || std::chrono::steady_clock::now() >= deadline_;
}

void MateSolver::orderAttackMoves(const Board& board, MoveList& moves, const Move& ttMove) const {
    const int enemyKing = board.side == Black ? board.whiteKingSquare : board.blackKingSquare;

    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        auto score = [&](const Move& m) -> int {
            int s = 0;
            if (ttMove.to >= 0 && sameMove(m, ttMove)) return 100000;
            if (!m.isDrop() && board.squares[m.to] != 0) s += 10000 + matePieceValue(typeOf(board.squares[m.to]));
            if (m.isDrop() && enemyKing >= 0) {
                int dist = std::abs(fileOf(m.to) - fileOf(enemyKing)) + std::abs(rankOf(m.to) - rankOf(enemyKing));
                s += 5000 - dist * 100;
            }
            if (m.promote) s += 3000;
            return s;
        };
        return score(a) > score(b);
    });
}

void MateSolver::orderDefenseMoves(const Board& board, MoveList& moves) const {
    const int myKing = board.side == Black ? board.blackKingSquare : board.whiteKingSquare;

    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        auto score = [&](const Move& m) -> int {
            int s = 0;
            if (!m.isDrop() && m.from == myKing) {
                s += 10000;
                if (board.squares[m.to] != 0) s += 5000;
            }
            if (!m.isDrop() && board.squares[m.to] != 0) s += 3000 + matePieceValue(typeOf(board.squares[m.to]));
            if (m.isDrop()) s += 1000;
            return s;
        };
        return score(a) > score(b);
    });
}

MateResult MateSolver::searchMate(const Board& board, Color attacker, int maxDepth, int timeLimitMs) {
    stopped_.store(false);
    nodes_.store(0);
    ++generation_;

    if (timeLimitMs > 0) {
        deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeLimitMs);
    } else {
        deadline_ = std::chrono::steady_clock::time_point::max();
    }

    MateResult result;

    if (board.side != attacker) {
        return result;
    }

    for (int depth = 1; depth <= maxDepth; depth += 2) {
        std::vector<Move> pv;
        Board work = board;
        if (attackerSearch(work, depth, attacker, pv)) {
            result.found = true;
            result.moves = (depth + 1) / 2;
            result.bestMove = pv.empty() ? Move{} : pv.front();
            result.pv = std::move(pv);
            result.nodes = nodes_.load();
            return result;
        }
        if (shouldStop()) break;
    }

    result.nodes = nodes_.load();
    return result;
}

bool MateSolver::attackerSearch(Board& board, int depth, Color attacker, std::vector<Move>& pv) {
    nodes_.fetch_add(1, std::memory_order_relaxed);
    if (shouldStop()) return false;

    const std::uint64_t key = board.hash;
    const MateEntry entry = probe(key);
    Move ttMove{};
    if (entry.key == key && entry.generation == generation_) {
        if (entry.depth >= depth) {
            if (entry.result == PROVEN) {
                if (entry.bestMove.to >= 0) pv.push_back(entry.bestMove);
                return true;
            }
            if (entry.result == DISPROVEN) return false;
        }
        ttMove = entry.bestMove;
    }

    auto legal = generateLegalMoves(board, true);

    MoveList checks;
    for (const Move& m : legal) {
        if (givesCheck(board, m)) {
            checks.push_back(m);
        }
    }

    if (checks.empty()) {
        store(key, depth, DISPROVEN);
        return false;
    }

    orderAttackMoves(board, checks, ttMove);

    for (const Move& move : checks) {
        UndoInfo undo;
        applyMove(board, move, undo);
        std::vector<Move> childPV;
        const bool mate = defenderSearch(board, depth - 1, attacker, childPV);
        undoMove(board, move, undo);
        if (mate) {
            pv.clear();
            pv.push_back(move);
            pv.insert(pv.end(), childPV.begin(), childPV.end());
            store(key, depth, PROVEN, move);
            return true;
        }
    }

    store(key, depth, DISPROVEN);
    return false;
}

bool MateSolver::defenderSearch(Board& board, int depth, Color attacker, std::vector<Move>& pv) {
    nodes_.fetch_add(1, std::memory_order_relaxed);
    if (shouldStop()) return false;

    auto evasions = generateLegalMoves(board, true);

    if (evasions.empty()) {
        return isKingAttacked(board, board.side);
    }

    if (depth <= 0) return false;

    const std::uint64_t key = board.hash;
    const MateEntry entry = probe(key);
    if (entry.key == key && entry.generation == generation_ && entry.depth >= depth) {
        if (entry.result == PROVEN) return true;
        if (entry.result == DISPROVEN) return false;
    }

    orderDefenseMoves(board, evasions);

    std::vector<Move> longestPV;

    for (const Move& move : evasions) {
        UndoInfo undo;
        applyMove(board, move, undo);
        std::vector<Move> childPV;
        const bool mate = attackerSearch(board, depth - 1, attacker, childPV);
        undoMove(board, move, undo);
        if (!mate) {
            store(key, depth, DISPROVEN);
            return false;
        }
        if (childPV.size() + 1 > longestPV.size()) {
            longestPV.clear();
            longestPV.push_back(move);
            longestPV.insert(longestPV.end(), childPV.begin(), childPV.end());
        }
    }

    pv = std::move(longestPV);
    store(key, depth, PROVEN);
    return true;
}

MateResult MateSolver::detectTsumero(const Board& board, Color attacker, int maxDepth) {
    if (board.side == attacker) {
        return MateResult{};
    }

    Board nullBoard = board;
    nullBoard.side = attacker;
    nullBoard.hash ^= zobrist::sideKey();

    return searchMate(nullBoard, attacker, maxDepth, 100);
}

} // namespace shogi
