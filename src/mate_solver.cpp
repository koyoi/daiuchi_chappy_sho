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

std::uint32_t saturatingAdd(std::uint32_t a, std::uint32_t b) {
    std::uint64_t sum = static_cast<std::uint64_t>(a) + b;
    return sum >= MateSolver::INF_PN ? MateSolver::INF_PN : static_cast<std::uint32_t>(sum);
}

} // namespace

MateSolver::MateSolver()
    : table_(TTSize) {}

void MateSolver::clearTable() {
    for (auto& e : table_) {
        e = DfpnEntry{};
    }
    generation_ = 0;
}

MateSolver::DfpnEntry MateSolver::probe(std::uint64_t key) const {
    const int idx = static_cast<int>(key & TTMask);
    std::lock_guard<std::mutex> lock(mutex_[idx % LockCount]);
    const auto& e = table_[idx];
    if (e.key == key && e.generation == generation_) {
        return e;
    }
    return DfpnEntry{};
}

void MateSolver::store(std::uint64_t key, std::uint32_t pn, std::uint32_t dn, const Move& bestMove) {
    const int idx = static_cast<int>(key & TTMask);
    std::lock_guard<std::mutex> lock(mutex_[idx % LockCount]);
    auto& e = table_[idx];
    if (e.key != key || e.generation != generation_
        || pn == 0 || dn == 0
        || (pn >= e.pn && dn >= e.dn)) {
        e.key = key;
        e.pn = pn;
        e.dn = dn;
        e.generation = generation_;
        if (bestMove.to >= 0) {
            e.bestMove = bestMove;
        }
    }
}

bool MateSolver::shouldStop() const {
    if (stopped_.load(std::memory_order_relaxed)) return true;
    if (std::chrono::steady_clock::now() >= deadline_) return true;
    if (nodeLimit_ > 0 && nodes_.load(std::memory_order_relaxed) >= nodeLimit_) return true;
    return false;
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

void MateSolver::dfpnSearch(Board& board, Color attacker, std::uint32_t thpn, std::uint32_t thdn,
                             std::vector<std::uint64_t>& path, bool attackerNode) {
    nodes_.fetch_add(1, std::memory_order_relaxed);
    if (shouldStop()) return;

    const std::uint64_t key = board.hash;

    // Cycle detection
    for (const auto& h : path) {
        if (h == key) {
            store(key, INF_PN, 0);
            return;
        }
    }

    auto legal = generateLegalMoves(board, true);

    if (attackerNode) {
        // OR node: attacker generates only checking moves
        MoveList checks;
        for (const Move& m : legal) {
            if (givesCheck(board, m)) {
                checks.push_back(m);
            }
        }

        if (checks.empty()) {
            store(key, INF_PN, 0);
            return;
        }

        DfpnEntry entry = probe(key);
        orderAttackMoves(board, checks, entry.bestMove);

        struct ChildInfo {
            Move move;
            std::uint32_t pn;
            std::uint32_t dn;
        };
        std::vector<ChildInfo> children;
        children.reserve(checks.size());

        for (const Move& m : checks) {
            UndoInfo undo;
            applyMove(board, m, undo);
            DfpnEntry childEntry = probe(board.hash);
            children.push_back({m, childEntry.pn, childEntry.dn});
            undoMove(board, m, undo);
        }

        // OR node: pn = min(child_pn), dn = sum(child_dn)
        auto computeOrPnDn = [&](std::uint32_t& nodePn, std::uint32_t& nodeDn, int& bestIdx) {
            nodePn = INF_PN;
            nodeDn = 0;
            bestIdx = 0;
            for (int i = 0; i < static_cast<int>(children.size()); ++i) {
                if (children[i].pn < nodePn) {
                    nodePn = children[i].pn;
                    bestIdx = i;
                }
                nodeDn = saturatingAdd(nodeDn, children[i].dn);
            }
        };

        std::uint32_t nodePn, nodeDn;
        int bestIdx;
        computeOrPnDn(nodePn, nodeDn, bestIdx);

        path.push_back(key);

        while (nodePn < thpn && nodeDn < thdn && !shouldStop()) {
            // Find second-best pn for threshold
            std::uint32_t secondPn = INF_PN;
            for (int i = 0; i < static_cast<int>(children.size()); ++i) {
                if (i != bestIdx && children[i].pn < secondPn) {
                    secondPn = children[i].pn;
                }
            }

            std::uint32_t childThpn = std::min(thpn, saturatingAdd(secondPn, 1));
            std::uint32_t childThdn = std::min(thdn, saturatingAdd(thdn - nodeDn + children[bestIdx].dn, 0));
            if (childThdn > thdn) childThdn = thdn;

            const Move& bestMove = children[bestIdx].move;
            UndoInfo undo;
            applyMove(board, bestMove, undo);
            dfpnSearch(board, attacker, childThpn, childThdn, path, false);
            DfpnEntry childEntry = probe(board.hash);
            children[bestIdx].pn = childEntry.pn;
            children[bestIdx].dn = childEntry.dn;
            undoMove(board, bestMove, undo);

            computeOrPnDn(nodePn, nodeDn, bestIdx);
        }

        path.pop_back();
        store(key, nodePn, nodeDn, children[bestIdx].move);

    } else {
        // AND node: defender, all legal moves
        if (legal.empty()) {
            if (isKingAttacked(board, board.side)) {
                store(key, 0, INF_PN);
            } else {
                store(key, INF_PN, 0);
            }
            return;
        }

        DfpnEntry entry = probe(key);
        orderDefenseMoves(board, legal);

        struct ChildInfo {
            Move move;
            std::uint32_t pn;
            std::uint32_t dn;
        };
        std::vector<ChildInfo> children;
        children.reserve(legal.size());

        for (const Move& m : legal) {
            UndoInfo undo;
            applyMove(board, m, undo);
            DfpnEntry childEntry = probe(board.hash);
            children.push_back({m, childEntry.pn, childEntry.dn});
            undoMove(board, m, undo);
        }

        // AND node: pn = sum(child_pn), dn = min(child_dn)
        auto computeAndPnDn = [&](std::uint32_t& nodePn, std::uint32_t& nodeDn, int& bestIdx) {
            nodePn = 0;
            nodeDn = INF_PN;
            bestIdx = 0;
            for (int i = 0; i < static_cast<int>(children.size()); ++i) {
                nodePn = saturatingAdd(nodePn, children[i].pn);
                if (children[i].dn < nodeDn) {
                    nodeDn = children[i].dn;
                    bestIdx = i;
                }
            }
        };

        std::uint32_t nodePn, nodeDn;
        int bestIdx;
        computeAndPnDn(nodePn, nodeDn, bestIdx);

        path.push_back(key);

        while (nodePn < thpn && nodeDn < thdn && !shouldStop()) {
            std::uint32_t secondDn = INF_PN;
            for (int i = 0; i < static_cast<int>(children.size()); ++i) {
                if (i != bestIdx && children[i].dn < secondDn) {
                    secondDn = children[i].dn;
                }
            }

            std::uint32_t childThpn = std::min(thpn, saturatingAdd(thpn - nodePn + children[bestIdx].pn, 0));
            if (childThpn > thpn) childThpn = thpn;
            std::uint32_t childThdn = std::min(thdn, saturatingAdd(secondDn, 1));

            const Move& bestMove = children[bestIdx].move;
            UndoInfo undo;
            applyMove(board, bestMove, undo);
            dfpnSearch(board, attacker, childThpn, childThdn, path, true);
            DfpnEntry childEntry = probe(board.hash);
            children[bestIdx].pn = childEntry.pn;
            children[bestIdx].dn = childEntry.dn;
            undoMove(board, bestMove, undo);

            computeAndPnDn(nodePn, nodeDn, bestIdx);
        }

        path.pop_back();
        Move bestDefense = bestIdx < static_cast<int>(children.size()) ? children[bestIdx].move : Move{};
        store(key, nodePn, nodeDn, bestDefense);
    }
}

std::vector<Move> MateSolver::restorePV(Board& board, Color attacker) const {
    std::vector<Move> pv;
    std::vector<UndoInfo> undos;
    bool isAttacker = (board.side == attacker);

    for (int ply = 0; ply < 600; ++ply) {
        DfpnEntry entry = probe(board.hash);
        if (entry.key != board.hash || entry.bestMove.to < 0) break;
        if (isAttacker && entry.pn != 0) break;
        if (!isAttacker && entry.dn != 0) break;

        pv.push_back(entry.bestMove);
        UndoInfo undo;
        applyMove(board, entry.bestMove, undo);
        undos.push_back(undo);
        isAttacker = !isAttacker;
    }

    for (int i = static_cast<int>(undos.size()) - 1; i >= 0; --i) {
        undoMove(board, pv[i], undos[i]);
    }

    return pv;
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

    nodeLimit_ = static_cast<std::uint64_t>(maxDepth) * 200000;

    MateResult result;

    if (board.side != attacker) {
        return result;
    }

    Board work = board;
    std::vector<std::uint64_t> path;
    dfpnSearch(work, attacker, INF_PN, INF_PN, path, true);

    DfpnEntry root = probe(board.hash);
    if (root.key == board.hash && root.pn == 0) {
        result.found = true;
        result.bestMove = root.bestMove;

        Board pvBoard = board;
        result.pv = restorePV(pvBoard, attacker);
        result.moves = (static_cast<int>(result.pv.size()) + 1) / 2;
        if (result.pv.empty() && result.bestMove.to >= 0) {
            result.pv.push_back(result.bestMove);
            result.moves = 1;
        }
    }

    result.nodes = nodes_.load();
    return result;
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
