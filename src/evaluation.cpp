#include "evaluation.h"

#include "movegen.h"
#include "position.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace shogi {

namespace {

constexpr std::array<double, FeatureCount> DefaultWeights = {
    100.0, 300.0, 320.0, 520.0, 620.0, 850.0, 1050.0,
    560.0, 1150.0, 1350.0,
    115.0, 330.0, 350.0, 560.0, 660.0, 900.0, 1100.0,
    45.0, 35.0, 22.0, 30.0,
    0.0, 42.0, 0.0, 0.0,
    0.0, 18.0, 10.0,
    35.0, 70.0, 45.0, 20.0,
    95.0, 55.0, 35.0, 180.0,
    30.0, 45.0, 45.0, 4.0,
    18.0, 12.0,
    10.0, 30.0,
    8.0, 5.0, 10.0, 12.0, 12.0, 8.0, 10.0,
    35.0, 25.0, 15.0, 20.0,
    -12.0, -8.0, 15.0,
    20.0, 12.0, 25.0,
    5.0, 8.0, 12.0,
    6.0, 5.0, 8.0, 5.0,
    25.0, 35.0,
    5.0, 0.0, -5.0, 0.0,
    -15.0, -20.0,
    25.0, 18.0, 40.0, -8.0,
    30.0, 20.0,
    22.0, 18.0,
    -10.0, 8.0, 0.0,
    50.0, 40.0,
    10.0,
    60.0, 30.0, 25.0, 20.0, 120.0, -20.0, -15.0, 18.0,
    50.0, 15.0,
};

constexpr int PstRank[PieceTypeCount][9] = {
    {},
    {3, 2, 2, 1, 0, 0, -1, -1, -1},
    {2, 2, 1, 1, 0, 0, -1, -1, -1},
    {3, 2, 2, 1, 0, 0, -1, -2, -2},
    {2, 2, 1, 1, 0, 0, -1, -1, -1},
    {1, 1, 1, 1, 0, 0, 0, 0, -1},
    {},
    {},
    {},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0},
};

int featureIndexForBoardPiece(PieceType type) {
    switch (type) {
    case Pawn: return 0;
    case Lance: return 1;
    case Knight: return 2;
    case Silver: return 3;
    case Gold: return 4;
    case Bishop: return 5;
    case Rook: return 6;
    case ProPawn:
    case ProLance:
    case ProKnight:
    case ProSilver:
        return 7;
    case Horse: return 8;
    case Dragon: return 9;
    default:
        return -1;
    }
}

int featureIndexForHandPiece(PieceType type) {
    switch (type) {
    case Pawn: return 10;
    case Lance: return 11;
    case Knight: return 12;
    case Silver: return 13;
    case Gold: return 14;
    case Bishop: return 15;
    case Rook: return 16;
    default:
        return -1;
    }
}

int signFor(Color owner, Color perspective) {
    return owner == perspective ? 1 : -1;
}

int colorIndex(Color color) {
    return color == Black ? 0 : 1;
}

int chebyshevDistance(int left, int right) {
    return std::max(std::abs(fileOf(left) - fileOf(right)), std::abs(rankOf(left) - rankOf(right)));
}

struct AttackMap {
    std::array<std::array<int, BoardSize>, 2> counts{};
};

AttackMap buildAttackMap(const Board& board) {
    AttackMap map;
    for (int square = 0; square < BoardSize; ++square) {
        map.counts[0][square] = countAttackers(board, square, Black);
        map.counts[1][square] = countAttackers(board, square, White);
    }
    return map;
}

int attackersFromMap(const AttackMap& map, int square, Color color) {
    return map.counts[colorIndex(color)][square];
}

int pieceValue(PieceType type) {
    switch (type) {
    case Pawn: return 100;
    case Lance: return 300;
    case Knight: return 320;
    case Silver: return 520;
    case Gold: return 620;
    case Bishop: return 850;
    case Rook: return 1050;
    case ProPawn:
    case ProLance:
    case ProKnight:
    case ProSilver:
        return 560;
    case Horse: return 1150;
    case Dragon: return 1350;
    default:
        return 0;
    }
}

bool goldLike(PieceType type) {
    return type == Gold || type == Silver || type == ProPawn || type == ProLance || type == ProKnight || type == ProSilver;
}

bool isAdvancedPawn(Color color, int rank) {
    return color == Black ? rank <= 5 : rank >= 5;
}

constexpr int PstFileTable[7][9] = {
    {0, 0, 1, 1, 2, 1, 1, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 1},
    {-1, 0, 1, 2, 2, 2, 1, 0, -1},
    {0, 1, 2, 2, 1, 1, 0, 0, 0},
    {1, 2, 2, 1, 0, 0, 0, 0, 0},
    {0, 0, 1, 1, 2, 1, 1, 0, 0},
    {0, 2, 1, 0, 1, 1, 1, 2, 0},
};

int fileFeatureIndex(PieceType type) {
    switch (type) {
    case Pawn: return 44;
    case Lance: return 45;
    case Knight: return 46;
    case Silver: return 47;
    case Gold: case ProPawn: case ProLance: case ProKnight: case ProSilver: return 48;
    case Bishop: case Horse: return 49;
    case Rook: case Dragon: return 50;
    default: return -1;
    }
}

int countSlide(const Board& board, int from, int df, int dr, Color owner) {
    int ci = owner == Black ? 0 : 1;
    Bitboard occ = board.occupied[0] | board.occupied[1];
    int count = 0;
    int f = fileOf(from) + df;
    int r = rankOf(from) + dr;
    while (inside(f, r)) {
        int sq = idx(f, r);
        if (board.occupied[ci].test(sq)) break;
        ++count;
        if (occ.test(sq)) break;
        f += df;
        r += dr;
    }
    return count;
}

int kingDefenders(const Board& board, const AttackMap& attacks, Color color) {
    const int king = findKing(board, color);
    if (king < 0) {
        return 0;
    }
    return attackersFromMap(attacks, king, color);
}

int kingEscapeSquares(const Board& board, Color color) {
    const int king = findKing(board, color);
    if (king < 0) {
        return 0;
    }
    int escapes = 0;
    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0) {
                continue;
            }
            const int file = fileOf(king) + df;
            const int rank = rankOf(king) + dr;
            if (!inside(file, rank)) {
                continue;
            }
            const int to = idx(file, rank);
            if (colorOf(board.squares[to]) == color) {
                continue;
            }
            Board next = board;
            next.squares[king] = 0;
            next.squares[to] = makePiece(color, King);
            if (!isSquareAttacked(next, to, opposite(color))) {
                ++escapes;
            }
        }
    }
    return escapes;
}

int attackedKingRingSquares(const Board& board, const AttackMap& attacks, Color attacker, Color kingColor) {
    const int king = findKing(board, kingColor);
    if (king < 0) {
        return 0;
    }
    int attacked = 0;
    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            const int file = fileOf(king) + df;
            const int rank = rankOf(king) + dr;
            if (inside(file, rank) && attackersFromMap(attacks, idx(file, rank), attacker) > 0) {
                ++attacked;
            }
        }
    }
    return attacked;
}

struct TacticalSummary {
    int legalMoves = 0;
    int checkMoves = 0;
    int bestCapture = 0;
    int promotionMoves = 0;
};

TacticalSummary summarizeLegalMoves(Board board, Color color) {
    board.side = color;
    TacticalSummary summary;
    const auto legal = generateLegalMoves(board, true);
    summary.legalMoves = static_cast<int>(legal.size());
    for (const Move& move : legal) {
        if (!move.isDrop() && board.squares[move.to] != 0) {
            summary.bestCapture = std::max(summary.bestCapture, pieceValue(typeOf(board.squares[move.to])));
        }
        if (move.promote) {
            ++summary.promotionMoves;
        }
        if (givesCheck(board, move)) {
            ++summary.checkMoves;
        }
    }
    return summary;
}

struct PinInfo { int count = 0; int value = 0; };

PinInfo countPins(const Board& board, Color side) {
    const int king = side == Black ? board.blackKingSquare : board.whiteKingSquare;
    if (king < 0) return {};
    const int ci = side == Black ? 0 : 1;
    const int ei = 1 - ci;
    const Color enemy = opposite(side);
    constexpr int dirs[][2] = {{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{-1,1},{1,-1},{1,1}};
    PinInfo info;
    for (auto& d : dirs) {
        int pinnedSq = -1;
        int f = fileOf(king) + d[0];
        int r = rankOf(king) + d[1];
        while (inside(f, r)) {
            int sq = idx(f, r);
            if (board.occupied[ci].test(sq)) {
                if (pinnedSq >= 0) break;
                pinnedSq = sq;
            } else if (board.occupied[ei].test(sq)) {
                if (pinnedSq >= 0) {
                    PieceType pt = typeOf(board.squares[sq]);
                    bool pins = false;
                    if (d[0] == 0 && d[1] == 0) { /* impossible */ }
                    else if (d[0] == 0 || d[1] == 0) {
                        pins = (pt == Rook || pt == Dragon);
                        if (!pins && d[0] == 0) {
                            int lanceDir = enemy == Black ? -1 : 1;
                            if (d[1] == lanceDir) pins = (pt == Lance);
                        }
                    } else {
                        pins = (pt == Bishop || pt == Horse);
                    }
                    if (pins) {
                        ++info.count;
                        info.value += pieceValue(typeOf(board.squares[pinnedSq]));
                    }
                }
                break;
            }
            f += d[0];
            r += d[1];
        }
    }
    return info;
}

int dropTargetsNearKing(const Board& board, const AttackMap& attacks, int kingSq, Color defender) {
    if (kingSq < 0) return 0;
    int count = 0;
    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            int f = fileOf(kingSq) + df;
            int r = rankOf(kingSq) + dr;
            if (!inside(f, r)) continue;
            int sq = idx(f, r);
            if (board.squares[sq] == 0 && attackersFromMap(attacks, sq, defender) == 0)
                ++count;
        }
    }
    return count;
}

int knightCheckDropSquares(const Board& board, int enemyKingSq, Color dropper) {
    if (enemyKingSq < 0) return 0;
    if (hand(board, dropper)[Knight] == 0) return 0;
    int kf = fileOf(enemyKingSq);
    int kr = rankOf(enemyKingSq);
    int fwd = dropper == Black ? -1 : 1;
    int count = 0;
    int candidates[][2] = {{kf - 1, kr - fwd * 2}, {kf + 1, kr - fwd * 2}};
    for (auto& c : candidates) {
        if (!inside(c[0], c[1])) continue;
        if (!canDropOnRank(dropper, Knight, c[1])) continue;
        int sq = idx(c[0], c[1]);
        if (board.squares[sq] == 0) ++count;
    }
    return count;
}

int canDropPawnOnKingFile(const Board& board, Color dropper, int enemyKingSq) {
    if (enemyKingSq < 0) return 0;
    if (hand(board, dropper)[Pawn] == 0) return 0;
    int kf = fileOf(enemyKingSq);
    int ci = dropper == Black ? 0 : 1;
    if (!(board.pieceBB[Pawn] & board.occupied[ci] & FileMask[kf]).empty()) return 0;
    Bitboard fileSq = FileMask[kf];
    while (!fileSq.empty()) {
        int sq = fileSq.lsb();
        fileSq.clear(sq);
        if (board.squares[sq] == 0 && canDropOnRank(dropper, Pawn, rankOf(sq)))
            return 1;
    }
    return 0;
}

int diagonalAlignmentToKing(const Board& board, Color side, int enemyKingSq) {
    if (enemyKingSq < 0) return 0;
    int ci = side == Black ? 0 : 1;
    Bitboard bishops = (board.pieceBB[Bishop] | board.pieceBB[Horse]) & board.occupied[ci];
    int count = 0;
    while (!bishops.empty()) {
        int sq = bishops.lsb();
        bishops.clear(sq);
        int df = fileOf(enemyKingSq) - fileOf(sq);
        int dr = rankOf(enemyKingSq) - rankOf(sq);
        if (df == 0 || dr == 0 || std::abs(df) != std::abs(dr)) continue;
        int stepF = df > 0 ? 1 : -1;
        int stepR = dr > 0 ? 1 : -1;
        bool blocked = false;
        int f = fileOf(sq) + stepF;
        int r = rankOf(sq) + stepR;
        while (f != fileOf(enemyKingSq) || r != rankOf(enemyKingSq)) {
            if (board.squares[idx(f, r)] != 0) { blocked = true; break; }
            f += stepF;
            r += stepR;
        }
        if (!blocked) ++count;
    }
    return count;
}

int lanceAimedAtKing(const Board& board, Color side, int enemyKingSq) {
    if (enemyKingSq < 0) return 0;
    int ci = side == Black ? 0 : 1;
    int kf = fileOf(enemyKingSq);
    Bitboard lances = board.pieceBB[Lance] & board.occupied[ci] & FileMask[kf];
    int fwd = side == Black ? -1 : 1;
    int count = 0;
    while (!lances.empty()) {
        int sq = lances.lsb();
        lances.clear(sq);
        int lr = rankOf(sq);
        int kr = rankOf(enemyKingSq);
        if ((kr - lr) * fwd <= 0) continue;
        bool blocked = false;
        int r = lr + fwd;
        while (r != kr) {
            if (board.squares[idx(kf, r)] != 0) { blocked = true; break; }
            r += fwd;
        }
        if (!blocked) ++count;
    }
    return count;
}

int pieceValueNearSquare(const Board& board, Color side, int sq, int maxDist) {
    if (sq < 0) return 0;
    int ci = side == Black ? 0 : 1;
    int total = 0;
    for (int i = 0; i < BoardSize; ++i) {
        if (!board.occupied[ci].test(i)) continue;
        PieceType pt = typeOf(board.squares[i]);
        if (pt == King) continue;
        if (chebyshevDistance(i, sq) <= maxDist)
            total += pieceValue(pt);
    }
    return total;
}

int openFilesNearKing(const Board& board, Color side, int kingSq) {
    if (kingSq < 0) return 0;
    int ci = side == Black ? 0 : 1;
    int kf = fileOf(kingSq);
    int count = 0;
    for (int f = std::max(1, kf - 1); f <= std::min(9, kf + 1); ++f) {
        if ((board.pieceBB[Pawn] & board.occupied[ci] & FileMask[f]).empty())
            ++count;
    }
    return count;
}

int passedPawns(const Board& board, Color side) {
    int ci = side == Black ? 0 : 1;
    int ei = 1 - ci;
    Bitboard pawns = board.pieceBB[Pawn] & board.occupied[ci];
    Bitboard enemyPawns = board.pieceBB[Pawn] & board.occupied[ei];
    int count = 0;
    while (!pawns.empty()) {
        int sq = pawns.lsb();
        pawns.clear(sq);
        int f = fileOf(sq);
        bool passed = true;
        for (int af = std::max(1, f - 1); af <= std::min(9, f + 1); ++af) {
            Bitboard epFile = enemyPawns & FileMask[af];
            while (!epFile.empty()) {
                int esq = epFile.lsb();
                epFile.clear(esq);
                int er = rankOf(esq);
                int pr = rankOf(sq);
                bool ahead = side == Black ? (er < pr) : (er > pr);
                if (ahead) { passed = false; break; }
            }
            if (!passed) break;
        }
        if (passed) ++count;
    }
    return count;
}

} // namespace

Evaluator::Evaluator()
    : weights_(DefaultWeights) {
}

FeatureVector Evaluator::extractFeatures(const Board& board, Color perspective) const {
    FeatureVector features{};
    const int blackKing = findKing(board, Black);
    const int whiteKing = findKing(board, White);
    const AttackMap attacks = buildAttackMap(board);
    int attackedEnemyValue = 0;
    int attackedOwnValue = 0;
    int hangingEnemyValue = 0;
    int hangingOwnValue = 0;
    int looseEnemyValue = 0;
    int looseOwnValue = 0;
    int defendedOwnValue = 0;
    int defendedEnemyValue = 0;

    for (int square = 0; square < BoardSize; ++square) {
        const int piece = board.squares[square];
        if (piece == 0) {
            continue;
        }
        const Color owner = static_cast<Color>(colorOf(piece));
        const PieceType type = typeOf(piece);
        const int sign = signFor(owner, perspective);
        const int boardIndex = featureIndexForBoardPiece(type);
        if (boardIndex >= 0) {
            features[boardIndex] += sign;
        }

        const int rank = rankOf(square);
        const int file = fileOf(square);
        const int relRank = owner == Black ? rank : 10 - rank;
        const int relFile = owner == Black ? file : 10 - file;
        if (type < PieceTypeCount) {
            features[42] += sign * PstRank[type][relRank - 1];
        }
        const int ffi = fileFeatureIndex(type);
        if (ffi >= 0) {
            features[ffi] += sign * PstFileTable[ffi - 44][relFile - 1];
        }
        if (inPromotionZone(owner, rank)) {
            features[17] += sign;
        }
        if (type == Pawn && isAdvancedPawn(owner, rank)) {
            features[18] += sign;
        }
        if (type != King) {
            const int value = pieceValue(type);
            const int enemyAttackers = attackersFromMap(attacks, square, opposite(owner));
            const int ownDefenders = attackersFromMap(attacks, square, owner);
            if (owner == perspective) {
                if (enemyAttackers > 0) {
                    attackedOwnValue += value;
                }
                if (enemyAttackers > 0 && ownDefenders == 0) {
                    hangingOwnValue += value;
                }
                if (enemyAttackers > ownDefenders) {
                    looseOwnValue += value;
                }
                if (ownDefenders > 0) {
                    defendedOwnValue += value;
                }
            } else {
                if (enemyAttackers > 0) {
                    attackedEnemyValue += value;
                }
                if (enemyAttackers > 0 && ownDefenders == 0) {
                    hangingEnemyValue += value;
                }
                if (enemyAttackers > ownDefenders) {
                    looseEnemyValue += value;
                }
                if (ownDefenders > 0) {
                    defendedEnemyValue += value;
                }
            }

            const int enemyKing = owner == Black ? whiteKing : blackKing;
            if (enemyKing >= 0 && chebyshevDistance(square, enemyKing) <= 2) {
                features[19] += sign;
            }
            const int ownKing = owner == Black ? blackKing : whiteKing;
            if (ownKing >= 0 && chebyshevDistance(square, ownKing) <= 2 && goldLike(type)) {
                features[20] += sign;
            }
        }
    }

    for (PieceType type : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        const int handIndex = featureIndexForHandPiece(type);
        features[handIndex] += hand(board, perspective)[type];
        features[handIndex] -= hand(board, opposite(perspective))[type];
    }

    const int ownKing = perspective == Black ? blackKing : whiteKing;
    const int enemyKing = perspective == Black ? whiteKing : blackKing;
    if (ownKing >= 0 && enemyKing >= 0) {
        features[21] = 9 - chebyshevDistance(ownKing, enemyKing);
    }

    features[22] = board.side == perspective ? 1.0 : -1.0;
    features[23] = board.moveNumber < 40 ? 1.0 : 0.0;
    features[24] = board.moveNumber >= 40 && board.moveNumber < 90 ? 1.0 : 0.0;
    features[25] = board.moveNumber >= 90 ? 1.0 : 0.0;

    int ownHandTotal = 0;
    int enemyHandTotal = 0;
    for (PieceType type : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        ownHandTotal += hand(board, perspective)[type];
        enemyHandTotal += hand(board, opposite(perspective))[type];
    }
    features[26] = ownHandTotal - enemyHandTotal;

    int ownPromoted = 0;
    int enemyPromoted = 0;
    for (int piece : board.squares) {
        const PieceType type = typeOf(piece);
        if (type == ProPawn || type == ProLance || type == ProKnight || type == ProSilver || type == Horse || type == Dragon) {
            if (colorOf(piece) == perspective) {
                ++ownPromoted;
            } else if (colorOf(piece) == opposite(perspective)) {
                ++enemyPromoted;
            }
        }
    }
    features[27] = ownPromoted - enemyPromoted;
    features[28] = (attackedEnemyValue - attackedOwnValue) / 100.0;
    features[29] = (hangingEnemyValue - hangingOwnValue) / 100.0;
    features[30] = (looseEnemyValue - looseOwnValue) / 100.0;
    features[31] = (defendedOwnValue - defendedEnemyValue) / 100.0;

    const Color enemy = opposite(perspective);
    const int ownKingAttackers = ownKing >= 0 ? attackersFromMap(attacks, ownKing, enemy) : 0;
    const int enemyKingAttackers = enemyKing >= 0 ? attackersFromMap(attacks, enemyKing, perspective) : 0;
    features[32] = enemyKingAttackers - ownKingAttackers;
    features[33] = kingDefenders(board, attacks, perspective) - kingDefenders(board, attacks, enemy);
    features[34] = kingEscapeSquares(board, perspective) - kingEscapeSquares(board, enemy);
    features[35] = (isKingAttacked(board, enemy) ? 1.0 : 0.0) - (isKingAttacked(board, perspective) ? 1.0 : 0.0);

    if (heavyFeatures_) {
        const TacticalSummary ownTactics = summarizeLegalMoves(board, perspective);
        const TacticalSummary enemyTactics = summarizeLegalMoves(board, enemy);
        features[36] = ownTactics.checkMoves - enemyTactics.checkMoves;
        features[37] = (ownTactics.bestCapture - enemyTactics.bestCapture) / 100.0;
        features[38] = ownTactics.promotionMoves - enemyTactics.promotionMoves;
        features[39] = ownTactics.legalMoves - enemyTactics.legalMoves;
        features[41] = (ownTactics.checkMoves + ownTactics.promotionMoves) - (enemyTactics.checkMoves + enemyTactics.promotionMoves);
    }
    features[40] = attackedKingRingSquares(board, attacks, perspective, enemy) - attackedKingRingSquares(board, attacks, enemy, perspective);

    auto pawnShelter = [&](Color color) -> int {
        const int king = color == Black ? board.blackKingSquare : board.whiteKingSquare;
        if (king < 0) return 0;
        const int kf = fileOf(king);
        const int ci = color == Black ? 0 : 1;
        Bitboard ownPawns = board.pieceBB[Pawn] & board.occupied[ci];
        Bitboard mask;
        for (int f = std::max(1, kf - 1); f <= std::min(9, kf + 1); ++f)
            mask |= FileMask[f];
        return (ownPawns & mask).popcount();
    };
    features[43] = pawnShelter(perspective) - pawnShelter(enemy);

    // King surround (51-54)
    {
        struct KingSurround { int gold = 0; int silver = 0; int friendly = 0; int enemy = 0; };
        auto countKS = [&](Color kingColor) -> KingSurround {
            KingSurround ks;
            int kingSq = kingColor == Black ? blackKing : whiteKing;
            if (kingSq < 0) return ks;
            for (int df = -1; df <= 1; ++df) {
                for (int dr = -1; dr <= 1; ++dr) {
                    if (df == 0 && dr == 0) continue;
                    int f = fileOf(kingSq) + df;
                    int r = rankOf(kingSq) + dr;
                    if (!inside(f, r)) continue;
                    int p = board.squares[idx(f, r)];
                    if (p == 0) continue;
                    Color pc = static_cast<Color>(colorOf(p));
                    PieceType pt = typeOf(p);
                    if (pc == kingColor) {
                        ++ks.friendly;
                        if (pt == Gold || pt == ProPawn || pt == ProLance
                            || pt == ProKnight || pt == ProSilver)
                            ++ks.gold;
                        if (pt == Silver) ++ks.silver;
                    } else {
                        ++ks.enemy;
                    }
                }
            }
            return ks;
        };
        auto ownKS = countKS(perspective);
        auto enemyKS = countKS(enemy);
        features[51] = ownKS.gold - enemyKS.gold;
        features[52] = ownKS.silver - enemyKS.silver;
        features[53] = ownKS.friendly - enemyKS.friendly;
        features[54] = enemyKS.enemy - ownKS.enemy;
    }

    // Pawn structure (55-57)
    {
        auto pawnStruct = [&](Color color) -> std::array<int, 3> {
            int ci = color == Black ? 0 : 1;
            int eci = 1 - ci;
            Bitboard pawns = board.pieceBB[Pawn] & board.occupied[ci];
            Bitboard enemyPawns = board.pieceBB[Pawn] & board.occupied[eci];
            int doubled = 0, isolated = 0, candidates = 0;
            for (int f = 1; f <= 9; ++f) {
                int cnt = (pawns & FileMask[f]).popcount();
                if (cnt > 1) doubled += cnt - 1;
                if (cnt > 0) {
                    bool adj = (f > 1 && !(pawns & FileMask[f - 1]).empty())
                            || (f < 9 && !(pawns & FileMask[f + 1]).empty());
                    if (!adj) isolated += cnt;
                    if ((enemyPawns & FileMask[f]).empty()) candidates += cnt;
                }
            }
            return {doubled, isolated, candidates};
        };
        auto own = pawnStruct(perspective);
        auto enem = pawnStruct(enemy);
        features[55] = own[0] - enem[0];
        features[56] = own[1] - enem[1];
        features[57] = own[2] - enem[2];
    }

    // Rook activity (58-60)
    {
        auto rookAct = [&](Color color) -> std::array<int, 3> {
            int ci = color == Black ? 0 : 1;
            int oppKingSq = color == perspective ? enemyKing : ownKing;
            int oppKingFile = oppKingSq >= 0 ? fileOf(oppKingSq) : -1;
            Bitboard rooks = (board.pieceBB[Rook] | board.pieceBB[Dragon]) & board.occupied[ci];
            int openF = 0, semiF = 0, kingAlign = 0;
            while (!rooks.empty()) {
                int sq = rooks.lsb();
                rooks.clear(sq);
                int f = fileOf(sq);
                bool ownPawn = !(board.pieceBB[Pawn] & board.occupied[ci] & FileMask[f]).empty();
                bool oppPawn = !(board.pieceBB[Pawn] & board.occupied[1 - ci] & FileMask[f]).empty();
                if (!ownPawn && !oppPawn) ++openF;
                else if (!ownPawn) ++semiF;
                if (f == oppKingFile) ++kingAlign;
            }
            return {openF, semiF, kingAlign};
        };
        auto ownRA = rookAct(perspective);
        auto enemyRA = rookAct(enemy);
        features[58] = ownRA[0] - enemyRA[0];
        features[59] = ownRA[1] - enemyRA[1];
        features[60] = ownRA[2] - enemyRA[2];
    }

    // Zone control (61-63)
    {
        int controlOwn = 0, controlCtr = 0, controlEne = 0;
        for (int sq = 0; sq < BoardSize; ++sq) {
            int relR = perspective == Black ? rankOf(sq) : 10 - rankOf(sq);
            bool ownCtrl = attackersFromMap(attacks, sq, perspective) > 0;
            bool eneCtrl = attackersFromMap(attacks, sq, enemy) > 0;
            int diff = (ownCtrl ? 1 : 0) - (eneCtrl ? 1 : 0);
            if (relR <= 3) controlEne += diff;
            else if (relR <= 6) controlCtr += diff;
            else controlOwn += diff;
        }
        features[61] = controlOwn;
        features[62] = controlCtr;
        features[63] = controlEne;
    }

    // Piece mobility (64-67)
    {
        auto mob = [&](Color color) -> std::array<int, 4> {
            int ci = color == Black ? 0 : 1;
            int rookMob = 0, bishopMob = 0, silverMob = 0, lanceMob = 0;

            Bitboard rooks = (board.pieceBB[Rook] | board.pieceBB[Dragon]) & board.occupied[ci];
            while (!rooks.empty()) {
                int sq = rooks.lsb();
                rooks.clear(sq);
                rookMob += countSlide(board, sq, 0, -1, color);
                rookMob += countSlide(board, sq, 0, 1, color);
                rookMob += countSlide(board, sq, -1, 0, color);
                rookMob += countSlide(board, sq, 1, 0, color);
                if (typeOf(board.squares[sq]) == Dragon) {
                    for (int df2 = -1; df2 <= 1; df2 += 2)
                        for (int dr2 = -1; dr2 <= 1; dr2 += 2) {
                            int f = fileOf(sq) + df2, r = rankOf(sq) + dr2;
                            if (inside(f, r) && !board.occupied[ci].test(idx(f, r)))
                                ++rookMob;
                        }
                }
            }

            Bitboard bishops = (board.pieceBB[Bishop] | board.pieceBB[Horse]) & board.occupied[ci];
            while (!bishops.empty()) {
                int sq = bishops.lsb();
                bishops.clear(sq);
                for (int df2 = -1; df2 <= 1; df2 += 2)
                    for (int dr2 = -1; dr2 <= 1; dr2 += 2)
                        bishopMob += countSlide(board, sq, df2, dr2, color);
                if (typeOf(board.squares[sq]) == Horse) {
                    int ortho[][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
                    for (auto& d : ortho) {
                        int f = fileOf(sq) + d[0], r = rankOf(sq) + d[1];
                        if (inside(f, r) && !board.occupied[ci].test(idx(f, r)))
                            ++bishopMob;
                    }
                }
            }

            int fwd = color == Black ? -1 : 1;
            Bitboard silvers = board.pieceBB[Silver] & board.occupied[ci];
            while (!silvers.empty()) {
                int sq = silvers.lsb();
                silvers.clear(sq);
                for (int df2 = -1; df2 <= 1; ++df2) {
                    int f = fileOf(sq) + df2, r = rankOf(sq) + fwd;
                    if (inside(f, r) && !board.occupied[ci].test(idx(f, r)))
                        ++silverMob;
                }
                for (int df2 = -1; df2 <= 1; df2 += 2) {
                    int f = fileOf(sq) + df2, r = rankOf(sq) - fwd;
                    if (inside(f, r) && !board.occupied[ci].test(idx(f, r)))
                        ++silverMob;
                }
            }

            int lDir = color == Black ? -1 : 1;
            Bitboard lances = board.pieceBB[Lance] & board.occupied[ci];
            while (!lances.empty()) {
                int sq = lances.lsb();
                lances.clear(sq);
                lanceMob += countSlide(board, sq, 0, lDir, color);
            }

            return {rookMob, bishopMob, silverMob, lanceMob};
        };
        auto ownMob = mob(perspective);
        auto enemyMob = mob(enemy);
        features[64] = (ownMob[0] - enemyMob[0]) / 10.0;
        features[65] = (ownMob[1] - enemyMob[1]) / 10.0;
        features[66] = ownMob[2] - enemyMob[2];
        features[67] = ownMob[3] - enemyMob[3];
    }

    // Threats (68-69)
    {
        auto pawnThr = [&](Color color) -> int {
            int ci = color == Black ? 0 : 1;
            Bitboard pawns = board.pieceBB[Pawn] & board.occupied[ci];
            int dir = color == Black ? -9 : 9;
            int threats = 0;
            while (!pawns.empty()) {
                int sq = pawns.lsb();
                pawns.clear(sq);
                int tgt = sq + dir;
                if (tgt >= 0 && tgt < BoardSize) {
                    int p = board.squares[tgt];
                    if (p != 0 && static_cast<Color>(colorOf(p)) != color
                        && typeOf(p) != Pawn)
                        ++threats;
                }
            }
            return threats;
        };
        features[68] = pawnThr(perspective) - pawnThr(enemy);

        auto minorMajor = [&](Color color) -> int {
            int ci = color == Black ? 0 : 1;
            int eci = 1 - ci;
            Bitboard enemyMaj = (board.pieceBB[Rook] | board.pieceBB[Bishop]
                | board.pieceBB[Horse] | board.pieceBB[Dragon]) & board.occupied[eci];
            Bitboard ownMin = (board.pieceBB[Pawn] | board.pieceBB[Lance]
                | board.pieceBB[Knight] | board.pieceBB[Silver]) & board.occupied[ci];
            int threats = 0;
            while (!enemyMaj.empty()) {
                int sq = enemyMaj.lsb();
                enemyMaj.clear(sq);
                Bitboard minors = ownMin;
                while (!minors.empty()) {
                    int msq = minors.lsb();
                    minors.clear(msq);
                    if (attacksSquare(board, msq, sq)) {
                        ++threats;
                        break;
                    }
                }
            }
            return threats;
        };
        features[69] = minorMajor(perspective) - minorMajor(enemy);
    }

    // King position (70-73)
    if (ownKing >= 0) {
        features[70] = std::abs(5 - fileOf(ownKing));
        features[71] = perspective == Black ? rankOf(ownKing) : 10 - rankOf(ownKing);
    }
    if (enemyKing >= 0) {
        features[72] = std::abs(5 - fileOf(enemyKing));
        features[73] = perspective == Black ? rankOf(enemyKing) : 10 - rankOf(enemyKing);
    }

    // Pin detection (74-75)
    {
        auto ownPins = countPins(board, perspective);
        auto enemyPins = countPins(board, enemy);
        features[74] = ownPins.count - enemyPins.count;
        features[75] = (ownPins.value - enemyPins.value) / 100.0;
    }

    // Drop threats (76-79)
    {
        int ownDropTargets = dropTargetsNearKing(board, attacks, enemyKing, enemy);
        int enemyDropTargets = dropTargetsNearKing(board, attacks, ownKing, perspective);
        features[76] = ownDropTargets - enemyDropTargets;

        int ownGS = hand(board, perspective)[Gold] + hand(board, perspective)[Silver];
        int enemyGS = hand(board, enemy)[Gold] + hand(board, enemy)[Silver];
        features[77] = ownGS * ownDropTargets - enemyGS * enemyDropTargets;

        features[78] = knightCheckDropSquares(board, enemyKing, perspective)
                     - knightCheckDropSquares(board, ownKing, enemy);

        features[79] = canDropPawnOnKingFile(board, perspective, enemyKing)
                     - canDropPawnOnKingFile(board, enemy, ownKing);
    }

    // Long-range alignment (80-81)
    {
        features[80] = diagonalAlignmentToKing(board, perspective, enemyKing)
                     - diagonalAlignmentToKing(board, enemy, ownKing);
        features[81] = lanceAimedAtKing(board, perspective, enemyKing)
                     - lanceAimedAtKing(board, enemy, ownKing);
    }

    // King area strength (82-83)
    {
        features[82] = (pieceValueNearSquare(board, perspective, enemyKing, 3)
                      - pieceValueNearSquare(board, enemy, ownKing, 3)) / 100.0;
        features[83] = (pieceValueNearSquare(board, perspective, ownKing, 2)
                      - pieceValueNearSquare(board, enemy, enemyKing, 2)) / 100.0;
    }

    // King safety details (84-86)
    {
        features[84] = openFilesNearKing(board, perspective, ownKing)
                     - openFilesNearKing(board, enemy, enemyKing);
        int ownEdge = ownKing >= 0 && (fileOf(ownKing) == 1 || fileOf(ownKing) == 9) ? 1 : 0;
        int enemyEdge = enemyKing >= 0 && (fileOf(enemyKing) == 1 || fileOf(enemyKing) == 9) ? 1 : 0;
        features[85] = ownEdge - enemyEdge;

        int totalMaterial = 0;
        for (int sq = 0; sq < BoardSize; ++sq) {
            if (board.squares[sq] != 0 && typeOf(board.squares[sq]) != King)
                totalMaterial += pieceValue(typeOf(board.squares[sq]));
        }
        features[86] = totalMaterial / 1000.0;
    }

    // Piece presence & passed pawns (87-89)
    {
        int ownCI = perspective == Black ? 0 : 1;
        int enemyCI = 1 - ownCI;
        bool ownRook = !(board.pieceBB[Rook] & board.occupied[ownCI]).empty()
                    || !(board.pieceBB[Dragon] & board.occupied[ownCI]).empty()
                    || hand(board, perspective)[Rook] > 0;
        bool enemyRook = !(board.pieceBB[Rook] & board.occupied[enemyCI]).empty()
                      || !(board.pieceBB[Dragon] & board.occupied[enemyCI]).empty()
                      || hand(board, enemy)[Rook] > 0;
        features[87] = (ownRook ? 1 : 0) - (enemyRook ? 1 : 0);

        bool ownBishop = !(board.pieceBB[Bishop] & board.occupied[ownCI]).empty()
                      || !(board.pieceBB[Horse] & board.occupied[ownCI]).empty()
                      || hand(board, perspective)[Bishop] > 0;
        bool enemyBishop = !(board.pieceBB[Bishop] & board.occupied[enemyCI]).empty()
                        || !(board.pieceBB[Horse] & board.occupied[enemyCI]).empty()
                        || hand(board, enemy)[Bishop] > 0;
        features[88] = (ownBishop ? 1 : 0) - (enemyBishop ? 1 : 0);

        features[89] = passedPawns(board, perspective) - passedPawns(board, enemy);
    }

    // Phase-gated interaction features (90-97)
    {
        const double endgameness = std::clamp(board.moveNumber / 120.0, 0.0, 1.0);
        const double openingness = 1.0 - endgameness;
        features[90] = features[32] * endgameness;
        features[91] = features[34] * endgameness;
        features[92] = features[82] * endgameness;
        features[93] = features[77] * endgameness;
        features[94] = features[35] * endgameness;
        features[95] = features[43] * openingness;
        features[96] = features[53] * openingness;
        features[97] = features[75] * endgameness;
    }

    // Fork detection (98)
    {
        auto forkScore = [&](Color side) -> int {
            int ci = side == Black ? 0 : 1;
            int ei = 1 - ci;
            int total = 0;
            for (int sq = 0; sq < BoardSize; ++sq) {
                if (!board.occupied[ci].test(sq)) continue;
                PieceType pt = typeOf(board.squares[sq]);
                if (pt == King || pt == Pawn) continue;
                int attackedCount = 0;
                int attackedValue = 0;
                for (int esq = 0; esq < BoardSize; ++esq) {
                    if (!board.occupied[ei].test(esq)) continue;
                    PieceType ept = typeOf(board.squares[esq]);
                    if (ept == King || ept == Pawn) continue;
                    if (attacksSquare(board, sq, esq)) {
                        ++attackedCount;
                        attackedValue += pieceValue(ept);
                    }
                }
                if (attackedCount >= 2) {
                    total += attackedValue;
                }
            }
            return total;
        };
        features[98] = (forkScore(perspective) - forkScore(enemy)) / 100.0;
    }

    // Piece coordination (99)
    {
        auto coordination = [&](Color side) -> int {
            int ci = side == Black ? 0 : 1;
            int count = 0;
            for (int sq = 0; sq < BoardSize; ++sq) {
                if (!board.occupied[ci].test(sq)) continue;
                PieceType pt = typeOf(board.squares[sq]);
                if (pt == King) continue;
                int defenders = attackersFromMap(attacks, sq, side);
                if (defenders > 0) ++count;
            }
            return count;
        };
        features[99] = coordination(perspective) - coordination(enemy);
    }

    return features;
}

int Evaluator::evaluate(const Board& board, Color perspective) const {
    const FeatureVector features = extractFeatures(board, perspective);
    if (mlpLoaded_) {
        return evaluateMlp(features);
    }
    const double score = std::inner_product(features.begin(), features.end(), weights_.begin(), 0.0);
    return static_cast<int>(std::clamp(score, -30000.0, 30000.0));
}

int Evaluator::evaluateMlp(const FeatureVector& features) const {
    double h1[MlpHidden1];
    for (int j = 0; j < MlpHidden1; ++j) {
        double sum = b1_[j];
        for (int i = 0; i < FeatureCount; ++i) {
            sum += w1_[j][i] * features[i];
        }
        h1[j] = sum > 0.0 ? sum : 0.0;
    }
    double h2[MlpHidden2];
    for (int j = 0; j < MlpHidden2; ++j) {
        double sum = b2_[j];
        for (int i = 0; i < MlpHidden1; ++i) {
            sum += w2_[j][i] * h1[i];
        }
        h2[j] = sum > 0.0 ? sum : 0.0;
    }
    double out = b3_;
    for (int i = 0; i < MlpHidden2; ++i) {
        out += w3_[i] * h2[i];
    }
    return static_cast<int>(std::clamp(out * 1000.0, -30000.0, 30000.0));
}

bool Evaluator::loadMlp(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    int inputDim = 0, hidden1 = 0, hidden2 = 0;
    if (!(in >> inputDim >> hidden1 >> hidden2)) {
        return false;
    }
    if (inputDim != FeatureCount || hidden1 != MlpHidden1 || hidden2 != MlpHidden2) {
        return false;
    }
    for (int j = 0; j < MlpHidden1; ++j)
        for (int i = 0; i < FeatureCount; ++i)
            if (!(in >> w1_[j][i])) return false;
    for (int j = 0; j < MlpHidden1; ++j)
        if (!(in >> b1_[j])) return false;
    for (int j = 0; j < MlpHidden2; ++j)
        for (int i = 0; i < MlpHidden1; ++i)
            if (!(in >> w2_[j][i])) return false;
    for (int j = 0; j < MlpHidden2; ++j)
        if (!(in >> b2_[j])) return false;
    for (int i = 0; i < MlpHidden2; ++i)
        if (!(in >> w3_[i])) return false;
    if (!(in >> b3_)) return false;
    mlpLoaded_ = true;
    return true;
}

void Evaluator::setHeavyFeatures(bool enabled) {
    heavyFeatures_ = enabled;
}

bool Evaluator::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::array<double, FeatureCount> loaded = DefaultWeights;
    bool readAny = false;
    for (double& weight : loaded) {
        if (!(input >> weight)) {
            break;
        }
        readAny = true;
    }
    if (!readAny) {
        return false;
    }
    for (int i = 0; i < FeatureCount; ++i) {
        double maxW = std::max(std::abs(DefaultWeights[i]) * 5.0, 5.0);
        loaded[i] = std::clamp(loaded[i], -maxW, maxW);
    }
    weights_ = loaded;
    return true;
}

bool Evaluator::save(const std::string& path) const {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    for (double weight : weights_) {
        output << weight << '\n';
    }
    return true;
}

bool Evaluator::computeGradient(const Board& board, const Move& correctMove, GradientResult& out, double temperature) const {
    auto legal = generateLegalMoves(board, true);
    if (legal.size() <= 1) return false;

    int correctIdx = -1;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        if (sameMove(legal[i], correctMove)) {
            correctIdx = static_cast<int>(i);
            break;
        }
    }
    if (correctIdx < 0) return false;

    std::vector<FeatureVector> features(legal.size());
    std::vector<double> scores(legal.size());
    for (std::size_t i = 0; i < legal.size(); ++i) {
        Board after = board;
        applyMove(after, legal[i]);
        features[i] = extractFeatures(after, board.side);
        scores[i] = std::inner_product(features[i].begin(), features[i].end(),
                                        weights_.begin(), 0.0);
    }

    int bestIdx = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    out.correct = (bestIdx == correctIdx);

    double maxScore = scores[bestIdx];
    double sumExp = 0.0;
    for (auto& s : scores) {
        s = std::exp((s - maxScore) / temperature);
        sumExp += s;
    }

    double correctProb = scores[correctIdx] / sumExp;
    out.loss = -std::log(std::max(correctProb, 1e-10));

    FeatureVector expected{};
    for (std::size_t i = 0; i < legal.size(); ++i) {
        double p = scores[i] / sumExp;
        for (int j = 0; j < FeatureCount; ++j) {
            expected[j] += p * features[i][j];
        }
    }

    out.delta = features[correctIdx] - expected;
    return true;
}

bool Evaluator::learnFromMove(const Board& board, const Move& correctMove, double lr) {
    GradientResult result;
    if (!computeGradient(board, correctMove, result)) return false;
    applyDelta(result.delta, lr);
    return true;
}

void Evaluator::applyDelta(const FeatureVector& delta, double scale) {
    for (int i = 0; i < FeatureCount; ++i) {
        double maxW = std::max(std::abs(DefaultWeights[i]) * 10.0, 50.0);
        weights_[i] = std::clamp(weights_[i] + delta[i] * scale, -maxW, maxW);
    }
}

FeatureVector operator-(const FeatureVector& left, const FeatureVector& right) {
    FeatureVector result{};
    for (int i = 0; i < FeatureCount; ++i) {
        result[i] = left[i] - right[i];
    }
    return result;
}

FeatureVector& operator+=(FeatureVector& left, const FeatureVector& right) {
    for (int i = 0; i < FeatureCount; ++i) {
        left[i] += right[i];
    }
    return left;
}

FeatureVector& operator/=(FeatureVector& left, double value) {
    if (value == 0.0) {
        return left;
    }
    for (double& item : left) {
        item /= value;
    }
    return left;
}

} // namespace shogi
