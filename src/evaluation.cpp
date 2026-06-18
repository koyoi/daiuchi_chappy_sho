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
    55.0, 42.0, 28.0, 32.0,
    24.0, 18.0, 10.0,
    0.35, 0.70, 0.45, 0.20,
    95.0, 55.0, 35.0, 180.0,
    30.0, 0.45, 45.0, 4.0,
    18.0, 12.0,
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
        Board next = board;
        applyMove(next, move);
        if (isKingAttacked(next, next.side)) {
            ++summary.checkMoves;
        }
    }
    return summary;
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
    features[28] = attackedEnemyValue - attackedOwnValue;
    features[29] = hangingEnemyValue - hangingOwnValue;
    features[30] = looseEnemyValue - looseOwnValue;
    features[31] = defendedOwnValue - defendedEnemyValue;

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
        features[37] = ownTactics.bestCapture - enemyTactics.bestCapture;
        features[38] = ownTactics.promotionMoves - enemyTactics.promotionMoves;
        features[39] = ownTactics.legalMoves - enemyTactics.legalMoves;
        features[41] = (ownTactics.checkMoves + ownTactics.promotionMoves) - (enemyTactics.checkMoves + enemyTactics.promotionMoves);
    }
    features[40] = attackedKingRingSquares(board, attacks, perspective, enemy) - attackedKingRingSquares(board, attacks, enemy, perspective);

    return features;
}

int Evaluator::evaluate(const Board& board, Color perspective) const {
    const FeatureVector features = extractFeatures(board, perspective);
    const double score = std::inner_product(features.begin(), features.end(), weights_.begin(), 0.0);
    return static_cast<int>(std::clamp(score, -30000.0, 30000.0));
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

void Evaluator::applyDelta(const FeatureVector& delta, double scale) {
    for (int i = 0; i < FeatureCount; ++i) {
        weights_[i] = std::clamp(weights_[i] + delta[i] * scale, -100000.0, 100000.0);
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
