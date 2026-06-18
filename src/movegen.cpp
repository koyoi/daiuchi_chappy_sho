#include "movegen.h"

#include "position.h"

#include <array>
#include <cmath>

namespace shogi {

namespace {

void addStep(const Board& board, std::vector<Move>& moves, int from, int df, int dr) {
    const Color color = static_cast<Color>(colorOf(board.squares[from]));
    const int file = fileOf(from) + df;
    const int rank = rankOf(from) + dr * static_cast<int>(color);
    if (!inside(file, rank)) {
        return;
    }
    const int to = idx(file, rank);
    if (colorOf(board.squares[to]) == color) {
        return;
    }
    const PieceType type = typeOf(board.squares[from]);
    const bool promotable = canPromote(type) && (inPromotionZone(color, rankOf(from)) || inPromotionZone(color, rankOf(to)));
    const bool mandatory = mustPromote(color, type, rankOf(to));
    if (promotable) {
        if (!mandatory) {
            moves.push_back(Move{from, to, type, Empty, false});
        }
        moves.push_back(Move{from, to, type, Empty, true});
    } else {
        moves.push_back(Move{from, to, type, Empty, false});
    }
}

void addSlide(const Board& board, std::vector<Move>& moves, int from, int df, int dr) {
    const Color color = static_cast<Color>(colorOf(board.squares[from]));
    const PieceType type = typeOf(board.squares[from]);
    int file = fileOf(from) + df;
    int rank = rankOf(from) + dr * static_cast<int>(color);
    while (inside(file, rank)) {
        const int to = idx(file, rank);
        if (colorOf(board.squares[to]) == color) {
            break;
        }
        const bool promotable = canPromote(type) && (inPromotionZone(color, rankOf(from)) || inPromotionZone(color, rankOf(to)));
        const bool mandatory = mustPromote(color, type, rankOf(to));
        if (promotable) {
            if (!mandatory) {
                moves.push_back(Move{from, to, type, Empty, false});
            }
            moves.push_back(Move{from, to, type, Empty, true});
        } else {
            moves.push_back(Move{from, to, type, Empty, false});
        }
        if (board.squares[to] != 0) {
            break;
        }
        file += df;
        rank += dr * static_cast<int>(color);
    }
}

bool hasUnpromotedPawnOnFile(const Board& board, Color color, int file) {
    for (int rank = 1; rank <= 9; ++rank) {
        const int piece = board.squares[idx(file, rank)];
        if (piece == makePiece(color, Pawn)) {
            return true;
        }
    }
    return false;
}

} // namespace

bool attacksSquare(const Board& board, int from, int target) {
    const int piece = board.squares[from];
    if (piece == 0) {
        return false;
    }
    const Color color = static_cast<Color>(colorOf(piece));
    const PieceType type = typeOf(piece);
    const int df = fileOf(target) - fileOf(from);
    const int dr = (rankOf(target) - rankOf(from)) * static_cast<int>(color);

    auto clearLine = [&](int stepFile, int stepRank) {
        int file = fileOf(from) + stepFile;
        int rank = rankOf(from) + stepRank * static_cast<int>(color);
        while (inside(file, rank)) {
            const int square = idx(file, rank);
            if (square == target) {
                return true;
            }
            if (board.squares[square] != 0) {
                return false;
            }
            file += stepFile;
            rank += stepRank * static_cast<int>(color);
        }
        return false;
    };

    switch (type) {
    case Pawn:
        return df == 0 && dr == -1;
    case Lance:
        return df == 0 && dr < 0 && clearLine(0, -1);
    case Knight:
        return std::abs(df) == 1 && dr == -2;
    case Silver:
        return (std::abs(df) == 1 && std::abs(dr) == 1) || (df == 0 && dr == -1);
    case Gold:
    case ProPawn:
    case ProLance:
    case ProKnight:
    case ProSilver:
        return std::abs(df) <= 1 && std::abs(dr) <= 1 && !(df == 0 && dr == 0) && !(std::abs(df) == 1 && dr == 1);
    case Bishop:
        return std::abs(df) == std::abs(dr) && clearLine(df > 0 ? 1 : -1, dr > 0 ? 1 : -1);
    case Rook:
        if (df == 0 && dr != 0) return clearLine(0, dr > 0 ? 1 : -1);
        if (dr == 0 && df != 0) return clearLine(df > 0 ? 1 : -1, 0);
        return false;
    case Horse:
        if (std::abs(df) == std::abs(dr) && df != 0) return clearLine(df > 0 ? 1 : -1, dr > 0 ? 1 : -1);
        return (std::abs(df) + std::abs(dr)) == 1;
    case Dragon:
        if (df == 0 && dr != 0) return clearLine(0, dr > 0 ? 1 : -1);
        if (dr == 0 && df != 0) return clearLine(df > 0 ? 1 : -1, 0);
        return std::abs(df) == 1 && std::abs(dr) == 1;
    case King:
        return std::abs(df) <= 1 && std::abs(dr) <= 1 && !(df == 0 && dr == 0);
    default:
        return false;
    }
}

int findKing(const Board& board, Color color) {
    for (int square = 0; square < BoardSize; ++square) {
        if (board.squares[square] == makePiece(color, King)) {
            return square;
        }
    }
    return -1;
}

bool isSquareAttacked(const Board& board, int square, Color byColor) {
    for (int from = 0; from < BoardSize; ++from) {
        if (colorOf(board.squares[from]) == byColor && attacksSquare(board, from, square)) {
            return true;
        }
    }
    return false;
}

int countAttackers(const Board& board, int square, Color byColor) {
    int count = 0;
    for (int from = 0; from < BoardSize; ++from) {
        if (colorOf(board.squares[from]) == byColor && attacksSquare(board, from, square)) {
            ++count;
        }
    }
    return count;
}

namespace {

bool hasAnyLegalMove(const Board& board, bool enforcePawnDropMate) {
    return !generateLegalMoves(board, enforcePawnDropMate).empty();
}

} // namespace

std::vector<Move> generatePseudoMoves(const Board& board) {
    std::vector<Move> moves;
    const Color color = board.side;
    for (int square = 0; square < BoardSize; ++square) {
        if (colorOf(board.squares[square]) != color) {
            continue;
        }
        switch (typeOf(board.squares[square])) {
        case Pawn:
            addStep(board, moves, square, 0, -1);
            break;
        case Lance:
            addSlide(board, moves, square, 0, -1);
            break;
        case Knight:
            addStep(board, moves, square, -1, -2);
            addStep(board, moves, square, 1, -2);
            break;
        case Silver:
            addStep(board, moves, square, -1, -1);
            addStep(board, moves, square, 0, -1);
            addStep(board, moves, square, 1, -1);
            addStep(board, moves, square, -1, 1);
            addStep(board, moves, square, 1, 1);
            break;
        case Gold:
        case ProPawn:
        case ProLance:
        case ProKnight:
        case ProSilver:
            addStep(board, moves, square, -1, -1);
            addStep(board, moves, square, 0, -1);
            addStep(board, moves, square, 1, -1);
            addStep(board, moves, square, -1, 0);
            addStep(board, moves, square, 1, 0);
            addStep(board, moves, square, 0, 1);
            break;
        case Bishop:
        case Horse:
            addSlide(board, moves, square, -1, -1);
            addSlide(board, moves, square, 1, -1);
            addSlide(board, moves, square, -1, 1);
            addSlide(board, moves, square, 1, 1);
            if (typeOf(board.squares[square]) == Horse) {
                addStep(board, moves, square, 0, -1);
                addStep(board, moves, square, -1, 0);
                addStep(board, moves, square, 1, 0);
                addStep(board, moves, square, 0, 1);
            }
            break;
        case Rook:
        case Dragon:
            addSlide(board, moves, square, 0, -1);
            addSlide(board, moves, square, -1, 0);
            addSlide(board, moves, square, 1, 0);
            addSlide(board, moves, square, 0, 1);
            if (typeOf(board.squares[square]) == Dragon) {
                addStep(board, moves, square, -1, -1);
                addStep(board, moves, square, 1, -1);
                addStep(board, moves, square, -1, 1);
                addStep(board, moves, square, 1, 1);
            }
            break;
        case King:
            addStep(board, moves, square, -1, -1);
            addStep(board, moves, square, 0, -1);
            addStep(board, moves, square, 1, -1);
            addStep(board, moves, square, -1, 0);
            addStep(board, moves, square, 1, 0);
            addStep(board, moves, square, -1, 1);
            addStep(board, moves, square, 0, 1);
            addStep(board, moves, square, 1, 1);
            break;
        default:
            break;
        }
    }

    const auto& ownHand = hand(board, color);
    const std::array<PieceType, 7> dropTypes = {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook};
    for (PieceType type : dropTypes) {
        if (ownHand[type] <= 0) {
            continue;
        }
        for (int rank = 1; rank <= 9; ++rank) {
            if (!canDropOnRank(color, type, rank)) {
                continue;
            }
            for (int file = 1; file <= 9; ++file) {
                const int to = idx(file, rank);
                if (board.squares[to] != 0) {
                    continue;
                }
                if (type == Pawn && hasUnpromotedPawnOnFile(board, color, file)) {
                    continue;
                }
                moves.push_back(Move{-1, to, Empty, type, false});
            }
        }
    }
    return moves;
}

bool isKingAttacked(const Board& board, Color color) {
    const int king = findKing(board, color);
    if (king < 0) {
        return true;
    }
    return isSquareAttacked(board, king, opposite(color));
}

std::vector<Move> generateLegalMoves(const Board& board, bool enforcePawnDropMate) {
    std::vector<Move> legal;
    for (const Move& move : generatePseudoMoves(board)) {
        Board next = board;
        const Color mover = board.side;
        applyMove(next, move);
        if (isKingAttacked(next, mover)) {
            continue;
        }
        if (enforcePawnDropMate && move.isDrop() && move.drop == Pawn && isKingAttacked(next, next.side) && !hasAnyLegalMove(next, false)) {
            continue;
        }
        legal.push_back(move);
    }
    return legal;
}

bool applyIfLegal(Board& board, const Move& requested) {
    const auto legal = generateLegalMoves(board, true);
    for (const Move& move : legal) {
        if (sameMove(move, requested)) {
            applyMove(board, move);
            return true;
        }
    }
    return false;
}

} // namespace shogi
