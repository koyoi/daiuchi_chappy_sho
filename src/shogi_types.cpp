#include "shogi_types.h"

#include <cmath>

namespace shogi {

int idx(int file, int rank) {
    return (rank - 1) * 9 + (file - 1);
}

int fileOf(int square) {
    return square % 9 + 1;
}

int rankOf(int square) {
    return square / 9 + 1;
}

Color opposite(Color color) {
    return color == Black ? White : Black;
}

int colorOf(int piece) {
    if (piece > 0) {
        return Black;
    }
    if (piece < 0) {
        return White;
    }
    return 0;
}

PieceType typeOf(int piece) {
    return static_cast<PieceType>(std::abs(piece));
}

int makePiece(Color color, PieceType type) {
    return static_cast<int>(type) * static_cast<int>(color);
}

std::array<int, 15>& hand(Board& board, Color color) {
    return color == Black ? board.blackHand : board.whiteHand;
}

const std::array<int, 15>& hand(const Board& board, Color color) {
    return color == Black ? board.blackHand : board.whiteHand;
}

PieceType unpromote(PieceType type) {
    switch (type) {
    case ProPawn:
        return Pawn;
    case ProLance:
        return Lance;
    case ProKnight:
        return Knight;
    case ProSilver:
        return Silver;
    case Horse:
        return Bishop;
    case Dragon:
        return Rook;
    default:
        return type;
    }
}

PieceType promote(PieceType type) {
    switch (type) {
    case Pawn:
        return ProPawn;
    case Lance:
        return ProLance;
    case Knight:
        return ProKnight;
    case Silver:
        return ProSilver;
    case Bishop:
        return Horse;
    case Rook:
        return Dragon;
    default:
        return type;
    }
}

bool canPromote(PieceType type) {
    return type == Pawn || type == Lance || type == Knight || type == Silver || type == Bishop || type == Rook;
}

bool inPromotionZone(Color color, int rank) {
    return color == Black ? rank <= 3 : rank >= 7;
}

bool mustPromote(Color color, PieceType type, int toRank) {
    if (type == Pawn || type == Lance) {
        return color == Black ? toRank == 1 : toRank == 9;
    }
    if (type == Knight) {
        return color == Black ? toRank <= 2 : toRank >= 8;
    }
    return false;
}

bool canDropOnRank(Color color, PieceType type, int rank) {
    if (type == Pawn || type == Lance) {
        return !(color == Black ? rank == 1 : rank == 9);
    }
    if (type == Knight) {
        return !(color == Black ? rank <= 2 : rank >= 8);
    }
    return true;
}

bool inside(int file, int rank) {
    return file >= 1 && file <= 9 && rank >= 1 && rank <= 9;
}

} // namespace shogi
