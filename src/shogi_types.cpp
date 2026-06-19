#include "shogi_types.h"

#include <cstdint>
#include <random>

namespace shogi {

namespace zobrist {
namespace {
// piece値は -14〜+14 の範囲。オフセット15で 1〜29 にマッピング
constexpr int PieceOffset = 15;
constexpr int PieceRange = 31;
constexpr int MaxHandCount = 19;  // 歩は最大18枚
constexpr int HandPieceTypes = 15;

std::uint64_t pieceTable[BoardSize][PieceRange];
std::uint64_t handTable[2][HandPieceTypes][MaxHandCount];
std::uint64_t sideValue;
bool initialized = false;
} // namespace

void init() {
    if (initialized) return;
    std::mt19937_64 rng(0xDEADBEEF12345678ULL);
    for (int sq = 0; sq < BoardSize; ++sq) {
        for (int p = 0; p < PieceRange; ++p) {
            pieceTable[sq][p] = rng();
        }
    }
    for (int c = 0; c < 2; ++c) {
        for (int t = 0; t < HandPieceTypes; ++t) {
            for (int n = 0; n < MaxHandCount; ++n) {
                handTable[c][t][n] = rng();
            }
        }
    }
    sideValue = rng();
    initialized = true;
}

std::uint64_t pieceKey(int square, int piece) {
    return pieceTable[square][piece + PieceOffset];
}

std::uint64_t handKey(int colorIndex, int pieceType, int count) {
    return handTable[colorIndex][pieceType][count];
}

std::uint64_t sideKey() {
    return sideValue;
}
} // namespace zobrist

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

void initKingSquares(Board& board) {
    board.blackKingSquare = -1;
    board.whiteKingSquare = -1;
    for (int sq = 0; sq < BoardSize; ++sq) {
        if (board.squares[sq] == makePiece(Black, King)) {
            board.blackKingSquare = sq;
        } else if (board.squares[sq] == makePiece(White, King)) {
            board.whiteKingSquare = sq;
        }
    }
}

void initBoardHash(Board& board) {
    zobrist::init();
    std::uint64_t h = 0;
    for (int sq = 0; sq < BoardSize; ++sq) {
        if (board.squares[sq] != 0) {
            h ^= zobrist::pieceKey(sq, board.squares[sq]);
        }
    }
    for (int t = 0; t < 15; ++t) {
        if (board.blackHand[t] > 0) {
            h ^= zobrist::handKey(0, t, board.blackHand[t]);
        }
        if (board.whiteHand[t] > 0) {
            h ^= zobrist::handKey(1, t, board.whiteHand[t]);
        }
    }
    if (board.side == White) {
        h ^= zobrist::sideKey();
    }
    board.hash = h;
}

int materialPieceValue(PieceType type) {
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

void initMaterialScore(Board& board) {
    int score = 0;
    for (int sq = 0; sq < BoardSize; ++sq) {
        const int piece = board.squares[sq];
        if (piece == 0 || typeOf(piece) == King) continue;
        const int value = materialPieceValue(typeOf(piece));
        score += colorOf(piece) == Black ? value : -value;
    }
    for (PieceType t : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        score += board.blackHand[t] * materialPieceValue(t);
        score -= board.whiteHand[t] * materialPieceValue(t);
    }
    board.materialScore = score;
}

void initBitboards(Board& board) {
    for (int i = 0; i < 2; ++i) board.occupied[i] = Bitboard{};
    for (int i = 0; i < PieceTypeCount; ++i) board.pieceBB[i] = Bitboard{};
    for (int sq = 0; sq < BoardSize; ++sq) {
        const int piece = board.squares[sq];
        if (piece == 0) continue;
        const int ci = colorOf(piece) == Black ? 0 : 1;
        board.occupied[ci].set(sq);
        board.pieceBB[typeOf(piece)].set(sq);
    }
}

} // namespace shogi
