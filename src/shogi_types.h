#pragma once

#include <array>

namespace shogi {

constexpr int BoardSize = 81;

enum Color {
    Black = 1,
    White = -1,
};

enum PieceType {
    Empty = 0,
    Pawn = 1,
    Lance = 2,
    Knight = 3,
    Silver = 4,
    Gold = 5,
    Bishop = 6,
    Rook = 7,
    King = 8,
    ProPawn = 9,
    ProLance = 10,
    ProKnight = 11,
    ProSilver = 12,
    Horse = 13,
    Dragon = 14,
};

struct Move {
    int from = -1;
    int to = -1;
    PieceType piece = Empty;
    PieceType drop = Empty;
    bool promote = false;

    bool isDrop() const { return drop != Empty; }
};

struct Board {
    std::array<int, BoardSize> squares{};
    std::array<int, 15> blackHand{};
    std::array<int, 15> whiteHand{};
    Color side = Black;
    int moveNumber = 1;
};

int idx(int file, int rank);
int fileOf(int square);
int rankOf(int square);
Color opposite(Color color);
int colorOf(int piece);
PieceType typeOf(int piece);
int makePiece(Color color, PieceType type);
std::array<int, 15>& hand(Board& board, Color color);
const std::array<int, 15>& hand(const Board& board, Color color);
PieceType unpromote(PieceType type);
PieceType promote(PieceType type);
bool canPromote(PieceType type);
bool inPromotionZone(Color color, int rank);
bool mustPromote(Color color, PieceType type, int toRank);
bool canDropOnRank(Color color, PieceType type, int rank);
bool inside(int file, int rank);

} // namespace shogi
