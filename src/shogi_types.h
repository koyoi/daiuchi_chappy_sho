#pragma once

#include <array>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace shogi {

constexpr int BoardSize = 81;
constexpr int PieceTypeCount = 15;

struct Bitboard {
    std::uint64_t lo = 0; // ビット0〜63 → マス0〜63
    std::uint32_t hi = 0; // ビット0〜16 → マス64〜80

    Bitboard() = default;
    Bitboard(std::uint64_t l, std::uint32_t h) : lo(l), hi(h) {}

    bool test(int sq) const {
        return sq < 64 ? (lo >> sq) & 1 : (hi >> (sq - 64)) & 1;
    }
    void set(int sq) {
        if (sq < 64) lo |= 1ULL << sq; else hi |= 1U << (sq - 64);
    }
    void clear(int sq) {
        if (sq < 64) lo &= ~(1ULL << sq); else hi &= ~(1U << (sq - 64));
    }
    bool empty() const { return lo == 0 && hi == 0; }
    Bitboard operator&(const Bitboard& o) const { return {lo & o.lo, hi & o.hi}; }
    Bitboard operator|(const Bitboard& o) const { return {lo | o.lo, hi | o.hi}; }
    Bitboard operator^(const Bitboard& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
    Bitboard operator~() const { return {~lo, (~hi) & 0x1FFFFU}; }
    Bitboard& operator|=(const Bitboard& o) { lo |= o.lo; hi |= o.hi; return *this; }
    Bitboard& operator&=(const Bitboard& o) { lo &= o.lo; hi &= o.hi; return *this; }
    Bitboard& operator^=(const Bitboard& o) { lo ^= o.lo; hi ^= o.hi; return *this; }
    bool operator==(const Bitboard& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const Bitboard& o) const { return !(*this == o); }

    inline int popcount() const {
#if defined(_MSC_VER)
        return static_cast<int>(__popcnt64(lo)) + static_cast<int>(__popcnt(hi));
#else
        return __builtin_popcountll(lo) + __builtin_popcount(hi);
#endif
    }

    inline int lsb() const {
        if (lo != 0) {
#if defined(_MSC_VER)
            unsigned long index;
            _BitScanForward64(&index, lo);
            return static_cast<int>(index);
#else
            return __builtin_ctzll(lo);
#endif
        }
#if defined(_MSC_VER)
        unsigned long index;
        _BitScanForward(&index, hi);
        return static_cast<int>(index) + 64;
#else
        return __builtin_ctz(hi) + 64;
#endif
    }

    inline Bitboard popLsb();
};

inline Bitboard squareBB(int sq) {
    Bitboard bb;
    bb.set(sq);
    return bb;
}

inline Bitboard Bitboard::popLsb() {
    const int sq = lsb();
    clear(sq);
    return squareBB(sq);
}

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
    int blackKingSquare = -1;
    int whiteKingSquare = -1;
    std::uint64_t hash = 0;
    int materialScore = 0;                         // 先手視点の駒割り(差分更新)
    Bitboard occupied[2]{};                        // [0]=先手, [1]=後手
    std::array<Bitboard, PieceTypeCount> pieceBB{}; // 駒種別Bitboard
};

inline int idx(int file, int rank) { return (rank - 1) * 9 + (file - 1); }
inline int fileOf(int square) { return square % 9 + 1; }
inline int rankOf(int square) { return square / 9 + 1; }
inline Color opposite(Color color) { return color == Black ? White : Black; }
inline int colorOf(int piece) { return piece > 0 ? Black : (piece < 0 ? White : 0); }

inline PieceType typeOf(int piece) {
    return static_cast<PieceType>(piece < 0 ? -piece : piece);
}

inline int makePiece(Color color, PieceType type) {
    return static_cast<int>(type) * static_cast<int>(color);
}

inline std::array<int, 15>& hand(Board& board, Color color) {
    return color == Black ? board.blackHand : board.whiteHand;
}

inline const std::array<int, 15>& hand(const Board& board, Color color) {
    return color == Black ? board.blackHand : board.whiteHand;
}

inline bool canPromote(PieceType type) {
    return type >= Pawn && type <= Rook && type != Gold && type != King;
}

inline bool inPromotionZone(Color color, int rank) {
    return color == Black ? rank <= 3 : rank >= 7;
}

inline bool inside(int file, int rank) {
    return static_cast<unsigned>(file - 1) < 9u && static_cast<unsigned>(rank - 1) < 9u;
}

PieceType unpromote(PieceType type);
PieceType promote(PieceType type);
bool mustPromote(Color color, PieceType type, int toRank);
bool canDropOnRank(Color color, PieceType type, int rank);

namespace zobrist {
void init();
std::uint64_t pieceKey(int square, int piece);
std::uint64_t handKey(int colorIndex, int pieceType, int count);
std::uint64_t sideKey();
} // namespace zobrist

void initBoardHash(Board& board);
void initKingSquares(Board& board);
void initBitboards(Board& board);
void initMaterialScore(Board& board);
int materialPieceValue(PieceType type);

} // namespace shogi
