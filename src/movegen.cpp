#include "movegen.h"

#include "position.h"

#include <array>
#include <cmath>

namespace shogi {

namespace {

Bitboard BlackStepAttacks[PieceTypeCount][BoardSize];
Bitboard WhiteStepAttacks[PieceTypeCount][BoardSize];
bool attackTablesInitialized = false;

void addStepBit(Bitboard table[BoardSize], int from, int df, int dr) {
    const int file = fileOf(from) + df;
    const int rank = rankOf(from) + dr;
    if (inside(file, rank)) {
        table[from].set(idx(file, rank));
    }
}

void initAttackTables() {
    if (attackTablesInitialized) return;
    for (int sq = 0; sq < BoardSize; ++sq) {
        addStepBit(BlackStepAttacks[Pawn], sq, 0, -1);
        addStepBit(WhiteStepAttacks[Pawn], sq, 0, 1);
        addStepBit(BlackStepAttacks[Knight], sq, -1, -2);
        addStepBit(BlackStepAttacks[Knight], sq, 1, -2);
        addStepBit(WhiteStepAttacks[Knight], sq, -1, 2);
        addStepBit(WhiteStepAttacks[Knight], sq, 1, 2);
        for (int df = -1; df <= 1; ++df) {
            addStepBit(BlackStepAttacks[Silver], sq, df, -1);
            addStepBit(WhiteStepAttacks[Silver], sq, df, 1);
        }
        addStepBit(BlackStepAttacks[Silver], sq, -1, 1);
        addStepBit(BlackStepAttacks[Silver], sq, 1, 1);
        addStepBit(WhiteStepAttacks[Silver], sq, -1, -1);
        addStepBit(WhiteStepAttacks[Silver], sq, 1, -1);
        for (PieceType t : {Gold, ProPawn, ProLance, ProKnight, ProSilver}) {
            for (int df = -1; df <= 1; ++df) {
                addStepBit(BlackStepAttacks[t], sq, df, -1);
                addStepBit(WhiteStepAttacks[t], sq, df, 1);
            }
            addStepBit(BlackStepAttacks[t], sq, -1, 0);
            addStepBit(BlackStepAttacks[t], sq, 1, 0);
            addStepBit(WhiteStepAttacks[t], sq, -1, 0);
            addStepBit(WhiteStepAttacks[t], sq, 1, 0);
            addStepBit(BlackStepAttacks[t], sq, 0, 1);
            addStepBit(WhiteStepAttacks[t], sq, 0, -1);
        }
        for (int df = -1; df <= 1; ++df) {
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                addStepBit(BlackStepAttacks[King], sq, df, dr);
                addStepBit(WhiteStepAttacks[King], sq, df, dr);
            }
        }
        addStepBit(BlackStepAttacks[Horse], sq, 0, -1);
        addStepBit(BlackStepAttacks[Horse], sq, -1, 0);
        addStepBit(BlackStepAttacks[Horse], sq, 1, 0);
        addStepBit(BlackStepAttacks[Horse], sq, 0, 1);
        addStepBit(WhiteStepAttacks[Horse], sq, 0, -1);
        addStepBit(WhiteStepAttacks[Horse], sq, -1, 0);
        addStepBit(WhiteStepAttacks[Horse], sq, 1, 0);
        addStepBit(WhiteStepAttacks[Horse], sq, 0, 1);
        addStepBit(BlackStepAttacks[Dragon], sq, -1, -1);
        addStepBit(BlackStepAttacks[Dragon], sq, 1, -1);
        addStepBit(BlackStepAttacks[Dragon], sq, -1, 1);
        addStepBit(BlackStepAttacks[Dragon], sq, 1, 1);
        addStepBit(WhiteStepAttacks[Dragon], sq, -1, -1);
        addStepBit(WhiteStepAttacks[Dragon], sq, 1, -1);
        addStepBit(WhiteStepAttacks[Dragon], sq, -1, 1);
        addStepBit(WhiteStepAttacks[Dragon], sq, 1, 1);
    }
    attackTablesInitialized = true;
}

Bitboard slideAttack(int from, int df, int dr, const Bitboard& allOccupied) {
    Bitboard result;
    int file = fileOf(from) + df;
    int rank = rankOf(from) + dr;
    while (inside(file, rank)) {
        const int sq = idx(file, rank);
        result.set(sq);
        if (allOccupied.test(sq)) break;
        file += df;
        rank += dr;
    }
    return result;
}

Bitboard allOccupied(const Board& board) {
    return board.occupied[0] | board.occupied[1];
}

const Bitboard& stepAttacks(PieceType type, int square, Color color) {
    return color == Black ? BlackStepAttacks[type][square] : WhiteStepAttacks[type][square];
}

Bitboard attackersOf(const Board& board, int square, Color byColor) {
    initAttackTables();
    const int ci = byColor == Black ? 0 : 1;
    const Bitboard occ = allOccupied(board);
    const Bitboard colorPieces = board.occupied[ci];
    Bitboard attackers;

    const Color rev = opposite(byColor);
    attackers |= stepAttacks(Pawn, square, rev) & board.pieceBB[Pawn] & colorPieces;
    attackers |= stepAttacks(Knight, square, rev) & board.pieceBB[Knight] & colorPieces;
    attackers |= stepAttacks(Silver, square, rev) & board.pieceBB[Silver] & colorPieces;
    Bitboard goldLike = board.pieceBB[Gold] | board.pieceBB[ProPawn] | board.pieceBB[ProLance] | board.pieceBB[ProKnight] | board.pieceBB[ProSilver];
    attackers |= stepAttacks(Gold, square, rev) & goldLike & colorPieces;
    attackers |= stepAttacks(King, square, rev) & board.pieceBB[King] & colorPieces;

    {
        const int lanceDir = byColor == Black ? 1 : -1;
        Bitboard lanceLine = slideAttack(square, 0, lanceDir, occ);
        attackers |= lanceLine & board.pieceBB[Lance] & colorPieces;
    }

    Bitboard bishopLike = board.pieceBB[Bishop] | board.pieceBB[Horse];
    for (int df = -1; df <= 1; df += 2) {
        for (int dr = -1; dr <= 1; dr += 2) {
            Bitboard line = slideAttack(square, df, dr, occ);
            attackers |= line & bishopLike & colorPieces;
        }
    }
    attackers |= stepAttacks(Horse, square, rev) & board.pieceBB[Horse] & colorPieces;

    Bitboard rookLike = board.pieceBB[Rook] | board.pieceBB[Dragon];
    for (auto [df, dr] : std::array<std::pair<int,int>,4>{{{0,-1},{0,1},{-1,0},{1,0}}}) {
        Bitboard line = slideAttack(square, df, dr, occ);
        attackers |= line & rookLike & colorPieces;
    }
    attackers |= stepAttacks(Dragon, square, rev) & board.pieceBB[Dragon] & colorPieces;

    return attackers;
}

} // namespace (anonymous)

Bitboard FileMask[10];

namespace {
bool fileMaskInitialized = false;

void initFileMasks() {
    if (fileMaskInitialized) return;
    for (int f = 1; f <= 9; ++f) {
        for (int r = 1; r <= 9; ++r) {
            FileMask[f].set(idx(f, r));
        }
    }
    fileMaskInitialized = true;
}

void addMovesFromBB(MoveList& moves, Bitboard targets, int from, PieceType type, Color color) {
    while (!targets.empty()) {
        const int to = targets.lsb();
        targets.clear(to);
        const int toRank = rankOf(to);
        const bool promotable = canPromote(type) && (inPromotionZone(color, rankOf(from)) || inPromotionZone(color, toRank));
        const bool mandatory = mustPromote(color, type, toRank);
        if (promotable) {
            if (!mandatory) {
                moves.push_back(Move(from, to, type, Empty, false));
            }
            moves.push_back(Move(from, to, type, Empty, true));
        } else {
            moves.push_back(Move(from, to, type, Empty, false));
        }
    }
}

void addSlideMovesDir(MoveList& moves, int from, PieceType type, Color color, int df, int dr, const Bitboard& occ, const Bitboard& ownPieces) {
    Bitboard targets = slideAttack(from, df, dr, occ) & ~ownPieces;
    addMovesFromBB(moves, targets, from, type, color);
}

} // namespace

bool attacksSquare(const Board& board, int from, int target) {
    const int piece = board.squares[from];
    if (piece == 0) return false;
    const Color color = static_cast<Color>(colorOf(piece));
    const PieceType type = typeOf(piece);
    const int df = fileOf(target) - fileOf(from);
    const int dr = (rankOf(target) - rankOf(from)) * static_cast<int>(color);

    auto clearLine = [&](int stepFile, int stepRank) {
        int file = fileOf(from) + stepFile;
        int rank = rankOf(from) + stepRank * static_cast<int>(color);
        while (inside(file, rank)) {
            const int square = idx(file, rank);
            if (square == target) return true;
            if (board.squares[square] != 0) return false;
            file += stepFile;
            rank += stepRank * static_cast<int>(color);
        }
        return false;
    };

    switch (type) {
    case Pawn:   return df == 0 && dr == -1;
    case Lance:  return df == 0 && dr < 0 && clearLine(0, -1);
    case Knight: return std::abs(df) == 1 && dr == -2;
    case Silver: return (std::abs(df) == 1 && std::abs(dr) == 1) || (df == 0 && dr == -1);
    case Gold: case ProPawn: case ProLance: case ProKnight: case ProSilver:
        return std::abs(df) <= 1 && std::abs(dr) <= 1 && !(df == 0 && dr == 0) && !(std::abs(df) == 1 && dr == 1);
    case Bishop: return std::abs(df) == std::abs(dr) && clearLine(df > 0 ? 1 : -1, dr > 0 ? 1 : -1);
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
    case King: return std::abs(df) <= 1 && std::abs(dr) <= 1 && !(df == 0 && dr == 0);
    default: return false;
    }
}

int findKing(const Board& board, Color color) {
    return color == Black ? board.blackKingSquare : board.whiteKingSquare;
}

bool isSquareAttacked(const Board& board, int square, Color byColor) {
    return !attackersOf(board, square, byColor).empty();
}

int countAttackers(const Board& board, int square, Color byColor) {
    return attackersOf(board, square, byColor).popcount();
}

bool isKingAttacked(const Board& board, Color color) {
    const int king = findKing(board, color);
    if (king < 0) return true;
    return isSquareAttacked(board, king, opposite(color));
}

bool givesCheck(const Board& board, const Move& move) {
    Board tmp = board;
    applyMove(tmp, move);
    return isKingAttacked(tmp, tmp.side);
}

MoveList generatePseudoMoves(const Board& board) {
    initAttackTables();
    initFileMasks();
    MoveList moves;
    const Color color = board.side;
    const int ci = color == Black ? 0 : 1;
    const Bitboard ownPieces = board.occupied[ci];
    const Bitboard occ = allOccupied(board);

    auto genStep = [&](PieceType type) {
        Bitboard pieces = board.pieceBB[type] & ownPieces;
        while (!pieces.empty()) {
            const int from = pieces.lsb();
            pieces.clear(from);
            Bitboard targets = stepAttacks(type, from, color) & ~ownPieces;
            addMovesFromBB(moves, targets, from, type, color);
        }
    };
    genStep(Pawn);
    genStep(Knight);
    genStep(Silver);
    genStep(Gold);
    genStep(ProPawn);
    genStep(ProLance);
    genStep(ProKnight);
    genStep(ProSilver);
    genStep(King);

    {
        Bitboard lances = board.pieceBB[Lance] & ownPieces;
        const int dir = color == Black ? -1 : 1;
        while (!lances.empty()) {
            const int from = lances.lsb();
            lances.clear(from);
            addSlideMovesDir(moves, from, Lance, color, 0, dir, occ, ownPieces);
        }
    }

    {
        Bitboard bishops = board.pieceBB[Bishop] & ownPieces;
        while (!bishops.empty()) {
            const int from = bishops.lsb();
            bishops.clear(from);
            for (int df = -1; df <= 1; df += 2)
                for (int dr = -1; dr <= 1; dr += 2)
                    addSlideMovesDir(moves, from, Bishop, color, df, dr, occ, ownPieces);
        }
    }

    {
        Bitboard rooks = board.pieceBB[Rook] & ownPieces;
        while (!rooks.empty()) {
            const int from = rooks.lsb();
            rooks.clear(from);
            for (auto [df, dr] : std::array<std::pair<int,int>,4>{{{0,-1},{0,1},{-1,0},{1,0}}})
                addSlideMovesDir(moves, from, Rook, color, df, dr, occ, ownPieces);
        }
    }

    {
        Bitboard horses = board.pieceBB[Horse] & ownPieces;
        while (!horses.empty()) {
            const int from = horses.lsb();
            horses.clear(from);
            for (int df = -1; df <= 1; df += 2)
                for (int dr = -1; dr <= 1; dr += 2)
                    addSlideMovesDir(moves, from, Horse, color, df, dr, occ, ownPieces);
            Bitboard stepTargets = stepAttacks(Horse, from, color) & ~ownPieces;
            addMovesFromBB(moves, stepTargets, from, Horse, color);
        }
    }

    {
        Bitboard dragons = board.pieceBB[Dragon] & ownPieces;
        while (!dragons.empty()) {
            const int from = dragons.lsb();
            dragons.clear(from);
            for (auto [df, dr] : std::array<std::pair<int,int>,4>{{{0,-1},{0,1},{-1,0},{1,0}}})
                addSlideMovesDir(moves, from, Dragon, color, df, dr, occ, ownPieces);
            Bitboard stepTargets = stepAttacks(Dragon, from, color) & ~ownPieces;
            addMovesFromBB(moves, stepTargets, from, Dragon, color);
        }
    }

    const Bitboard emptySquares = ~occ;
    const auto& ownHand = hand(board, color);
    for (PieceType type : {Pawn, Lance, Knight, Silver, Gold, Bishop, Rook}) {
        if (ownHand[type] <= 0) continue;
        Bitboard dropTargets = emptySquares;
        if (type == Pawn || type == Lance) {
            const int forbiddenRank = color == Black ? 1 : 9;
            for (int f = 1; f <= 9; ++f)
                dropTargets.clear(idx(f, forbiddenRank));
        }
        if (type == Knight) {
            const int r1 = color == Black ? 1 : 9;
            const int r2 = color == Black ? 2 : 8;
            for (int f = 1; f <= 9; ++f) {
                dropTargets.clear(idx(f, r1));
                dropTargets.clear(idx(f, r2));
            }
        }
        if (type == Pawn) {
            Bitboard ownPawns = board.pieceBB[Pawn] & ownPieces;
            for (int f = 1; f <= 9; ++f) {
                if (!(ownPawns & FileMask[f]).empty())
                    dropTargets &= ~FileMask[f];
            }
        }
        while (!dropTargets.empty()) {
            const int to = dropTargets.lsb();
            dropTargets.clear(to);
            moves.push_back(Move(-1, to, Empty, type, false));
        }
    }
    return moves;
}

MoveList generateLegalMoves(const Board& board, bool enforcePawnDropMate) {

    Board temp = board;
    const Color mover = board.side;
    MoveList pseudo = generatePseudoMoves(board);
    MoveList legal;
    for (const Move& move : pseudo) {
        UndoInfo undo;
        applyMove(temp, move, undo);
        if (isKingAttacked(temp, mover)) {
            undoMove(temp, move, undo);
            continue;
        }
        if (enforcePawnDropMate && move.isDrop() && move.drop == Pawn
            && isKingAttacked(temp, temp.side)
            && generateLegalMoves(temp, false).empty()) {
            undoMove(temp, move, undo);
            continue;
        }
        undoMove(temp, move, undo);
        legal.push_back(move);
    }
    return legal;
}

bool applyIfLegal(Board& board, const Move& requested) {
    MoveList legal = generateLegalMoves(board, true);
    for (const Move& move : legal) {
        if (sameMove(move, requested)) {
            applyMove(board, move);
            return true;
        }
    }
    return false;
}

} // namespace shogi
