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

} // namespace (anonymous)

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

namespace {

Bitboard allAttackersOfOcc(const Board& board, int square, const Bitboard& occ) {
    initAttackTables();
    Bitboard attackers;
    for (int ci = 0; ci < 2; ++ci) {
        Color c = ci == 0 ? Black : White;
        Color rev = opposite(c);
        Bitboard cp = board.occupied[ci];
        attackers |= stepAttacks(Pawn, square, rev) & board.pieceBB[Pawn] & cp;
        attackers |= stepAttacks(Knight, square, rev) & board.pieceBB[Knight] & cp;
        attackers |= stepAttacks(Silver, square, rev) & board.pieceBB[Silver] & cp;
        Bitboard goldLike = board.pieceBB[Gold] | board.pieceBB[ProPawn] | board.pieceBB[ProLance]
                          | board.pieceBB[ProKnight] | board.pieceBB[ProSilver];
        attackers |= stepAttacks(Gold, square, rev) & goldLike & cp;
        attackers |= stepAttacks(King, square, rev) & board.pieceBB[King] & cp;
        attackers |= stepAttacks(Horse, square, rev) & board.pieceBB[Horse] & cp;
        attackers |= stepAttacks(Dragon, square, rev) & board.pieceBB[Dragon] & cp;
    }
    attackers |= slideAttack(square, 0, 1, occ) & board.pieceBB[Lance] & board.occupied[0];
    attackers |= slideAttack(square, 0, -1, occ) & board.pieceBB[Lance] & board.occupied[1];
    Bitboard bishopLike = board.pieceBB[Bishop] | board.pieceBB[Horse];
    for (int df = -1; df <= 1; df += 2)
        for (int dr = -1; dr <= 1; dr += 2)
            attackers |= slideAttack(square, df, dr, occ) & bishopLike;
    Bitboard rookLike = board.pieceBB[Rook] | board.pieceBB[Dragon];
    for (auto [df, dr] : std::array<std::pair<int,int>,4>{{{0,-1},{0,1},{-1,0},{1,0}}})
        attackers |= slideAttack(square, df, dr, occ) & rookLike;
    return attackers & occ;
}

int seeValue(PieceType type) {
    static constexpr int V[] = {0,100,300,320,520,620,850,1050,10000,560,560,560,560,1150,1350};
    return (type >= 1 && type <= 14) ? V[type] : 0;
}

} // anonymous namespace

int staticExchangeEval(const Board& board, const Move& move) {
    if (move.isDrop()) return 0;
    PieceType victim = typeOf(board.squares[move.to]);
    if (victim == 0 && !move.promote) return 0;

    int gain[32];
    int d = 0;
    gain[0] = victim != 0 ? seeValue(victim) : 0;
    PieceType attacker = typeOf(board.squares[move.from]);
    if (move.promote) {
        PieceType prom = promote(attacker);
        gain[0] += seeValue(prom) - seeValue(attacker);
        attacker = prom;
    }

    Bitboard occ = board.occupied[0] | board.occupied[1];
    occ.clear(move.from);
    Bitboard attackers = allAttackersOfOcc(board, move.to, occ);
    Color side = opposite(board.side);

    static const PieceType ptOrder[] = {
        Pawn, Lance, Knight, Silver, ProPawn, ProLance, ProKnight, ProSilver,
        Gold, Bishop, Rook, Horse, Dragon, King
    };

    while (d < 30) {
        d++;
        gain[d] = seeValue(attacker) - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;

        int ci = side == Black ? 0 : 1;
        Bitboard sideAtt = attackers & board.occupied[ci];
        if (sideAtt.empty()) break;

        PieceType nextPt = King;
        int nextSq = -1;
        for (PieceType pt : ptOrder) {
            Bitboard cands = sideAtt & board.pieceBB[pt];
            if (!cands.empty()) { nextPt = pt; nextSq = cands.lsb(); break; }
        }

        if (nextPt == King) {
            Bitboard opAtt = attackers & board.occupied[1 - ci];
            if (!opAtt.empty()) break;
        }

        occ.clear(nextSq);
        attackers = allAttackersOfOcc(board, move.to, occ);
        attacker = nextPt;
        side = opposite(side);
    }

    while (--d) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

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
