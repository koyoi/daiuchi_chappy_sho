#pragma once

#include "shogi_types.h"

namespace shogi {

MoveList generatePseudoMoves(const Board& board);
MoveList generateLegalMoves(const Board& board, bool enforcePawnDropMate = true);
bool applyIfLegal(Board& board, const Move& requested);
bool isKingAttacked(const Board& board, Color color);
bool attacksSquare(const Board& board, int from, int target);
bool isSquareAttacked(const Board& board, int square, Color byColor);
int countAttackers(const Board& board, int square, Color byColor);
int findKing(const Board& board, Color color);
bool givesCheck(const Board& board, const Move& move);

extern Bitboard FileMask[10];

} // namespace shogi
