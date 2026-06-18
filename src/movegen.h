#pragma once

#include "shogi_types.h"

#include <vector>

namespace shogi {

std::vector<Move> generatePseudoMoves(const Board& board);
std::vector<Move> generateLegalMoves(const Board& board, bool enforcePawnDropMate = true);
bool applyIfLegal(Board& board, const Move& requested);
bool isKingAttacked(const Board& board, Color color);
bool attacksSquare(const Board& board, int from, int target);
bool isSquareAttacked(const Board& board, int square, Color byColor);
int countAttackers(const Board& board, int square, Color byColor);
int findKing(const Board& board, Color color);

extern Bitboard FileMask[10];

} // namespace shogi
