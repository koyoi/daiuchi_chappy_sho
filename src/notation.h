#pragma once

#include "shogi_types.h"

#include <string>

namespace shogi {

char usiRank(int rank);
int parseUsiRank(char c);
PieceType pieceFromSfen(char c);
char sfenFromPiece(PieceType type);
std::string usiSquare(int square);
PieceType pieceFromUsiDrop(char c);
std::string toUsi(const Move& move);
Move parseUsiMove(const Board& board, const std::string& text);
std::string csaPiece(PieceType type);
PieceType pieceFromCsa(const std::string& code);
std::string toCsa(const Board& board, const Move& move, Color color);
Move parseCsaMove(const Board& board, const std::string& text);

} // namespace shogi
