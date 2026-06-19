#pragma once

#include "shogi_types.h"

#include <string>
#include <vector>

namespace shogi {

Board startpos();
bool setFromSfen(Board& board, const std::string& boardPart, const std::string& sidePart, const std::string& handPart, const std::string& movePart);
bool setPosition(Board& board, const std::vector<std::string>& words);
void applyMove(Board& board, const Move& move);
void applyMove(Board& board, const Move& move, UndoInfo& undo);
void undoMove(Board& board, const Move& move, const UndoInfo& undo);
bool sameMove(const Move& left, const Move& right);

} // namespace shogi
