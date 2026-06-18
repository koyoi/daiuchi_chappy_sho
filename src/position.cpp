#include "position.h"

#include "movegen.h"
#include "notation.h"

#include <algorithm>
#include <cctype>

namespace shogi {

Board startpos() {
    Board board;
    const std::string sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL";
    int rank = 1;
    int file = 9;
    bool promoted = false;
    for (char c : sfen) {
        if (c == '/') {
            ++rank;
            file = 9;
            promoted = false;
            continue;
        }
        if (c == '+') {
            promoted = true;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file -= c - '0';
            promoted = false;
            continue;
        }
        PieceType type = pieceFromSfen(c);
        if (promoted) {
            type = promote(type);
        }
        const Color color = std::isupper(static_cast<unsigned char>(c)) ? Black : White;
        board.squares[idx(file, rank)] = makePiece(color, type);
        --file;
        promoted = false;
    }
    board.side = Black;
    board.moveNumber = 1;
    return board;
}

bool setFromSfen(Board& board, const std::string& boardPart, const std::string& sidePart, const std::string& handPart, const std::string& movePart) {
    board = Board{};
    int rank = 1;
    int file = 9;
    bool promoted = false;
    for (char c : boardPart) {
        if (c == '/') {
            if (file != 0) {
                return false;
            }
            ++rank;
            file = 9;
            promoted = false;
            continue;
        }
        if (c == '+') {
            promoted = true;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file -= c - '0';
            promoted = false;
            continue;
        }
        if (!inside(file, rank)) {
            return false;
        }
        PieceType type = pieceFromSfen(c);
        if (type == Empty) {
            return false;
        }
        if (promoted) {
            type = promote(type);
        }
        const Color color = std::isupper(static_cast<unsigned char>(c)) ? Black : White;
        board.squares[idx(file, rank)] = makePiece(color, type);
        --file;
        promoted = false;
    }
    board.side = sidePart == "w" ? White : Black;

    if (handPart != "-") {
        int count = 0;
        for (char c : handPart) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                count = count * 10 + (c - '0');
                continue;
            }
            PieceType type = pieceFromSfen(c);
            if (type == Empty || type == King) {
                return false;
            }
            const Color color = std::isupper(static_cast<unsigned char>(c)) ? Black : White;
            hand(board, color)[type] += count == 0 ? 1 : count;
            count = 0;
        }
    }

    try {
        board.moveNumber = std::max(1, std::stoi(movePart));
    } catch (...) {
        board.moveNumber = 1;
    }
    return true;
}

void applyMove(Board& board, const Move& move) {
    Color color = board.side;
    if (move.isDrop()) {
        board.squares[move.to] = makePiece(color, move.drop);
        --hand(board, color)[move.drop];
    } else {
        const int moving = board.squares[move.from];
        const int captured = board.squares[move.to];
        PieceType type = typeOf(moving);
        if (captured != 0) {
            ++hand(board, color)[unpromote(typeOf(captured))];
        }
        if (move.promote) {
            type = promote(type);
        }
        board.squares[move.from] = 0;
        board.squares[move.to] = makePiece(color, type);
    }
    board.side = opposite(board.side);
    ++board.moveNumber;
}

bool sameMove(const Move& left, const Move& right) {
    return left.from == right.from && left.to == right.to && left.drop == right.drop && left.promote == right.promote;
}

bool setPosition(Board& board, const std::vector<std::string>& words) {
    if (words.size() < 2 || words[0] != "position") {
        return false;
    }
    std::size_t index = 1;
    if (words[index] == "startpos") {
        board = startpos();
        ++index;
    } else if (words[index] == "sfen" && words.size() >= index + 5) {
        if (!setFromSfen(board, words[index + 1], words[index + 2], words[index + 3], words[index + 4])) {
            return false;
        }
        index += 5;
    } else {
        return false;
    }
    if (index < words.size() && words[index] == "moves") {
        ++index;
        for (; index < words.size(); ++index) {
            const Move move = parseUsiMove(board, words[index]);
            applyIfLegal(board, move);
        }
    }
    return true;
}

} // namespace shogi
