#include "notation.h"

#include <cctype>

namespace shogi {

char usiRank(int rank) {
    return static_cast<char>('a' + rank - 1);
}

int parseUsiRank(char c) {
    return c - 'a' + 1;
}

PieceType pieceFromSfen(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
    case 'P':
        return Pawn;
    case 'L':
        return Lance;
    case 'N':
        return Knight;
    case 'S':
        return Silver;
    case 'G':
        return Gold;
    case 'B':
        return Bishop;
    case 'R':
        return Rook;
    case 'K':
        return King;
    default:
        return Empty;
    }
}

char sfenFromPiece(PieceType type) {
    switch (type) {
    case Pawn:
        return 'P';
    case Lance:
        return 'L';
    case Knight:
        return 'N';
    case Silver:
        return 'S';
    case Gold:
        return 'G';
    case Bishop:
        return 'B';
    case Rook:
        return 'R';
    case King:
        return 'K';
    default:
        return '?';
    }
}

std::string usiSquare(int square) {
    std::string result;
    result += static_cast<char>('0' + fileOf(square));
    result += usiRank(rankOf(square));
    return result;
}

PieceType pieceFromUsiDrop(char c) {
    return pieceFromSfen(c);
}

std::string toUsi(const Move& move) {
    std::string result;
    if (move.isDrop()) {
        result += sfenFromPiece(move.drop);
        result += '*';
        result += usiSquare(move.to);
    } else {
        result += usiSquare(move.from);
        result += usiSquare(move.to);
        if (move.promote) {
            result += '+';
        }
    }
    return result;
}

Move parseUsiMove(const Board& board, const std::string& text) {
    Move move;
    if (text.size() < 4) {
        return move;
    }
    if (text[1] == '*') {
        move.drop = pieceFromUsiDrop(text[0]);
        const int toFile = text[2] - '0';
        const int toRank = parseUsiRank(text[3]);
        if (inside(toFile, toRank)) {
            move.to = static_cast<std::int8_t>(idx(toFile, toRank));
        }
        return move;
    }
    const int fromFile = text[0] - '0';
    const int fromRank = parseUsiRank(text[1]);
    const int toFile = text[2] - '0';
    const int toRank = parseUsiRank(text[3]);
    if (inside(fromFile, fromRank) && inside(toFile, toRank)) {
        move.from = static_cast<std::int8_t>(idx(fromFile, fromRank));
        move.to = static_cast<std::int8_t>(idx(toFile, toRank));
        move.piece = typeOf(board.squares[move.from]);
        move.promote = text.size() >= 5 && text[4] == '+';
    }
    return move;
}

std::string csaPiece(PieceType type) {
    switch (type) {
    case Pawn:
        return "FU";
    case Lance:
        return "KY";
    case Knight:
        return "KE";
    case Silver:
        return "GI";
    case Gold:
        return "KI";
    case Bishop:
        return "KA";
    case Rook:
        return "HI";
    case King:
        return "OU";
    case ProPawn:
        return "TO";
    case ProLance:
        return "NY";
    case ProKnight:
        return "NK";
    case ProSilver:
        return "NG";
    case Horse:
        return "UM";
    case Dragon:
        return "RY";
    default:
        return "??";
    }
}

PieceType pieceFromCsa(const std::string& code) {
    if (code == "FU") return Pawn;
    if (code == "KY") return Lance;
    if (code == "KE") return Knight;
    if (code == "GI") return Silver;
    if (code == "KI") return Gold;
    if (code == "KA") return Bishop;
    if (code == "HI") return Rook;
    if (code == "OU") return King;
    if (code == "TO") return ProPawn;
    if (code == "NY") return ProLance;
    if (code == "NK") return ProKnight;
    if (code == "NG") return ProSilver;
    if (code == "UM") return Horse;
    if (code == "RY") return Dragon;
    return Empty;
}

std::string toCsa(const Board& board, const Move& move, Color color) {
    std::string result;
    result += color == Black ? '+' : '-';
    if (move.isDrop()) {
        result += "00";
        result += static_cast<char>('0' + fileOf(move.to));
        result += static_cast<char>('0' + rankOf(move.to));
        result += csaPiece(move.drop);
        return result;
    }
    result += static_cast<char>('0' + fileOf(move.from));
    result += static_cast<char>('0' + rankOf(move.from));
    result += static_cast<char>('0' + fileOf(move.to));
    result += static_cast<char>('0' + rankOf(move.to));
    PieceType type = typeOf(board.squares[move.from]);
    if (move.promote) {
        type = promote(type);
    }
    result += csaPiece(type);
    return result;
}

Move parseCsaMove(const Board& board, const std::string& text) {
    Move move;
    if (text.size() < 7 || (text[0] != '+' && text[0] != '-')) {
        return move;
    }
    const int fromFile = text[1] - '0';
    const int fromRank = text[2] - '0';
    const int toFile = text[3] - '0';
    const int toRank = text[4] - '0';
    const PieceType csaType = pieceFromCsa(text.substr(5, 2));
    if (!inside(toFile, toRank) || csaType == Empty) {
        return move;
    }
    move.to = static_cast<std::int8_t>(idx(toFile, toRank));
    if (fromFile == 0 && fromRank == 0) {
        move.drop = csaType;
        return move;
    }
    if (!inside(fromFile, fromRank)) {
        return Move{};
    }
    move.from = static_cast<std::int8_t>(idx(fromFile, fromRank));
    const PieceType current = typeOf(board.squares[move.from]);
    move.piece = current;
    move.promote = csaType != current;
    return move;
}

} // namespace shogi
