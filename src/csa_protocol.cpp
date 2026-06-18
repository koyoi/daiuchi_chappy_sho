#include "csa_protocol.h"

#include "engine.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <cctype>
#include <iostream>
#include <string>

namespace shogi {

namespace {

void parseCsaBoardLine(Board& board, const std::string& line) {
    if (line.size() < 29 || line[0] != 'P' || !std::isdigit(static_cast<unsigned char>(line[1]))) {
        return;
    }
    const int rank = line[1] - '0';
    if (rank < 1 || rank > 9) {
        return;
    }
    if (rank == 1) {
        board.squares.fill(0);
        board.blackHand.fill(0);
        board.whiteHand.fill(0);
    }
    for (int cell = 0; cell < 9; ++cell) {
        const int pos = 2 + cell * 3;
        const std::string token = line.substr(pos, 3);
        const int file = 9 - cell;
        if (token[0] == '+' || token[0] == '-') {
            const Color color = token[0] == '+' ? Black : White;
            const PieceType type = pieceFromCsa(token.substr(1, 2));
            board.squares[idx(file, rank)] = makePiece(color, type);
        } else {
            board.squares[idx(file, rank)] = 0;
        }
    }
}

void parseCsaHandLine(Board& board, const std::string& line) {
    if (line.size() < 5 || line[0] != 'P' || (line[1] != '+' && line[1] != '-')) {
        return;
    }
    const Color color = line[1] == '+' ? Black : White;
    auto& ownHand = hand(board, color);
    ownHand.fill(0);
    for (std::size_t pos = 2; pos + 3 < line.size(); pos += 4) {
        if (line.substr(pos, 2) != "00") {
            continue;
        }
        const PieceType type = pieceFromCsa(line.substr(pos + 2, 2));
        if (type != Empty && type != King) {
            ++ownHand[type];
        }
    }
}

void maybePlayCsa(Board& board, Color engineSide, LearningEngine& engine) {
    if (board.side != engineSide) {
        return;
    }
    const Move move = engine.chooseMove(board);
    if (move.to < 0) {
        std::cout << "%TORYO" << std::endl;
        return;
    }
    engine.recordMove(board, move, true);
    std::cout << toCsa(board, move, board.side) << std::endl;
    applyMove(board, move);
}

int csaGameResultForEngine(const std::string& line) {
    if (line == "#WIN") {
        return 1;
    }
    if (line == "#LOSE") {
        return -1;
    }
    if (line == "#DRAW" || line == "#CHUDAN" || line == "#SENNICHITE" || line == "#JISHOGI") {
        return 0;
    }
    if (line == "%TORYO") {
        return 1;
    }
    return 0;
}

} // namespace

void csaLoop() {
    Board board = startpos();
    Color engineSide = Black;
    LearningEngine engine;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (line.rfind("LOGIN", 0) == 0) {
            const auto words = splitWords(line);
            const std::string user = words.size() >= 2 ? words[1] : "user";
            std::cout << "LOGIN:" << user << " OK" << std::endl;
        } else if (line == "LOGOUT") {
            std::cout << "LOGOUT:completed" << std::endl;
            break;
        } else if (line == "PI") {
            board = startpos();
        } else if (line.rfind("P", 0) == 0 && line.size() >= 2 && std::isdigit(static_cast<unsigned char>(line[1]))) {
            parseCsaBoardLine(board, line);
        } else if (line.rfind("P+", 0) == 0 || line.rfind("P-", 0) == 0) {
            parseCsaHandLine(board, line);
        } else if (line == "+" || line == "-") {
            board.side = line == "+" ? Black : White;
        } else if (line.rfind("Your_Turn:", 0) == 0 && line.size() >= 11) {
            engineSide = line[10] == '+' ? Black : White;
        } else if (line.rfind("To_Move:", 0) == 0 && line.size() >= 9) {
            board.side = line[8] == '+' ? Black : White;
        } else if (line.rfind("START:", 0) == 0 || line == "go") {
            maybePlayCsa(board, engineSide, engine);
        } else if ((line[0] == '+' || line[0] == '-') && line.size() >= 7 && std::isdigit(static_cast<unsigned char>(line[1]))) {
            const Move move = parseCsaMove(board, line);
            if (board.side != engineSide) {
                engine.recordMove(board, move, false);
            }
            applyIfLegal(board, move);
            maybePlayCsa(board, engineSide, engine);
        } else if (line == "#END") {
            engine.clearGame();
            board = startpos();
        } else if (line == "%TORYO" || line.rfind("#", 0) == 0) {
            engine.finishGame(csaGameResultForEngine(line), engineSide);
            board = startpos();
        }
    }
}

} // namespace shogi
