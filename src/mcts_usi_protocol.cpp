#include "mcts_usi_protocol.h"

#include "mate_solver.h"
#include "mcts_engine.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

namespace shogi {

namespace {

std::vector<std::string> extractUsiMoves(const std::vector<std::string>& words) {
    std::vector<std::string> moves;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (words[i] == "moves") {
            for (std::size_t j = i + 1; j < words.size(); ++j) {
                moves.push_back(words[j]);
            }
            break;
        }
    }
    return moves;
}

void handleSetOption(MCTSEngineWrapper& engine, const std::vector<std::string>& words) {
    const auto nameIt = std::find(words.begin(), words.end(), "name");
    if (nameIt == words.end() || std::next(nameIt) == words.end()) {
        return;
    }
    const auto valueIt = std::find(words.begin(), words.end(), "value");
    const auto nameBegin = std::next(nameIt);
    const auto nameEnd = valueIt == words.end() ? words.end() : valueIt;
    std::string name;
    for (auto it = nameBegin; it != nameEnd; ++it) {
        if (!name.empty()) name += ' ';
        name += *it;
    }
    std::string value;
    if (valueIt != words.end()) {
        for (auto it = std::next(valueIt); it != words.end(); ++it) {
            if (!value.empty()) value += ' ';
            value += *it;
        }
    }
    if (name == "MaxMoveTimeMs") {
        try { engine.setMaxMoveTimeMs(std::stoi(value)); } catch (...) {}
    } else if (name == "MctsSimulations") {
        try { engine.setSimulations(std::stoi(value)); } catch (...) {}
    }
#ifndef HAS_ONNXRUNTIME
    else if (name == "NNPython") {
        engine.setNNPython(value);
    } else if (name == "NNScript") {
        engine.setNNScript(value);
    }
#endif
    else if (name == "Book") {
        engine.setBookEnabled(value != "false" && value != "0");
    } else if (name == "WarnOnNoModel") {
        engine.setWarnOnNoModel(value != "false" && value != "0");
    } else if (name == "NNModel") {
        engine.setNNModel(value);
    } else if (name == "NNDevice") {
        engine.setNNDevice(value);
    } else if (name == "MctsBatchSize") {
        try { engine.setBatchSize(std::stoi(value)); } catch (...) {}
    } else if (name == "ReuseTree") {
        engine.setReuseTree(value != "false" && value != "0");
    }
}

int valueAfter(const std::vector<std::string>& words, const std::string& key, int fallback) {
    const auto it = std::find(words.begin(), words.end(), key);
    if (it == words.end() || std::next(it) == words.end()) return fallback;
    try { return std::stoi(*std::next(it)); } catch (...) { return fallback; }
}

SearchLimits parseSearchLimits(const std::vector<std::string>& words, Color side) {
    const int moveTime = valueAfter(words, "movetime", -1);
    if (moveTime > 0) return SearchLimits{std::max(50, moveTime - 30)};

    const int byoyomi = valueAfter(words, "byoyomi", 0);
    const int increment = valueAfter(words, side == Black ? "binc" : "winc", 0);
    const int remaining = valueAfter(words, side == Black ? "btime" : "wtime", 0);
    int budget = -1;
    if (byoyomi > 0) budget = std::max(50, byoyomi - 100);
    else if (increment > 0) budget = std::max(50, increment + remaining / 40 - 100);
    else if (remaining > 0) budget = std::max(50, remaining / 40);
    if (budget < 0) return SearchLimits{-1};
    return SearchLimits{std::clamp(budget, 50, 10000)};
}

void printSearchInfo(const SearchInfo& info) {
    if (!info.hasBestMove) return;
    const std::uint64_t nps = info.timeMs > 0
        ? info.nodes * 1000ull / static_cast<std::uint64_t>(info.timeMs) : info.nodes;
    std::cout << "info depth " << info.depth
              << " score cp " << info.scoreCp
              << " nodes " << info.nodes
              << " nps " << nps
              << " time " << info.timeMs
              << " pv";
    if (!info.pv.empty()) {
        for (const Move& m : info.pv) std::cout << " " << toUsi(m);
    } else {
        std::cout << " " << toUsi(info.bestMove);
    }
    std::cout << std::endl;
}

} // namespace

void mctsUsiLoop() {
    Board board = startpos();
    MCTSEngineWrapper engine;
    Color engineSide = Black;
    bool engineSideKnown = false;
    std::vector<std::string> recordedMoves;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto words = splitWords(line);
        const std::string& command = words[0];
        if (command == "usi") {
            std::cout << "id name KishiTo-MCTS" << std::endl;
            std::cout << "id author kishi_to" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 3000 min 50 max 600000" << std::endl;
            std::cout << "option name MctsSimulations type spin default 800 min 1 max 100000" << std::endl;
#ifndef HAS_ONNXRUNTIME
#ifdef _WIN32
            std::cout << "option name NNPython type string default ..\\..\\.venv\\Scripts\\python.exe" << std::endl;
#else
            std::cout << "option name NNPython type string default python" << std::endl;
#endif
            std::cout << "option name NNScript type string default tools/nn_eval.py" << std::endl;
            std::cout << "option name NNModel type string default nn_model.pt" << std::endl;
#else
            std::cout << "option name NNModel type string default nn_model.onnx" << std::endl;
#endif
            std::cout << "option name NNDevice type string default auto" << std::endl;
            std::cout << "option name Book type check default true" << std::endl;
            std::cout << "option name WarnOnNoModel type check default true" << std::endl;
            std::cout << "option name MctsBatchSize type spin default 8 min 1 max 64" << std::endl;
            std::cout << "option name ReuseTree type check default true" << std::endl;
            std::cout << "usiok" << std::endl;
        } else if (command == "isready") {
            if (!engine.ensureNN()) {
                if (!fileExists(engine.nnModelPath()))
                    std::cout << "info string ERROR: " << engine.nnModelPath() << " not found" << std::endl;
                else
                    std::cout << "info string ERROR: " << engine.nnModelPath() << " load failed" << std::endl;
                if (!engine.nnLastError().empty()) {
                    std::cout << "info string  -> " << engine.nnLastError() << std::endl;
                }
            } else {
                std::cout << "info string NN model loaded: " << engine.nnModelPath() << std::endl;
            }
            if (engine.loadBook())
                std::cout << "info string Opening book loaded" << std::endl;
            std::cout << "readyok" << std::endl;
        } else if (command == "setoption") {
            handleSetOption(engine, words);
        } else if (command == "usinewgame") {
            board = startpos();
            recordedMoves.clear();
            engine.clearGame();
            engineSideKnown = false;
        } else if (command == "position") {
            if (!setPosition(board, words)) continue;
        } else if (command == "go" || command == "stop") {
            const bool isMateSearch = std::find(words.begin(), words.end(), "mate") != words.end();
            if (isMateSearch) {
                int mateMoveTime = 10000;
                auto mateIt = std::find(words.begin(), words.end(), "mate");
                if (std::next(mateIt) != words.end()) {
                    const std::string& val = *std::next(mateIt);
                    if (val != "infinite") {
                        try { mateMoveTime = std::stoi(val); } catch (...) {}
                        if (mateMoveTime <= 0) mateMoveTime = 10000;
                    } else {
                        mateMoveTime = 300000;
                    }
                }
                MateResult result = engine.searchMate(board, mateMoveTime);
                if (result.found) {
                    std::cout << "checkmate";
                    for (const Move& m : result.pv) {
                        std::cout << " " << toUsi(m);
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "checkmate notimplemented" << std::endl;
                }
            } else {
            if (!engineSideKnown) {
                engineSide = board.side;
                engineSideKnown = true;
            }
            const Move move = engine.chooseMove(board, parseSearchLimits(words, board.side),
                [](const SearchInfo& info) { printSearchInfo(info); });
            if (move.to < 0) {
                std::cout << "bestmove resign" << std::endl;
            } else {
                printSearchInfo(engine.lastSearchInfo());
                std::cout << "bestmove " << toUsi(move) << std::endl;
            }
            }
        } else if (command == "gameover") {
            engine.clearGame();
            board = startpos();
            recordedMoves.clear();
            engineSideKnown = false;
        } else if (command == "quit") {
            break;
        }
    }
}

} // namespace shogi
