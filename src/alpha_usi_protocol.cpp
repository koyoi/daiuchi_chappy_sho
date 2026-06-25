#include "alpha_usi_protocol.h"

#include "alpha_engine.h"
#include "mate_solver.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace shogi {

namespace {

void handleSetOption(AlphaEngineWrapper& engine, const std::vector<std::string>& words) {
    const auto nameIt = std::find(words.begin(), words.end(), "name");
    if (nameIt == words.end() || std::next(nameIt) == words.end()) return;
    const auto valueIt = std::find(words.begin(), words.end(), "value");
    const auto nameEnd = valueIt == words.end() ? words.end() : valueIt;
    std::string name;
    for (auto it = std::next(nameIt); it != nameEnd; ++it) {
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
    } else if (name == "NNModel") {
        engine.setNNModel(value);
    } else if (name == "NNDevice") {
        engine.setNNDevice(value);
    } else if (name == "MctsBatchSize") {
        try { engine.setBatchSize(std::stoi(value)); } catch (...) {}
    } else if (name == "FPUReduction") {
        try { engine.setFPUReduction(std::stoi(value) / 100.0); } catch (...) {}
    } else if (name == "TemperatureDropMove") {
        try { engine.setTemperatureDropMove(std::stoi(value)); } catch (...) {}
    } else if (name == "Book") {
        engine.setBookEnabled(value != "false" && value != "0");
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
    if (byoyomi > 0) {
        budget = std::max(50, byoyomi - 100);
    } else if (increment > 0) {
        budget = std::max(50, increment + remaining / 30 - 100);
    } else if (remaining > 0) {
        budget = std::max(50, remaining / 30);
    }
    if (budget < 0) return SearchLimits{-1};
    return SearchLimits{std::clamp(budget, 50, 30000)};
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

void printWDL(const AlphaMCTSResult& result) {
    int w = static_cast<int>(result.wdl[0] * 1000 + 0.5);
    int d = static_cast<int>(result.wdl[1] * 1000 + 0.5);
    int l = static_cast<int>(result.wdl[2] * 1000 + 0.5);
    std::cout << "info string wdl " << w << " " << d << " " << l << std::endl;
}

void printVisitDistribution(const AlphaEngineWrapper& engine, Color side) {
    const auto& result = engine.lastMCTSResult();
    std::cout << "visits";
    for (const auto& [move, count] : result.visitDistribution) {
        int idx = AlphaOnnxInference::moveToIndex(move, side);
        std::cout << " " << toUsi(move) << ":" << count << ":" << idx;
    }
    std::cout << std::endl;
}

} // namespace

void alphaUsiLoop() {
    Board board = startpos();
    AlphaEngineWrapper engine;
    Color lastSearchSide = Black;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto words = splitWords(line);
        const std::string& command = words[0];
        if (command == "usi") {
            std::cout << "id name KishiTo-Alpha" << std::endl;
            std::cout << "id author kishi_to" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 3000 min 50 max 600000" << std::endl;
            std::cout << "option name MctsSimulations type spin default 10000 min 1 max 1000000" << std::endl;
            std::cout << "option name NNModel type string default alpha_model.onnx" << std::endl;
            std::cout << "option name NNDevice type string default auto" << std::endl;
            std::cout << "option name MctsBatchSize type spin default 16 min 1 max 64" << std::endl;
            std::cout << "option name FPUReduction type spin default 20 min 0 max 100" << std::endl;
            std::cout << "option name TemperatureDropMove type spin default 30 min 0 max 200" << std::endl;
            std::cout << "option name Book type check default true" << std::endl;
            std::cout << "option name ReuseTree type check default true" << std::endl;
            std::cout << "usiok" << std::endl;
        } else if (command == "isready") {
            if (!engine.ensureNN()) {
                if (!fileExists(engine.nnModelPath()))
                    std::cout << "info string ERROR: " << engine.nnModelPath() << " not found" << std::endl;
                else
                    std::cout << "info string ERROR: " << engine.nnModelPath() << " load failed" << std::endl;
                if (!engine.nnLastError().empty())
                    std::cout << "info string  -> " << engine.nnLastError() << std::endl;
            } else {
                std::cout << "info string NN model loaded: " << engine.nnModelPath()
                          << " [" << engine.nnDeviceUsed() << "]" << std::endl;
                if (!engine.nnCudaError().empty())
                    std::cout << "info string WARNING: CUDA unavailable, using CPU -> " << engine.nnCudaError() << std::endl;
            }
            if (engine.loadBook())
                std::cout << "info string Opening book loaded" << std::endl;
            std::cout << "readyok" << std::endl;
        } else if (command == "setoption") {
            handleSetOption(engine, words);
        } else if (command == "usinewgame") {
            board = startpos();
            engine.clearGame();
        } else if (command == "position") {
            if (!setPosition(board, words)) continue;
        } else if (command == "go") {
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
                    for (const Move& m : result.pv) std::cout << " " << toUsi(m);
                    std::cout << std::endl;
                } else {
                    std::cout << "checkmate notimplemented" << std::endl;
                }
            } else {
                lastSearchSide = board.side;
                auto lastInfoTime = std::chrono::steady_clock::now();
                const Move move = engine.chooseMove(
                    board, parseSearchLimits(words, board.side),
                    [&lastInfoTime](const SearchInfo& info) {
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastInfoTime).count() >= 333) {
                            printSearchInfo(info);
                            lastInfoTime = now;
                        }
                    });
                if (move.to < 0) {
                    std::cout << "bestmove resign" << std::endl;
                } else {
                    printSearchInfo(engine.lastSearchInfo());
                    printWDL(engine.lastMCTSResult());
                    std::cout << "bestmove " << toUsi(move) << std::endl;
                }
            }
        } else if (command == "getvisits") {
            printVisitDistribution(engine, lastSearchSide);
        } else if (command == "gameover") {
            engine.clearGame();
            board = startpos();
        } else if (command == "quit") {
            break;
        }
    }
}

} // namespace shogi
