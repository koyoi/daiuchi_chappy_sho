#include "nnue_usi_protocol.h"

#include "mate_solver.h"
#include "movegen.h"
#include "nnue_engine.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace shogi {

namespace {

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
    const std::uint64_t nps = info.timeMs > 0 ? info.nodes * 1000ull / static_cast<std::uint64_t>(info.timeMs) : info.nodes;
    std::cout << "info depth " << info.depth;
    if (info.isMate) std::cout << " score mate " << info.mateInMoves;
    else std::cout << " score cp " << info.scoreCp;
    std::cout << " nodes " << info.nodes << " nps " << nps << " time " << info.timeMs << " pv";
    if (!info.pv.empty()) for (const Move& m : info.pv) std::cout << " " << toUsi(m);
    else std::cout << " " << toUsi(info.bestMove);
    std::cout << std::endl;
}

void handleSetOption(NNUEEngine& engine, const std::vector<std::string>& words) {
    const auto nameIt = std::find(words.begin(), words.end(), "name");
    if (nameIt == words.end() || std::next(nameIt) == words.end()) return;
    const auto valueIt = std::find(words.begin(), words.end(), "value");
    const auto nameBegin = std::next(nameIt);
    const auto nameEnd = valueIt == words.end() ? words.end() : valueIt;
    std::string name;
    for (auto it = nameBegin; it != nameEnd; ++it) { if (!name.empty()) name += ' '; name += *it; }
    std::string value;
    if (valueIt != words.end()) { for (auto it = std::next(valueIt); it != words.end(); ++it) { if (!value.empty()) value += ' '; value += *it; } }

    if (name == "SearchDepth") {
        try { engine.setSearchDepth(std::stoi(value)); } catch (...) {}
    } else if (name == "MaxMoveTimeMs") {
        try { engine.setMaxMoveTimeMs(std::stoi(value)); } catch (...) {}
    } else if (name == "Threads") {
        try { engine.setThreads(std::stoi(value)); } catch (...) {}
    } else if (name == "Book") {
        engine.setBookEnabled(value != "false" && value != "0");
    } else if (name == "WarnOnNoWeights") {
        engine.setWarnOnNoWeights(value != "false" && value != "0");
    } else if (name == "NNUEFile") {
        if (!value.empty()) {
            if (engine.loadNNUE(value))
                std::cout << "info string NNUE loaded: " << value << std::endl;
            else if (!fileExists(value))
                std::cout << "info string ERROR: " << value << " not found" << std::endl;
            else
                std::cout << "info string ERROR: " << value << " format error (bad magic or truncated)" << std::endl;
        }
    } else {
        try { engine.setParam(name, std::stoi(value)); } catch (...) {}
    }
}

} // namespace

void nnueUsiLoop() {
    Board board = startpos();
    auto enginePtr = std::make_unique<NNUEEngine>();
    NNUEEngine& engine = *enginePtr;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto words = splitWords(line);
        const std::string& command = words[0];
        if (command == "usi") {
            std::cout << "id name KishiTo-NNUE" << std::endl;
            std::cout << "id author Ryohei Fujita" << std::endl;
            std::cout << "option name SearchDepth type spin default 0 min 0 max 128" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 1000 min 50 max 600000" << std::endl;
            std::cout << "option name Threads type spin default " << engine.threadCount() << " min 1 max 256" << std::endl;
            std::cout << "option name NNUEFile type string default nnue.bin" << std::endl;
            std::cout << "option name Book type check default true" << std::endl;
            std::cout << "option name WarnOnNoWeights type check default true" << std::endl;
            std::cout << "option name LMRFullDepthMoves type spin default 4 min 1 max 20" << std::endl;
            std::cout << "option name LMRMinDepth type spin default 3 min 1 max 10" << std::endl;
            std::cout << "option name NMPMinDepth type spin default 3 min 1 max 10" << std::endl;
            std::cout << "option name NMPReduction type spin default 3 min 1 max 8" << std::endl;
            std::cout << "option name FutilityMargin1 type spin default 400 min 50 max 2000" << std::endl;
            std::cout << "option name FutilityMargin2 type spin default 900 min 100 max 3000" << std::endl;
            std::cout << "option name AspirationWindow type spin default 50 min 10 max 500" << std::endl;
            std::cout << "option name IIDMinDepth type spin default 5 min 2 max 10" << std::endl;
            std::cout << "option name DeltaMargin type spin default 1400 min 200 max 5000" << std::endl;
            std::cout << "option name QDepth type spin default 6 min 1 max 20" << std::endl;
            std::cout << "option name QCheckDepthMin type spin default 4 min 1 max 10" << std::endl;
            std::cout << "option name RootPruneWidth type spin default 15 min 1 max 100" << std::endl;
            std::cout << "usiok" << std::endl;
        } else if (command == "isready") {
            if (!engine.loadNNUE("nnue.bin")) {
                if (!fileExists("nnue.bin"))
                    std::cout << "info string WARNING: nnue.bin not found -- using random weights" << std::endl;
                else
                    std::cout << "info string ERROR: nnue.bin format error (bad magic or truncated) -- using random weights" << std::endl;
            } else {
                std::cout << "info string NNUE evaluation loaded" << std::endl;
            }
            if (engine.loadBook())
                std::cout << "info string Opening book loaded" << std::endl;
            std::cout << "readyok" << std::endl;
        } else if (command == "setoption") {
            handleSetOption(engine, words);
        } else if (command == "usinewgame") {
            board = startpos();
        } else if (command == "position") {
            Board next;
            if (setPosition(next, words)) board = next;
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
                const Move move = engine.chooseMove(board, parseSearchLimits(words, board.side), [](const SearchInfo& info) {
                    printSearchInfo(info);
                });
                if (move.to < 0) {
                    std::cout << "bestmove resign" << std::endl;
                } else {
                    printSearchInfo(engine.lastSearchInfo());
                    std::cout << "bestmove " << toUsi(move) << std::endl;
                }
            }
        } else if (command == "stop") {
            // synchronous
        } else if (command == "gameover" || command == "usinewgame") {
            board = startpos();
        } else if (command == "quit") {
            break;
        }
    }
}

} // namespace shogi
