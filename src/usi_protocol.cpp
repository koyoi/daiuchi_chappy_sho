#include "usi_protocol.h"

#include "engine.h"
#include "mate_solver.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
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

bool samePrefix(const std::vector<std::string>& left, const std::vector<std::string>& right, std::size_t count) {
    if (left.size() < count || right.size() < count) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (left[i] != right[i]) {
            return false;
        }
    }
    return true;
}

void syncPositionAndLearning(
    Board& board,
    const std::vector<std::string>& words,
    LearningEngine& engine,
    bool engineSideKnown,
    Color engineSide,
    std::vector<std::string>& recordedMoves) {
    Board replay;
    if (!setPosition(replay, words)) {
        return;
    }

    const std::vector<std::string> incomingMoves = extractUsiMoves(words);
    std::size_t common = std::min(recordedMoves.size(), incomingMoves.size());
    if (!samePrefix(recordedMoves, incomingMoves, common)) {
        common = 0;
    }
    if (common < recordedMoves.size()) {
        engine.clearGame();
        recordedMoves.clear();
        common = 0;
    }

    Board cursor;
    const std::vector<std::string> positionOnly(words.begin(), std::find(words.begin(), words.end(), "moves"));
    if (!setPosition(cursor, positionOnly.empty() ? words : positionOnly)) {
        board = replay;
        return;
    }

    for (std::size_t i = 0; i < incomingMoves.size(); ++i) {
        const Move move = parseUsiMove(cursor, incomingMoves[i]);
        if (i >= common) {
            engine.recordMove(cursor, move, engineSideKnown && cursor.side == engineSide);
            recordedMoves.push_back(incomingMoves[i]);
        }
        applyIfLegal(cursor, move);
    }
    board = replay;
}

void handleSetOption(LearningEngine& engine, const std::vector<std::string>& words) {
    const auto nameIt = std::find(words.begin(), words.end(), "name");
    if (nameIt == words.end() || std::next(nameIt) == words.end()) {
        return;
    }
    const auto valueIt = std::find(words.begin(), words.end(), "value");
    const auto nameBegin = std::next(nameIt);
    const auto nameEnd = valueIt == words.end() ? words.end() : valueIt;
    std::string name;
    for (auto it = nameBegin; it != nameEnd; ++it) {
        if (!name.empty()) {
            name += ' ';
        }
        name += *it;
    }
    std::string value;
    if (valueIt != words.end()) {
        for (auto it = std::next(valueIt); it != words.end(); ++it) {
            if (!value.empty()) {
                value += ' ';
            }
            value += *it;
        }
    }
    if (name == "Learning") {
        engine.setLearningEnabled(value != "false" && value != "0");
    } else if (name == "SearchDepth") {
        try {
            engine.setSearchDepth(std::stoi(value));
        } catch (...) {
        }
    } else if (name == "MaxMoveTimeMs") {
        try {
            engine.setMaxMoveTimeMs(std::stoi(value));
        } catch (...) {
        }
    } else if (name == "Threads") {
        try {
            engine.setThreads(std::stoi(value));
        } catch (...) {
        }
    } else if (name == "HeavyEvaluation") {
        engine.setHeavyEvaluation(value != "false" && value != "0");
    } else if (name == "OpeningSafety") {
        engine.setOpeningSafety(value != "false" && value != "0");
    } else if (name == "RecordOnly") {
        engine.setRecordOnly(value != "false" && value != "0");
    } else if (name == "WeightsFile") {
        engine.setWeightsPath(value);
    } else if (name == "TrainingDataFile") {
        engine.setTrainingDataPath(value);
    } else if (name == "MlpWeightsFile") {
        if (!value.empty()) {
            if (!engine.loadMlpWeights(value)) {
                if (!fileExists(value))
                    std::cout << "info string ERROR: " << value << " not found" << std::endl;
                else
                    std::cout << "info string ERROR: " << value << " format error (dimension mismatch?)" << std::endl;
            } else {
                std::cout << "info string MLP weights loaded: " << value << std::endl;
            }
        }
    } else if (name == "UseMLP") {
        engine.setUseMlp(value != "false" && value != "0");
    } else if (name == "Book") {
        engine.setBookEnabled(value != "false" && value != "0");
    } else if (name == "WarnOnNoWeights") {
        engine.setWarnOnNoWeights(value != "false" && value != "0");
    } else if (name == "RootPruneWidth") {
        try {
            engine.setRootPruneWidth(std::stoi(value));
        } catch (...) {
        }
    } else if (name == "ReuseCache") {
        engine.setReuseCache(value != "false" && value != "0");
    }
}

int parseGameoverResult(const std::vector<std::string>& words) {
    if (words.size() < 2) {
        return 0;
    }
    if (words[1] == "win") {
        return 1;
    }
    if (words[1] == "lose") {
        return -1;
    }
    return 0;
}

int valueAfter(const std::vector<std::string>& words, const std::string& key, int fallback) {
    const auto it = std::find(words.begin(), words.end(), key);
    if (it == words.end() || std::next(it) == words.end()) {
        return fallback;
    }
    try {
        return std::stoi(*std::next(it));
    } catch (...) {
        return fallback;
    }
}

SearchLimits parseSearchLimits(const std::vector<std::string>& words, Color side) {
    const int moveTime = valueAfter(words, "movetime", -1);
    if (moveTime > 0) {
        return SearchLimits{std::max(50, moveTime - 30)};
    }

    const int byoyomi = valueAfter(words, "byoyomi", 0);
    const int increment = valueAfter(words, side == Black ? "binc" : "winc", 0);
    const int remaining = valueAfter(words, side == Black ? "btime" : "wtime", 0);
    int budget = -1;
    if (byoyomi > 0) {
        budget = std::max(50, byoyomi - 100);
    } else if (increment > 0) {
        budget = std::max(50, increment + remaining / 40 - 100);
    } else if (remaining > 0) {
        budget = std::max(50, remaining / 40);
    }
    if (budget < 0) {
        return SearchLimits{-1};
    }
    return SearchLimits{std::clamp(budget, 50, 10000)};
}

void printSearchInfo(const SearchInfo& info) {
    if (!info.hasBestMove) {
        return;
    }
    const std::uint64_t nps = info.timeMs > 0 ? info.nodes * 1000ull / static_cast<std::uint64_t>(info.timeMs) : info.nodes;
    std::cout << "info depth " << info.depth;
    if (info.isMate) {
        std::cout << " score mate " << info.mateInMoves;
    } else {
        std::cout << " score cp " << info.scoreCp;
    }
    std::cout << " nodes " << info.nodes
              << " nps " << nps
              << " time " << info.timeMs
              << " pv";
    if (!info.pv.empty()) {
        for (const Move& m : info.pv) {
            std::cout << " " << toUsi(m);
        }
    } else {
        std::cout << " " << toUsi(info.bestMove);
    }
    std::cout << std::endl;
}

void printSearchInfo(const LearningEngine& engine) {
    printSearchInfo(engine.lastSearchInfo());
}

} // namespace

void usiLoop() {
    Board board = startpos();
    LearningEngine engine;
    Color engineSide = Black;
    bool engineSideKnown = false;
    std::vector<std::string> recordedMoves;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto words = splitWords(line);
        const std::string& command = words[0];
        if (command == "usi") {
            std::cout << "id name LearningShogiEngine" << std::endl;
            std::cout << "id author OpenAI Codex" << std::endl;
            std::cout << "option name Learning type check default true" << std::endl;
            std::cout << "option name SearchDepth type spin default 0 min 0 max 128" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 1000 min 50 max 600000" << std::endl;
            std::cout << "option name Threads type spin default " << engine.threadCount() << " min 1 max 256" << std::endl;
            std::cout << "option name WeightsFile type string default linear.weights" << std::endl;
            std::cout << "option name MlpWeightsFile type string default " << std::endl;
            std::cout << "option name UseMLP type check default true" << std::endl;
            std::cout << "option name Book type check default true" << std::endl;
            std::cout << "option name WarnOnNoWeights type check default true" << std::endl;
            std::cout << "option name ReuseCache type check default true" << std::endl;
            std::cout << "usiok" << std::endl;
        } else if (command == "isready") {
            if (!engine.loadWeights()) {
                if (!fileExists(engine.weightsPath()))
                    std::cout << "info string WARNING: " << engine.weightsPath() << " not found (using defaults)" << std::endl;
                else
                    std::cout << "info string ERROR: " << engine.weightsPath() << " format error (using defaults)" << std::endl;
            }
            if (engine.loadBook()) {
                std::cout << "info string Opening book loaded" << std::endl;
            }
            if (engine.loadMlpWeights("mlp.weights")) {
                std::cout << "info string MLP evaluation enabled (mlp.weights)" << std::endl;
            } else {
                if (fileExists("mlp.weights"))
                    std::cout << "info string ERROR: mlp.weights format error, using linear evaluation" << std::endl;
                else
                    std::cout << "info string mlp.weights not found, using linear evaluation" << std::endl;
            }
            std::cout << "readyok" << std::endl;
        } else if (command == "setoption") {
            handleSetOption(engine, words);
        } else if (command == "usinewgame") {
            board = startpos();
            recordedMoves.clear();
            engine.clearGame();
            engineSideKnown = false;
        } else if (command == "position") {
            syncPositionAndLearning(board, words, engine, engineSideKnown, engineSide, recordedMoves);
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
                int lastInfoDepth = -1;
                auto lastInfoTime = std::chrono::steady_clock::now();
                const Move move = engine.chooseMove(board, parseSearchLimits(words, board.side),
                    [&lastInfoDepth, &lastInfoTime](const SearchInfo& info) {
                        auto now = std::chrono::steady_clock::now();
                        if (info.depth != lastInfoDepth ||
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastInfoTime).count() >= 333) {
                            printSearchInfo(info);
                            lastInfoDepth = info.depth;
                            lastInfoTime = now;
                        }
                    });
                if (move.to < 0) {
                    std::cout << "bestmove resign" << std::endl;
                } else {
                    printSearchInfo(engine);
                    engine.recordMove(board, move, true);
                    recordedMoves.push_back(toUsi(move));
                    std::cout << "bestmove " << toUsi(move) << std::endl;
                }
            }
        } else if (command == "stop") {
            // Search is synchronous, so stop is a no-op.
        } else if (command == "gameover") {
            if (engineSideKnown) {
                engine.finishGame(parseGameoverResult(words), engineSide);
            } else {
                engine.clearGame();
            }
            board = startpos();
            recordedMoves.clear();
            engineSideKnown = false;
        } else if (command == "quit") {
            break;
        }
    }
}

} // namespace shogi
