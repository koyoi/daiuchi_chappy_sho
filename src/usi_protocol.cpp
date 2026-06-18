#include "usi_protocol.h"

#include "engine.h"
#include "movegen.h"
#include "notation.h"
#include "position.h"
#include "text_util.h"

#include <iostream>
#include <algorithm>
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
    } else if (name == "WeightsFile") {
        engine.setWeightsPath(value);
    } else if (name == "TrainingDataFile") {
        engine.setTrainingDataPath(value);
    } else if (name == "UseGpu") {
        engine.setGpuEnabled(value == "true" || value == "1");
    } else if (name == "GpuTrainOnGameEnd") {
        engine.setGpuTrainOnGameEnd(value == "true" || value == "1");
    } else if (name == "GpuPython") {
        engine.setGpuPython(value);
    } else if (name == "GpuScript") {
        engine.setGpuScript(value);
    } else if (name == "GpuModel") {
        engine.setGpuModel(value);
    } else if (name == "GpuDevice") {
        engine.setGpuDevice(value);
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
            std::cout << "option name SearchDepth type spin default 2 min 1 max 6" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 1000 min 50 max 600000" << std::endl;
            std::cout << "option name Threads type spin default " << engine.threadCount() << " min 1 max 256" << std::endl;
            std::cout << "option name HeavyEvaluation type check default true" << std::endl;
            std::cout << "option name WeightsFile type string default random-shogi.weights" << std::endl;
            std::cout << "option name TrainingDataFile type string default gpu_training.tsv" << std::endl;
            std::cout << "option name UseGpu type check default false" << std::endl;
            std::cout << "option name GpuTrainOnGameEnd type check default false" << std::endl;
#ifdef _WIN32
            std::cout << "option name GpuPython type string default ..\\..\\.venv\\Scripts\\python.exe" << std::endl;
#else
            std::cout << "option name GpuPython type string default python" << std::endl;
#endif
            std::cout << "option name GpuScript type string default tools/gpu_eval.py" << std::endl;
            std::cout << "option name GpuModel type string default gpu_model.pt" << std::endl;
            std::cout << "option name GpuDevice type string default auto" << std::endl;
            std::cout << "usiok" << std::endl;
        } else if (command == "isready") {
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
        } else if (command == "go" || command == "stop") {
            if (!engineSideKnown) {
                engineSide = board.side;
                engineSideKnown = true;
            }
            const Move move = engine.chooseMove(board, parseSearchLimits(words, board.side));
            if (move.to < 0) {
                std::cout << "bestmove resign" << std::endl;
            } else {
                engine.recordMove(board, move, true);
                recordedMoves.push_back(toUsi(move));
                std::cout << "bestmove " << toUsi(move) << std::endl;
            }
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
