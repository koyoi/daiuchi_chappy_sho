#include "gpu_usi_protocol.h"

#include "gpu_eval_engine.h"
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

void handleSetOption(GpuEvalEngine& engine, const std::vector<std::string>& words) {
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
    } else if (name == "OpeningSafety") {
        engine.setOpeningSafety(value == "true" || value == "1");
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

void gpuUsiLoop() {
    Board board = startpos();
    GpuEvalEngine engine;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto words = splitWords(line);
        const std::string& command = words[0];
        if (command == "usi") {
            std::cout << "id name KishiTo-GPU" << std::endl;
            std::cout << "id author kishi_to" << std::endl;
            std::cout << "option name MaxMoveTimeMs type spin default 3000 min 50 max 600000" << std::endl;
            std::cout << "option name OpeningSafety type check default true" << std::endl;
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
        } else if (command == "position") {
            if (!setPosition(board, words)) continue;
        } else if (command == "go" || command == "stop") {
            const Move move = engine.chooseMove(board, parseSearchLimits(words, board.side),
                [](const SearchInfo& info) { printSearchInfo(info); });
            if (move.to < 0) {
                std::cout << "bestmove resign" << std::endl;
            } else {
                printSearchInfo(engine.lastSearchInfo());
                std::cout << "bestmove " << toUsi(move) << std::endl;
            }
        } else if (command == "gameover") {
            board = startpos();
        } else if (command == "quit") {
            break;
        }
    }
}

} // namespace shogi
