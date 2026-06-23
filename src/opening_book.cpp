#include "opening_book.h"

#include "position.h"

#include <fstream>
#include <sstream>
#include <string>

namespace shogi {

bool OpeningBook::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;

    entries_.clear();
    std::string line;
    std::uint64_t currentHash = 0;
    bool hasPosition = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line.rfind("position", 0) == 0) {
            std::istringstream iss(line);
            std::vector<std::string> words;
            std::string word;
            while (iss >> word) words.push_back(word);

            Board board;
            if (setPosition(board, words)) {
                currentHash = board.hash;
                hasPosition = true;
            } else {
                hasPosition = false;
            }
        } else if (hasPosition) {
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                auto colon = token.find(':');
                if (colon == std::string::npos) continue;

                BookEntry entry;
                entry.usiMove = token.substr(0, colon);
                try {
                    entry.weight = std::stoi(token.substr(colon + 1));
                } catch (...) {
                    entry.weight = 1;
                }
                if (entry.weight > 0) {
                    entries_[currentHash].push_back(std::move(entry));
                }
            }
            hasPosition = false;
        }
    }

    return !entries_.empty();
}

std::string OpeningBook::selectMove(std::uint64_t hash, std::mt19937& rng) const {
    auto it = entries_.find(hash);
    if (it == entries_.end() || it->second.empty()) return {};

    const auto& moves = it->second;
    int totalWeight = 0;
    for (const auto& e : moves) totalWeight += e.weight;
    if (totalWeight <= 0) return {};

    std::uniform_int_distribution<int> dist(0, totalWeight - 1);
    int roll = dist(rng);
    int cumulative = 0;
    for (const auto& e : moves) {
        cumulative += e.weight;
        if (roll < cumulative) return e.usiMove;
    }
    return moves.back().usiMove;
}

} // namespace shogi
