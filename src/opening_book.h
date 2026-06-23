#pragma once

#include "shogi_types.h"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace shogi {

struct BookEntry {
    std::string usiMove;
    int weight = 1;
};

class OpeningBook {
public:
    bool load(const std::string& path);
    std::string selectMove(std::uint64_t hash, std::mt19937& rng) const;
    bool empty() const { return entries_.empty(); }
    std::size_t size() const { return entries_.size(); }

private:
    std::unordered_map<std::uint64_t, std::vector<BookEntry>> entries_;
};

} // namespace shogi
