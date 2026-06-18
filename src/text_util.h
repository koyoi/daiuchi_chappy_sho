#pragma once

#include <string>
#include <vector>

namespace shogi {

std::string trim(const std::string& text);
std::vector<std::string> splitWords(const std::string& line);

} // namespace shogi
