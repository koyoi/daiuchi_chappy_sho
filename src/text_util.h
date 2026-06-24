#pragma once

#include <string>
#include <vector>

namespace shogi {

std::string trim(const std::string& text);
std::vector<std::string> splitWords(const std::string& line);
std::string fileModTime(const std::string& path);
bool fileExists(const std::string& path);

} // namespace shogi
