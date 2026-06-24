#include "text_util.h"

#include <cstring>
#include <ctime>
#include <sstream>
#include <sys/stat.h>

namespace shogi {

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> splitWords(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> words;
    std::string word;
    while (input >> word) {
        words.push_back(word);
    }
    return words;
}

std::string fileModTime(const std::string& path) {
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path.c_str(), &st) != 0) return "not found";
    std::tm tm{};
    localtime_s(&tm, &st.st_mtime);
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return "not found";
    std::tm tm{};
    localtime_r(&st.st_mtime, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

bool fileExists(const std::string& path) {
#ifdef _WIN32
    struct _stat64 st;
    return _stat64(path.c_str(), &st) == 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0;
#endif
}

} // namespace shogi
