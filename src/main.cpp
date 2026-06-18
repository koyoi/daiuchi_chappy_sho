#include "csa_protocol.h"
#include "usi_protocol.h"

#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        if (executable.has_parent_path()) {
            std::filesystem::current_path(executable.parent_path());
        }
    } catch (...) {
    }

    std::string protocol = "usi";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--protocol" || arg == "-p") && i + 1 < argc) {
            protocol = argv[++i];
        } else if (arg == "--csa") {
            protocol = "csa";
        } else if (arg == "--usi") {
            protocol = "usi";
        }
    }

    if (protocol == "csa") {
        shogi::csaLoop();
    } else {
        shogi::usiLoop();
    }
    return 0;
}
