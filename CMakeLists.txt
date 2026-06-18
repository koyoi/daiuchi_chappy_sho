cmake_minimum_required(VERSION 3.16)

project(kishi_to LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 共通ライブラリ (盤面・指し手生成・局面操作・表記) ---
add_library(shogi_core STATIC
    src/shogi_types.cpp
    src/movegen.cpp
    src/position.cpp
    src/notation.cpp
    src/text_util.cpp
)
target_compile_features(shogi_core PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(shogi_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(shogi_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- 従来エンジン (alpha-beta + 44特徴量 + オンライン学習) ---
add_executable(kishi-to-classic
    src/engine.cpp
    src/evaluation.cpp
    src/learning.cpp
    src/gpu_bridge.cpp
    src/usi_protocol.cpp
    src/csa_protocol.cpp
    src/main.cpp
)
target_link_libraries(kishi-to-classic PRIVATE shogi_core)
target_compile_features(kishi-to-classic PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to-classic PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to-classic PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET kishi-to-classic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kishi-to-classic>/tools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_eval.py"
        "$<TARGET_FILE_DIR:kishi-to-classic>/tools/gpu_eval.py"
)

# --- MCTSエンジン (Transformer + MCTS) ---
add_executable(kishi-to
    src/mcts.cpp
    src/nn_bridge.cpp
    src/mcts_engine.cpp
    src/mcts_usi_protocol.cpp
    src/mcts_main.cpp
)
target_link_libraries(kishi-to PRIVATE shogi_core)
target_compile_features(kishi-to PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET kishi-to POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kishi-to>/tools"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/nn_eval.py"
        "$<TARGET_FILE_DIR:kishi-to>/tools/nn_eval.py"
)
