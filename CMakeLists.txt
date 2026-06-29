cmake_minimum_required(VERSION 3.16)

project(kishi_to LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release: 速度最優先の最適化 (古いCPU非対応)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /GL /GS- /fp:fast /arch:AVX2 /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -ffast-math -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

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

# --- 従来エンジン (alpha-beta + 98特徴量 + オンライン学習) ---
add_executable(kishi-to-classic
    src/engine.cpp
    src/evaluation.cpp
    src/learning.cpp
    src/kifu_learner.cpp
    src/mate_solver.cpp
    src/opening_book.cpp
    src/usi_protocol.cpp
    src/csa_protocol.cpp
    src/main.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(kishi-to-classic PRIVATE shogi_core Threads::Threads)
target_compile_features(kishi-to-classic PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to-classic PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to-classic PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- MCTSエンジン (Transformer + MCTS, ONNX Runtime 必須) ---
option(USE_ONNXRUNTIME "Use ONNX Runtime for direct C++ NN inference" OFF)

if(USE_ONNXRUNTIME)
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" ENV ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" ENV ONNXRUNTIME_ROOT
        PATH_SUFFIXES onnxruntime)
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
        message(WARNING "ONNX Runtime not found (set ONNXRUNTIME_ROOT). kishi-to (MCTS) will not be built.")
        set(USE_ONNXRUNTIME OFF)
    else()
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    endif()
endif()

if(USE_ONNXRUNTIME)
    add_executable(kishi-to
        src/mcts.cpp
        src/nn_bridge.cpp
        src/onnx_inference.cpp
        src/mate_solver.cpp
        src/opening_book.cpp
        src/mcts_engine.cpp
        src/mcts_usi_protocol.cpp
        src/mcts_main.cpp
    )
    target_link_libraries(kishi-to PRIVATE shogi_core)
    target_compile_features(kishi-to PRIVATE cxx_std_17)
    target_include_directories(kishi-to PRIVATE ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(kishi-to PRIVATE ${ONNXRUNTIME_LIB})
    if(MSVC)
        target_compile_options(kishi-to PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(kishi-to PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# --- NNUEエンジン (NNUE評価 + alpha-beta) ---
add_executable(kishi-to-nnue
    src/nnue.cpp
    src/nnue_engine.cpp
    src/mate_solver.cpp
    src/opening_book.cpp
    src/nnue_usi_protocol.cpp
    src/nnue_main.cpp
)
target_link_libraries(kishi-to-nnue PRIVATE shogi_core Threads::Threads)
target_compile_features(kishi-to-nnue PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(kishi-to-nnue PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(kishi-to-nnue PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Alphaエンジン (ResNet-SE + 改良MCTS) ---
set(ALPHA_SOURCES
    src/alpha_mcts.cpp
    src/alpha_engine.cpp
    src/mate_solver.cpp
    src/opening_book.cpp
    src/alpha_usi_protocol.cpp
    src/alpha_main.cpp
)

if(USE_ONNXRUNTIME)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        list(APPEND ALPHA_SOURCES src/alpha_onnx_inference.cpp)

        add_executable(kishi-to-alpha ${ALPHA_SOURCES})
        target_link_libraries(kishi-to-alpha PRIVATE shogi_core Threads::Threads)
        target_compile_features(kishi-to-alpha PRIVATE cxx_std_17)
        target_compile_definitions(kishi-to-alpha PRIVATE HAS_ONNXRUNTIME=1)
        target_include_directories(kishi-to-alpha PRIVATE ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(kishi-to-alpha PRIVATE ${ONNXRUNTIME_LIB})
        if(MSVC)
            target_compile_options(kishi-to-alpha PRIVATE /W4 /permissive- /utf-8)
        else()
            target_compile_options(kishi-to-alpha PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()
