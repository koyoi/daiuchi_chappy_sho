# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C++17 shogi (Japanese chess) engine with online learning. Supports USI and simplified CSA protocols. Features alpha-beta search with iterative deepening, 44-feature linear evaluation, online weight learning from game results, and optional GPU inference/training via PyTorch bridge.

## Build

```bash
cmake -B build
cmake --build build --config Release
```

Executable: `build/Release/random-shogi-engine.exe` (Windows) or `build/random-shogi-engine` (Linux/macOS).

MSVC builds require `/utf-8` (already in CMakeLists.txt) because source files contain Japanese comments.

## Running

```bash
# USI protocol (default)
random-shogi-engine

# CSA protocol
random-shogi-engine --csa

# Quick test via USI
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./random-shogi-engine
```

The executable sets its working directory to its own location on startup, so `random-shogi.weights` and `tools/gpu_eval.py` are resolved relative to the exe.

## Architecture

Entry point (`main.cpp`) selects protocol → `usiLoop()` or `csaLoop()` → `LearningEngine` for search/learning.

### Core Layers

- **shogi_types** — `Board`, `Move`, `Bitboard` (81-bit: `uint64_t lo` + `uint32_t hi`), `Color`, `PieceType`. Board holds piece array, hand pieces, Zobrist hash (XOR-incremental), king position cache, material score (incremental), and per-color/per-piece-type Bitboards.
- **position** — `applyMove()` updates all Board state atomically (squares, Bitboards, hash, material, king positions). `startpos()` / `setFromSfen()` initialize from SFEN.
- **movegen** — Bitboard-based move generation. Pre-computed step-attack tables for short-range pieces; slide functions for lance/bishop/rook. `attackersOf()` returns Bitboard of all attackers of a square. `generateLegalMoves()` filters pseudo-legal moves by checking king safety and pawn-drop mate.
- **evaluation** — `Evaluator` extracts 44 features (piece counts, hand pieces, king safety, attack maps, PST, king shelter, tactical summaries) and computes score as `dot(features, weights)`. Features 28-31,37 are /100 normalized. L2 regularization in weight updates. `buildAttackMap()` uses Bitboard-based `countAttackers()`. Optional heavy features (index 36-41) require full legal move generation per side.
- **engine** — `LearningEngine::chooseMove()` runs iterative deepening. `search()` is alpha-beta with transposition table (fixed-size 2^20 array, 64-stripe lock sharding, generation counter). `quiescence()` extends search for tactical moves. Root moves scored in parallel via thread pool. Move ordering: MVV-LVA captures > checks > promotions > drops.
- **learning** — `OnlineLearner` records moves during a game. On `finishGame()`, updates weights by comparing chosen-move features against average-of-legal-moves features, scaled by win/loss outcome, actor type (engine vs human), and recency.
- **gpu_bridge** — Spawns `tools/gpu_eval.py` as subprocess for batch inference (`score`) or training (`train`). Model: 44→64→32→1 MLP. Falls back to CPU evaluation on failure.

### Piece Encoding

`makePiece(color, type) = type * color` where Black=1, White=-1. Board squares are signed ints: positive=Black, negative=White, 0=empty. PieceType 1-8 unpromoted (Pawn→King), 9-14 promoted.

### Zobrist Hashing

Board.hash is maintained incrementally in `applyMove()` via XOR. Tables: `zobristPiece[81][31]` (piece value + offset 15), `zobristHand[2][15][19]` (color × type × count), `zobristSide`. Engine's `boardHash()` mixes `board.hash` with rootSide for TT keying.

### Transposition Table

Fixed-size array of 2^20 entries, indexed by `hash & mask`. 64 stripe mutexes selected by `(hash >> 20) % 64`. Generation counter (`ttGeneration_`) avoids clearing the table between searches — stale entries are overwritten by deeper or newer results.

## Key Files

| File | Role |
|------|------|
| `src/shogi_types.h/cpp` | Board, Bitboard, Move, Zobrist, init functions |
| `src/movegen.cpp` | Attack tables, move generation, legality checking |
| `src/engine.cpp` | Search (alpha-beta, quiescence, mate detection), move ordering |
| `src/evaluation.cpp` | 44-feature extraction, linear evaluation, PST, king shelter |
| `src/position.cpp` | applyMove, SFEN parsing, board initialization |
| `src/learning.cpp` | Online weight learning from game outcomes |
| `src/gpu_bridge.cpp` | Python subprocess for GPU inference/training |
| `src/usi_protocol.cpp` | USI command loop and option handling |
| `src/csa_protocol.cpp` | CSA protocol loop |
| `tools/gpu_eval.py` | PyTorch MLP model for GPU evaluation |

## Not Implemented

Perpetual check detection, kachi (entering king) declaration, dedicated mate search, TCP CSA server connection.

## Testing

No automated test suite. Validate manually via USI protocol or ShogiGUI/将棋所.
