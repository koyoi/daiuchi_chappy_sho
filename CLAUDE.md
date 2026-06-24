# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C++17 shogi (Japanese chess) engine with online learning. Four engines: Classic (alpha-beta with 98-feature linear or MLP evaluation), MCTS (Transformer via Python subprocess or native ONNX Runtime), NNUE (alpha-beta with efficiently updatable neural network), and Alpha (ResNet-SE with improved MCTS, AlphaZero-style). Supports USI and simplified CSA protocols.

## Build

```bash
cmake -B build
cmake --build build --config Release
```

Executables: `kishi-to-classic` (Classic), `kishi-to` (MCTS), `kishi-to-nnue` (NNUE), and `kishi-to-alpha` (Alpha, requires `USE_ONNXRUNTIME=ON`).

MSVC builds require `/utf-8` (already in CMakeLists.txt) because source files contain Japanese comments.

## Running

```bash
# Classic engine (USI, default)
./build/kishi-to-classic

# CSA protocol
./build/kishi-to-classic --csa

# Quick test via USI
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to-classic
```

The executable sets its working directory to its own location on startup, so `linear.weights`, `mlp.weights`, and `book.txt` are resolved relative to the exe.

To generate an opening book: `python tools/gen_book.py --engine build/Release/kishi-to-classic.exe --movetime 10000 --plies 8`

## Architecture

Entry point (`main.cpp`) selects protocol → `usiLoop()` or `csaLoop()` → `LearningEngine` for search/learning.

### Core Layers

- **shogi_types** — `Board`, `Move`, `Bitboard` (81-bit: `uint64_t lo` + `uint32_t hi`), `Color`, `PieceType`. Board holds piece array, hand pieces, Zobrist hash (XOR-incremental), king position cache, material score (incremental), and per-color/per-piece-type Bitboards.
- **position** — `applyMove()` updates all Board state atomically (squares, Bitboards, hash, material, king positions). `startpos()` / `setFromSfen()` initialize from SFEN.
- **movegen** — Bitboard-based move generation. Pre-computed step-attack tables for short-range pieces; slide functions for lance/bishop/rook. `attackersOf()` returns Bitboard of all attackers of a square. `generateLegalMoves()` filters pseudo-legal moves by checking king safety and pawn-drop mate.
- **evaluation** — `Evaluator` extracts 98 features and computes score as `dot(features, weights)` (linear) or MLP forward pass (98→128→64→1 with ReLU). MLP weights loaded from text file via `loadMlp()`. `buildAttackMap()` uses Bitboard-based `countAttackers()`. Optional heavy features (index 36-39,41) require full legal move generation per side. Feature 40 (attacked king ring) is always computed. Feature groups:
  - 0-20: Piece counts, PST, hand pieces, king proximity, gold-like king defenders, near-king piece pressure
  - 21-27: King distance, side to move, game phase (opening/middle/endgame), hand piece totals, promoted piece count
  - 28-31: Threat values (attacked/hanging/loose/defended), /100 normalized
  - 32-35: King safety (attackers, defenders, escape squares, check status)
  - 36-41: Tactical summaries (check/capture/promotion/legal move counts, king ring attacks). 36-39,41 are heavy features
  - 42-43: Feature 42 unused (gap). Feature 43: pawn shelter
  - 51-54: King surround (gold, silver, friendly, enemy piece counts around king)
  - 55-57: Pawn structure (doubled, isolated, passed pawn candidates)
  - 58-60: Rook activity (open file, semi-open file, king file alignment)
  - 61-63: Zone control (own territory, center, enemy territory)
  - 64-67: Piece mobility (rook /10, bishop /10, silver, lance)
  - 68-69: Threats (pawn threats to non-pawns, minor piece threats to major pieces)
  - 70-73: King position (file centrality and rank for both sides)
  - 74-75: Pin detection (count, value /100)
  - 76-79: Drop threats (king-adjacent drops, gold/silver drop potential, knight check drops, pawn file drops)
  - 80-81: Long-range alignment (diagonal to enemy king, lance aimed at enemy king)
  - 82-83: King area strength (piece value near enemy/own king, /100)
  - 84-86: King safety details (open files near king, edge king, total material /1000)
  - 87-89: Piece presence (rook, bishop availability) and passed pawns
  - 90-97: Phase-gated interaction features (endgame-scaled king attackers/escape/strength/drops/check/shelter/surround/pins)
- **opening_book** — `OpeningBook` loads a text-based opening book (`book.txt`) mapping position hashes to weighted candidate moves. `chooseMove()` probes the book before search when `moveNumber <= 8`. Book file uses USI `position` command syntax; `gen_book.py` generates it via deep engine search.
- **engine** — `LearningEngine::chooseMove()` first checks the opening book, then runs iterative deepening. `search()` is alpha-beta with transposition table (fixed-size 2^20 array, 64-stripe lock sharding, generation counter). `quiescence()` extends search for tactical moves. Root moves scored in parallel via thread pool, reordered after each depth iteration. Late Move Reduction: moves ranked below `rootPruneWidth_` at depth≥3 searched at reduced depth. Move ordering: MVV-LVA captures > checks > promotions > drops. Before normal search, runs a quick mate search (up to 7 plies, 10% of time budget). When the main search detects a mate score, reports `score mate N` via USI.
- **mate_solver** — `MateSolver` provides dedicated tsumi (checkmate) search and tsumero (threatmate) detection. `searchMate()` uses iterative deepening with check-only move generation for the attacker and evasion-only for the defender. Dedicated 256K-entry TT stores proven/disproven status per position. `detectTsumero()` uses the null-move approach: if the defender skips their turn, is there tsumi? USI `go mate [time]` command invokes the solver directly.
- **learning** — `OnlineLearner` records moves during a game. On `finishGame()`, updates weights by comparing chosen-move features against average-of-legal-moves features, scaled by win/loss outcome, actor type (engine vs human), and recency.

### Alpha Engine (ResNet-SE + Improved MCTS)

- **alpha_onnx_inference** — `AlphaOnnxInference` handles ONNX Runtime inference for the 15-block, 192-channel ResNet-SE model. Input: 45-channel 9×9 spatial features (28 piece planes + 14 hand planes + 2 attack count planes + 1 side plane). Output: WDL (win/draw/loss) 3-class value head and 2187-logit policy head (81 squares × 27 channels). `encodeBoardSpatial()` builds the spatial tensor from Board. Supports both CUDA and CPU execution providers; CPU mode uses `SetIntraOpNumThreads(hardware_concurrency())`.
- **alpha_mcts** — `AlphaMCTSEngine` implements improved MCTS with: dynamic c_puct = `log((1+N+19652)/19652) + 2.5`, FPU (First Play Urgency) reduction where unvisited nodes get Q = parentQ - 0.2, temperature schedule (stochastic for first 30 moves, argmax after), early termination when best move has >90% of visits after 25% of simulations, and batch NN inference. `AlphaMCTSResult` includes `visitDistribution` for training data export.
- **alpha_engine** — `AlphaEngineWrapper` composes `AlphaOnnxInference`, `AlphaMCTSEngine`, `MateSolver`, and `OpeningBook`. `chooseMove()` flow: opening book → mate search (10% of time budget, max 200ms) → tsumero detection (1.5x simulation extension) → MCTS search. Stores `lastMCTSResult_` for self-play training data access via `getvisits` USI command.
- **Training pipeline** — Two-phase: (1) Supervised learning from Floodgate R3000+ games via `train_alpha.py`, (2) Self-play RL via `alpha_train_loop.py` (200 games/iter at 400 sims → train on 2M-position replay buffer → evaluate 20 games → gate at 55% win rate). CPU deployment via INT8 quantization (`quantize_alpha.py`) or knowledge distillation to 10-block/128ch student (`train_alpha_small.py`).

### Piece Encoding

`makePiece(color, type) = type * color` where Black=1, White=-1. Board squares are signed ints: positive=Black, negative=White, 0=empty. PieceType 1-8 unpromoted (Pawn→King), 9-14 promoted.

### Zobrist Hashing

Board.hash is maintained incrementally in `applyMove()` via XOR. Tables: `zobristPiece[81][31]` (piece value + offset 15), `zobristHand[2][15][19]` (color × type × count), `zobristSide`. Engine's `boardHash()` mixes `board.hash` with rootSide for TT keying.

### Transposition Table

Fixed-size array of 2^20 entries, indexed by `hash & mask`. 64 stripe mutexes selected by `(hash >> 20) % 64`. Generation counter (`ttGeneration_`) avoids clearing the table between searches — stale entries are overwritten by deeper or newer results.

## Key Files

| File | Role |
| ---- | ---- |
| `src/shogi_types.h/cpp` | Board, Bitboard, Move, Zobrist, init functions |
| `src/movegen.cpp` | Attack tables, move generation, legality checking |
| `src/engine.cpp` | Search (alpha-beta, quiescence, mate detection), move ordering, root LMR |
| `src/mate_solver.h/cpp` | Tsumi (checkmate) search and tsumero (threatmate) detection |
| `src/opening_book.h/cpp` | Opening book loader and weighted random move selection |
| `src/evaluation.cpp` | 98-feature extraction, linear/MLP evaluation, PST, king shelter |
| `src/position.cpp` | applyMove, SFEN parsing, board initialization |
| `src/learning.cpp` | Online weight learning from game outcomes |
| `src/usi_protocol.cpp` | USI command loop and option handling |
| `src/csa_protocol.cpp` | CSA protocol loop |
| `src/nn_bridge.h/cpp` | NN inference bridge (Python subprocess or ONNX Runtime) for MCTS engine |
| `src/onnx_inference.cpp` | Native ONNX Runtime inference (compiled with `HAS_ONNXRUNTIME`) |
| `src/mcts_usi_protocol.cpp` | USI protocol loop for MCTS engine |
| `tools/mlp_eval.py` | PyTorch MLP model training (Classic engine) |
| `tools/export_mlp.py` | PyTorch MLP → text weights converter |
| `tools/export_onnx.py` | PyTorch Transformer → ONNX model converter |
| `tools/train.py` | Transformer model training (MCTS engine) |
| `tools/nn_eval.py` | Python-side NN evaluation server for MCTS engine |
| `tools/self_play.py` | Self-play game generation for training data |
| `tools/train_classic.py` | Classic engine training pipeline |
| `tools/csa_parser.py` | CSA game record parser for training data |
| `tools/gen_book.py` | Opening book generator via deep USI search |
| `src/alpha_onnx_inference.h/cpp` | ResNet-SE ONNX inference (45ch spatial input, WDL+policy output) |
| `src/alpha_mcts.h/cpp` | Improved MCTS (dynamic c_puct, FPU reduction, temperature schedule) |
| `src/alpha_engine.h/cpp` | Alpha engine wrapper (ONNX + MCTS + MateSolver + OpeningBook) |
| `src/alpha_usi_protocol.h/cpp` | USI protocol loop for Alpha engine (getvisits command) |
| `src/alpha_main.cpp` | Alpha engine entry point |
| `tools/train_alpha.py` | ResNet-SE model definition and supervised learning |
| `tools/export_alpha_onnx.py` | ResNet-SE PyTorch → ONNX converter |
| `tools/alpha_self_play.py` | Self-play with MCTS visit distribution extraction |
| `tools/train_alpha_rl.py` | RL training from self-play data (KL-div policy + WDL value) |
| `tools/alpha_train_loop.py` | RL loop orchestrator (self-play → train → evaluate → gate) |
| `tools/quantize_alpha.py` | INT8 static quantization for CPU inference |
| `tools/train_alpha_small.py` | Knowledge distillation (15blk/192ch → 10blk/128ch) |
| `tools/gen_alpha_book.py` | Opening book generator via high-simulation MCTS analysis |

## Not Implemented

Perpetual check detection, kachi (entering king) declaration, df-pn mate search (current solver uses iterative deepening), TCP CSA server connection.

## Testing

No automated test suite. Validate manually via USI protocol or ShogiGUI/将棋所.
