# NNUE エンジン (kishi-to-nnue)

Efficiently Updatable Neural Network (NNUE) 評価関数を用いたαβ探索エンジン。
差分計算により指し手ごとにネットワーク入力層を高速更新し、深い探索を実現する。
ソースコード: `src/nnue.cpp` / `src/nnue_engine.cpp`

---

## 全体の流れ

```text
chooseMove()
  ├── 定石照合 (序盤8手以内)
  ├── 詰み探索 (最大31手, 持ち時間の10%)
  ├── 詰めろ検出 → 探索時間延長 (×1.5)
  └── 反復深化
        ├── Aspiration Windows
        ├── ルート並列探索 (Lazy SMP)
        └── αβ探索
              ├── NNUE 差分更新評価
              └── 静止探索
```

Classic エンジンと同じ探索フレームワーク (αβ + 置換表 + 各種枝刈り) を使い、評価関数のみを NNUE に置き換えた構成。

---

## NNUE ネットワーク

ソースコード: `src/nnue.h` / `src/nnue.cpp`

### 入力特徴量

盤面と持ち駒を 2344 次元のバイナリ (0/1) 特徴量で表現:

| 範囲 | 内容 | 次元数 |
|------|------|--------|
| 0-2267 | 盤上の駒 (14 駒種 × 81 マス × 2 視点) | 2268 |
| 2268-2343 | 持ち駒 (7 種 × 最大枚数 × 2 色) | 76 |
| **合計** | | **2344** |

**盤上特徴量**: `boardFeatureIndex(perspective, pieceColor, type, square)` で計算。自駒と敵駒を分離し、perspective (先手/後手) ごとに別の特徴量を生成。

**持ち駒特徴量**: 歩 1〜18 枚、香桂銀金 1〜4 枚、角飛 1〜2 枚の各枚数を独立した特徴量としてエンコード。`handFeatureIndex(perspective, handColor, type, count)` で計算。

### アーキテクチャ

```text
入力 (2344次元, バイナリ)
  │
  ▼  L0: Accumulator (差分更新可能)
┌────────────────────────────────────────┐
│ 先手視点 → linear(2344→256) → ClampedReLU  │
│ 後手視点 → linear(2344→256) → ClampedReLU  │
│ 手番に応じて [自分|相手] の順で concat      │
└────────────────────────────────────────┘
  │ 512次元
  ▼
  L1: linear(512→64) → ClampedReLU
  │ 64次元
  ▼
  L2: linear(64→32) → ClampedReLU
  │ 32次元
  ▼
  L3: linear(32→1) → ×600 → int
```

- **パラメータ数**: L0 = 2344×256×2 ≈ 120万、L1-L3 ≈ 4万、合計 約 124 万
- **ClampedReLU**: `clamp(x, 0.0, 1.0)` — 出力を [0, 1] に制限
- **出力スケール**: L3 の float 出力を 600 倍して int に変換 (cp 単位の評価値)
- **L0 量子化**: int16 で重みを保持 (scale=64)、Accumulator は int32。L1-L3 は float32

### 差分更新 (Efficiently Updatable)

NNUE の核心。指し手ごとにアキュムレータ全体を再計算せず、変化した特徴量のみを加減算:

```text
指し手: 7七銀 → 6六銀
  removed: boardFeature(perspective, Black, Silver, 7七)
  added:   boardFeature(perspective, Black, Silver, 6六)

指し手: 7七銀 × 6六角 (駒取り)
  removed: boardFeature(perspective, Black, Silver, 7七)
  removed: boardFeature(perspective, White, Bishop, 6六)
  added:   boardFeature(perspective, Black, Silver, 6六)
  added:   handFeature(perspective, Black, Bishop, 新枚数)
```

`computeMoveDelta()` が先手・後手両視点のデルタを計算し、`updateAccumulatorIncremental()` が L0 出力を更新:

```
child.accumulator[j] = parent.accumulator[j]
  - Σ l0Weights[removed_feature][j]
  + Σ l0Weights[added_feature][j]
```

更新される特徴量は 1 手あたり通常 2〜4 個。L0 の 256 次元 × 2 視点分の加減算で済み、2344 次元すべてを通す完全評価の数十倍高速。

### Accumulator スタック

`thread_local std::array<Accumulator, 129>` を探索 ply ごとに保持:

- ply 0: ルート局面のフル計算
- ply N: ply N-1 の Accumulator を差分更新
- undoMove() でスタックを巻き戻すだけで復元 (再計算不要)

### 重みファイル形式

`nnue.bin` — バイナリ形式、マジック `"NNU2"`:

```text
[4B] "NNU2"
[L0 weights] int16 × 2344 × 256
[L0 biases]  int32 × 256
[L1 weights] float32 × 512 × 64
[L1 biases]  float32 × 64
[L2 weights] float32 × 64 × 32
[L2 biases]  float32 × 32
[L3 weights] float32 × 32
[L3 bias]    float32 × 1
```

---

## αβ探索

NNUE エンジンの探索部は Classic エンジンとほぼ同一の構成。評価関数のみが NNUE に置き換わっている。

### 探索テクニック (Classic と共通)

| テクニック | 説明 |
|-----------|------|
| 反復深化 | depth 1 から順に深化、時間制限で打ち切り |
| 置換表 | 2^20 エントリ、64 ストライプ mutex、世代管理。`ReuseCache` で前回の探索結果を再利用 |
| Aspiration Windows | 前回スコア ±50 の狭窓で開始 |
| PVS | 最善候補のみ全窓、以降は null window |
| Null Move Pruning | depth >= 3 で R=3 の null move |
| Reverse Futility Pruning | depth <= 3 で静的評価が beta を大幅に超える場合に枝刈り |
| Razoring | depth <= 2 で評価値が alpha を大幅に下回る場合に静止探索へ |
| Late Move Reduction | 後ろの静かな手を縮小探索 |
| Futility Pruning | depth 1-2 で見込みのない静かな手をスキップ |
| Late Move Pruning | depth <= 3 で `3 + depth²` 手以降の静かな手をスキップ |
| Check Extension | 王手をかける手で depth+1 |
| IID | TT 最善手がなく depth >= 5 で浅い予備探索 |
| Killer Move | β-cutoff した静かな手を同 ply で優先 |
| History Heuristic | 過去の cutoff 統計で静かな手の順序改善 |
| Counter Move | 相手の直前の手に対する反撃手を記憶 |

### NNUE 固有の最適化

- **差分更新**: 各指し手で `computeMoveDelta()` → `updateAccumulatorIncremental()` を呼び、ply+1 の Accumulator を差分更新
- **Lazy SMP**: ヘルパースレッドが深さをスキップしながら同じ探索木を並列探索 (Skip/Phase パターン)

### 静止探索

- 対象: 駒取り、成り、王手 (depth >= 4)
- 最大深さ: 6 手
- Standing Pat + Delta Pruning (マージン 1400)

### 定数一覧

| 定数 | 値 | 説明 |
|------|-----|------|
| `TTSize` | 2^20 | 置換表エントリ数 |
| `LMRFullDepthMoves` | 4 | LMR 適用前のフル探索手数 |
| `LMRMinDepth` | 3 | LMR 最低 depth |
| `NMPReduction` | 3 | NMP 縮小量 |
| `FutilityMargin1` | 400 | depth 1 の Futility マージン |
| `FutilityMargin2` | 900 | depth 2 の Futility マージン |
| `AspirationWindow` | 50 | Aspiration 初期窓幅 |
| `IIDMinDepth` | 5 | IID 最低 depth |
| `DeltaMargin` | 1400 | 静止探索のデルタ枝刈り |
| `QDepth` | 6 | 静止探索最大深さ |
| `RootPruneWidth` | 15 | Root LMR の全幅探索手数 |

---

## 実装ステップ

### Phase 1: NNUE 評価関数 (完了)

- `src/nnue.h/cpp` — 2344→256→64→32→1 ネットワーク、L0 int16 量子化
- `computeMoveDelta()` / `updateAccumulatorIncremental()` による差分更新
- `NNU2` バイナリ形式の load/save

### Phase 2: αβ探索統合 (完了)

- `src/nnue_engine.cpp` — Classic エンジンと同じ探索フレームワークに NNUE 評価を組み込み
- thread_local Accumulator スタックで探索ツリー全体の差分更新
- Lazy SMP によるルート並列探索
- Reverse Futility Pruning, Razoring, Late Move Pruning を追加

### Phase 3: 教師あり学習 (完了)

- `tools/train_nnue.py` — Floodgate 棋譜から BCEWithLogitsLoss で学習
- ブートストラップ学習 (eval_drop + ラベルブレンド)
- GPU 対応 (バッチサイズ 65536 推奨)

### 今後の改善候補

| 機能 | 概要 | 状態 |
|------|------|------|
| HalfKP 特徴量 | 玉の位置を組み合わせた特徴量 (Stockfish 方式) | 未実装 |
| SIMD 最適化 | AVX2/NEON による L0 差分更新の高速化 | 未実装 |
| RL 学習 | 自己対局による NNUE 重みの強化 | 未実装 |
| 千日手検出 | 同一局面反復の検出と引分評価 | 未実装 |
| 入玉宣言 | 相入玉時の点数計算による勝敗判定 | 未実装 |

---

## 関連ドキュメント

- [Classic 探索](search.md) — 共通の探索アルゴリズム詳細
- [Classic 評価関数](evaluation.md) — 98 次元線形/MLP 評価
- [Alpha エンジン](alpha.md) — ResNet-SE + 改良 MCTS
- [学習パイプライン](training.md) — NNUE 学習方法
- [使い方](usage.md) — USI オプション・GUI 設定
