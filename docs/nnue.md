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

### 入力特徴量 — HalfKP

玉の位置と盤上の各駒の位置関係を直接符号化する方式。

| 範囲 | 内容 | 次元数 |
|------|------|--------|
| 0-170585 | HalfKP: 玉マス(81) × 駒種色(26) × 駒マス(81) | 170,586 |
| 170586-170661 | 持ち駒 (7 種 × 最大枚数 × 2 色) | 76 |
| **合計** | | **170,662** |

**HalfKP 特徴量**: 各 perspective (先手/後手) ごとに、自玉の位置と盤上の全駒 (玉を除く) の位置の組み合わせを特徴量とする。駒種は自駒 13 種 + 敵駒 13 種 = 26 種。1 局面あたりのアクティブ特徴量は約 30〜40 個 (疎)。

**持ち駒特徴量**: 歩 1〜18 枚、香桂銀金 1〜4 枚、角飛 1〜2 枚の各枚数を独立した特徴量としてエンコード。

### アーキテクチャ

```text
入力 (170,662次元, バイナリ, 疎)
  │
  ▼  L0: Accumulator (差分更新可能)
┌────────────────────────────────────────────┐
│ 先手視点 → EmbeddingBag(170662→512) + bias │
│ 後手視点 → EmbeddingBag(170662→512) + bias │
│ 手番に応じて [自分|相手] の順で concat        │
└────────────────────────────────────────────┘
  │ 1024次元 (512×2)
  ▼  SCReLU: clamp(x, 0, 1)²
  │
  ▼  Output: linear(1024→1) → ×600 → int
```

- **パラメータ数**: L0 = 170,662 × 512 ≈ 8,740 万、Output = 1,024 + 1、合計 約 8,740 万
- **SCReLU**: `clamp(x, 0.0, 1.0)²` — 入力を [0,1] にクランプ後、二乗。通常の ReLU よりも滑らかな勾配
- **出力スケール**: Output の float 出力を 600 倍して int に変換 (cp 単位の評価値)
- **L0 量子化**: int16 で重みを保持 (scale=64)、Accumulator は int32。Output 層は float32
- **SIMD 最適化**: AVX2 (16-wide int16) または SSE4.1 (8-wide) で差分更新・SCReLU・内積を高速化

#### なぜ隠れ層が 1 層なのか

旧構成 (256→64→32→1) では 256→64 のボトルネックで情報の 75% を捨てていた。HalfKP 特徴量は玉×駒種×駒位置の 3 要素交互作用を入力レベルで符号化しているため、L0 の SCReLU で十分な非線形性がある。L0 層自体が 512 ユニットの隠れ層であり、Stockfish NNUE も同様の直結構成を採用している。L0Size を 1024 以上に拡大する場合は、後段に小さな隠れ層 (16〜32 ユニット) を入れる価値が出てくる。

### 差分更新 (Efficiently Updatable)

NNUE の核心。指し手ごとにアキュムレータ全体を再計算せず、変化した特徴量のみを加減算:

```text
指し手: 7七銀 → 6六銀
  removed: HalfKP(自玉, Black, Silver, 7七)
  added:   HalfKP(自玉, Black, Silver, 6六)

指し手: 7七銀 × 6六角 (駒取り)
  removed: HalfKP(自玉, Black, Silver, 7七)
  removed: HalfKP(自玉, White, Bishop, 6六)
  added:   HalfKP(自玉, Black, Silver, 6六)
  added:   handFeature(Black, Bishop, 新枚数)
```

`computeMoveDelta()` が先手・後手両視点のデルタを計算し、`updateAccumulatorIncremental()` が L0 出力を更新。**玉が動いた場合**: その perspective のアキュムレータを全再計算 (`needsFullRecompute` フラグ)。

更新される特徴量は 1 手あたり通常 2〜4 個。L0 の 512 次元 × 2 視点分の加減算で済み、170,662 次元すべてを通す完全評価の数千倍高速。

### Accumulator スタック

`thread_local std::array<Accumulator, 129>` を探索 ply ごとに保持:

- ply 0: ルート局面のフル計算
- ply N: ply N-1 の Accumulator を差分更新
- undoMove() でスタックを巻き戻すだけで復元 (再計算不要)

### 重みファイル形式

`nnue.bin` — バイナリ形式、マジック `"NNU5"`:

```text
[4B] "NNU5"
[4B] L0Size (int32, 現在 512)
[L0 weights] int16 × 170662 × 512
[L0 biases]  int32 × 512
[out weights] float32 × 1024
[out bias]    float32 × 1
```

---

## αβ探索

NNUE エンジンの探索部は Classic エンジンとほぼ同一の構成。評価関数のみが NNUE に置き換わっている。

### 探索テクニック

| テクニック | 説明 |
|-----------|------|
| 反復深化 | depth 1 から順に深化、時間制限で打ち切り |
| 置換表 | バケット方式 (4 エントリ/バケット)、USI `Hash` オプション (1〜65536 MB、デフォルト 256 MB)。`ReuseCache` で前回の探索結果を再利用 |
| Aspiration Windows | 前回スコア ±200 の狭窓で開始 |
| PVS | 最善候補のみ全窓、以降は null window |
| Null Move Pruning | depth >= 3 で R=3 の null move |
| Reverse Futility Pruning | depth <= 3 で静的評価が beta を大幅に超える場合に枝刈り |
| Razoring | depth <= 2 で評価値が alpha を大幅に下回る場合に静止探索へ |
| Late Move Reduction | テーブルベース: R = 0.75 + ln(depth) × ln(moveIndex) / 2.25。ヒストリ値で調整 |
| Singular Extension | depth >= 8 で TT 手が代替手より十分強い場合に延長 |
| Futility Pruning | depth 1-2 で見込みのない静かな手をスキップ |
| SEE Pruning | depth <= 4 で SEE 負けの駒取り/打ちをスキップ |
| Late Move Pruning | depth <= 3 で `3 + depth²` 手以降の静かな手をスキップ |
| Check Extension | 王手をかける手で depth+1 |
| IID | TT 最善手がなく depth >= 5 で浅い予備探索 |
| Killer Move | β-cutoff した静かな手を同 ply で優先 |
| History Heuristic | Main `[2][81][81]` + Continuation `[14*81][14*81]` + Capture `[14][81][14]` + Drop `[2][7][81]`。gravity 更新 |
| Counter Move | 相手の直前の手に対する反撃手を記憶 |

### NNUE 固有の最適化

- **差分更新**: 各指し手で `computeMoveDelta()` → `updateAccumulatorIncremental()` を呼び、ply+1 の Accumulator を差分更新
- **Lazy SMP**: ヘルパースレッドが深さをスキップしながら同じ探索木を並列探索 (Skip/Phase パターン)

### 静止探索

- 対象: 駒取り、成り、王手 (depth >= 4)
- 最大深さ: 6 手
- Standing Pat + Delta Pruning (マージン 1400)

### 手順序

TT 手 (1M) > MVV-LVA 駒取り (100K+) > Killer (90K) > Counter Move (85K) > History/Continuation/Drop history で静かな手を順序付け。駒取りは capture history でも評価。

### 時間管理

`btime/wtime/binc/winc/byoyomi` から動的に配分:

- **秒読み (残り時間なし)**: hardLimit = byoyomi - 800ms、optimumTime = hardLimit × 80%
- **秒読み (残り時間あり)**: 残り手数から base を算出、hardLimit = 残り + byoyomi - 800ms
- **持ち時間のみ**: 残り手数ベースの配分
- **MaxMoveTimeMs**: hardLimit の上限を制約
- **延長**: 評価値 80cp 以上の急落で ×1.5、最善手変化で ×1.3、詰めろ検出で optimum/maximum ×1.5
- **早期終了**: 最善手が 3 深さ以上安定 + optimumTime の 60% 経過で打ち切り

### 定数一覧

| 定数 | 値 | 説明 |
|------|-----|------|
| `Hash` (USI) | 256 MB | 置換表サイズ |
| `LMRFullDepthMoves` | 4 | LMR 適用前のフル探索手数 |
| `LMRMinDepth` | 3 | LMR 最低 depth |
| `NMPReduction` | 3 | NMP 縮小量 |
| `SEMinDepth` | 8 | Singular Extension 最低 depth |
| `FutilityMargin1` | 600 | depth 1 の Futility マージン |
| `FutilityMargin2` | 1200 | depth 2 の Futility マージン |
| `AspirationWindow` | 200 | Aspiration 初期窓幅 |
| `IIDMinDepth` | 5 | IID 最低 depth |
| `DeltaMargin` | 2000 | 静止探索のデルタ枝刈り |
| `QDepth` | 6 | 静止探索最大深さ |
| `RootPruneWidth` | 15 | Root LMR の全幅探索手数 |

---

## 改訂履歴

### v5 — HalfKP + SCReLU + 直結出力 (現行)

- **入力特徴量**: HalfKP 方式に変更。玉マス(81) × 駒種色(26) × 駒マス(81) = 170,586 + 持ち駒 76 = **170,662 次元**
  - 玉と各駒の位置関係を直接学習可能に (+100〜200 Elo 相当の改善ポテンシャル)
  - 玉移動時は該当 perspective のアキュムレータを全再計算
- **活性化関数**: ClampedReLU → **SCReLU** (`clamp(x,0,1)²`) に変更。より滑らかな勾配で学習安定性向上
- **ネットワーク構成**: 512→64→32→1 → **2×512→1** (直結出力) に変更
  - 旧構成の 512→64 ボトルネック (情報の 75% 喪失) を解消
  - L0 層 (170,662→512) 自体が隠れ層。HalfKP 特徴量は 3 要素の交互作用を入力レベルで符号化しているため、後段の追加層は不要
- **SIMD 最適化**: AVX2/SSE4.1 による差分更新・SCReLU・内積の高速化
- **ファイル形式**: **NNU5** (L0Size をヘッダに記録、出力層は float32)
- **探索改善**:
  - LMR テーブル化: `R = 0.75 + ln(depth) × ln(moveIndex) / 2.25`、ヒストリ値で調整
  - Singular Extension (depth ≥ 8)
  - Continuation History, Capture History, Drop History 追加
  - SEE ベースの枝刈り (depth ≤ 4)
  - バケット方式 TT (4 エントリ/バケット、デフォルト 256 MB)
  - 動的時間管理 (評価値急変延長、最善手安定早期終了)
- **学習パイプライン改善**:
  - EmbeddingBag ベースのモデル (GPU 最適化)
  - sigmoid + cross-entropy 損失 (rescaling 不要)
  - **結果割引ラベル**: `discount = 0.5^(残り手数/30)`, `label = 0.5 + discount × (勝率 - 0.5)` — 終局に近い局面ほど結果を強く反映
  - **序盤確率的ランプアップ**: `include_prob = min(1.0, ply/30)` — 機械的な序盤スキップを廃止、序盤は低確率で採用しつつレアな変化を拾う
  - Eval bootstrapping (学習済みモデルの評価値とのブレンド)
  - Mixed precision training (AMP)

### v4 — SCReLU + L0 512 ニューロン

- L0Size を 256→512 に拡大
- ClampedReLU → SCReLU に変更
- NNU4 形式

### v3 — HalfKP 特徴量導入

- HalfKP 入力特徴量 (170,662 次元)
- 玉移動時のフル再計算
- NNU3 形式

### v2 — 初期実装

- 2344 次元入力 (駒種×位置 + 持ち駒)
- 4 層ネットワーク: 2344→256→64→32→1 (ClampedReLU)
- NNU2 形式
- 基本的な αβ + 置換表 + LMR

---

## 開発記録 — 試行錯誤と知見

### 評価値スケールと探索パラメータの不整合 (v5)

NNUEの出力は勝率のロジット（sigmoid の逆関数）を 600 倍して cp に変換する。
学習データ（R2300+、42万局、2700万局面）で訓練すると、cp の標準偏差が約 918 になった。

旧パラメータ（Aspiration=50, Futility=400/900 など）は cp_std≈200 程度を想定した値であり、
cp_std=918 の評価関数ではほぼ毎回 aspiration fail → full re-search が発生し、探索効率が著しく低下していた。
実測で depth iteration の 78% が fail していた。

**対応**: 探索パラメータを評価値スケールに合わせて再校正。

| パラメータ       | 旧値          | 新値           | 根拠               |
| ---------------- | ------------- | -------------- | ------------------ |
| AspirationWindow | 50            | 200            | fail rate 78%→28% |
| FutilityMargin1  | 400           | 600            | cp_std×0.65        |
| FutilityMargin2  | 900           | 1200           | cp_std×1.3         |
| DeltaMargin      | 1400          | 2000           | cp_std×2.2         |
| RFP margin       | depth×200     | depth×300      | —                  |
| Razoring         | 300+depth×200 | 500+depth×300  | —                  |

### ゲーム結果ラベルの限界

ゲーム結果（勝ち=1, 負け=0）を直接ラベルにすると、同一局面でも試合によってラベルが異なる。
**val_loss ≈ 0.52 がノイズフロア**であり、モデル容量やデータ量を増やしても改善しない。

局面固有の評価ではなく「この局面からどちらが勝ちやすいか」を学習してしまうため:

- 矢倉の普通の序盤が +376cp（矢倉を指す棋士の勝率バイアス）
- 飛車先交換後の明確な有利局面が -84cp（局面の良し悪しが学べていない）

### 結果割引 halflife の調整

`discount = 0.5^(残り手数/halflife)` の halflife は学習品質に大きく影響する。

| halflife     | 効果                   | 問題点                                                                                       |
| ------------ | ---------------------- | -------------------------------------------------------------------------------------------- |
| 30（初期値） | 終局付近に強いシグナル | ラベルの 42% が [0.4, 0.6] に集中。モデルは「全部 0.5 と予測」を学習してしまう               |
| 60（現行）   | ラベル分布がフラット化 | [0.4, 0.6] の集中度は 7% に改善。val_loss は 0.52→0.54 に微増だが、モデルの局面理解は向上    |

halflife=30 での val_loss=0.52 は「何でも 0.5 と答える」安直な戦略の結果であり、halflife=60 で val_loss=0.54 の方が実際には良いモデル。
ただし根本的にはゲーム結果ラベル自体のノイズが問題であり、損失関数の改善ではなくラベル品質の改善が必要。

### MCTS 教師による自己対局データ生成

ゲーム結果ラベルの限界を打破するため、**より強いエンジン（MCTS）の探索評価値を教師ラベルとする**アプローチを導入。

```text
MCTS エンジン (kishi-to)
  │  自己対局（200ms/手、~920 sims）
  │
  ▼
selfplay_mcts.npz  ← 各局面の探索評価値がラベル
  │
  ▼
train_nnue.py --data selfplay_mcts.npz
  │
  ▼
nnue.bin  ← MCTS の知識を蒸留した NNUE
```

**なぜ効くのか**: MCTS の探索評価値は局面固有（「この局面で歩得は有利」）であり、ゲーム結果（「この試合は先手が勝った」）より遥かに高品質。
実測で、ラベルの標準偏差が 0.237（NNUE自己対局）→ 0.427（MCTS教師）に増加 = 情報量が約 2 倍。

**movetime の選択**: 200ms（~920 sims）で十分。AlphaZero の学習データ生成は 800 sims/手が標準。
500ms にすると質は √(2.5) ≈ 1.58 倍だが、同じ時間でデータ量は 2.5 分の 1。NNUE の 87M パラメータに対してはデータ量がボトルネックなので 200ms が最適点。

**スクリプト**: `tools/nnue_self_play.py`。`--mcts` フラグで MCTS のスコア変換 (cp/1000→勝率) を使う。
`train_nnue.py --data` で生成した NPZ を直接学習データとして使用可能。

### GPU 利用率の改善

RTX PRO 6000 (48GB VRAM) を使う場合、batch_size=4096 では GPU 利用率が 3% 程度。
batch_size=262144 に引き上げて VRAM をフル活用し、学習速度を大幅に改善。
AMP (混合精度) も有効にして fp16 演算を活用。

### NPZ キャッシュシステム

42万局のCSA棋譜パースに 56 秒かかっていたのをNPZキャッシュで数秒にした。
自動無効化: `fingerprint = ファイル数 + 合計サイズ + 最新mtime` と学習パラメータ (min_rate, halflife 等) の一致チェック。

### 今後の課題

1. **MCTS 教師データで学習し、MCTS に匹敵する評価品質を得る**
2. **反復改善**: MCTS 教師 → NNUE 学習 → 改良 NNUE で自己対局 → 再学習
3. **Eval bootstrapping の活用**: MCTS 教師モデルが十分強くなったら、Floodgate の 2700 万局面にも MCTS ラベルを付けて大規模学習
4. **評価値スケールの見直し**: 現在の ×600 倍は Stockfish (×173) と比べて大きく、将来的に見直す可能性あり

---

## 関連ドキュメント

- [Classic 探索](search.md) — 共通の探索アルゴリズム詳細
- [Classic 評価関数](evaluation.md) — 98 次元線形/MLP 評価
- [Alpha エンジン](alpha.md) — ResNet-SE + 改良 MCTS
- [学習パイプライン](training.md) — NNUE 学習方法
- [使い方](usage.md) — USI オプション・GUI 設定
