# Alpha エンジン (kishi-to-alpha)

ResNet-SE ニューラルネットワークと改良 MCTS を組み合わせた AlphaZero 方式のエンジン。
ソースコード: `src/alpha_mcts.cpp` / `src/alpha_engine.cpp` / `src/alpha_onnx_inference.cpp`

---

## 全体の流れ

```text
chooseMove()
  ├── 定石照合 (序盤8手以内)
  ├── 詰み探索 (最大31手, 持ち時間の10%, 上限200ms)
  ├── 詰めろ検出 → 探索延長 (sim × 1.5)
  └── 改良 MCTS 探索
        ├── NN バッチ推論
        ├── 動的 c_puct + FPU reduction
        ├── Virtual Loss 付き並列選択
        └── 温度制御による着手選択
```

1. **定石照合**: `book.txt` からハッシュ引き (手数 <= 8)
2. **詰み探索**: `MateSolver` が王手のみの手で最大 31 手詰めまで探索
3. **詰めろ検出**: 相手にパスした場合に詰みがあれば、シミュレーション数を 1.5 倍に延長
4. **MCTS 探索**: NN 評価に基づくモンテカルロ木探索で最善手を選択

---

## ResNet-SE モデル

ソースコード: `src/alpha_onnx_inference.cpp` / `tools/train_alpha.py`

### 入力表現

盤面を 45 チャンネル × 9×9 の空間テンソルで表現:

| チャンネル | 内容 | エンコーディング |
|-----------|------|----------------|
| 0-13 | 先手の駒面 (14 駒種) | one-hot |
| 14-27 | 後手の駒面 (14 駒種) | one-hot |
| 28-34 | 先手の持ち駒 (7 種) | count / max で正規化 (全マス定数) |
| 35-41 | 後手の持ち駒 (7 種) | count / max で正規化 (全マス定数) |
| 42 | 自分の利き数 | `countAttackers() / 8.0` |
| 43 | 相手の利き数 | `countAttackers() / 8.0` |
| 44 | 手番 | 先手=1.0, 後手=0.0 (全マス定数) |

持ち駒の最大枚数: 歩18, 香4, 桂4, 銀4, 金4, 角2, 飛2

### アーキテクチャ

```text
入力 [B, 45, 9, 9]
  │
  ▼
Initial Conv2d(45→192, 3×3, pad=1) → BN → ReLU
  │
  ▼  ×15 Residual Blocks
┌─────────────────────────────────────────┐
│ Conv2d(192→192, 3×3, pad=1) → BN → ReLU│
│ Conv2d(192→192, 3×3, pad=1) → BN       │
│ SE(192→48→192)                          │
│ + skip connection → ReLU                │
└─────────────────────────────────────────┘
  │
  ├──────────────────────────┐
  ▼                          ▼
Value Head                  Policy Head
Conv(192→1,1×1) → BN → ReLU  Conv(192→27,1×1)
Flatten(81)                    Flatten → 2187 logits
FC(81→256) → ReLU
FC(256→3)  ← WDL 3クラス
```

- **パラメータ数**: 約 7-8M
- **SE (Squeeze-and-Excitation)**: チャンネル間の相互作用を動的に学習。GlobalAvgPool → FC(192→48) → ReLU → FC(48→192) → Sigmoid でチャンネルごとの重みを生成し、各チャンネルにスケーリング
- **WDL 出力**: [勝ち, 引分, 負け] の 3 クラス。期待値 = wdl[0] - wdl[2]
- **Policy**: 81 マス × 27 チャンネル = 2187 次元 (MCTS エンジンと共通のエンコーディング)

### Policy エンコーディング

`index = toSquare × 27 + channel`

| チャンネル | 内容 |
|-----------|------|
| 0-9 | 移動方向 (成りなし): 上,下,左,右,左上,右上,左下,右下,桂左,桂右 |
| 10-19 | 移動方向 (成り): 同上 |
| 20-26 | 打ち: 歩,香,桂,銀,金,角,飛 |

方向は手番側の視点で計算される。

### ONNX 推論

- ONNX Runtime で推論 (Python 不要)
- 入力テンソル名: `"features"` [B, 45, 9, 9]
- 出力テンソル名: `"value_wdl"` [B, 3], `"policy_logits"` [B, 2187]
- GPU (CUDA) と CPU の両方に対応。`NNDevice` オプションで切替
- CPU 推論時は `SetIntraOpNumThreads(hardware_concurrency())` で全コア使用
- `cuda:N` 形式で GPU デバイス ID 指定可

---

## 改良 MCTS 探索

ソースコード: `src/alpha_mcts.cpp` / `src/alpha_mcts.h`

AlphaZero / KataGo の知見を取り込んだ MCTS 実装。

### PUCT 選択

```
UCB(child) = Q(child) + cPuct(N) × prior(child) × √(parentVisits) / (1 + childVisits)
```

- `Q` = 累積価値 / 訪問回数 (勝率の近似)
- `prior` = NN の policy 出力 (softmax 済み、合法手で正規化)

### 動的 c_puct

AlphaZero 方式の対数スケーリング:

```
cPuct(N) = log((1 + N + cPuctBase) / cPuctBase) + cPuctInit
```

- `cPuctBase` = 19652, `cPuctInit` = 2.5
- 訪問数が少ないうちは探索寄り、増えるにつれ活用寄りに自動調整
- 固定 c_puct よりも序盤の探索幅と終盤の収束性を両立

### FPU Reduction (First Play Urgency)

未訪問の子ノードの Q 値を親ノードの Q から減算して設定:

```
Q_unvisited = Q_parent - fpuReduction
```

- `fpuReduction` = 0.2 (デフォルト)
- 親ノードが有利な局面では未訪問手に楽観的なスコアを付けず、実績のある手を優先
- KataGo で有効性が実証された手法

### ディリクレノイズ

ルートノードの prior にディリクレ分布のノイズを加えて探索の多様性を確保:

```
prior' = (1 - ε) × prior + ε × Dir(α)
```

- α = 0.15, ε = 0.25

### Virtual Loss

バッチ推論時の並列選択で同じノードに集中するのを防ぐ:

- 選択時にリーフまでのパスに Virtual Loss = 3 を加算 (訪問数+、価値-)
- NN 推論完了後に Virtual Loss を除去し、実際の評価値で逆伝播
- 複数のリーフを同時に選択でき、バッチ推論の効率を最大化

### バッチ NN 推論

1 シミュレーションサイクルで最大 `batchSize` 個のリーフを同時に選択し、まとめて NN 推論:

1. `select()` でリーフノードを選択 (Virtual Loss 付き)
2. `expandMoves()` で合法手を展開
3. リーフの局面をバッチに追加
4. バッチがたまったら `evaluateBatch()` で一括推論
5. 各リーフに policy を適用し、value で逆伝播

### 早期終了

25% のシミュレーションを消化した後、最善手の訪問数が全体の 90% を超えていたら探索を打ち切り:

```
if (simCount > simulations / 4 && bestVisits * 10 > totalVisits * 9) break;
```

### 温度制御

- `moveNumber <= temperatureDropMove` (デフォルト 30): 訪問数に比例した確率で選択 (自己対局の多様性確保)
- `moveNumber > temperatureDropMove`: 最多訪問の手を deterministic に選択 (対局時の安定性)

### ツリー再利用 (Tree Reuse)

前回の探索ツリーを保持し、次の探索時に再利用する:

1. 探索完了後、ルートツリー全体を `retainedTree_` として保持
2. 次の `search()` 呼び出し時、保持ツリーの子 (自分の前手) × 孫 (相手の応手) を走査
3. Zobrist ハッシュが一致するノードをサブツリーごと新ルートに昇格
4. 前回のシミュレーション結果 (訪問数・評価値) がそのまま引き継がれる
5. 新ルートにはディリクレノイズを再適用

USI オプション `ReuseTree` (デフォルト `true`) で有効/無効を切替。`usinewgame` / `gameover` でツリーはクリアされる。

### 逆伝播

評価値は各階層で符号を反転して伝播 (自分の勝ちは相手の負け)。`std::atomic` で CAS ループを使い、ロックフリーで更新。

### パラメータ一覧

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `simulations` | 10000 | 1 手あたりのシミュレーション上限 |
| `cPuctBase` | 19652 | 動的 c_puct の底 |
| `cPuctInit` | 2.5 | 動的 c_puct の初期値 |
| `fpuReduction` | 0.2 | 未訪問ノードの Q 値減算量 |
| `dirichletAlpha` | 0.15 | ノイズの集中度 |
| `dirichletEpsilon` | 0.25 | ノイズの混合率 |
| `batchSize` | 16 | NN バッチ推論サイズ |
| `virtualLoss` | 3 | 並列選択時の仮損失量 |
| `temperatureDropMove` | 30 | 温度を 0 に下げる手数 |

---

## エンジンラッパー

ソースコード: `src/alpha_engine.cpp` / `src/alpha_engine.h`

`AlphaEngineWrapper` が NN 推論・MCTS・詰み探索・定石を統合:

```text
AlphaOnnxInference (NN推論)
AlphaMCTSEngine    (MCTS探索)
MateSolver         (詰み/詰めろ)
OpeningBook        (定石)
```

### chooseMove() の流れ

1. 合法手が 1 手なら即返し
2. 定石照合 (序盤 8 手以内)
3. 詰み探索 (持ち時間の 10%, 最大 200ms, 最大 31 手詰め)
4. 詰めろ検出 → シミュレーション数を 1.5 倍に延長
5. MCTS 探索 (movetime と simulations の早い方で打ち切り)

### 自己対局データ出力

`getvisits` USI コマンドで直前の探索の訪問分布を出力:

```
visits 7g7f:1200:42 3c3d:300:15 ...
```

`AlphaMCTSResult::visitDistribution` に全子ノードの (手, 訪問数) を保持。自己対局スクリプト (`alpha_self_play.py`) がこの情報を policy の教師信号として `.npz` に保存。

---

## 実装ステップ

### Phase 1: ResNet-SE モデル + 教師あり学習 (完了)

- `tools/train_alpha.py` で Floodgate R3000+ 棋譜から SL 学習
- 15 ブロック × 192ch の ResNet-SE モデル
- WDL 3 クラス value head + 2187 次元 policy head
- `tools/export_alpha_onnx.py` で ONNX エクスポート

### Phase 2: 改良 MCTS + 新 exe (完了)

- `src/alpha_mcts.cpp` — 動的 c_puct, FPU reduction, 温度制御, Virtual Loss
- `src/alpha_engine.cpp` — MCTS + MateSolver + OpeningBook 統合
- `src/alpha_onnx_inference.cpp` — ONNX Runtime 推論 (CUDA/CPU)
- `src/alpha_usi_protocol.cpp` — USI プロトコル + `getvisits` コマンド
- Linux / Windows クロスビルド対応 (`#ifdef _WIN32` で `ORTCHAR_T` 分岐)

### Phase 3: 自己対局 RL パイプライン (完了)

- `tools/alpha_self_play.py` — Rich ベース並列表示、マルチワーカー自己対局、`--gpus N` 対応
- `tools/train_alpha_rl.py` — KL-div policy + CE value loss、200 万局面リプレイバッファ
- `tools/alpha_train_loop.py` — RL ループ (自己対局 → 学習 → 評価 → 55% ゲート)、`--resume` 自動復帰
- `tools/remote_alpha_loop.ps1` — リモート GPU サーバーでの一括実行スクリプト

### Phase 4: CPU 推論 + 量子化 (完了)

- `tools/quantize_alpha.py` — ONNX INT8 静的量子化。Floodgate 局面でキャリブレーション
- `tools/train_alpha_small.py` — 知識蒸留 (15blk/192ch → 10blk/128ch)。温度 T=3.0 のソフトターゲット
- CPU 推論時に `SetIntraOpNumThreads(hardware_concurrency())` で全コア活用

### Phase 5: 高度な機能 (TODO)

| 機能 | 概要 | 状態 |
|------|------|------|
| ツリー再利用 | 前回のMCTSツリーを保持し、相手の応手に対応するサブツリーから探索継続 | **実装済** |
| ポンダリング | 相手の手番中に予測応手から探索継続 (`go ponder` / `ponderhit`) | 未実装 |
| 改良時間管理 | 局面複雑度に応じた持ち時間配分、安定時早切り | 未実装 |
| マルチ GPU 自己対局 | ワーカーごとに異なる GPU を割り当て | 部分実装 (1 GPU のみ動作確認) |
| 千日手検出 | 同一局面反復の検出と引分評価 | 未実装 |
| 入玉宣言 | 相入玉時の点数計算による勝敗判定 | 未実装 |

---

## 定石生成

ソースコード: `tools/gen_alpha_book.py`

高シミュレーション MCTS で序盤を BFS 展開し、定跡ファイルを生成:

- 各局面で上位 N 手を分岐 (`--branches`, デフォルト 3)
- 訪問数を定石の重みとして `book.txt` に出力
- `--plies` で展開手数を指定 (デフォルト 12)

---

## 関連ドキュメント

- [MCTS + Transformer](mcts.md) — 旧世代の NN 探索エンジン
- [NNUE 探索](nnue.md) — 差分更新ニューラルネット + αβ探索
- [Classic 探索](search.md) — αβ探索の詳細
- [学習パイプライン](training.md) — 全エンジンの学習方法
- [使い方](usage.md) — USI オプション・GUI 設定
