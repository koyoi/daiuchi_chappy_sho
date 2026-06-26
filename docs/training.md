# 学習パイプライン

5 つのエンジンが Floodgate CSA 棋譜から学習できます。
入力は共通、各エンジン専用のスクリプトがパース→学習→モデル保存を行います。

---

## 全体像

```text
Floodgate CSA 棋譜 (kifu/floodgate/*.csa)
        │
        ├── train_classic.py ──→ linear.weights          (Classic 線形評価)
        │
        ├── train_mlp.py    ──→ mlp_model.pt + mlp.weights  (Classic MLP 評価)
        │
        ├── train_nnue.py   ──→ nnue.bin                 (NNUE 評価)
        │
        ├── train.py        ──→ nn_model.pt + nn_model.onnx (MCTS Transformer)
        │
        └── train_alpha.py  ──→ alpha_model.pt + alpha_model.onnx (Alpha ResNet-SE)
                │
                └── alpha_train_loop.py ──→ 自己対局 RL ループで継続強化
```

| エンジン | 学習スクリプト | 出力ファイル | 評価方式 |
|---------|-------------|-------------|---------|
| Classic (αβ探索 線形) | `train_classic.py` | `linear.weights` | 98次元線形 |
| Classic (αβ探索 MLP) | `train_mlp.py` | `mlp.weights` | MLP (98→128→64→1) |
| NNUE (αβ探索) | `train_nnue.py` | `nnue.bin` | NNUE HalfKP (2×512→1, SCReLU) |
| MCTS (Transformer) | `train.py` | `nn_model.onnx` | Transformer (方策+価値) |
| Alpha (改良MCTS) | `train_alpha.py` | `alpha_model.onnx` | ResNet-SE (方策+WDL価値) |

---

## 棋譜の準備

Floodgate (<http://wdoor.c.u-tokyo.ac.jp/shogi/floodgate.html>) の CSA 棋譜を使用します。

```text
kifu/
  floodgate/
    2026/
      01-06/
        *.csa
```

棋譜ファイルは `*.csa` 形式で、`kifu/` 以下に配置します。

---

## 1. Classic エンジン — 線形評価 (`train_classic.py`)

98次元の手作り特徴量に対する線形重みを Bonanza 法で学習します。

### クイックスタート

```sh
python tools/train_classic.py \
  --kifu kifu/floodgate \
  --engine build/kishi-to-classic
```

### 内部の流れ

1. CSA 棋譜をパースし、勝者（高レート側）の指し手を SFEN+USI 形式で抽出
2. `classic_training.tsv` に書き出し
3. `kishi-to-classic --learn classic_training.tsv` を実行
4. 各局面で正解手のスコアが高くなるよう重みを勾配降下で更新
5. `linear.weights` に保存

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--engine` | (必須) | `kishi-to-classic` 実行ファイル |
| `--weights` | `linear.weights` | 重みファイルパス |
| `--min-rate` | 1500 | 最低レーティング |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--skip-opening` | 10 | 序盤 N 手をスキップ |
| `--sample-rate` | 0.5 | 局面のサンプリング率 |
| `--lr` | 0.01 | 学習率 |
| `--epochs` | 1 | エポック数 |
| `--temperature` | 100.0 | softmax 温度 |

### モデルの配備

```sh
cp linear.weights build/Release/linear.weights
```

---

## 2. Classic エンジン — MLP 評価 (`train_mlp.py`)

Classic と同じ 98 次元特徴量を入力とする MLP を学習し、テキスト形式の重みファイルにエクスポートします。
αβ探索のリーフノード評価を MLP に置き換えることで、非線形パターン（駒の連携・囲い強度など）を捉えた評価が可能になります。

### クイックスタート

```sh
python tools/train_mlp.py \
  --kifu kifu/floodgate \
  --engine build/kishi-to-classic
```

### 内部の流れ

1. CSA 棋譜をパースし、SFEN+USI 形式で抽出（Classic と共通）
2. `kishi-to-classic --extract-features` で 98 次元特徴量を抽出
   - 正解手の局面 → ラベル +1.0
   - ランダムな不正解手の局面 → ラベル -1.0
3. `mlp_eval.py train` で MLP (98→128→64→1) を BCEWithLogitsLoss で学習
4. `mlp_model.pt` に保存
5. `export_mlp.py` で `mlp.weights`（テキスト形式）にエクスポート

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--engine` | (必須) | `kishi-to-classic` 実行ファイル |
| `--model` | `mlp_model.pt` | モデル保存先 |
| `--output` | `mlp.weights` | エンジン用重みファイル |
| `--min-rate` | 1500 | 最低レーティング |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--skip-opening` | 10 | 序盤 N 手をスキップ |
| `--sample-rate` | 0.5 | 局面のサンプリング率 |
| `--negatives` | 1 | 局面あたりの負例数 |
| `--epochs` | 5 | 学習エポック数 |
| `--batch-size` | 512 | バッチサイズ |
| `--lr` | 1e-3 | 学習率 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

### モデルの配備

```sh
cp mlp.weights build/Release/mlp.weights
```

USI で `setoption name UseMLP value true` を設定すると MLP 評価に切り替わります。

### 手動エクスポート

```sh
python tools/export_mlp.py --model mlp_model.pt --output mlp.weights
```

---

## 3. NNUE エンジン (`kishi-to-nnue`)

NNUE (Efficiently Updatable Neural Network) 方式の評価関数を学習します。
HalfKP 特徴量 (170,662 次元) から差分計算可能なネットワーク (2×512→1, SCReLU) で局面を評価します。

### クイックスタート

```sh
python tools/train_nnue.py --kifu kifu/floodgate
```

GPU がある場合はバッチサイズを大きくすると高速化できます:

```sh
python tools/train_nnue.py --kifu kifu/floodgate --batch-size 65536 --epochs 20
```

### 内部の流れ

1. CSA 棋譜をパースし、各局面の HalfKP 特徴量を抽出
2. 結果割引ラベル (`discount = 0.5^(残り手数/30)`) で勝率予測を学習 (sigmoid cross-entropy)
3. 序盤は確率的にランプアップ (`include_prob = min(1.0, ply/30)`) — 機械的スキップではなく確率的に間引き
4. `nnue_model.pt` (PyTorch チェックポイント) に保存
5. NNU5 バイナリ形式 (`nnue.bin`) にエクスポート
   - L0 重みは int16 量子化 (scale=64)、L0 バイアスは int32
   - 出力層は float32

### ラベル設計

| 要素 | 旧方式 | 現方式 |
|------|--------|--------|
| 勝敗ラベル | 全局面に一律 0/1 | 結果割引: 終局に近い局面ほど結果を強く反映 |
| 序盤スキップ | 最初の 24 手を機械的に除外 | `include_prob = min(1.0, ply/30)` で確率的にランプアップ |
| 損失関数 | sigmoid/scale=361 | sigmoid cross-entropy (rescaling 不要) |

結果割引とランプアップの組み合わせにより、序盤は「採用されにくい + 採用されてもラベルが 0.5 に近い」の二重ガードで自然にダウンウェイトされる。

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--output` | `nnue.bin` | エンジン用バイナリ重みファイル |
| `--model-pt` | `nnue_model.pt` | PyTorch チェックポイント |
| `--min-rate` | 2300 | 最低レーティング |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--skip-opening` | 0 | 序盤 N 手をスキップ (ランプアップがあるため通常 0) |
| `--sample-rate` | 1.0 | 局面のサンプリング率 |
| `--epochs` | 20 | 学習エポック数 |
| `--batch-size` | 65536 | バッチサイズ |
| `--lr` | 1e-3 | 学習率 |
| `--workers` | 0 | パース並列数 (0=自動) |
| `--resume` | - | 既存チェックポイントから再開 |
| `--bootstrap` | - | eval bootstrapping 用モデル |
| `--lambda-blend` | 0.5 | bootstrap blend: lambda×engine + (1-lambda)×outcome |

### MCTS 教師による自己対局学習

ゲーム結果ラベルでは val_loss ≈ 0.52 がノイズフロアとなり、それ以上の評価品質は得られない。
より強い MCTS エンジンの探索評価値を教師ラベルとすることで、局面固有の評価を学習できる。

#### Step 1: 自己対局データ生成（ローカル）

```sh
python tools/nnue_self_play.py \
  --engine build/Release/kishi-to.exe \
  --mcts --games 3000 --movetime 200 \
  --output selfplay_mcts.npz
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--engine` | (必須) | 教師エンジンの実行ファイル |
| `--mcts` | - | MCTS スコア変換 (cp/1000→勝率)。省略時は NNUE 方式 (sigmoid(cp/600)) |
| `--games` | 1000 | 対局数 |
| `--movetime` | 200 | 1 手あたりの思考時間 (ms) |
| `--random-plies` | 4 | 序盤のランダム手数（開局多様性のため） |
| `--workers` | 1 | 並列エンジン数 |
| `--output` | `selfplay_data.npz` | 出力ファイル |

所要時間の目安: 3000 局 × movetime=200ms ≈ 10 時間。

#### Step 2: 学習（リモート GPU）

```sh
python tools/train_nnue.py \
  --data selfplay_mcts.npz \
  --epochs 20 --batch-size 8192 --lr 4e-3
```

`--data` オプションは NPZ ファイルを直接読み込む（棋譜パース不要）。
`--kifu` と `--data` は排他。自己対局データは既にソフトラベル付きなので bootstrap は不要。

### モデルの配備

```sh
cp nnue.bin build/Release/nnue.bin
```

---

## 4. MCTS エンジン (`kishi-to`)

Transformer で方策（次の一手の確率分布）と価値（勝率）を同時に学習します。
学習完了後に自動で ONNX 形式にエクスポートされます。

データは CPU に保持し、バッチごとに GPU に転送する方式のため、VRAM が小さい GPU (12GB 等) でも大規模データセットで学習可能です。

### クイックスタート

```sh
python tools/train.py \
  --kifu kifu/floodgate \
  --model nn_model.pt
```

### キャッシュ付き学習 (推奨)

棋譜パース (数分) とテンソル変換 (数十秒) の結果を `.npz` ファイルにキャッシュできます。
2回目以降はキャッシュから数秒でロードされます。

```sh
# 初回: パース + キャッシュ保存
python tools/train.py --kifu kifu/floodgate --cache kifu_cache.npz --epochs 10

# 2回目以降: キャッシュからロード (--kifu 不要)
python tools/train.py --cache kifu_cache.npz --epochs 10

# 別フィルタには別キャッシュを使う
python tools/train.py --kifu kifu/floodgate --min-rate 2000 --cache cache_2000.npz
```

`--cache` 指定時は全局面をサンプリングなしでキャッシュするため、bootstrap 学習にもそのまま使えます。

### ブートストラップ学習

学習済みモデル自身の評価値から悪手 (eval_drop) を検出し、ラベルを改善して再学習するセルフ評価方式です。

```sh
python tools/train.py --cache kifu_cache.npz --epochs 10 \
  --bootstrap-rounds 2 --bootstrap-blend 0.5 --drop-margin 0.8
```

### 内部の流れ

1. CSA 棋譜をパースし、盤面を 258 整数にエンコード (numpy 経由で高速変換)
2. 方策ラベル: 正解手の policy インデックス (0-2186)
3. 価値ラベル: 指し手側の勝敗 (+1/-1)
4. Transformer (d=128, 4層, 8ヘッド) を CrossEntropy + MSE で学習
5. `nn_model.pt` に保存
6. `nn_model.onnx` に自動エクスポート

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | - | CSA 棋譜のルートディレクトリ (`--cache` 未使用時は必須) |
| `--cache` | - | パース済みデータのキャッシュファイル (.npz) |
| `--model` | `nn_model.pt` | モデルの保存先 |
| `--output` | `nn_model.onnx` | ONNX 出力先 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--epochs` | 20 | エポック数 |
| `--batch-size` | 1024 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--min-rate` | 2500 | 最低レーティング (0=フィルタなし) |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--sample-rate` | 0.5 | 中盤局面のサンプリング率 |
| `--opening-n` | 40 | 序盤 N 手は必ず使用 |
| `--endgame-n` | 40 | 終盤 N 手は必ず使用 |
| `--resume` | - | 既存モデルから再開 |
| `--bootstrap-rounds` | 0 | セルフ評価ブートストラップ回数 (0=無効) |
| `--bootstrap-blend` | 0.5 | 勝敗とモデル評価値の混合比 (1=勝敗のみ) |
| `--drop-margin` | 0.8 | eval_drop がこの値で policy weight が最小になる |

### バージョン管理付き学習ループ (`train_loop.py`)

段階的に棋譜数を増やしながら学習と自己対戦評価を繰り返します。

```sh
python tools/train_loop.py \
  --kifu kifu/floodgate \
  --engine build/kishi-to \
  --rounds 10
```

1 ラウンドごとに:
1. 累積棋譜で `train.py --resume` を実行
2. `models/nn_model_v{N}.pt` に保存
3. 前バージョンと自己対戦 (勝率を比較)
4. `models/training_log.txt` に記録

### 自己対戦 (`self_play.py`)

```sh
python tools/self_play.py \
  --engine build/kishi-to \
  --model1 models/nn_model_v002.pt \
  --model2 models/nn_model_v001.pt \
  --games 20
```

### モデルの配備

```sh
cp nn_model.onnx build/Release/nn_model.onnx
```

### 手動 ONNX エクスポート

```sh
python tools/export_onnx.py --model nn_model.pt --output nn_model.onnx --verify
```

---

## 5. Alpha エンジン (`kishi-to-alpha`)

AlphaZero 方式の ResNet-SE + 改良 MCTS エンジンです。
教師あり学習 (SL) で初期モデルを作り、自己対局強化学習 (RL) で継続的に強化します。

### アーキテクチャ

```text
入力: [B, 45, 9, 9] float32
  28ch: 駒面 (14 駒種 × 2 色, one-hot)
  14ch: 持ち駒 (7 種 × 2 色, count/max 正規化)
   2ch: 利き数 (先手/後手, /8.0)
   1ch: 手番 (先手=1.0, 後手=0.0)

Initial Conv(45→192) → BN → ReLU
× 15 Residual Blocks:
  Conv(192→192) → BN → ReLU → Conv(192→192) → BN → SE(192→48→192) → Skip → ReLU

Value Head (WDL 3 クラス):
  Conv(192→1) → BN → ReLU → FC(81→256) → ReLU → FC(256→3)

Policy Head:
  Conv(192→27) → Flatten → 2187 logits (81 マス × 27 チャンネル)
```

- パラメータ数: 約 7-8M
- WDL (Win/Draw/Loss) 3 クラス出力。期待値 = wdl[0] - wdl[2]
- Policy: 既存と同じ 81×27=2187 エンコーディング
- SE (Squeeze-and-Excitation) ブロックにより、チャンネル間の相互作用を動的に調整

### ステップ 1: 教師あり学習 (SL)

Floodgate 棋譜から初期モデルを学習します。

```sh
python tools/train_alpha.py \
  --kifu kifu/floodgate \
  --model alpha_model.pt \
  --output alpha_model.onnx \
  --min-rate 3000 \
  --epochs 20
```

GPU が 2 枚ある場合は自動で DataParallel になります。

### ステップ 2: 自己対局 RL

#### 手動実行

```sh
# 1. 自己対局でデータ生成
python tools/alpha_self_play.py \
  --engine build/Release/kishi-to-alpha.exe \
  --model alpha_model.onnx \
  --games 200 --simulations 400 \
  --output selfplay_001.npz

# 2. 自己対局データから学習
python tools/train_alpha_rl.py \
  --data selfplay_001.npz selfplay_002.npz \
  --model alpha_model.pt \
  --output alpha_model.onnx
```

#### 自動 RL ループ (推奨)

自己対局→学習→評価→ゲートの繰り返しを自動化します。

```sh
python tools/alpha_train_loop.py \
  --engine build/Release/kishi-to-alpha.exe \
  --iterations 100
```

1 イテレーションの流れ:

1. 現モデルで自己対局 200 局 (400 sim/手)
2. リプレイバッファから学習 (直近 200 万局面, FIFO)
3. 新モデル vs 現ベストで 20 局評価
4. 勝率 55% 以上で昇格、以下なら棄却
5. 繰り返し

ログは `alpha_rl_work/train_log.jsonl` に記録されます。

### ステップ 3: CPU 用モデルの作成

GPU がない環境向けに、軽量モデルと INT8 量子化の 2 つの手段があります。

#### 知識蒸留 (小型モデル)

フルモデル (15 ブロック, 192ch) を教師として、小型モデル (10 ブロック, 128ch) を学習します。

```sh
python tools/train_alpha_small.py \
  --teacher alpha_model.pt \
  --data selfplay_*.npz \
  --output alpha_model_small
```

- 温度付きソフトターゲット (T=3.0) で教師の出力分布を模倣
- ソフト:ハード = 70%:30% のブレンド比率
- 圧縮率: 約 3x (7-8M → 2-3M パラメータ)

#### INT8 量子化

```sh
python tools/quantize_alpha.py \
  --model alpha_model.onnx \
  --output alpha_model_int8.onnx \
  --calibration-data kifu_cache.npz \
  --verify
```

- `--verify` で元モデルとの精度比較を実行
- Policy top-1 一致率 98%+、Value MAE +0.02 以内が目標

### ステップ 4: 定跡生成

高シミュレーションの MCTS で序盤を解析し、定跡ファイルを生成します。

```sh
python tools/gen_alpha_book.py \
  --engine build/Release/kishi-to-alpha.exe \
  --plies 12 --simulations 6400 --branches 3 \
  --output book.txt
```

BFS で上位 3 手を分岐・展開し、訪問数を定跡の重みとして出力します。

### オプション一覧

#### `train_alpha.py` (教師あり学習)

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | - | CSA 棋譜のルートディレクトリ (`--cache` 未使用時は必須) |
| `--cache` | - | パース済みデータのキャッシュファイル (.npz) |
| `--model` | `alpha_model.pt` | モデルの保存先 |
| `--output` | `alpha_model.onnx` | ONNX 出力先 |
| `--channels` | 192 | ResNet チャンネル数 |
| `--blocks` | 15 | Residual ブロック数 |
| `--epochs` | 20 | エポック数 |
| `--batch-size` | 1024 | バッチサイズ |
| `--lr` | 2e-4 | 学習率 |
| `--min-rate` | 2500 | 最低レーティング |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

#### `train_alpha_rl.py` (RL 学習)

| オプション | デフォルト | 説明 |
| ---------- | ---------- | ---- |
| `--data` | (必須) | 自己対局 .npz ファイル (複数指定可) |
| `--model` | `alpha_model.pt` | モデルパス (ロード・保存兼用) |
| `--output` | `alpha_model.onnx` | ONNX 出力先 |
| `--buffer-size` | 2000000 | リプレイバッファサイズ |
| `--epochs` | 5 | エポック数 |
| `--batch-size` | 1024 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |

#### `alpha_train_loop.py` (RL ループ)

| オプション | デフォルト | 説明 |
| ---------- | ---------- | ---- |
| `--engine` | (必須) | `kishi-to-alpha` 実行ファイル |
| `--work-dir` | `alpha_rl_work` | 作業ディレクトリ |
| `--iterations` | 100 | RL イテレーション数 |
| `--selfplay-games` | 200 | 自己対局数 |
| `--selfplay-sims` | 400 | 自己対局時シミュレーション数 |
| `--eval-games` | 20 | 評価対局数 |
| `--skip-eval` | - | 評価をスキップ (常に昇格) |
| `--resume` | - | 既存作業ディレクトリから再開 |

### モデルの配備

```sh
# GPU 推論用 (フルモデル)
cp alpha_model.onnx build/Release/alpha_model.onnx

# CPU 推論用 (小型モデル)
cp alpha_model_small.onnx build/Release/alpha_model.onnx

# CPU 推論用 (INT8 量子化)
cp alpha_model_int8.onnx build/Release/alpha_model.onnx
```

### Alpha 用 ONNX エクスポート

```sh
python tools/export_alpha_onnx.py --model alpha_model.pt --output alpha_model.onnx --verify
```

---

## 共通ツール

| ファイル | 役割 |
|---------|------|
| `tools/csa_parser.py` | Floodgate CSA 棋譜パーサー (全スクリプト共通) |
| `tools/export_mlp.py` | PyTorch MLP → テキスト重みファイル変換 |
| `tools/export_onnx.py` | PyTorch Transformer → ONNX 変換 |
| `tools/export_alpha_onnx.py` | PyTorch ResNet-SE → ONNX 変換 (Alpha 用) |
| `tools/self_play.py` | USI 自己対戦 (MCTS 用) |
| `tools/alpha_self_play.py` | 学習データ生成付き自己対局 (Alpha 用) |
| `tools/train_loop.py` | バージョン管理付き学習ループ (MCTS 用) |
| `tools/alpha_train_loop.py` | RL 学習ループ (Alpha 用) |
| `tools/quantize_alpha.py` | ONNX INT8 静的量子化 (Alpha 用) |
| `tools/train_alpha_small.py` | 知識蒸留で軽量モデル作成 (Alpha 用) |
| `tools/gen_alpha_book.py` | MCTS 解析ベースの定跡生成 (Alpha 用) |

---

## 関連ドキュメント

- [使い方](usage.md) — GUI 登録・USI オプション
- [Classic 評価関数](evaluation.md) — 98 次元線形特徴量の詳細
- [MCTS + Transformer](mcts.md) — ニューラルネット探索のアルゴリズム
