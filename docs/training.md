# 学習パイプライン

Transformer モデルの学習に関するツール群の説明。

---

## 概要

```text
Floodgate CSA 棋譜
        ↓
   csa_parser.py (パース)
        ↓
   train.py (学習 → nn_model.pt)
        ↓
   train_loop.py (バージョン管理 + 自己対戦評価)
        ↓
   nn_model.pt → kishi-to.exe で使用
```

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

## 単発学習 (`train.py`)

CSA 棋譜から直接 Transformer を学習します。

```sh
python tools/train.py --kifu kifu/floodgate --model nn_model.pt
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--model` | `nn_model.pt` | モデルの保存先 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--epochs` | 20 | エポック数 |
| `--batch-size` | 256 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--min-rate` | 0 | 最低レーティング (0=フィルタなし) |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--sample-rate` | 0.3 | 中盤局面のサンプリング率 |
| `--opening-n` | 20 | 各棋譜の序盤 N 手は必ず使用 |
| `--endgame-n` | 20 | 各棋譜の終盤 N 手は必ず使用 |
| `--resume` | - | 既存モデルから再開 |

### サンプリング戦略

全局面を均等に使うのではなく、序盤と終盤を重視:

- **序盤 20 手**: 必ず含める (定跡の学習)
- **終盤 20 手**: 必ず含める (寄せの学習)
- **中盤**: 30% の確率でサンプリング (局面の偏りを防ぐ)

### 進捗表示

```text
Device: cuda
Found 60485 CSA files
  Parsing 1000/60485... (28341 samples)
  ...
Training: 185234 samples, 20 epochs, batch=256, lr=0.0001
  epoch 1/20: [=====>              ]  25% | v=0.8523 p=5.1234
  epoch 1/20: value_loss=0.7891 policy_loss=4.8765 lr=0.000100 (42.3s)
  ...
Saved model to nn_model.pt
Model parameters: 6,987,788
```

---

## バージョン管理付き学習ループ (`train_loop.py`)

500 局 × 5 エポックを 1 ラウンドとして、段階的に学習とモデルの性能評価を行います。

```sh
python tools/train_loop.py \
  --kifu kifu/floodgate \
  --engine build/Release/kishi-to.exe \
  --rounds 10
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--engine` | (必須) | USI エンジン実行ファイル |
| `--rounds` | 10 | 学習ラウンド数 |
| `--games-per-round` | 500 | 1 ラウンドあたりの棋譜数 (累積) |
| `--epochs` | 5 | 1 ラウンドのエポック数 |
| `--eval-games` | 20 | 自己対戦の局数 |
| `--movetime` | 1000 | 自己対戦の 1 手あたり時間 (ms) |
| `--models-dir` | `models` | モデル保存ディレクトリ |
| `--device` | `auto` | PyTorch デバイス |
| `--python` | (現在の Python) | Python インタープリタ |
| `--nn-python` | - | エンジン用の Python パス |
| `--min-rate` | 0 | 最低レーティング |
| `--sample-rate` | 0.3 | 中盤サンプリング率 |

### 1 ラウンドの流れ

1. `train.py --resume` で累積学習 (ラウンド N は最初の N×500 局を使用)
2. `models/nn_model_v{N}.pt` に保存
3. `models/nn_model.pt` にコピー (最新版)
4. N≥2: v(N) vs v(N-1) で自己対戦
5. N≥3: v(N) vs v(N-2) で自己対戦
6. 結果を `models/training_log.txt` に記録

### 出力ディレクトリ

```text
models/
  nn_model_v001.pt
  nn_model_v002.pt
  nn_model_v003.pt
  nn_model.pt          ← 最新版のコピー
  training_log.txt     ← 学習ログ + 対戦結果
```

### 出力例

```text
==================================================
=== Round 1/10 (v001) ===
==================================================
Training with 500 games, 5 epochs...
  epoch 1/5: value_loss=0.8234 policy_loss=5.1234 lr=0.000100 (38.2s)
  ...
Saved models/nn_model_v001.pt (190s)
(No previous model for comparison)

==================================================
=== Round 2/10 (v002) ===
==================================================
Training with 1000 games, 5 epochs...
  ...
Saved models/nn_model_v002.pt (245s)
Evaluating v002 vs v001 (20 games)...
  v002 vs v001: 12W-6L-2D (60.0%) -> improved
```

---

## 自己対戦 (`self_play.py`)

2 つのモデルを USI プロトコルで対局させて勝率を比較します。

```sh
python tools/self_play.py \
  --engine build/Release/kishi-to.exe \
  --model1 models/nn_model_v002.pt \
  --model2 models/nn_model_v001.pt \
  --games 20 \
  --movetime 1000
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--engine` | (必須) | USI エンジン実行ファイル |
| `--model1` | - | エンジン 1 のモデル |
| `--model2` | - | エンジン 2 のモデル |
| `--games` | 20 | 対局数 |
| `--movetime` | 1000 | 1 手あたり時間 (ms) |
| `--python` | - | エンジン用 Python パス |
| `--device` | - | PyTorch デバイス |

### 動作

- 同じエンジン実行ファイルを 2 プロセス起動
- `setoption name NNModel value <path>` でモデルを切り替え
- 先後を毎局交替
- `bestmove resign` で投了、512 手で引き分け
- model1 の勝率 50% 以上なら終了コード 0、未満なら 1

---

## CSA パーサー (`csa_parser.py`)

Floodgate 形式の CSA 棋譜を解析するモジュール。`train.py` から呼ばれます。

### 機能

- 棋譜ファイルから盤面・指し手・結果を抽出
- `'summary:` 行から勝敗を判定
- レーティングフィルタ (`min_rate`)
- 96 整数の盤面エンコーディングと policy インデックスを出力

---

## モデルの配備

学習したモデルを MCTS エンジンで使うには、`nn_model.pt` をエンジン実行ファイルと同じディレクトリに配置します。

```sh
cp models/nn_model.pt build/Release/nn_model.pt
```

エンジンは起動時に作業ディレクトリを exe の場所に変更するため、相対パス `nn_model.pt` で自動的にロードされます。ShogiGUI / 将棋所からの起動でも同様です。

---

## 関連ドキュメント

- [使い方](usage.md) — USI オプションの一覧
- [MCTS + Transformer](mcts.md) — 探索アルゴリズムの詳細
- [Classic 評価関数](evaluation.md) — 線形特徴量方式
