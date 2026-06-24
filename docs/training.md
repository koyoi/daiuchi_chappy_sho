# 学習パイプライン

4 つのエンジンが Floodgate CSA 棋譜から学習できます。
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
        └── train.py        ──→ nn_model.pt + nn_model.onnx (MCTS Transformer)
```

| エンジン | 学習スクリプト | 出力ファイル | 評価方式 |
|---------|-------------|-------------|---------|
| Classic (αβ探索 線形) | `train_classic.py` | `linear.weights` | 98次元線形 |
| Classic (αβ探索 MLP) | `train_mlp.py` | `mlp.weights` | MLP (98→128→64→1) |
| NNUE (αβ探索) | `train_nnue.py` | `nnue.bin` | NNUE (2344→256→64→32→1) |
| MCTS (Transformer) | `train.py` | `nn_model.onnx` | Transformer (方策+価値) |

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
2344 次元の盤面特徴量（駒種×位置 + 持ち駒）から差分計算可能なネットワーク (256→64→32→1) で局面を評価します。

### クイックスタート

```sh
python tools/train_nnue.py --kifu kifu/floodgate
```

GPU (RTX 4070 等) がある場合はバッチサイズを大きくすると高速化できます:

```sh
python tools/train_nnue.py --kifu kifu/floodgate --batch-size 65536 --epochs 10 --lr 4e-3
```

### 内部の流れ

1. CSA 棋譜をパースし、各局面の盤面＋持ち駒を特徴量に変換
2. 勝敗ラベル (+1/-1) で勝率予測を学習 (BCEWithLogitsLoss)
3. `nnue_model.pt` (PyTorch チェックポイント) に保存
4. NNU2 バイナリ形式 (`nnue.bin`) にエクスポート
   - L0 重みは int16 量子化 (scale=64)、L0 バイアスは int32
   - L1-L3 は float32

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--output` | `nnue.bin` | エンジン用バイナリ重みファイル |
| `--model-pt` | `nnue_model.pt` | PyTorch チェックポイント |
| `--min-rate` | 1500 | 最低レーティング |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--skip-opening` | 10 | 序盤 N 手をスキップ |
| `--sample-rate` | 0.3 | 局面のサンプリング率 |
| `--epochs` | 5 | 学習エポック数 |
| `--batch-size` | 4096 | バッチサイズ (GPU ありなら 65536 推奨) |
| `--lr` | 1e-3 | 学習率 |
| `--workers` | 0 | パース並列数 (0=自動) |
| `--resume` | - | 既存チェックポイントから再開 |

### モデルの配備

```sh
cp nnue.bin build/Release/nnue.bin
```

---

## 4. MCTS エンジン (`kishi-to`)

Transformer で方策（次の一手の確率分布）と価値（勝率）を同時に学習します。
学習完了後に自動で ONNX 形式にエクスポートされます。

### クイックスタート

```sh
python tools/train.py \
  --kifu kifu/floodgate \
  --model nn_model.pt
```

### 内部の流れ

1. CSA 棋譜をパースし、盤面を 258 整数にエンコード
2. 方策ラベル: 正解手の policy インデックス (0-2186)
3. 価値ラベル: 指し手側の勝敗 (+1/-1)
4. Transformer (d=128, 4層, 8ヘッド) を CrossEntropy + MSE で学習
5. `nn_model.pt` に保存
6. `nn_model.onnx` に自動エクスポート

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
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

## 共通ツール

| ファイル | 役割 |
|---------|------|
| `tools/csa_parser.py` | Floodgate CSA 棋譜パーサー (全スクリプト共通) |
| `tools/export_mlp.py` | PyTorch MLP → テキスト重みファイル変換 |
| `tools/export_onnx.py` | PyTorch Transformer → ONNX 変換 |
| `tools/self_play.py` | USI 自己対戦 (MCTS 用) |
| `tools/train_loop.py` | バージョン管理付き学習ループ (MCTS 用) |

---

## 関連ドキュメント

- [使い方](usage.md) — GUI 登録・USI オプション
- [Classic 評価関数](evaluation.md) — 98 次元線形特徴量の詳細
- [MCTS + Transformer](mcts.md) — ニューラルネット探索のアルゴリズム
