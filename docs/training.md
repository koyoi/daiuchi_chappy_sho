# 学習パイプライン

3 つのエンジンすべてが Floodgate CSA 棋譜から学習できます。
入力は共通、各エンジン専用のスクリプトがパース→学習→モデル保存を行います。

---

## 全体像

```text
Floodgate CSA 棋譜 (kifu/floodgate/*.csa)
        │
        ├── train_classic.py ──→ random-shogi.weights  (kishi-to-classic 用)
        │
        ├── train_gpu.py    ──→ gpu_model.pt           (kishi-to-gpu 用)
        │
        └── train.py        ──→ nn_model.pt            (kishi-to 用)
```

| エンジン | 学習スクリプト | モデルファイル | 評価方式 |
|---------|-------------|-------------|---------|
| Classic (αβ探索) | `train_classic.py` | `random-shogi.weights` | 74次元線形 |
| GPU (NN 1手評価) | `train_gpu.py` | `gpu_model.pt` | MLP (74→64→32→1) |
| MCTS (Transformer) | `train.py` | `nn_model.pt` | Transformer (方策+価値) |

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

## 1. Classic エンジン (`kishi-to-classic`)

74次元の手作り特徴量に対する線形重みを Bonanza 法で学習します。

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
5. `random-shogi.weights` に保存

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--engine` | (必須) | `kishi-to-classic` 実行ファイル |
| `--weights` | `random-shogi.weights` | 重みファイルパス |
| `--min-rate` | 1500 | 最低レーティング |
| `--max-games` | 0 | 最大棋譜数 (0=全部) |
| `--skip-opening` | 10 | 序盤 N 手をスキップ |
| `--sample-rate` | 0.5 | 局面のサンプリング率 |
| `--lr` | 0.01 | 学習率 |
| `--epochs` | 1 | エポック数 |
| `--temperature` | 100.0 | softmax 温度 |

### モデルの配備

```sh
cp random-shogi.weights build/random-shogi.weights
```

---

## 2. GPU エンジン (`kishi-to-gpu`)

Classic と同じ 74 次元特徴量を入力とする MLP を学習します。

### クイックスタート

```sh
python tools/train_gpu.py \
  --kifu kifu/floodgate \
  --engine build/kishi-to-classic
```

### 内部の流れ

1. CSA 棋譜をパースし、SFEN+USI 形式で抽出（Classic と共通）
2. `kishi-to-classic --extract-features` で 74 次元特徴量を抽出
   - 正解手の局面 → ラベル +1.0
   - ランダムな不正解手の局面 → ラベル -1.0
3. `gpu_eval.py train` で MLP (74→64→32→1) を BCEWithLogitsLoss で学習
4. `gpu_model.pt` に保存

### オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--kifu` | (必須) | CSA 棋譜のルートディレクトリ |
| `--engine` | (必須) | `kishi-to-classic` 実行ファイル |
| `--model` | `gpu_model.pt` | モデル保存先 |
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
cp gpu_model.pt build/gpu_model.pt
```

---

## 3. MCTS エンジン (`kishi-to`)

Transformer で方策（次の一手の確率分布）と価値（勝率）を同時に学習します。

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
| `--opening-n` | 20 | 序盤 N 手は必ず使用 |
| `--endgame-n` | 20 | 終盤 N 手は必ず使用 |
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
cp nn_model.pt build/nn_model.pt
```

---

## 共通ツール

| ファイル | 役割 |
|---------|------|
| `tools/csa_parser.py` | Floodgate CSA 棋譜パーサー (全スクリプト共通) |
| `tools/self_play.py` | USI 自己対戦 (MCTS 用) |
| `tools/train_loop.py` | バージョン管理付き学習ループ (MCTS 用) |

---

## 関連ドキュメント

- [使い方](usage.md) — GUI 登録・USI オプション
- [Classic 評価関数](evaluation.md) — 74 次元線形特徴量の詳細
- [MCTS + Transformer](mcts.md) — ニューラルネット探索のアルゴリズム
