# KishiTo — 将棋エンジン

C++17 製の将棋エンジンです。USI プロトコルに対応し、ShogiGUI / 将棋所から利用できます。

2 種類のエンジンを同梱しています。

| 実行ファイル | 思考方式 | 評価関数 |
|-------------|---------|---------|
| `kishi-to-classic` | αβ探索（反復深化＋静止探索＋置換表） | 44次元線形特徴量 |
| `kishi-to-gpu` | NN 1手評価（全合法手をNNでバッチ評価） | MLP（44→64→32→1） |
| `kishi-to` | MCTS（モンテカルロ木探索） | Transformer ニューラルネット |

## ビルド

```sh
cmake -B build
cmake --build build --config Release
```

Windows (MSVC) の場合:

```
build\Release\kishi-to-classic.exe
build\Release\kishi-to-gpu.exe
build\Release\kishi-to.exe
```

Linux / macOS の場合:

```
build/kishi-to-classic
build/kishi-to-gpu
build/kishi-to
```

## クイックスタート

```sh
# Classic エンジン（すぐ使える）
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to-classic

# GPU エンジン（gpu_model.pt + Python/PyTorch が必要）
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to-gpu

# MCTS エンジン（nn_model.pt + Python/PyTorch が必要）
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to
```

GPU エンジン・MCTS エンジンは Python + PyTorch 環境と学習済みモデルが必要です。モデルがない場合はランダムな重みで動作します。

## ソース構成

### 共通 (`shogi_core` ライブラリ)

| ファイル | 役割 |
|---------|------|
| `src/shogi_types.*` | 駒・手番・盤・指し手・Bitboard・Zobrist ハッシュ |
| `src/movegen.*` | Bitboard ベースの合法手生成・利き計算 |
| `src/position.*` | 初期局面・SFEN 解析・指し手適用 |
| `src/notation.*` | USI/CSA/SFEN 表記変換 |
| `src/text_util.*` | 文字列処理ユーティリティ |

### Classic エンジン (`kishi-to-classic`)

| ファイル | 役割 |
|---------|------|
| `src/engine.*` | αβ探索・静止探索・置換表・手の並べ替え |
| `src/evaluation.*` | 44 次元特徴量抽出と線形評価 |
| `src/learning.*` | 対局結果からのオンライン重み学習 |
| `src/usi_protocol.*` | USI ループ |
| `src/csa_protocol.*` | CSA ループ |

### GPU 評価エンジン (`kishi-to-gpu`)

| ファイル | 役割 |
|---------|------|
| `src/gpu_eval_engine.*` | NN 1手評価による指し手選択 |
| `src/gpu_bridge.*` | Python/PyTorch GPU 推論ブリッジ |
| `src/gpu_usi_protocol.*` | GPU エンジン用 USI ループ |

### MCTS エンジン (`kishi-to`)

| ファイル | 役割 |
|---------|------|
| `src/mcts.*` | MCTS 探索（PUCT選択・展開・逆伝播） |
| `src/nn_bridge.*` | Transformer 推論用 Python サブプロセス管理 |
| `src/mcts_engine.*` | MCTS エンジンラッパー |
| `src/mcts_usi_protocol.*` | MCTS 用 USI ループ |

### ツール

| ファイル | 役割 |
|---------|------|
| `tools/nn_eval.py` | Transformer モデルの推論サーバー |
| `tools/train.py` | CSA 棋譜からの学習 |
| `tools/train_loop.py` | バージョン管理付き学習ループ |
| `tools/self_play.py` | USI 自己対戦 |
| `tools/csa_parser.py` | Floodgate CSA 棋譜パーサー |
| `tools/gpu_eval.py` | Classic 用 GPU 推論（旧方式） |

## ドキュメント

- [使い方](docs/usage.md) — GUI 登録・USI オプション・CSA モード
- [Classic 評価関数](docs/evaluation.md) — 44 次元線形特徴量の詳細
- [MCTS + Transformer](docs/mcts.md) — ニューラルネット探索のアルゴリズム
- [学習パイプライン](docs/training.md) — 棋譜学習・バージョン管理・自己対戦

## USI 探索情報 (`info` 出力) の読み方

USI プロトコルで出力される `info depth ... nodes ...` の意味はエンジン・モードによって異なります。

| エンジン | `depth` | `nodes` |
|---------|---------|---------|
| Classic（αβ探索） | 反復深化で完了した探索深さ（半手単位） | 探索した局面数（search + quiescence の合計） |
| GPU（NN 1手評価） | 固定 `1`（1手先の全合法手をNNで一括評価） | 合法手の数（＝NNに渡した局面数） |
| MCTS（Transformer） | シミュレーション回数 | シミュレーション回数（depth と同値） |

- Classic αβでは depth が大きいほど深く読んでおり、nodes は計算量の指標です。
- GPU エンジンは探索木を作らず、現局面から1手先の全局面をNNでバッチ評価して最高スコアの手を選ぶため、depth は常に1です。
- MCTS では depth/nodes ともにシミュレーション（選択→展開→評価→逆伝播）の実行回数を表します。

## 未実装

千日手検出、入玉宣言、詰み専用探索、CSA サーバーへの TCP 接続。
