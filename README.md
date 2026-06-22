# KishiTo — 将棋エンジン

C++17 製の将棋エンジンです。USI プロトコルに対応し、ShogiGUI / 将棋所から利用できます。

2 種類のエンジンを同梱しています。

| 実行ファイル | 思考方式 | 評価関数 |
|-------------|---------|---------|
| `kishi-to-classic` | αβ探索（反復深化＋静止探索＋置換表） | 74次元線形 or MLP（74→64→32→1） |
| `kishi-to` | MCTS（モンテカルロ木探索） | Transformer ニューラルネット |

Classic エンジンは `mlp.weights` ファイルを配置すると MLP 評価に切り替わり、非線形パターン（駒の連携・囲い強度など）を捉えた高精度な評価が可能になります。MLP がない場合は従来の線形評価で動作します。

## ビルド

```sh
cmake -B build
cmake --build build --config Release
```

Windows (MSVC) の場合:

```
build\Release\kishi-to-classic.exe
build\Release\kishi-to.exe
```

Linux / macOS の場合:

```
build/kishi-to-classic
build/kishi-to
```

## クイックスタート

```sh
# Classic エンジン（すぐ使える）
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to-classic

# Classic エンジン（MLP 評価を使用）
echo -e "usi\nsetoption name MlpWeightsFile value mlp.weights\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to-classic

# MCTS エンジン（nn_model.pt + Python/PyTorch が必要）
echo -e "usi\nisready\nposition startpos\ngo movetime 3000\nquit" | ./build/kishi-to
```

MCTS エンジンは Python + PyTorch 環境と学習済みモデルが必要です。モデルがない場合はランダムな重みで動作します。

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
| `src/engine.*` | αβ探索・静止探索・置換表・手の並べ替え・Root LMR |
| `src/evaluation.*` | 74 次元特徴量抽出、線形評価 / MLP 評価 |
| `src/learning.*` | 対局結果からのオンライン重み学習 |
| `src/usi_protocol.*` | USI ループ |
| `src/csa_protocol.*` | CSA ループ |

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
| `tools/train.py` | CSA 棋譜からの MCTS 学習 |
| `tools/train_classic.py` | CSA 棋譜からの Classic 線形学習 |
| `tools/train_mlp.py` | CSA 棋譜からの MLP 学習 |
| `tools/export_mlp.py` | PyTorch MLP → テキスト重みファイル変換 |
| `tools/mlp_eval.py` | MLP モデル定義・学習 |
| `tools/train_loop.py` | バージョン管理付き学習ループ |
| `tools/self_play.py` | USI 自己対戦 |
| `tools/csa_parser.py` | Floodgate CSA 棋譜パーサー |

## ドキュメント

- [使い方](docs/usage.md) — GUI 登録・USI オプション・CSA モード
- [Classic 評価関数](docs/evaluation.md) — 74 次元特徴量の詳細
- [MCTS + Transformer](docs/mcts.md) — ニューラルネット探索のアルゴリズム
- [学習パイプライン](docs/training.md) — 棋譜学習・バージョン管理・自己対戦

## USI 探索情報 (`info` 出力) の読み方

USI プロトコルで出力される `info depth ... nodes ...` の意味はエンジンによって異なります。

| エンジン | `depth` | `nodes` |
|---------|---------|---------|
| Classic（αβ探索） | 反復深化で完了した探索深さ（半手単位） | 探索した局面数（search + quiescence の合計） |
| MCTS（Transformer） | シミュレーション回数 | シミュレーション回数（depth と同値） |

- Classic αβでは depth が大きいほど深く読んでおり、nodes は計算量の指標です。
- MCTS では depth/nodes ともにシミュレーション（選択→展開→評価→逆伝播）の実行回数を表します。

## Classic エンジンの USI オプション

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `MlpWeightsFile` | string | (空) | MLP 重みファイルのパス。設定するとリーフ評価が MLP に切り替わる |
| `RootPruneWidth` | spin | 15 | Root で上位何手をフル深さで読むか（0=無効、Late Move Reduction） |
| `SearchDepth` | spin | 0 | 最大探索深さ（0=時間制限のみ） |
| `MaxMoveTimeMs` | spin | 1000 | 1手あたりの最大思考時間 (ms) |
| `Threads` | spin | (自動) | 探索スレッド数 |
| `HeavyEvaluation` | check | false | 重い特徴量（合法手生成ベース）を有効化 |
| `OpeningSafety` | check | true | 序盤の安全性ペナルティ |
| `Learning` | check | true | オンライン学習の有効/無効 |
| `WeightsFile` | string | random-shogi.weights | 線形重みファイルのパス |

## 未実装

千日手検出、入玉宣言、詰み専用探索、CSA サーバーへの TCP 接続。
