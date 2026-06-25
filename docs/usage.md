# 使い方

## Windows GUI で使う

ShogiGUI か将棋所に USI エンジンとして登録するのが最も簡単です。

- ShogiGUI: <https://shogigui.siganus.com/>
- 将棋所: <https://shogidokoro.starfree.jp/>

登録する実行ファイル:

```text
build\Release\kishi-to-classic.exe   (Classic αβ探索)
build\Release\kishi-to.exe           (MCTS + Transformer)
build\Release\kishi-to-alpha.exe     (Alpha ResNet-SE + 改良MCTS)
```

エンジン起動時に作業ディレクトリを exe のある場所へ移すため、重みファイルやスクリプトは exe 基準の相対パスで解決されます。

### Classic エンジンの準備

`kishi-to-classic.exe` はそのままで動作します（線形評価）。

MLP 評価を使う場合は、exe と同じディレクトリに `mlp.weights` を配置し、USI オプション `MlpWeightsFile` にパスを設定します。

### MCTS エンジンの準備

`kishi-to.exe` を使うには、exe と同じディレクトリに以下を配置します:

- `nn_model.pt` — 学習済み Transformer モデル（[学習パイプライン](training.md) で作成）
- `tools/nn_eval.py` — 推論サーバースクリプト

Python + PyTorch がインストールされた環境が必要です。Python のパスは USI オプション `NNPython` で指定できます。

### Alpha エンジンの準備

`kishi-to-alpha.exe` を使うには、exe と同じディレクトリに以下を配置します:

- `alpha_model.onnx` — 学習済み ResNet-SE モデル（[学習パイプライン](training.md) で作成）

ONNX Runtime で推論するため、Python は不要です。ビルド時に `USE_ONNXRUNTIME=ON` と ONNX Runtime SDK が必要です。

GPU 推論 (CUDA) と CPU 推論の両方に対応しています。USI オプション `NNDevice` で切り替えます。CPU 用には INT8 量子化モデルや知識蒸留による小型モデルも利用可能です（[学習パイプライン](training.md) 参照）。

### 自己対戦時の注意 (Classic)

Classic エンジンで自己対戦する場合、先手と後手が同じ `linear.weights` を同時に書くと競合する可能性があります。以下のどちらかにしてください:

- 片方だけ `Learning=true`、もう片方は `Learning=false`
- 先手と後手で `WeightsFile` を別名にする

```text
先手: WeightsFile black.weights
後手: WeightsFile white.weights
```

---

## USI プロトコル

どちらのエンジンもデフォルトは USI モードです。

```text
usi
isready
position startpos moves 7g7f 3c3d
go movetime 3000
quit
```

`go` に対して `bestmove 2g2f` のように返します。

### 共通コマンド

- `usi` / `isready` / `usinewgame`
- `position startpos [moves ...]`
- `position sfen <sfen> [moves ...]`
- `go` / `go movetime <ms>` / `go btime <ms> wtime <ms> [byoyomi <ms>]`
- `stop` / `quit` / `gameover`

### Classic エンジンのオプション

```text
setoption name Learning value true|false
setoption name SearchDepth value 0..128
setoption name MaxMoveTimeMs value 50..600000
setoption name Threads value 1..256
setoption name HeavyEvaluation value true|false
setoption name OpeningSafety value true|false
setoption name WeightsFile value linear.weights
setoption name TrainingDataFile value mlp_training.tsv
setoption name MlpWeightsFile value mlp.weights
setoption name RootPruneWidth value 0..256
setoption name ReuseCache value true|false
```

**主要オプションの説明:**

- `MlpWeightsFile` — MLP 重みファイルのパス。設定するとリーフ評価が線形から MLP に切り替わる。空欄なら線形評価。
- `RootPruneWidth` — Root で上位何手をフル深さで読むか（デフォルト 15）。0 で無効化（全手フル深さ）。Late Move Reduction により有望手に探索時間を集中させる。
- `SearchDepth` — 読みの上限。`0` なら持ち時間内で可能な限り反復深化。
- `MaxMoveTimeMs` — GUI から持ち時間指定がない場合の 1 手あたり上限 (ms)。
- `Threads` — CPU 探索スレッド数。デフォルトは物理コア数 - 2。
- `HeavyEvaluation` — `false` にすると重い評価特徴を省略。弱くなるが短い持ち時間で安定。
- `OpeningSafety` — 序盤の罠回避ヒューリスティック。候補手の後に相手が即王手・大駒取り・危険な打ち込みをできる場合に減点。
- `Learning` — `true` で対局終了時に `linear.weights` を更新。
- `gameover win/lose` を受けると、勝った側の手を模倣・負けた側の手を抑制する方向に重みを更新。
- `ReuseCache` — `true` で前回の探索の置換表エントリを次の探索でも参照可能に（デフォルト `true`）。相手が予測手を指した場合に探索効率が向上。

### MCTS エンジンのオプション

```text
setoption name MaxMoveTimeMs value 50..600000
setoption name MctsSimulations value 1..100000
setoption name NNPython value python
setoption name NNScript value tools/nn_eval.py
setoption name NNModel value nn_model.pt
setoption name NNDevice value auto|cuda|cpu
setoption name ReuseTree value true|false
```

- `MctsSimulations` — 1 手あたりの MCTS シミュレーション回数（デフォルト 800）。
- `NNModel` — Transformer モデルファイルのパス。
- `NNDevice` — `auto` で CUDA があれば GPU、なければ CPU。
- `ReuseTree` — `true` で前回の MCTS ツリーを保持し、次の探索で再利用（デフォルト `true`）。相手の指した手に対応するサブツリーを新しいルートとして活用し、シミュレーション結果を引き継ぐ。

### Alpha エンジンのオプション

```text
setoption name MaxMoveTimeMs value 50..600000
setoption name MctsSimulations value 1..100000
setoption name NNModel value alpha_model.onnx
setoption name NNDevice value auto|cuda|cpu
setoption name MctsBatchSize value 1..64
setoption name FPUReduction value 0..100
setoption name TemperatureDropMove value 0..200
setoption name Book value true|false
setoption name ReuseTree value true|false
```

- `MctsSimulations` — 1 手あたりの MCTS シミュレーション回数（デフォルト 1600）。
- `NNModel` — ResNet-SE モデルファイルのパス（デフォルト `alpha_model.onnx`）。
- `NNDevice` — `auto` で CUDA があれば GPU、なければ CPU。
- `MctsBatchSize` — NN 推論のバッチサイズ（デフォルト 16）。
- `FPUReduction` — First Play Urgency 減算値 ×100（デフォルト 20 = 0.20）。未訪問ノードの Q を `parentQ - FPU` に設定。
- `TemperatureDropMove` — この手数以降は温度を 0 に下げて argmax 選択に切り替え（デフォルト 30）。0 で常時 argmax。
- `Book` — 定跡使用の有無（デフォルト `true`）。
- `ReuseTree` — MCTS ツリー再利用（デフォルト `true`）。前回の探索ツリーから相手の指した手に対応するサブツリーを引き継ぎ、探索効率を向上。

Alpha エンジン固有のコマンド:

- `getvisits` — 直前の探索の各手の訪問回数を出力（自己対局データ生成用）。形式: `visits 7g7f:1200:42 3c3d:300:15 ...`

---

## CSA プロトコル (Classic のみ)

Classic エンジンは CSA モードにも対応しています。

```sh
./build/kishi-to-classic --csa
```

対応入力:

- `LOGIN user pass` / `LOGOUT`
- `PI` / `P1`〜`P9` の盤面行
- `P+00FU...` / `P-00FU...` の持ち駒行
- `+` / `-` の手番行
- `Your_Turn:+` / `Your_Turn:-`
- `START:...`
- `+7776FU` / `-3334FU` 形式の指し手
- `go`

---

## 速度設定のガイド

| 持ち時間 | おすすめ設定 |
|---------|------------|
| 秒読み 10 秒以下 | Classic: `HeavyEvaluation=false` |
| 秒読み 30 秒以上 | Classic: `HeavyEvaluation=true` |
| 1 手 1〜3 秒 | MCTS: `MctsSimulations=200` |
| 1 手 5 秒以上 | MCTS: `MctsSimulations=800`（デフォルト） |
| 1 手 3〜5 秒 | Alpha: `MctsSimulations=800` |
| 1 手 5 秒以上 | Alpha: `MctsSimulations=1600`（デフォルト） |
| CPU 推論 | Alpha: INT8 量子化モデル + `MctsSimulations=400` |
