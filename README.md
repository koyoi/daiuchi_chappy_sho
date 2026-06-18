# learning-shogi-engine

USI と簡易 CSA を話す、C++17 製の将棋エンジンです。

局面特徴量に対する線形評価関数、αβ探索（反復深化＋静止探索＋置換表）、対局結果からのオンライン学習を持っています。勝った側の指し手を模倣し、負けた側の指し手を抑制する方向に `random-shogi.weights` を更新します。

この学習は、表形式 Q 学習ではありません。局面を丸ごと覚えず、駒得、持ち駒、成り駒、玉周辺、敵玉付近の駒などの特徴量に分解し、その重みを育てます。

## ソース構成

- `src/main.cpp`: 起動引数とプロトコル選択
- `src/usi_protocol.*`: USI 標準入出力ループ
- `src/csa_protocol.*`: CSA 標準入出力ループ
- `src/engine.*`: 思考ルーチン。反復深化αβ探索＋静止探索＋Zobrist置換表（固定配列＋ロックストライピング）
- `src/evaluation.*`: 局面特徴量と線形評価関数
- `src/learning.*`: 対局結果から評価重みを更新するオンライン学習
- `src/gpu_bridge.*`: Python/PyTorch GPU 推論・学習ブリッジ
- `src/movegen.*`: Bitboard ベースの疑似合法手生成、事前計算利きテーブル、王手判定、合法手フィルタ
- `src/position.*`: 初期局面、SFEN局面設定、指し手適用
- `src/notation.*`: USI/CSA/SFEN の表記変換
- `src/shogi_types.*`: 駒、手番、盤、指し手などの基本型。Bitboard (64+17ビット)、Zobrist ハッシュテーブル
- `src/text_util.*`: プロトコル入力用の小さな文字列処理

## ビルド

```sh
cmake -S . -B build
cmake --build build
```

Windows の Visual Studio Generator では実行ファイルは構成別ディレクトリに出る場合があります。

```sh
build\Debug\random-shogi-engine.exe
```

Ninja や Makefile では通常こちらです。

```sh
./build/random-shogi-engine
```

## USI

デフォルトは USI モードです。

```text
usi
isready
position startpos moves 7g7f 3c3d
go
quit
```

`go` に対して `bestmove 2g2f` のように返します。対応済みの主なコマンドは次の通りです。

- `usi`
- `isready`
- `setoption name Learning value true|false`
- `setoption name SearchDepth value 0..128`
- `setoption name MaxMoveTimeMs value 50..600000`
- `setoption name Threads value 1..256`
- `setoption name HeavyEvaluation value true|false`
- `setoption name OpeningSafety value true|false`
- `setoption name WeightsFile value random-shogi.weights`
- `setoption name TrainingDataFile value gpu_training.tsv`
- `setoption name UseGpu value true|false`
- `setoption name GpuTrainOnGameEnd value true|false`
- `setoption name GpuPython value python`
- `setoption name GpuScript value tools/gpu_eval.py`
- `setoption name GpuModel value gpu_model.pt`
- `setoption name GpuDevice value auto|cuda|cpu`
- `usinewgame`
- `position startpos ...`
- `position sfen ...`
- `go`
- `stop`
- `quit`
- `gameover`

`gameover win` / `gameover lose` を受けると、その対局中の自分と相手の指し手から `random-shogi.weights` を更新します。勝った側の手は模倣方向、負けた側の手は抑制方向に評価関数の重みを動かします。

`OpeningSafety=true` は序盤用の罠回避ヒューリスティックです。定跡手順は持たず、候補手の後に相手が即王手、大駒取り、成り、玉周辺への危険な打ち込みをできる場合に減点します。序盤ほど強く働き、手数が進むと自動的に弱まります。

学習ファイルは起動時に自動ロードされ、対局終了時に保存されます。別の重みファイルを使う場合は次のように指定します。

```text
setoption name WeightsFile value my-engine.weights
```

## Windows GUI で使う

Windowsでは、まず `ShogiGUI` か `将棋所` にUSIエンジンとして登録するのが手早いです。

- ShogiGUI: https://shogigui.siganus.com/
- 将棋所: https://shogidokoro.starfree.jp/

登録する実行ファイルは、Visual Studio Generatorでビルドした場合は次です。

```text
C:\noscan\kishi_to\build\Debug\random-shogi-engine.exe
```

ShogiGUIの場合は、エンジン管理画面から上記exeを追加します。対人戦では片側を人間、片側を `LearningShogiEngine` にします。自己対戦では先手・後手の両方に `LearningShogiEngine` を指定します。

将棋所の場合も、対局用エンジンとして上記exeを登録します。エンジン同士の対局機能を使えば、別プロセスで2個起動して自己対戦できます。

自己対戦で学習させる場合、先手エンジンと後手エンジンが同じ `random-shogi.weights` を同時に書くと競合する可能性があります。最初は次のどちらかにしてください。

- 片方だけ `Learning=true`、もう片方は `Learning=false`
- 先手と後手で `WeightsFile` と `TrainingDataFile` を別名にする

例:

```text
先手: WeightsFile black.weights / TrainingDataFile black_gpu_training.tsv
後手: WeightsFile white.weights / TrainingDataFile white_gpu_training.tsv
```

エンジン起動時に作業ディレクトリをexeのある場所へ移すため、相対パスの重みファイルや `tools/gpu_eval.py` はエンジンexe基準で扱われます。

## GPU 推論・学習

GPUを使う場合は Python と PyTorch を用意し、USIで次を指定します。

```text
setoption name UseGpu value true
setoption name GpuPython value python
setoption name GpuScript value tools/gpu_eval.py
setoption name GpuModel value gpu_model.pt
setoption name GpuDevice value auto
```

`UseGpu=true` のとき、エンジンは候補手ごとの局面特徴量をまとめて `tools/gpu_eval.py score` に渡し、PyTorchモデルのスコアで手を選びます。Python、PyTorch、CUDA、モデルファイルのいずれかが使えない場合は、静かにC++側の評価関数と探索へフォールバックします。

対局終了時にGPUモデルも更新したい場合は、次を指定します。

```text
setoption name GpuTrainOnGameEnd value true
setoption name TrainingDataFile value gpu_training.tsv
```

`gameover win` / `gameover lose` で `gpu_training.tsv` に教師データを追記し、`GpuTrainOnGameEnd=true` なら `tools/gpu_eval.py train` を呼んで `gpu_model.pt` を更新します。終了時学習は `UseGpu=false` のままでも実行されます。手動でまとめて学習する場合は次のように実行できます。

```sh
python tools/gpu_eval.py train --data gpu_training.tsv --model gpu_model.pt --device auto
```

推論スクリプト単体の形式は次です。

```sh
python tools/gpu_eval.py score --input features.tsv --output scores.tsv --model gpu_model.pt --device auto
```

## CSA

CSA は標準入出力で動く簡易モードです。

```sh
./build/random-shogi-engine --protocol csa
```

対応している主な入力は次の通りです。

- `LOGIN user pass`
- `LOGOUT`
- `PI`
- `P1` から `P9` の盤面行
- `P+00FU...` / `P-00FU...` の持ち駒行
- `+` / `-` の手番行
- `Your_Turn:+` / `Your_Turn:-`
- `START:...`
- `+7776FU` / `-3334FU` 形式の指し手
- `go`

エンジンの手番で `go` または `START:` を受け取ると、`+7776FU` のような CSA 指し手を返します。

## 学習の考え方

1手ごとに、実際に指された手の後の特徴量と、その局面の合法手全体の平均特徴量を比較します。勝った側の手ならその差分を強め、負けた側の手なら弱めます。

これにより、相手が勝った場合は相手の良かった手も評価関数に取り込み、自分が負けた場合は自分の悪かった手を避ける方向に学習します。同一エンジン同士の自己対局でも、勝敗が付けば同じ仕組みで更新されます。

現在のパラメータ数は `src/evaluation.h` の `FeatureCount` で決まります。重みだけで頭打ちになった場合は、特徴量を増やすか、評価器をニューラルネット等に差し替えるのが次の段階です。

## 速度設定

`SearchDepth` は読みの上限です。`0` の場合は、持ち時間内で可能な限り反復深化します。`MaxMoveTimeMs` はGUIから持ち時間指定が来ない場合の1手あたり上限です。ShogiGUIから `go btime ... wtime ... byoyomi ...` が来る場合は、その持ち時間から自動で考慮時間を決めます。

`Threads` はCPU探索スレッド数です。デフォルトはWindowsでは物理コア数を検出し、`物理コア数 - 2` にします。検出できない環境では論理スレッド数から控えめに推定します。並列化はルート合法手ごとの探索に対して行います。

`HeavyEvaluation=false` にすると、合法手を追加生成して数える重い評価特徴を止めます。弱くなりますが、短い持ち時間ではこちらの方が安定します。

探索には、時間制御、短い詰み探索、静止探索（取り合い延長）、MVV-LVA による手の並べ替え、Zobrist ハッシュ差分更新による固定サイズ置換表（2^20 エントリ、64 分割ロックストライピング、世代管理）、Bitboard ベースの利き計算、駒割り差分更新、ルート並列探索が入っています。

## 実装範囲

合法手生成では、駒の移動、成り、不成、持ち駒打ち、自玉を王手にさらす手の除外、二歩、行き所のない駒打ち、打ち歩詰めの除外を扱っています。

千日手、入玉宣言、CSAサーバへのTCP接続、詰み専用探索、ニューラルネット評価は未実装です。
