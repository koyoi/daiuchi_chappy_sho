# MCTS + Transformer (kishi-to)

MCTS エンジン (`kishi-to`) が使用するモンテカルロ木探索と Transformer ニューラルネットワークの詳細。

---

## 全体の流れ

```text
局面 → Transformer推論 → (value, policy) → MCTS探索 → bestmove
```

1. MCTS がノードを選択 (PUCT)
2. 未展開ノードを展開: 合法手を列挙し、Transformer から policy を取得
3. Transformer の value 出力で評価
4. 結果を逆伝播

探索 1 回につき Transformer 推論は 1〜2 回。αβ探索と異なり、全ノードで全幅探索せず、有望な手を集中的に読む。

---

## Transformer モデル

ソースコード: `tools/nn_eval.py`

### 入力表現

盤面を 96 個の整数で表現:

| 範囲 | 内容 | サイズ |
|------|------|--------|
| 0〜80 | 盤上の駒 (0=空, 1-14=先手, 15-28=後手) | 81 |
| 81〜87 | 先手持ち駒 (歩〜飛の枚数) | 7 |
| 88〜94 | 後手持ち駒 (歩〜飛の枚数) | 7 |
| 95 | 手番 (0=先手, 1=後手) | 1 |

### アーキテクチャ

```text
盤面81マス → piece_embed(29, 128) + file_embed(9, 128) + rank_embed(9, 128)
                ↓
          CNN前段 (Conv2d 3×3 × 2層 + 残差接続)
                ↓
持ち駒14枠 → hand_count_embed(20, 128) + hand_pos_embed(14, 128)
                ↓
          concat(board 81, hand 14) + side_embed(2, 128)
                ↓
          TransformerEncoder (4層, 8ヘッド, FFN=256)
              ↓              ↓
          mean pool      board部分(81×128)
             ↓                ↓
       value_head       policy_head
       128→64→1→Tanh   81×128→512→2187
```

- **CNN前段**: 9×9 盤面を画像として畳み込み、利き・紐付き等の局所パターンを自動検出
- **2D位置エンコーディング**: 筋 (file) と段 (rank) を分離した埋め込みで盤面の空間構造を明示
- **パラメータ数**: 約 740 万
- **value**: -1.0 (負け) 〜 +1.0 (勝ち) の評価値
- **policy**: 2187 次元 = 81 マス × 27 チャンネル

### Policy エンコーディング

`index = toSquare × 27 + channel`

| チャンネル | 内容 |
|-----------|------|
| 0〜9 | 移動方向 (成りなし): 上,下,左,右,左上,右上,左下,右下,桂左,桂右 |
| 10〜19 | 移動方向 (成り): 同上 |
| 20〜26 | 打ち: 歩,香,桂,銀,金,角,飛 |

方向は手番側の視点で計算される (先手は上が前進、後手は下が前進)。

---

## MCTS 探索

ソースコード: `src/mcts.cpp` / `src/mcts.h`

### PUCT 選択

```
UCB(child) = Q(child) + cPuct × prior(child) × √(parentVisits) / (1 + childVisits)
```

- `Q` = 訪問回数で割った累積価値 (勝率の近似)
- `cPuct` = 探索と活用のバランス (デフォルト 1.5)
- `prior` = Transformer の policy 出力 (softmax 済み)

訪問数が少ない子ノードは prior に引かれて選ばれやすく、訪問が増えると Q 値で選ばれるようになる。

### ディリクレノイズ

ルートノードの prior にディリクレ分布のノイズを加えて探索の多様性を確保:

```
prior' = (1 - ε) × prior + ε × Dir(α)
```

- α = 0.15, ε = 0.25 (デフォルト)

### 逆伝播

評価値は各階層で符号を反転して伝播する (自分の局面の勝ちは相手の局面の負け)。

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `simulations` | 800 | 1 手あたりのシミュレーション回数 |
| `cPuct` | 1.5 | 探索/活用バランス |
| `dirichletAlpha` | 0.15 | ノイズの集中度 |
| `dirichletEpsilon` | 0.25 | ノイズの混合率 |

---

## NN ブリッジ

ソースコード: `src/nn_bridge.cpp` / `src/nn_bridge.h`

C++ エンジンと Python の Transformer 推論サーバー間の通信。

### プロセス管理

- エンジン起動時に `nn_eval.py serve` を子プロセスとして起動
- stdin/stdout でパイプ通信 (stderr はログ用に親プロセスへ中継)
- `ready` 行の受信で初期化完了を確認
- エンジン終了時に `quit` を送信してクリーンアップ

### 通信プロトコル

```text
→ ready                              (起動完了)
← eval <96 ints>                     (1局面評価)
→ <value> <2187 policy floats>       (結果)
← batch <N> | <state1> | <state2>    (バッチ評価)
→ <value> <policy>                   (N行の結果)
← quit                               (終了)
```

### フォールバック

Python プロセスの起動に失敗した場合、一様分布の policy と value=0.0 を返す。

---

## 関連ドキュメント

- [使い方](usage.md) — USI オプションの一覧
- [Classic 評価関数](evaluation.md) — 線形特徴量方式
- [学習パイプライン](training.md) — Transformer の学習方法
