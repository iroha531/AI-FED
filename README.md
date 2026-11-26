# AI-FED: Pコン跡の有無判定

SDNET2018 の壁面画像(256×256)に、コンクリート施工時のPコン跡が含まれるかを判定し、1(あり)/0(なし)を出力します。評価指標は F1 スコアです。

最終的な提出ファイルは、周囲のタイルを使う ResNet18(`resnet18_ctx`)と、使わない EfficientNet-B0 をブレンドし、しきい値を決めて作成します。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `main.py` | 学習・予測・結果の保存・上位3モデルの表示 |
| `resnet18_ctx-optimize.py` | ブレンドとしきい値の後処理(最終提出ファイルを作成) |
| `analyze.py` | 見落とし・誤検出のワーストを一覧画像にする(提出ファイルの生成には不要) |
| `train_label.csv` / `test.txt` / `sample_submission.csv` | 配布データ |

画像(`W/`)、学習済みの重み(`*.pt`)、実行結果(`result/`)は `.gitignore` で除外しています。

## 動作環境

- Python 3.11、NVIDIA RTX 4090(CUDA)
- PyTorch 2.3 以上、torchvision、scikit-learn、pandas、numpy、pillow(動作確認したバージョン: ＜記入＞)
- 初回の学習時に、torchvision の ImageNet 学習済み重み(ResNet18 / EfficientNet-B0)をダウンロードするため、インターネット接続が必要です。
- 任意: `--clahe` を使う場合のみ `opencv-python-headless` が必要です。

```
pip install "torch>=2.3" torchvision scikit-learn pandas numpy pillow
```

## 事前準備

SDNET2018 の壁面画像(`W/`)と、`train_label.csv`、`test.txt`、`sample_submission.csv` を、スクリプトと同じフォルダに置きます。配布の `AI-FED-main` をそのまま使えます。

```
AI-FED-main/
├── W/            # 壁面画像 (CW/ と UW/)
├── train_label.csv
├── test.txt
├── main.py
└── resnet18_ctx-optimize.py
```

## 提出ファイルの再現手順

```
python main.py --models efficientnet_b0 --no_ensemble
python main.py --models resnet18 --context --no_ensemble
python resnet18_ctx-optimize.py
```

1. 周囲のタイルを使わない EfficientNet-B0 を、5分割で学習します(`result/efficientnet_b0/`)。
2. 周囲8タイルを貼り合わせた画像を入力にする ResNet18 を学習します(`result/resnet18_ctx/`)。
3. 2つのモデルの予測をブレンドし、しきい値を決めて提出ファイルを作成します。

提出ファイルは `result/resnet18_ctx_opt/submission.csv` です(ヘッダーなし、`test.txt` の順、1=Pコン跡あり)。

所要時間の目安(RTX 4090): EfficientNet-B0 約6分、ResNet18(周囲あり) ＜実測値を記入＞、後処理は数秒です。

> 乱数seedは42で固定していますが、GPUの演算の非決定性により、実行ごとにスコアがわずかに変わる場合があります。

`--no_ensemble` は、再現時にブレンドの候補を EfficientNet-B0 だけにするために付けています。

## 結果の保存先

`./result/{モデル名}/` に、次のファイルが保存されます。

- `submission.csv`: 提出用
- `metrics.json`: F1、AP、しきい値、fold別F1、所要時間
- `oof.csv`: 学習画像の検証予測
- `test_probs.csv`: テスト画像の予測確率
- `fold*.pt`: CNNの重み

## 手法の概要

### 1. 評価データと同じ分け方のCV

学習用と評価用は撮影元の写真単位で分かれているため、検証も写真番号(`7071-82.jpg` の `7071`)単位で5分割します(`StratifiedGroupKFold`)。タイル単位で分けると、隣のタイルが学習と検証にまたがり、精度が高く出すぎます。

### 2. 周囲8タイルの貼り合わせ(`--context`)

見落としの多くは、穴の本体が隣のタイルにあり、縁の欠片しか写っていないものでした。1枚の写真は18列×14行=252タイルで、連番が左→右、上→下に並ぶことを、タイルの縁の連続性から確認しています。

- 自分のタイルの周囲8タイルの縁64pxを貼り合わせた、384×384の画像を入力にします(`--margin` で変更可)。
- 4つ目のチャンネルに「判定するタイルの範囲」を示すマスクを追加し、学習済みモデルの最初の畳み込みを4チャンネルに拡張します(RGBの重みは引き継ぎ、4つ目は0で初期化)。
- 写真の端で隣がない部分は、灰色で埋めます。
- ファイル名の連番から隣接関係を復元して使っています。

### 3. ブレンドとしきい値(`resnet18_ctx-optimize.py`)

- 周囲を使うモデルと、使わないモデルは間違い方が違うため、予測をlogit空間で平均します。ベースの重みを0.9〜0.5で探索し、改善が小さい場合はブレンドしません。
- しきい値は、F1が最大の1点ではなく、前後±0.05の平均F1が最大の点を選びます(正例が約2%と少なく、F1が揺れやすいため)。

検証(撮影元の写真単位のCV)の結果は次のとおりです。

| | F1 | AP |
| --- | --- | --- |
| `resnet18_ctx` 単独 | 0.8895 | 0.9562 |
| ブレンド後 | 0.9430 | 0.9777 |

手元のF1は、検証で設定を選んでいるため、やや楽観的な値です。

### 4. 学習設定

- 損失: `BCEWithLogitsLoss`(正例の重み `pos_weight` = min(√(負例/正例), 10))
- 最適化: AdamW(学習率 3e-4、weight decay 1e-4)、コサイン減衰、8エポック、バッチ64、AMP
- 拡張: 反転、90度回転、明るさ・コントラスト・彩度の変動(端の穴が切れないよう、切り抜き系は使いません)
- 検証のAPが最良のエポックを採用し、予測は4方向の反転でTTAします。

## その他のオプション

```
python main.py --models resnet18 efficientnet_b0 --context --strong_aug   # 明るさ・ガンマの拡張を強める
python main.py --models efficientnet_b0 --clahe                           # CLAHE(要 opencv-python-headless)
python main.py --limit 600 --epochs 2                                     # 動作確認用(提出には使えません)
python analyze.py                                                         # 見落とし・誤検出の一覧画像
python resnet18_ctx-optimize.py --retrain                                 # 前処理を強めた学習を追加して後処理
```

## 補足

- 周囲タイルの利用は、ファイル名の連番に依存しています。
- 試したが採用しなかった方法: 隣のタイルの予測確率で判定し直す2段階目(スタッキング)は、CNNでは効果がありませんでした(F1 0.9251 → 0.9208)。
