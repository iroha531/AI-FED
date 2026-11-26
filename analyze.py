#!/usr/bin/env python3
"""誤判定の確認 (analyze.py)

result/{モデル}/oof.csv から、
  ・見落とし (正解は「あり」なのに確率が低い) ワースト
  ・誤検出   (正解は「なし」なのに確率が高い) ワースト
の画像を並べた一覧画像を作る。次の改善の方針を決めるために、まず目で見て傾向をつかむ。

使い方 (main.py と同じフォルダで):
  python analyze.py                        # F1が最良のモデルを自動で選ぶ
  python analyze.py --model efficientnet_b0 --n 24
出力: result/{モデル}/errors_missed.png , errors_false_alarm.png
各画像の左上の数字は 予測確率 (正解/予測のしきい値は画面に表示)。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TILE, COLS = 192, 6


def montage(root, df, out_path, title):
    rows = (len(df) + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * TILE, rows * TILE), "white")
    for k, r in enumerate(df.itertuples()):
        im = Image.open(Path(root) / r.path).convert("RGB").resize((TILE, TILE))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 118, 16], fill="black")
        d.text((3, 2), f"p={r.prob:.2f} {Path(r.path).stem}", fill="yellow")
        canvas.paste(im, ((k % COLS) * TILE, (k // COLS) * TILE))
    canvas.save(out_path)
    print(f"{title}: {len(df)}枚 → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--result_dir", default="result")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=24, help="各一覧に並べる枚数")
    args = ap.parse_args()

    res = Path(args.result_dir)
    if args.model is None:
        scores = {d.name: json.load(open(d / "metrics.json"))["f1"]
                  for d in res.iterdir() if (d / "metrics.json").exists()}
        args.model = max(scores, key=scores.get)
    m = json.load(open(res / args.model / "metrics.json"))
    thr = m["threshold"]
    df = pd.read_csv(res / args.model / "oof.csv")
    print(f"モデル: {args.model} / F1 {m['f1']:.4f} / しきい値 {thr:.2f}")

    pred = df["prob"] >= thr
    fn, fp = df[(df.label == 1) & ~pred], df[(df.label == 0) & pred]
    print(f"正例 {int(df.label.sum())} 枚のうち 見落とし {len(fn)} 枚 / 誤検出 {len(fp)} 枚 (予測は正例 {int(pred.sum())} 枚)")
    montage(args.data_root, fn.sort_values("prob").head(args.n), res / args.model / "errors_missed.png", "見落とし")
    montage(args.data_root, fp.sort_values("prob", ascending=False).head(args.n),
            res / args.model / "errors_false_alarm.png", "誤検出")


if __name__ == "__main__":
    main()