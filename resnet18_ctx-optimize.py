#!/usr/bin/env python3
"""resnet18_ctx をベースにした前処理・後処理の最適化 (resnet18_ctx-optimize.py)

main.py と同じフォルダで実行する。再学習なしなら数秒で終わる。
  python resnet18_ctx-optimize.py            # 後処理だけ (すぐ終わる)
  python resnet18_ctx-optimize.py --retrain  # 前処理を強めた resnet18_ctx_aug を追加学習 → 後処理

やること:
  [前処理] --retrain: main.py を呼び、明るさ・コントラスト・ガンマの拡張を強めた resnet18_ctx_aug を作る
           (--clahe も付ければ CLAHE も入れる)。既存の結果は上書きしない (ensemble も触らない)。
  [後処理1] ブレンド: result/ 内の他モデル (ensemble など) と logit 平均を取り、
            「ベースの重み x 相手」を OOF のF1で選ぶ。改善が小さければベース単独のまま
            (周囲タイルを使う ctx と、使わないモデルは間違い方が違うので、混ぜると補い合う)
  [後処理2] しきい値: F1最大の1点ではなく、前後±0.05の平均F1が最大の点を選ぶ
            (正例が少なくF1が揺れやすいので、尖った点を避けて本番で落ちにくくする)
結果: result/resnet18_ctx_opt/ (submission.csv など。main.py と同じ形式)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from main import group_id, load_result, save_result

GRID = np.arange(0.05, 0.96, 0.01)
HALF = 5  # しきい値の平滑化: 前後5点 (±0.05)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def blend(ps, ws):
    """logit空間での加重平均 (重みの合計は1)"""
    return 1 / (1 + np.exp(-sum(w * logit(p) for w, p in zip(ws, ps))))


def robust_threshold(y, p):
    """(しきい値, その点のF1, 平滑化F1, 尖った最大F1) を返す"""
    f = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in GRID])
    sm = pd.Series(f).rolling(2 * HALF + 1, center=True, min_periods=1).mean().values
    i = int(np.argmax(sm))
    return float(GRID[i]), float(f[i]), float(sm[i]), float(f.max())


def retrain(args):
    cmd = [sys.executable, str(Path(__file__).with_name("main.py")), "--models", "resnet18",
           "--context", "--strong_aug", "--no_ensemble",
           "--result_dir", args.result_dir, "--data_root", args.data_root]
    if args.clahe:
        cmd.append("--clahe")
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    print("前処理を強めたモデルを学習します:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resnet18_ctx", help="ベースにする result/ 内のモデル名")
    ap.add_argument("--result_dir", default="result")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--retrain", action="store_true", help="前処理を強めた resnet18_ctx_aug を追加学習する")
    ap.add_argument("--clahe", action="store_true", help="--retrain 時に CLAHE も使う")
    ap.add_argument("--epochs", type=int, default=0, help="--retrain 時のエポック数 (0 = main.py の既定)")
    ap.add_argument("--min_gain", type=float, default=0.003, help="ブレンドを採用する最小の改善幅 (平滑化F1)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.retrain:
        retrain(args)

    res = Path(args.result_dir)
    base = load_result(args.result_dir, args.base)
    if base is None:
        raise SystemExit(f"result/{args.base}/ がありません。先に main.py で作ってください。")
    t0 = time.time()
    oof_df = pd.read_csv(res / args.base / "oof.csv")
    te_df = pd.read_csv(res / args.base / "test_probs.csv")
    paths, y, test_paths = oof_df["path"].values, oof_df["label"].values.astype(int), te_df["path"].values
    oof_b, te_b = base[1], base[2]

    # 相手候補: result/ 内の他モデル (ensemble も含める)。以前の opt / stack は除く
    cands = {}
    for d in sorted(res.iterdir()):
        if not d.is_dir() or d.name in (args.base, "stack", "resnet18_ctx_opt"):
            continue
        r = load_result(args.result_dir, d.name)
        if r is not None and len(r[1]) == len(y) and len(r[2]) == len(test_paths):
            cands[d.name] = r
    print(f"ベース: {args.base} / ブレンド候補: {', '.join(cands) or 'なし'}")

    thr0, f1_0, sm0, raw0 = robust_threshold(y, oof_b)
    ap0 = average_precision_score(y, oof_b)
    print(f"\n[ベース単独] AP {ap0:.4f} / F1 {f1_0:.4f} (しきい値 {thr0:.2f}) / 平滑化F1 {sm0:.4f}")

    # --- 後処理1: ブレンドの探索 (ベースの重みを 0.9〜0.5 で試す) ---
    rows = []
    for name, (_, oof_c, te_c) in cands.items():
        for w in (0.9, 0.8, 0.7, 0.6, 0.5):
            p = blend([oof_b, oof_c], [w, 1 - w])
            thr, f1, sm, _ = robust_threshold(y, p)
            rows.append((sm, f1, average_precision_score(y, p), name, w, thr))
    rows.sort(reverse=True)
    if rows:
        print("\nブレンド候補 (平滑化F1の上位5件)")
        print(f"{'partner':<24}{'w(base)':>8}{'smoothF1':>10}{'F1':>8}{'AP':>8}")
        for sm, f1, apv, name, w, thr in rows[:5]:
            print(f"{name:<24}{w:>8.1f}{sm:>10.4f}{f1:>8.4f}{apv:>8.4f}")

    if rows and rows[0][0] >= sm0 + args.min_gain:
        sm, _, _, name, w, _ = rows[0]
        oof_f = blend([oof_b, cands[name][1]], [w, 1 - w])
        te_f = blend([te_b, cands[name][2]], [w, 1 - w])
        desc = f"{args.base} {w:.1f} + {name} {1 - w:.1f}"
    else:
        oof_f, te_f, desc = oof_b, te_b, f"{args.base} 単独 (ブレンドの改善が {args.min_gain} 未満)"

    # --- 後処理2: 平滑化したしきい値 ---
    thr, f1, sm, raw = robust_threshold(y, oof_f)
    apf = average_precision_score(y, oof_f)
    groups = np.array([group_id(p) for p in paths])
    splits = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                       random_state=args.seed).split(paths, y, groups))
    fold_f1 = [float(f1_score(y[va], (oof_f[va] >= thr).astype(int), zero_division=0)) for _, va in splits]
    metrics = {"f1": round(f1, 4), "threshold": round(thr, 2), "ap": round(float(apf), 4),
               "f1_at_0.5": round(float(f1_score(y, (oof_f >= 0.5).astype(int), zero_division=0)), 4),
               "smoothed_f1": round(sm, 4), "fold_f1": [round(v, 4) for v in fold_f1],
               "fold_f1_std": round(float(np.std(fold_f1)), 4), "seconds": round(time.time() - t0),
               "recipe": desc}
    m = save_result(args.result_dir, "resnet18_ctx_opt", paths, y, oof_f, test_paths, te_f, metrics)

    expected = (oof_f >= thr).mean() * len(test_paths)
    print(f"\n[最終] {desc}")
    print(f"  AP {apf:.4f} / F1 {f1:.4f} (しきい値 {thr:.2f}) / 平滑化F1 {sm:.4f} / fold別F1の標準偏差 {m['fold_f1_std']:.3f}")
    print(f"  テストの正例予測 {m['test_positive_pred']} 枚 (OOFの正例予測率から見込む枚数: 約{expected:.0f} 枚)")
    print(f"  提出用ファイル: {res / 'resnet18_ctx_opt' / 'submission.csv'}")
    print("  ※ OOFで選んだ設定なので、手元のF1はやや楽観的です。提出スコアで確かめてください。")


if __name__ == "__main__":
    main()