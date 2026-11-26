#!/usr/bin/env python3
"""Pコン跡の有無判定: 学習 → 予測 → 結果保存 → 上位3モデル表示 を一括実行する

置き場所: AI-FED-main/ 直下 (W/ , train_label.csv , test.txt と同じ階層)
実行例:
  python main.py                                   # 既定のモデルをすべて実行
  python main.py --models resnet18 efficientnet_b0 # モデルを指定
  python main.py --limit 600 --epochs 2            # 動作確認 (画像を間引くので提出用には使わない)

評価指標はコンペと同じ F1 スコア。
各モデルの F1 は「撮影元の写真単位のCV (OOF予測)」で、F1最大のしきい値を使って算出する。

保存先: ./result/{モデル名}/
  submission.csv  提出用 (ヘッダーなし, test.txt の順)
  metrics.json    F1, AP, しきい値, fold別F1, 所要時間など
  oof.csv         学習画像の検証予測 (path,label,prob)
  test_probs.csv  テスト画像の予測確率 (path,prob)
  fold*.pt        CNNの重み (CNNモデルのみ)
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

try:  # PyTorch が無くても hgb / extratrees だけなら動く
    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
except ImportError:
    torch = None

SK_MODELS = ["hgb", "extratrees"]  # 画像の統計量 + 木モデル (深層学習なし)
CNN_MODELS = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"]
DEFAULT_MODELS = ["hgb", "extratrees", "resnet18", "efficientnet_b0", "resnet50"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- 共通部品
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def group_id(path):
    """W/UW/7071-1.jpg -> 7071 (撮影元の写真番号。学習/評価はこの単位で分かれている)"""
    m = re.search(r"/(\d+)-", path)
    if m is None:
        raise ValueError(f"グループ番号を読み取れません: {path}")
    return int(m.group(1))


def best_threshold(y, p):
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def summarize(y, oof, splits, seconds):
    thr, f1 = best_threshold(y, oof)
    fold_f1 = [float(f1_score(y[va], (oof[va] >= thr).astype(int), zero_division=0)) for _, va in splits]
    return {
        "f1": round(f1, 4),
        "threshold": round(thr, 2),
        "ap": round(float(average_precision_score(y, oof)), 4),
        "f1_at_0.5": round(float(f1_score(y, (oof >= 0.5).astype(int), zero_division=0)), 4),
        "fold_f1": [round(v, 4) for v in fold_f1],
        "fold_f1_std": round(float(np.std(fold_f1)), 4),
        "seconds": round(seconds),
    }


def save_result(out_dir, name, paths, y, oof, test_paths, test_prob, metrics):
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    pred = (test_prob >= metrics["threshold"]).astype(int)
    pd.DataFrame({"p": test_paths, "y": pred}).to_csv(d / "submission.csv", header=False, index=False)
    pd.DataFrame({"path": test_paths, "prob": test_prob}).to_csv(d / "test_probs.csv", index=False)
    pd.DataFrame({"path": paths, "label": y, "prob": oof}).to_csv(d / "oof.csv", index=False)
    metrics = {"model": name, **metrics, "test_positive_pred": int(pred.sum()), "n_test": len(pred)}
    json.dump(metrics, open(d / "metrics.json", "w"), ensure_ascii=False, indent=2)
    return metrics


def load_result(out_dir, name):
    d = Path(out_dir) / name
    if not (d / "metrics.json").exists():
        return None
    return (json.load(open(d / "metrics.json")),
            pd.read_csv(d / "oof.csv")["prob"].values,
            pd.read_csv(d / "test_probs.csv")["prob"].values)


# ----------------------------------------------------------------- 画像の統計量 (hgb / extratrees)
def image_features(root, path):
    """48x48グレー化 → 4x4グリッドごとの平均/最小/標準偏差 + 暗い画素の割合 (Pコンの穴は暗い塊)"""
    a = np.asarray(Image.open(Path(root) / path).convert("L").resize((48, 48), Image.BILINEAR),
                   dtype=np.float32) / 255
    g = a.reshape(4, 12, 4, 12)
    dark = [(a < a.mean() - d).mean() for d in (0.1, 0.2, 0.3)]
    return np.concatenate([g.mean((1, 3)).ravel(), g.min((1, 3)).ravel(), g.std((1, 3)).ravel(),
                           dark, [a.mean(), a.std(), a.min()]])


def run_sklearn(name, X, y, Xte, splits, seed):
    oof, te = np.zeros(len(y)), np.zeros(len(Xte))
    for k, (tr, va) in enumerate(splits):
        if name == "hgb":
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                               class_weight={0: 1, 1: 8}, random_state=seed)
        else:
            m = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / len(splits)
        print(f"  [{name}] fold {k + 1}/{len(splits)} done", flush=True)
    return oof, te


# ----------------------------------------------------------------- CNN (PyTorch)
if torch is not None:

    class PconDataset(Dataset):
        def __init__(self, root, paths, labels=None, train=False, size=256):
            self.root, self.paths, self.labels, self.train = Path(root), list(paths), labels, train
            # 画像の端に一部だけ写るPコンも「あり」なので、切り抜き系の拡張は使わない
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
            self.color = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.root / self.paths[i]).convert("RGB")
            if self.train:
                img = self.color(img)
            x = self.tf(img)
            if self.train:
                if random.random() < 0.5:
                    x = torch.flip(x, [2])
                if random.random() < 0.5:
                    x = torch.flip(x, [1])
                x = torch.rot90(x, random.randint(0, 3), (1, 2))
            return x if self.labels is None else (x, torch.tensor(float(self.labels[i])))

    def build_model(name):
        tv = torchvision.models
        if name == "resnet18":
            m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
            m.fc = nn.Linear(m.fc.in_features, 1)
        elif name == "resnet50":
            m = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
            m.fc = nn.Linear(m.fc.in_features, 1)
        elif name == "efficientnet_b0":
            m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        elif name == "mobilenet_v3_large":
            m = tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.DEFAULT)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
        elif name == "convnext_tiny":
            m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.DEFAULT)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        else:
            raise ValueError(name)
        return m

    @torch.no_grad()
    def predict(model, loader, device, tta=True):
        """確率を返す。tta=True なら 元/左右/上下/両方 反転 の平均"""
        model.eval()
        outs = []
        for x in loader:
            x = (x[0] if isinstance(x, (list, tuple)) else x).to(device)
            views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])] if tta else [x]
            outs.append(torch.stack([torch.sigmoid(model(v).squeeze(1)) for v in views]).mean(0).float().cpu())
        return torch.cat(outs).numpy()

    def run_cnn(name, args, paths, y, test_paths, splits, device, out_dir):
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        dl_te = DataLoader(PconDataset(args.data_root, test_paths, None, size=args.size),
                           batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
                           pin_memory=device.type == "cuda")
        oof, te = np.zeros(len(y)), np.zeros(len(test_paths))
        use_amp = device.type == "cuda"
        for k, (tr, va) in enumerate(splits):
            dl_tr = DataLoader(PconDataset(args.data_root, paths[tr], y[tr], train=True, size=args.size),
                               batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=use_amp)
            dl_va = DataLoader(PconDataset(args.data_root, paths[va], y[va], size=args.size),
                               batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
                               pin_memory=use_amp)
            model = build_model(name).to(device)
            pos, neg = y[tr].sum(), (1 - y[tr]).sum()
            pw = torch.tensor([min(float(np.sqrt(neg / max(pos, 1))), 10.0)], device=device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            ckpt, best_ap = d / f"fold{k}.pt", -1.0
            for ep in range(args.epochs):
                model.train()
                total = 0.0
                for x, t in dl_tr:
                    x, t = x.to(device), t.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss = crit(model(x).squeeze(1), t)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    total += loss.item() * len(x)
                sched.step()
                p = predict(model, dl_va, device, tta=False)
                ap = average_precision_score(y[va], p) if y[va].sum() > 0 else 0.0
                print(f"  [{name}] fold {k + 1}/{len(splits)} epoch {ep + 1}/{args.epochs} "
                      f"loss {total / max(len(tr), 1):.4f} val AP {ap:.4f}", flush=True)
                if ap > best_ap:  # 検証APが最良のエポックを採用
                    best_ap = ap
                    torch.save(model.state_dict(), ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            oof[va] = predict(model, dl_va, device, tta=True)
            te += predict(model, dl_te, device, tta=True) / len(splits)
        return oof, te


# ----------------------------------------------------------------- 表示
def print_top3(results, n=3):
    ranked = sorted(results.items(), key=lambda kv: (kv[1][0]["f1"], kv[1][0]["ap"]), reverse=True)
    print("\n" + "=" * 62)
    print(f"上位{min(n, len(ranked))}モデル (評価指標: F1 / 撮影元の写真単位CVの予測から算出)")
    print("=" * 62)
    print(f"{'rank':<5}{'model':<22}{'F1':>8}{'AP':>8}{'thr':>7}{'time(s)':>9}")
    for i, (name, (m, _, _)) in enumerate(ranked[:n], 1):
        print(f"{i:<5}{name:<22}{m['f1']:>8.4f}{m['ap']:>8.4f}{m['threshold']:>7.2f}{m['seconds']:>9}")
    best = ranked[0][0]
    print(f"\n提出用ファイル(F1最良): result/{best}/submission.csv")
    return ranked


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".", help="W/ , train_label.csv , test.txt があるフォルダ")
    ap.add_argument("--result_dir", default="result")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=SK_MODELS + CNN_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="動作確認用: 画像をこの枚数に間引く")
    ap.add_argument("--rerun", action="store_true", help="結果が既にあるモデルも再実行する")
    ap.add_argument("--no_ensemble", action="store_true", help="上位3モデルの平均 (ensemble) を作らない")
    args = ap.parse_args()

    seed_everything(args.seed)
    root = Path(args.data_root)
    df = pd.read_csv(root / "train_label.csv", header=None, names=["path", "label"])
    test_paths = [l.strip() for l in open(root / "test.txt", encoding="utf-8") if l.strip()]
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
        test_paths = list(pd.Series(test_paths).sample(n=min(args.limit, len(test_paths)), random_state=args.seed))
        print(f"[注意] --limit {args.limit}: 画像を間引いています。この結果は提出に使えません。")
    paths, y = df["path"].values, df["label"].values.astype(int)
    groups = np.array([group_id(p) for p in paths])
    print(f"学習 {len(df)} 枚 (正例 {y.sum()} = {y.mean():.2%}) / テスト {len(test_paths)} 枚 / "
          f"撮影元の写真 {len(set(groups))} 枚")

    # 全モデル共通の分割 (撮影元の写真単位 + 正例の割合をそろえる) → モデル同士を公平に比較できる
    splits = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                       random_state=args.seed).split(paths, y, groups))
    device = None
    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device if device is not None else "PyTorchなし (CNNモデルはスキップ)")

    results, feats = {}, None
    for name in args.models:
        if not args.rerun and (r := load_result(args.result_dir, name)) is not None and len(r[1]) == len(y):
            print(f"\n--- {name}: 既存の結果を読み込み (再実行は --rerun) ---")
            results[name] = r
            continue
        print(f"\n--- {name}: 学習開始 ---")
        t0 = time.time()
        if name in SK_MODELS:
            if feats is None:
                print("  画像の特徴量を計算中...", flush=True)
                feats = (np.stack([image_features(root, p) for p in paths]),
                         np.stack([image_features(root, p) for p in test_paths]))
            oof, te = run_sklearn(name, feats[0], y, feats[1], splits, args.seed)
        else:
            if torch is None:
                print(f"  [スキップ] {name} には PyTorch が必要です: pip install torch torchvision")
                continue
            oof, te = run_cnn(name, args, paths, y, test_paths, splits, device, args.result_dir)
        m = save_result(args.result_dir, name, paths, y, oof, test_paths, te,
                        summarize(y, oof, splits, time.time() - t0))
        results[name] = (m, oof, te)
        print(f"  → {name}: F1 {m['f1']:.4f} / AP {m['ap']:.4f} (しきい値 {m['threshold']:.2f})")

    if not results:
        raise SystemExit("実行できたモデルがありません。")

    # 上位3モデルの確率を平均 (アンサンブル)
    if not args.no_ensemble and len(results) >= 2:
        top = sorted((k for k in results if k != "ensemble"),
                     key=lambda k: results[k][0]["f1"], reverse=True)[:3]
        oof = np.mean([results[k][1] for k in top], axis=0)
        te = np.mean([results[k][2] for k in top], axis=0)
        m = save_result(args.result_dir, "ensemble", paths, y, oof, test_paths, te,
                        {**summarize(y, oof, splits, 0), "members": top})
        results["ensemble"] = (m, oof, te)
        print(f"\n--- ensemble (平均: {', '.join(top)}) ---\n  → F1 {m['f1']:.4f} / AP {m['ap']:.4f}")

    print_top3(results)


if __name__ == "__main__":
    main()