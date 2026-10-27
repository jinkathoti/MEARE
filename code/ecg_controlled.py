"""Controlled comparison of the ECG-image branch and the raw-signal baseline (editor point 7).

Both models are trained under one identical protocol:
  * training data: the same class-balanced 2,000-record subset of PTB-XL folds 1-8 (sampling seed 42,
    identical to ecg_pipeline.train --subset 2000)
  * validation: the complete fold 9 (MI/NORM); test: the complete fold 10, scored once per run
  * optimiser AdamW (weight decay 1e-4), batch 32, 15 epochs, learning rate chosen from {1e-4, 1e-3}
    by validation AUC with seed 0, then the chosen rate is re-run with seeds 1 and 2
  * checkpoint = epoch with best validation AUC; threshold = Youden index on fold 9
  * light augmentation of the same strength for both (image: random translation of up to 2 %;
    signal: random time shift of up to 0.5 s)
Models: ImageNet-pretrained ResNet-18 on the rendered 224x224 images (the MEARE image branch) and the
1D residual network on the raw 100 Hz signals (12 x 1000).
Usage: python ecg_controlled.py  -> results/ecg_controlled.json (+ per-run probabilities)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
APP = Path("D:/CDSIMER-Lastday/cvd_app")
sys.path.insert(0, str(HERE))
from ecg_raw_baseline import ResNet1D  # noqa: E402
from tab_analysis import counts, wilson  # noqa: E402

OUT = HERE / "results"
EPOCHS, BS = 15, 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def split():
    lab = pd.read_csv(APP / "data_ecg" / "labels.csv")
    tr = lab[lab.strat_fold <= 8]
    parts = [g.sample(min(1000, len(g)), random_state=42) for _, g in tr.groupby("label")]
    tr = pd.concat(parts).sample(frac=1, random_state=42)
    return lab, tr, lab[lab.strat_fold == 9], lab[lab.strat_fold == 10]


def load_images(df):
    a = np.zeros((len(df), 224, 224), np.uint8)
    for i, p in enumerate(df.image.values):
        a[i] = np.asarray(Image.open(APP / "data_ecg" / p).convert("L").resize((224, 224)))
    return a


def img_batch(u8, rng=None):
    x = u8.astype(np.float32)[:, None] / 255.0
    if rng is not None:                         # random translation up to 2 % (~4 px), p = 0.3
        for k in range(len(x)):
            if rng.random() < 0.3:
                dx, dy = rng.integers(-4, 5, 2)
                x[k] = np.roll(x[k], (int(dy), int(dx)), axis=(1, 2))
    x = np.repeat(x, 3, axis=1)
    return torch.from_numpy((x - MEAN) / STD)


def build_image():
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


@torch.no_grad()
def predict(net, X, kind, bs=128):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        xb = img_batch(X[i:i + bs]) if kind == "image" else torch.from_numpy(X[i:i + bs])
        out.append(torch.sigmoid(net(xb)).numpy().ravel())
    return np.concatenate(out)


def train_one(kind, data, lr, seed):
    Xtr, ytr, Xva, yva = data
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build_image() if kind == "image" else ResNet1D()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()               # the subset is balanced 1:1
    best, state, log = -1, None, []
    idx = np.arange(len(ytr))
    for ep in range(EPOCHS):
        net.train(); rng.shuffle(idx); t0 = time.time()
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            if kind == "image":
                xb = img_batch(Xtr[b], rng)
            else:
                xb = torch.roll(torch.from_numpy(Xtr[b]), int(rng.integers(-50, 51)), dims=-1)
            yb = torch.from_numpy(ytr[b].astype(np.float32)).unsqueeze(1)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        auc = roc_auc_score(yva, predict(net, Xva, kind))
        log.append(round(float(auc), 4))
        print(f"{kind} lr={lr} seed={seed} epoch {ep + 1} val-AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best = auc; state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, log


def evaluate(net, kind, Xva, yva, Xte, yte):
    pv, pt = predict(net, Xva, kind), predict(net, Xte, kind)
    fpr, tpr, thr = roc_curve(yva, pv); t = float(thr[np.argmax(tpr - fpr)])
    tp, fn, tn, fp = counts(yte, (pt >= t).astype(int))
    rng = np.random.default_rng(42); bs = []
    for _ in range(2000):
        b = rng.integers(0, len(yte), len(yte))
        bs.append(roc_auc_score(yte[b], pt[b]))
    return pt, dict(test_auc=float(roc_auc_score(yte, pt)), test_auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    test_ap=float(average_precision_score(yte, pt)), threshold=t,
                    sens=[tp / (tp + fn), *wilson(tp, tp + fn)], spec=[tn / (tn + fp), *wilson(tn, tn + fp)])


def main():
    torch.set_num_threads(12)
    lab, tr, va, te = split()
    z = np.load(HERE / "cache_ptbxl100.npz")
    pos = {e: i for i, e in enumerate(z["ecg_id"])}
    Xsig = z["X"]
    mu = Xsig[[pos[e] for e in tr.ecg_id]].mean((0, 2), keepdims=True); sd = Xsig[[pos[e] for e in tr.ecg_id]].std((0, 2), keepdims=True) + 1e-6
    sig = lambda d: ((Xsig[[pos[e] for e in d.ecg_id]] - mu) / sd).astype(np.float32)
    data = {"raw": (sig(tr), tr.label.values, sig(va), va.label.values, sig(te), te.label.values)}
    print("loading images ...", flush=True)
    data["image"] = (load_images(tr), tr.label.values, load_images(va), va.label.values, load_images(te), te.label.values)
    res = {}
    for kind in ("raw", "image"):
        Xtr, ytr, Xva, yva, Xte, yte = data[kind]
        runs = {}
        for lr in (1e-4, 1e-3):                       # learning-rate selection on validation, seed 0
            net, best, log = train_one(kind, (Xtr, ytr, Xva, yva), lr, 0)
            runs[lr] = (net, best, log)
        lr_star = max(runs, key=lambda k: runs[k][1])
        seeds = {}
        for seed in (0, 1, 2):
            if seed == 0:
                net, best, log = runs[lr_star]
            else:
                net, best, log = train_one(kind, (Xtr, ytr, Xva, yva), lr_star, seed)
            pt, ev = evaluate(net, kind, Xva, yva, Xte, yte)
            ev.update(best_val_auc=float(best), val_log=log)
            seeds[seed] = ev
            pd.DataFrame({"ecg_id": te.ecg_id.values, "label": yte, "p": pt}).to_csv(OUT / f"ecg_ctrl_{kind}_s{seed}.csv", index=False)
        aucs = [seeds[s]["test_auc"] for s in seeds]
        res[kind] = dict(lr_grid={str(k): float(v[1]) for k, v in runs.items()}, lr=lr_star, seeds=seeds,
                         test_auc_mean=float(np.mean(aucs)), test_auc_sd=float(np.std(aucs, ddof=1)))
        (OUT / "ecg_controlled.json").write_text(json.dumps(res, indent=1))
    # paired comparison on the test fold: seed-averaged probabilities, paired bootstrap of the AUC difference
    yte = data["raw"][5]
    pr = np.mean([pd.read_csv(OUT / f"ecg_ctrl_raw_s{s}.csv").p.values for s in (0, 1, 2)], 0)
    pi = np.mean([pd.read_csv(OUT / f"ecg_ctrl_image_s{s}.csv").p.values for s in (0, 1, 2)], 0)
    rng = np.random.default_rng(42); d = []
    for _ in range(2000):
        b = rng.integers(0, len(yte), len(yte))
        d.append(roc_auc_score(yte[b], pr[b]) - roc_auc_score(yte[b], pi[b]))
    res["paired_diff_raw_minus_image"] = dict(point=float(roc_auc_score(yte, pr) - roc_auc_score(yte, pi)),
                                              ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                              p_two_sided=float(2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))
    (OUT / "ecg_controlled.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k == "paired_diff_raw_minus_image" else {a: b for a, b in v.items() if a != "seeds"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
