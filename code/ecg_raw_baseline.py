"""Raw-signal benchmark for the PTB-XL MI-vs-NORM task, on exactly the records and patient-disjoint folds
used by the ECG-image branch (data_ecg/labels.csv: folds 1-8 train, 9 validation, 10 test).

Model: a 1D residual network in the style of resnet1d_wang from the PTB-XL benchmark (Strodthoff et al.
2021): three residual blocks (kernels 8-5-3), global average pooling, linear output; input 12 x 1000
(10 s at 100 Hz), per-lead standardisation with training statistics. Adam, class-weighted BCE,
checkpoint selected by validation AUC, test fold scored once.
Usage: python ecg_raw_baseline.py full | subset2000
"""
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from tab_analysis import counts, wilson

HERE = Path(__file__).resolve().parent
PTB = Path("D:/CDSIMER-Lastday/cvd_app/DatasetPTB")
LAB = Path("D:/CDSIMER-Lastday/cvd_app/data_ecg/labels.csv")
CACHE = HERE / "cache_ptbxl100.npz"
RNG = np.random.default_rng(42)


def load_signals():
    lab = pd.read_csv(LAB)
    if CACHE.exists():
        z = np.load(CACHE)
        assert np.array_equal(z["ecg_id"], lab.ecg_id.values)
        return lab, z["X"]
    meta = pd.read_csv(PTB / "ptbxl_database.csv", index_col="ecg_id")
    X = np.zeros((len(lab), 12, 1000), np.float32)
    t0 = time.time()
    for i, eid in enumerate(lab.ecg_id.values):
        sig, _ = wfdb.rdsamp(str(PTB / meta.loc[eid, "filename_lr"]))
        X[i] = sig.T[:, :1000]
        if i % 2000 == 0:
            print(f"loaded {i}/{len(lab)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(CACHE, X=X, ecg_id=lab.ecg_id.values)
    return lab, X


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(cin, cout, 8, padding="same", bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 5, padding="same", bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding="same", bias=False), nn.BatchNorm1d(cout))
        self.s = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return torch.relu(self.f(x) + self.s(x))


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(Block(12, 64), Block(64, 128), Block(128, 128))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.b(x).mean(-1))


@torch.no_grad()
def predict(net, X, bs=256):
    net.eval()
    return np.concatenate([torch.sigmoid(net(torch.from_numpy(X[i:i + bs]))).numpy().ravel()
                           for i in range(0, len(X), bs)])


def boot(y, p, f, n=2000):
    out = []
    for _ in range(n):
        b = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            out.append(f(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    torch.manual_seed(42); torch.set_num_threads(12)
    lab, X = load_signals()
    tr = lab.strat_fold.values <= 8
    if mode == "subset2000":   # same balanced sampling as ecg_pipeline.train(--subset 2000)
        trdf = lab[tr]
        parts = [g.sample(min(1000, len(g)), random_state=42) for _, g in trdf.groupby("label")]
        idx = pd.concat(parts).sample(frac=1, random_state=42).index.values
        tr = np.zeros(len(lab), bool); tr[idx] = True
    va, te = lab.strat_fold.values == 9, lab.strat_fold.values == 10
    mu = X[tr].mean((0, 2), keepdims=True); sd = X[tr].std((0, 2), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    y = lab.label.values.astype(np.float32)
    net = ResNet1D()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    tri = np.where(tr)[0]; best, best_state, log = -1, None, []
    epochs = 15 if mode == "full" else 30
    for ep in range(epochs):
        net.train(); RNG.shuffle(tri); t0 = time.time()
        for i in range(0, len(tri), 64):
            b = tri[i:i + 64]
            xb = torch.from_numpy(Xn[b]); yb = torch.from_numpy(y[b]).unsqueeze(1)
            shift = int(RNG.integers(-50, 51))           # small random time shift
            xb = torch.roll(xb, shift, dims=-1)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        auc = roc_auc_score(y[va], predict(net, Xn[va]))
        log.append(f"epoch {ep + 1} val-AUC {auc:.4f} ({time.time() - t0:.0f}s)"); print(log[-1], flush=True)
        if auc > best:
            best = auc; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    pv, pt = predict(net, Xn[va]), predict(net, Xn[te])
    fpr, tpr, thr = roc_curve(y[va], pv); t = float(thr[np.argmax(tpr - fpr)])
    tp, fn, tn, fp = counts(y[te].astype(int), (pt >= t).astype(int))
    res = dict(mode=mode, n_train=int(tr.sum()), best_val_auc=best, threshold=t, n_test=int(te.sum()),
               test_auc=float(roc_auc_score(y[te], pt)), test_auc_ci=boot(y[te], pt, roc_auc_score),
               test_ap=float(average_precision_score(y[te], pt)), test_ap_ci=boot(y[te], pt, average_precision_score),
               tp=tp, fn=fn, tn=tn, fp=fp, sens=[tp / (tp + fn), *wilson(tp, tp + fn)],
               spec=[tn / (tn + fp), *wilson(tn, tn + fp)], log=log)
    (HERE / "results" / f"ecg_raw_{mode}.json").write_text(json.dumps(res, indent=1))
    torch.save(best_state, HERE / "results" / f"ecg_raw_{mode}.pt")
    pd.DataFrame({"ecg_id": lab.ecg_id.values[te], "label": y[te], "p_raw": pt}).to_csv(
        HERE / "results" / f"ecg_raw_{mode}_probs.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
