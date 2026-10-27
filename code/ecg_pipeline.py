"""
ecg_pipeline.py -- ECG-image branch of the multimodal Acute-MI system.

Adds an imaging modality (12-lead ECG rendered to an image) to the tabular
biomarker model, with Grad-CAM explainability and late fusion.

Reputed data source (NOT Kaggle):
    PTB-XL, a large publicly available ECG dataset -- PhysioNet
    https://physionet.org/content/ptb-xl/   (Wagner et al., Sci Data 2020)
    21,799 clinical 12-lead ECGs with SCP diagnostic labels incl. MI.

This module runs on a machine with a deep-learning stack and internet
(the project sandbox has neither). Typical workflow:

    pip install -r requirements-multimodal.txt
    python ecg_pipeline.py prepare   --ptbxl_dir /path/to/ptbxl --out data_ecg
    python ecg_pipeline.py train     --data data_ecg --epochs 20
    python ecg_pipeline.py evaluate  --data data_ecg --ckpt ecg_resnet18.pt
    python ecg_pipeline.py gradcam   --ckpt ecg_resnet18.pt --image sample.png

Design choices mirroring the tabular pipeline:
  * Patient-level train/val/test split (no leakage across records of one patient).
  * ResNet-18 backbone; Grad-CAM on the last conv block (layer4).
  * Late fusion: fused_prob = sigma(w0 + w_ecg*logit_ecg + w_tab*logit_tab),
    the fusion weights fit on the VALIDATION set only.
"""
from __future__ import annotations
import os
import argparse
import numpy as np

# Heavy deps imported lazily so the module can be inspected without them.
def _torch():
    import torch  # noqa
    return torch


IMG_SIZE = 224
MI_CLASSES = ("NORM", "MI")          # binary: myocardial infarction vs normal
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------------
# 1. DATA PREPARATION : PTB-XL signals -> 12-lead ECG images + labels
# ----------------------------------------------------------------------
def prepare_ptbxl(ptbxl_dir: str, out_dir: str, sampling_rate: int = 100, limit=None):
    """Render each PTB-XL record to a standard 12-lead ECG image and write a
    labels CSV (MI vs NORM). Requires wfdb, pandas, matplotlib."""
    import pandas as pd
    import wfdb
    import ast
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"),
                       index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg = pd.read_csv(os.path.join(ptbxl_dir, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def superclass(codes):
        s = set()
        for k in codes:
            if k in agg.index:
                s.add(agg.loc[k, "diagnostic_class"])
        return s

    # Phase 1: collect candidate records (MI vs NORM) WITHOUT rendering (fast).
    cand = []
    for ecg_id, r in meta.iterrows():
        sup = superclass(r.scp_codes)
        if "MI" in sup:
            label = 1
        elif "NORM" in sup:
            label = 0
        else:
            continue  # keep the task binary and clean: MI vs NORM only
        fn = r.filename_lr if sampling_rate == 100 else r.filename_hr
        cand.append((ecg_id, r.patient_id, r.strat_fold, label, fn))

    # Optional --limit: balanced subsample across (fold, class) so val/test folds
    # (9,10) are still represented. Rendering is the slow step, so we cut here.
    if limit:
        import random as _rnd
        _rnd.seed(42)
        buckets = {}
        for c in cand:
            buckets.setdefault((c[2], c[3]), []).append(c)
        per = max(1, limit // max(1, len(buckets)))
        sub = []
        for lst in buckets.values():
            _rnd.shuffle(lst)
            sub += lst[:per]
        cand = sub
        print(f"[--limit {limit}] rendering {len(cand)} of the available records")

    # Phase 2: render only the selected records to 12-lead images (the slow step).
    print(f"Rendering {len(cand)} ECG images to {img_dir} ... "
          f"(~0.1-0.3s each on CPU; this is normal, not a freeze)")
    try:
        from tqdm import tqdm as _tqdm
        cand_iter = _tqdm(cand, unit="ecg")
    except Exception:
        cand_iter = cand
    rows = []
    for ecg_id, patient_id, strat_fold, label, fn in cand_iter:
        sig, _ = wfdb.rdsamp(os.path.join(ptbxl_dir, fn))
        img_path = os.path.join(img_dir, f"{ecg_id}.png")
        _render_ecg(sig, img_path)
        rows.append({"ecg_id": ecg_id, "patient_id": patient_id,
                     "image": os.path.relpath(img_path, out_dir),
                     "label": label, "strat_fold": strat_fold})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print(f"Prepared {len(df)} images "
          f"(MI={int(df.label.sum())}, NORM={int((df.label==0).sum())}) -> {out_dir}")


def _render_ecg(sig: np.ndarray, path: str):
    """Render a 12-lead signal (n_samples x 12) to a compact grid image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 2, figsize=(6, 6), dpi=IMG_SIZE / 6)
    for i, ax in enumerate(axes.T.ravel()):
        if i < sig.shape[1]:
            ax.plot(sig[:, i], lw=0.5, color="black")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=IMG_SIZE / 6, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. MODEL : ResNet-18 for ECG images
# ----------------------------------------------------------------------
def build_model(pretrained: bool = True):
    torch = _torch()
    import torch.nn as nn
    import torchvision
    net = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 1)   # single logit (MI probability)
    return net


class ECGImageDataset:
    """Minimal dataset over the labels.csv produced by prepare_ptbxl."""
    def __init__(self, root, df, train=False):
        import torchvision.transforms as T
        self.root, self.df = root, df.reset_index(drop=True)
        aug = [T.RandomApply([T.RandomAffine(0, translate=(0.02, 0.02))], p=0.3)] if train else []
        self.tf = T.Compose([T.Grayscale(3), T.Resize((IMG_SIZE, IMG_SIZE)), *aug,
                             T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        torch = _torch()
        r = self.df.iloc[i]
        from PIL import Image
        img = Image.open(os.path.join(self.root, r["image"])).convert("RGB")
        return self.tf(img), torch.tensor([float(r["label"])])


# ----------------------------------------------------------------------
# 3. TRAINING (patient-level split, val-based selection -> leakage-free)
# ----------------------------------------------------------------------
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ecg")

def _save(name, text):
    import datetime, json
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    p = os.path.join(_RESULTS_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[saved] {p}")
    return p


def train(data_dir, epochs=20, bs=32, lr=1e-4, ckpt="ecg_resnet18.pt", workers=0, subset=None):
    import pandas as pd
    torch = _torch()
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {dev}")
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    # PTB-XL recommended folds: 1-8 train, 9 val, 10 test (patient-disjoint)
    tr = df[df.strat_fold <= 8]; va = df[df.strat_fold == 9]
    if subset:
        def _bal(frame, n):
            half = max(1, n // 2)
            parts = [g.sample(min(half, len(g)), random_state=42)
                     for _, g in frame.groupby("label")]
            return pd.concat(parts).sample(frac=1, random_state=42)
        tr = _bal(tr, subset)
        va = _bal(va, max(200, subset // 4))
        print(f"[--subset {subset}] train={len(tr)} val={len(va)} "
              f"(images reused from disk; nothing re-rendered)")
    net = build_model(True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor([(tr.label == 0).sum() / max(1, (tr.label == 1).sum())]).to(dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dl_tr = DataLoader(ECGImageDataset(data_dir, tr, True), bs, shuffle=True, num_workers=workers)
    dl_va = DataLoader(ECGImageDataset(data_dir, va, False), bs, num_workers=workers)
    import datetime
    log = [f"ECG branch training log  ({datetime.datetime.now():%Y-%m-%d %H:%M})",
           f"config: subset={subset} epochs={epochs} bs={bs} lr={lr} device={dev}",
           f"train n={len(tr)}  val n={len(va)}", "-" * 44]
    best = -1
    for ep in range(epochs):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(net(x), y); loss.backward(); opt.step()
        auc = _auc_loader(net, dl_va, dev)
        if dev == "cuda":
            torch.cuda.empty_cache()
        line = f"epoch {ep+1}/{epochs}  val-AUC {auc:.4f}"
        print(line); log.append(line)
        if auc > best:
            best = auc
            torch.save({"state_dict": net.state_dict(), "val_auc": float(best)}, ckpt)
    log.append("-" * 44)
    log.append(f"Best val AUC {best:.4f}; checkpoint -> {ckpt}")
    print(f"Best val AUC {best:.4f}; checkpoint -> {ckpt}")
    _save("train_log.txt", "\n".join(log) + "\n")


def _auc_loader(net, dl, dev):
    torch = _torch()
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.sigmoid(net(x.to(dev))).cpu().numpy().ravel()
            ps += list(p); ys += list(y.numpy().ravel())
    ys, ps = np.array(ys), np.array(ps)
    o = np.argsort(ps); r = np.empty_like(o, float); r[o] = np.arange(len(ps))
    P = ys.sum(); N = len(ys) - P
    return (r[ys == 1].sum() - P * (P - 1) / 2) / (P * N + 1e-9)


def evaluate(data_dir, ckpt="ecg_resnet18.pt"):
    import pandas as pd
    torch = _torch()
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    te = df[df.strat_fold == 10]
    net = build_model(False).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["state_dict"])
    import datetime, json
    dl = DataLoader(ECGImageDataset(data_dir, te, False), 64)
    test_auc = _auc_loader(net, dl, dev)
    n_pos = int((te.label == 1).sum()); n_neg = int((te.label == 0).sum())
    val_auc = torch.load(ckpt, map_location="cpu", weights_only=False).get("val_auc")
    line = f"TEST AUC (ECG branch): {test_auc:.4f}  (n={len(te)}, MI={n_pos}, NORM={n_neg})"
    print(line)
    report = (f"ECG branch held-out evaluation  ({datetime.datetime.now():%Y-%m-%d %H:%M})\n"
              f"checkpoint: {ckpt}  (best val-AUC {val_auc:.4f})\n"
              f"test fold (strat_fold==10): n={len(te)}  MI={n_pos}  NORM={n_neg}\n"
              f"TEST ROC-AUC (ECG image branch): {test_auc:.4f}\n")
    _save("eval.txt", report)
    _save("eval.json", json.dumps({"test_auc": float(test_auc), "n": int(len(te)),
          "n_MI": n_pos, "n_NORM": n_neg, "val_auc": float(val_auc), "checkpoint": ckpt}, indent=2))


# ----------------------------------------------------------------------
# 4. GRAD-CAM  (trustworthiness / explainability)
# ----------------------------------------------------------------------
class GradCAM:
    """Grad-CAM for the ResNet-18 last conv block (layer4)."""
    def __init__(self, net, target_layer=None):
        torch = _torch()
        self.net = net.eval()
        self.tl = target_layer or net.layer4[-1]
        self.acts = None; self.grads = None
        self.tl.register_forward_hook(self._fwd)
        self.tl.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x):
        torch = _torch()
        logit = self.net(x)                       # (1,1)
        self.net.zero_grad(); logit.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP over spatial
        cam = torch.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return float(torch.sigmoid(logit).item()), cam


def gradcam_overlay(ckpt, image_path, out_path="gradcam.png"):
    """Save an ECG image with its Grad-CAM heat overlay; return MI probability."""
    torch = _torch()
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    net = build_model(False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    tf = T.Compose([T.Grayscale(3), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0); x.requires_grad_(True)
    prob, cam = GradCAM(net)(x)
    import numpy as np
    base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(float) / 255
    cam_r = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((IMG_SIZE, IMG_SIZE))) / 255
    plt.figure(figsize=(4, 4)); plt.imshow(base); plt.imshow(cam_r, cmap="jet", alpha=0.45)
    plt.axis("off"); plt.title(f"Grad-CAM  P(MI)={prob:.2f}")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"P(MI)={prob:.3f}; overlay -> {out_path}")
    return prob, out_path


# ----------------------------------------------------------------------
# 5. LATE FUSION with the tabular biomarker model
# ----------------------------------------------------------------------
def fuse_probabilities(p_ecg, p_tab, weights=(0.0, 1.0, 1.0)):
    """Logistic late fusion. weights=(w0,w_ecg,w_tab) fit on validation set.
    Default = equal-weight average in logit space."""
    def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    w0, we, wt = weights
    z = w0 + we * logit(np.asarray(p_ecg)) + wt * logit(np.asarray(p_tab))
    return 1 / (1 + np.exp(-z))


def _cli():
    ap = argparse.ArgumentParser(description="ECG multimodal pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--ptbxl_dir", required=True); p.add_argument("--out", default="data_ecg"); p.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("train"); t.add_argument("--data", default="data_ecg"); t.add_argument("--epochs", type=int, default=20); t.add_argument("--ckpt", default="ecg_resnet18.pt"); t.add_argument("--bs", type=int, default=32); t.add_argument("--workers", type=int, default=0); t.add_argument("--subset", type=int, default=None)
    e = sub.add_parser("evaluate"); e.add_argument("--data", default="data_ecg"); e.add_argument("--ckpt", default="ecg_resnet18.pt")
    g = sub.add_parser("gradcam"); g.add_argument("--ckpt", default="ecg_resnet18.pt"); g.add_argument("--image", required=True); g.add_argument("--out", default="gradcam.png")
    a = ap.parse_args()
    if a.cmd == "prepare": prepare_ptbxl(a.ptbxl_dir, a.out, limit=a.limit)
    elif a.cmd == "train": train(a.data, a.epochs, bs=a.bs, ckpt=a.ckpt, workers=a.workers, subset=a.subset)
    elif a.cmd == "evaluate": evaluate(a.data, a.ckpt)
    elif a.cmd == "gradcam": gradcam_overlay(a.ckpt, a.image, a.out)


if __name__ == "__main__":
    _cli()
