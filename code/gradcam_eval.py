"""Systematic evaluation of the Grad-CAM explanations of the deployed ECG-image checkpoint (editor point 9).

Test fold 10 of PTB-XL (MI vs NORM), checkpoint D:/CDSIMER-Lastday/cvd_app/ecg_resnet18.pt.
1. Faithfulness (deletion test): for every MI record, the 10 % of waveform (trace) pixels with the highest
   Grad-CAM value are erased (set to the white background) and the drop in P(MI) is compared with erasing the
   same number of randomly chosen trace pixels (paired Wilcoxon signed-rank test).
2. Anatomical plausibility: the rendered image is a 6 x 2 grid (left column I, II, III, aVR, aVL, aVF; right
   column V1-V6). The share of Grad-CAM mass falling in the inferior leads (II, III, aVF) and in the anterior/
   septal leads (V1-V4) is compared between test records with inferior MI (IMI, ILMI, IPLMI, IPMI) and
   anterior MI (AMI, ASMI, ALMI) statements (Mann-Whitney U).
3. Trace focus: share of Grad-CAM mass on trace pixels relative to the share of trace pixels in the image.
Writes results/gradcam_eval.json
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
APP = Path("D:/CDSIMER-Lastday/cvd_app")
sys.path.insert(0, str(HERE))
import ecg_pipeline_copy as E  # noqa: E402

INFERIOR = {"IMI", "ILMI", "IPLMI", "IPMI"}
ANTERIOR = {"AMI", "ASMI", "ALMI"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def to_x(gray):
    x = np.repeat(gray.astype(np.float32)[None] / 255.0, 3, 0)
    return torch.from_numpy((x - MEAN) / STD)[None]


def lead_masks(h=224, w=224):
    """Boolean masks for the 12 lead panels of the 6 x 2 rendering (column-major lead order)."""
    m = np.zeros((12, h, w), bool)
    for lead in range(12):
        col, row = divmod(lead, 6)
        r0, r1 = int(round(row * h / 6)), int(round((row + 1) * h / 6))
        c0, c1 = int(round(col * w / 2)), int(round((col + 1) * w / 2))
        m[lead, r0:r1, c0:c1] = True
    return m


def main():
    torch.set_num_threads(8)
    lab = pd.read_csv(APP / "data_ecg" / "labels.csv")
    te = lab[lab.strat_fold == 10].reset_index(drop=True)
    meta = pd.read_csv(APP / "DatasetPTB" / "ptbxl_database.csv", index_col="ecg_id")
    codes = meta.scp_codes.apply(ast.literal_eval)
    net = E.build_model(False)
    net.load_state_dict(torch.load(APP / "ecg_resnet18.pt", map_location="cpu", weights_only=False)["state_dict"])
    net.eval()
    cam_fn = E.GradCAM(net)
    LM = lead_masks()
    rng = np.random.default_rng(42)
    rows = []
    for r in te.itertuples():
        g = np.asarray(Image.open(APP / "data_ecg" / r.image).convert("L").resize((224, 224)))
        x = to_x(g).requires_grad_(True)
        p, cam = cam_fn(x)
        cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
        trace = g < 128
        mass = cam.sum() + 1e-9
        lead_share = [(cam * LM[k]).sum() / mass for k in range(12)]
        row = dict(ecg_id=r.ecg_id, label=r.label, p=p, trace_mass=float((cam * trace).sum() / mass),
                   trace_frac=float(trace.mean()), inferior=float(sum(lead_share[k] for k in (1, 2, 5))),
                   anterior=float(sum(lead_share[k] for k in (6, 7, 8, 9))))
        sc = set(codes.loc[r.ecg_id].keys())
        row["territory"] = "inferior" if sc & INFERIOR and not sc & ANTERIOR else (
            "anterior" if sc & ANTERIOR and not sc & INFERIOR else "")
        if r.label == 1 and trace.sum() > 50:
            ti = np.flatnonzero(trace.ravel())
            k = max(1, int(0.10 * len(ti)))
            top = ti[np.argsort(-cam.ravel()[ti])[:k]]
            rnd = rng.choice(ti, k, replace=False)
            out = {}
            for name, sel in (("top", top), ("rand", rnd)):
                g2 = g.copy().ravel(); g2[sel] = 255
                with torch.no_grad():
                    out[name] = float(torch.sigmoid(net(to_x(g2.reshape(224, 224)))).item())
            row.update(del_top=p - out["top"], del_rand=p - out["rand"])
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "gradcam_records.csv", index=False)
    mi = d[d.label == 1].dropna(subset=["del_top"])
    w = wilcoxon(mi.del_top, mi.del_rand)
    inf, ant = d[d.territory == "inferior"], d[d.territory == "anterior"]
    res = dict(
        n_test=len(d), n_mi=int((d.label == 1).sum()),
        deletion=dict(n=len(mi), mean_drop_top=float(mi.del_top.mean()), mean_drop_random=float(mi.del_rand.mean()),
                      median_drop_top=float(mi.del_top.median()), median_drop_random=float(mi.del_rand.median()),
                      frac_top_greater=float((mi.del_top > mi.del_rand).mean()), wilcoxon_p=float(w.pvalue)),
        territory=dict(n_inferior=len(inf), n_anterior=len(ant),
                       inferior_share_in_inferior_mi=float(inf.inferior.mean()), inferior_share_in_anterior_mi=float(ant.inferior.mean()),
                       anterior_share_in_inferior_mi=float(inf.anterior.mean()), anterior_share_in_anterior_mi=float(ant.anterior.mean()),
                       p_inferior_leads=float(mannwhitneyu(inf.inferior, ant.inferior, alternative="greater").pvalue),
                       p_anterior_leads=float(mannwhitneyu(ant.anterior, inf.anterior, alternative="greater").pvalue),
                       uniform_inferior=3 / 12, uniform_anterior=4 / 12),
        trace=dict(mean_cam_mass_on_trace=float(d.trace_mass.mean()), mean_trace_pixel_fraction=float(d.trace_frac.mean())))
    (HERE / "results" / "gradcam_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
