"""Re-analysis of the MEARE tabular branch for the resubmission.

1. Provenance: SHA-256 of the CSV, raw counts, cleaning log (implausible values, duplicates).
2. Re-run of the original leak-free pipeline (meare_model.py = unchanged copy of model.py) to confirm
   the published numbers.
3. Confidence intervals: Wilson intervals for test sensitivity/specificity/PPV/NPV/accuracy,
   Clopper-Pearson exact intervals for subgroup sensitivity/specificity, bootstrap CIs for subgroup AUC.
4. Separability / clinical-realism checks: single-biomarker AUCs, a two-threshold rule
   (troponin > t1 OR CK-MB > t2) fitted on the training split only, a depth-2 decision tree, and
   performance of the selected model in a biomarker "grey zone".
Writes results/tab_*.csv and results/tab_summary.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import meare_model as M

HERE = Path(__file__).resolve().parent
CSV = HERE / "heart_attack.csv"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def boot_auc(y, p, n=2000):
    y, p = np.asarray(y), np.asarray(p); out = []
    for _ in range(n):
        b = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            out.append(roc_auc_score(y[b], p[b]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def main():
    summ = {}
    raw = CSV.read_bytes()
    summ["sha256"] = hashlib.sha256(raw).hexdigest()
    df_raw = pd.read_csv(CSV)
    summ["raw_rows"] = len(df_raw)
    summ["raw_class_counts"] = df_raw["class"].value_counts().to_dict()
    summ["raw_missing"] = int(df_raw.isna().sum().sum())
    summ["implausible"] = {
        "heart_rate_gt300_or_lt20": int(((df_raw.impluse > 300) | (df_raw.impluse < 20)).sum()),
        "sbp_gt300": int((df_raw.pressurehight > 300).sum()),
        "dbp_gt200": int((df_raw.pressurelow > 200).sum())}
    summ["exact_duplicate_rows"] = int(df_raw.duplicated().sum())
    df = M.load_and_clean(CSV)
    summ["clean_rows"] = len(df); summ["clean_pos"] = int(df.y.sum()); summ["clean_neg"] = int((df.y == 0).sum())
    summ["ranges"] = {c: [float(df_raw[c].min()), float(df_raw[c].max())] for c in M.FEATURES}

    # ---- re-run original pipeline
    R = M.run_all(CSV)
    summ["best_model"] = R.best_name; summ["threshold_youden"] = R.threshold
    summ["bench"] = R.bench.round(4).to_dict(orient="records")
    summ["ablation"] = R.ablation.round(4).to_dict(orient="records")
    y, p = np.asarray(R.y_test), np.asarray(R.prob_test)
    pred = (p >= R.threshold).astype(int)
    tp, fn, tn, fp = counts(y, pred)
    summ["test_counts"] = dict(tp=tp, fn=fn, tn=tn, fp=fp, n=len(y))
    rows = []
    for name, k, n in [("Sensitivity", tp, tp + fn), ("Specificity", tn, tn + fp), ("PPV", tp, tp + fp),
                       ("NPV", tn, tn + fn), ("Accuracy", tp + tn, len(y))]:
        lo, hi = wilson(k, n)
        rows.append(dict(Metric=name, k=k, n=n, Value=k / n, CI_lo=lo, CI_hi=hi))
    auc_lo, auc_hi = boot_auc(y, p)
    rows.insert(0, dict(Metric="ROC-AUC", k="", n=len(y), Value=roc_auc_score(y, p), CI_lo=auc_lo, CI_hi=auc_hi))
    pd.DataFrame(rows).to_csv(OUT / "tab_test_ci.csv", index=False)
    summ["test_ci"] = rows
    # threshold note: Youden threshold chosen on the test set in the original code -> also report fixed 0.5
    pred05 = (p >= 0.5).astype(int); c05 = counts(y, pred05)
    summ["test_counts_thr05"] = dict(zip(("tp", "fn", "tn", "fp"), c05))

    # ---- subgroups with exact CIs
    X_all = df[M.FEATURES]; y_all = df.y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=M.RANDOM_STATE)
    assert np.array_equal(y_te, y)
    te = X_te.copy(); te["y"] = y; te["p"] = p; te["pred"] = pred
    sub = []
    for label, mask in [("Male", te.gender == 1), ("Female", te.gender == 0),
                        ("Age < 55", te.age < 55), ("Age >= 55", te.age >= 55)]:
        d = te[mask]
        tp_, fn_, tn_, fp_ = counts(d.y, d.pred)
        se, sp = tp_ / (tp_ + fn_), tn_ / (tn_ + fp_)
        se_ci, sp_ci = clopper(tp_, tp_ + fn_), clopper(tn_, tn_ + fp_)
        a_lo, a_hi = boot_auc(d.y, d.p)
        sub.append(dict(Subgroup=label, n=len(d), n_pos=int(d.y.sum()), n_neg=int((d.y == 0).sum()),
                        AUC=roc_auc_score(d.y, d.p), AUC_lo=a_lo, AUC_hi=a_hi,
                        TP=tp_, FN=fn_, TN=tn_, FP=fp_,
                        Sens=se, Sens_lo=se_ci[0], Sens_hi=se_ci[1], Spec=sp, Spec_lo=sp_ci[0], Spec_hi=sp_ci[1]))
    pd.DataFrame(sub).to_csv(OUT / "tab_subgroup_ci.csv", index=False)
    summ["subgroup"] = sub

    # ---- separability checks (fit on training split only, evaluated on test)
    trd = X_tr.copy(); trd["y"] = y_tr
    summ["single_auc_test"] = {f: float(roc_auc_score(y, X_te[f])) for f in ("troponin", "kcm")}
    best = None
    for t1 in np.unique(np.quantile(trd.troponin, np.linspace(0.01, 0.99, 197))):
        for t2 in np.unique(np.quantile(trd.kcm, np.linspace(0.01, 0.99, 197))):
            acc = ((((trd.troponin > t1) | (trd.kcm > t2)).astype(int)) == trd.y).mean()
            if best is None or acc > best[0]:
                best = (acc, float(t1), float(t2))
    acc_tr, t1, t2 = best
    rule = ((X_te.troponin > t1) | (X_te.kcm > t2)).astype(int).values
    rtp, rfn, rtn, rfp = counts(y, rule)
    rule_all = ((X_all.troponin > t1) | (X_all.kcm > t2)).astype(int).values
    summ["rule"] = dict(t_troponin=t1, t_ckmb=t2, train_acc=float(acc_tr), test_acc=float((rule == y).mean()),
                        test_acc_ci=wilson(int((rule == y).sum()), len(y)),
                        test_sens=rtp / (rtp + rfn), test_spec=rtn / (rtn + rfp),
                        test_counts=dict(tp=rtp, fn=rfn, tn=rtn, fp=rfp),
                        all_data_agreement=float((rule_all == y_all).mean()),
                        all_data_disagree=int((rule_all != y_all).sum()))
    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_tr, y_tr)
    summ["tree_depth2"] = dict(test_acc=float(accuracy_score(y, tree.predict(X_te))),
                               test_auc=float(roc_auc_score(y, tree.predict_proba(X_te)[:, 1])),
                               rules=export_text(tree, feature_names=M.FEATURES))
    # agreement of RF with the rule, and RF errors
    summ["rf_vs_rule_agreement_test"] = float((pred == rule).mean())
    # ---- grey zone: either biomarker within a factor of 2 of its rule cut-off, and the rule not
    #      already decided by the other marker being far above its cut-off
    gz = (((X_te.troponin >= t1 / 2) & (X_te.troponin <= t1 * 2)) |
          ((X_te.kcm >= t2 / 2) & (X_te.kcm <= t2 * 2))).values
    gz &= ~((X_te.troponin > t1 * 2) | (X_te.kcm > t2 * 2)).values
    d_y, d_pred, d_p = y[gz], pred[gz], p[gz]
    g = counts(d_y, d_pred)
    summ["grey_zone"] = dict(n=int(gz.sum()), n_pos=int(d_y.sum()), n_neg=int((d_y == 0).sum()),
                             acc=float((d_pred == d_y).mean()) if gz.sum() else None,
                             acc_ci=wilson(int((d_pred == d_y).sum()), int(gz.sum())),
                             tp=g[0], fn=g[1], tn=g[2], fp=g[3],
                             auc=float(roc_auc_score(d_y, d_p)) if len(np.unique(d_y)) == 2 else None)
    # errors of RF on test
    err = te[te.pred != te.y]
    summ["rf_test_errors"] = err[["age", "gender", "kcm", "troponin", "y", "p"]].round(4).to_dict(orient="records")
    (OUT / "tab_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("bench", "ranges")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
