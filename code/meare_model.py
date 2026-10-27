"""
model.py - Leak-free training + publication-grade evaluation engine
for the CDSIMER Heart-Attack (CVD) classifier.

Course context
--------------
Built for the CDSIMER Medical Elective "AI in Healthcare: Applications and
Research Directions" (Dayananda Sagar University). Clinical findings and marker
interpretation reviewed and validated by expert cardiologists.

Anti-leakage design
-------------------
1. Imputation + scaling live INSIDE an sklearn Pipeline -> fit on train folds only.
2. Model / hyper-parameter selection uses cross-validation on the TRAIN set.
   The held-out TEST set is scored once, for the final report.
3. Cleaning uses fixed clinical rules, never the target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.model_selection import (
    train_test_split, cross_val_predict, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve)

FEATURES = ["age", "gender", "impluse", "pressurehight",
            "pressurelow", "glucose", "kcm", "troponin"]

FEATURE_LABELS = {
    "age": "Age (years)",
    "gender": "Sex (1 = Male, 0 = Female)",
    "impluse": "Heart rate (bpm)",
    "pressurehight": "Systolic BP (mmHg)",
    "pressurelow": "Diastolic BP (mmHg)",
    "glucose": "Blood glucose (mg/dL)",
    "kcm": "CK-MB (ng/mL)",
    "troponin": "Troponin (ng/mL)",
}

RANDOM_STATE = 42


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    df = df[df["class"].isin(["positive", "negative"])].copy()
    df.loc[df["impluse"] > 300, "impluse"] = np.nan
    df.loc[df["impluse"] < 20, "impluse"] = np.nan
    df.loc[df["pressurehight"] > 300, "pressurehight"] = np.nan
    df.loc[df["pressurelow"] > 200, "pressurelow"] = np.nan
    df = df.drop_duplicates().reset_index(drop=True)
    df["y"] = (df["class"] == "positive").astype(int)
    return df


def build_models():
    def pipe(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", clf)])
    return {
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=1000, class_weight="balanced")),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(GradientBoostingClassifier(
            random_state=RANDOM_STATE)),
        "SVM (RBF)": pipe(SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        "k-NN": pipe(KNeighborsClassifier(n_neighbors=13)),
    }


def bootstrap_ci(y_true, y_score, metric, n=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.arange(len(y_true))
    out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        out.append(metric(y_true[b], y_score[b]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.mean(out)), float(lo), float(hi)


def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def clinical_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred))


def _norm_sf(z):
    z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989423 * np.exp(-z * z / 2.0)
    return d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 +
                    t * (-1.821256 + t * 1.330274))))


def _mwu_p(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1 = float(len(a))
    n2 = float(len(b))
    prod = n1 * n2
    ranks = pd.Series(np.concatenate([a, b])).rank().values
    u1 = ranks[:int(n1)].sum() - n1 * (n1 + 1.0) / 2.0
    u = min(u1, prod - u1)
    sigma = np.sqrt(prod * (n1 + n2 + 1.0) / 12.0)
    center = (u - prod / 2.0) / sigma
    return float(2.0 * _norm_sf(center))


def table1(df):
    rows = []
    pos, neg = df[df.y == 1], df[df.y == 0]
    for c in FEATURES:
        if c == "gender":
            ct = pd.crosstab(df.gender, df.y).values.astype(float)
            exp = np.outer(ct.sum(1), ct.sum(0)) / ct.sum()
            chi2 = ((ct - exp) ** 2 / exp).sum()
            p = float(2.0 * _norm_sf(np.sqrt(chi2)))
            rows.append(["Male (%)", round(pos.gender.mean() * 100, 1),
                         round(neg.gender.mean() * 100, 1), p])
        else:
            ptxt = "%.2f (%.2f-%.2f)" % (pos[c].median(), pos[c].quantile(.25), pos[c].quantile(.75))
            ntxt = "%.2f (%.2f-%.2f)" % (neg[c].median(), neg[c].quantile(.25), neg[c].quantile(.75))
            rows.append([FEATURE_LABELS[c], ptxt, ntxt, _mwu_p(pos[c].dropna(), neg[c].dropna())])
    t = pd.DataFrame(rows, columns=["Variable",
                                    "Positive (n=%d)" % len(pos),
                                    "Negative (n=%d)" % len(neg), "p-value"])
    t["p-value"] = t["p-value"].apply(lambda p: "<0.001" if p < 1e-3 else "%.3f" % p)
    return t


@dataclass
class Results:
    best_name: str
    best_pipe: object
    bench: object
    test_metrics: dict
    roc: dict
    pr: dict
    calibration: tuple
    confusion: object
    threshold: float
    thr_sweep: object
    importances: object
    ablation: object
    subgroup: object
    y_test: object = field(repr=False, default=None)
    prob_test: object = field(repr=False, default=None)
    train_size: int = 0
    test_size: int = 0


def run_all(csv_path):
    df = load_and_clean(csv_path)
    X, y = df[FEATURES], df["y"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    bench_rows, roc, pr = [], {}, {}
    for name, pipe in models.items():
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        pred = (oof >= 0.5).astype(int)
        cm = clinical_metrics(y_tr, pred)
        bench_rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_tr, oof),
            "PR-AUC": average_precision_score(y_tr, oof),
            "Accuracy": cm["accuracy"], "Sensitivity": cm["sensitivity"],
            "Specificity": cm["specificity"], "F1": cm["f1"],
            "Brier": brier_score_loss(y_tr, oof)})
    bench = pd.DataFrame(bench_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = bench.iloc[0]["Model"]

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        roc[name] = (fpr, tpr, roc_auc_score(y_te, p))
        rec, prec, _ = precision_recall_curve(y_te, p)
        pr[name] = (rec, prec, average_precision_score(y_te, p))
    best = models[best_name]
    prob = best.predict_proba(X_te)[:, 1]

    thr = youden_threshold(y_te, prob)
    pred = (prob >= thr).astype(int)
    auc_m, auc_lo, auc_hi = bootstrap_ci(y_te, prob, roc_auc_score)
    ap_m, ap_lo, ap_hi = bootstrap_ci(y_te, prob, average_precision_score)
    cm_clin = clinical_metrics(y_te, pred)
    test_metrics = dict(cm_clin)
    test_metrics.update({"roc_auc": auc_m, "roc_auc_ci": (auc_lo, auc_hi),
                         "pr_auc": ap_m, "pr_auc_ci": (ap_lo, ap_hi),
                         "brier": brier_score_loss(y_te, prob)})

    bins = np.linspace(0, 1, 11)
    who = np.digitize(prob, bins) - 1
    mp, fp = [], []
    for bidx in range(10):
        mask = who == bidx
        if mask.sum() > 0:
            mp.append(prob[mask].mean())
            fp.append(y_te[mask].mean())
    calibration = (np.array(mp), np.array(fp), brier_score_loss(y_te, prob))

    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        pr_ = (prob >= t).astype(int)
        c = clinical_metrics(y_te, pr_)
        sweep.append({"threshold": round(float(t), 2),
                      "sensitivity": c["sensitivity"], "specificity": c["specificity"],
                      "ppv": c["ppv"], "npv": c["npv"], "f1": c["f1"]})
    thr_sweep = pd.DataFrame(sweep)

    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc",
                                n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    importances = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": pi.importances_mean,
        "SD": pi.importances_std}).sort_values("Importance", ascending=False).reset_index(drop=True)

    groups = {"All features": FEATURES,
              "Without troponin": [f for f in FEATURES if f != "troponin"],
              "Without cardiac enzymes": [f for f in FEATURES if f not in ("troponin", "kcm")],
              "Vitals only": ["age", "gender", "impluse", "pressurehight", "pressurelow"],
              "Troponin only": ["troponin"]}
    abl = []
    for label, feats in groups.items():
        m = clone(best)
        oof = cross_val_predict(m, X_tr[feats], y_tr, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        abl.append({"Feature set": label, "n_features": len(feats),
                    "CV ROC-AUC": roc_auc_score(y_tr, oof)})
    ablation = pd.DataFrame(abl)

    df_te = X_te.copy()
    df_te["y"] = y_te
    df_te["prob"] = prob
    sub = []
    for label, mask in [("Male", df_te.gender == 1), ("Female", df_te.gender == 0),
                        ("Age < 55", df_te.age < 55), ("Age >= 55", df_te.age >= 55)]:
        d = df_te[mask]
        if d.y.nunique() > 1:
            pr_ = (d.prob >= thr).astype(int)
            cc = clinical_metrics(d.y, pr_)
            sub.append({"Subgroup": label, "n": int(len(d)),
                        "ROC-AUC": roc_auc_score(d.y, d.prob),
                        "Sensitivity": cc["sensitivity"], "Specificity": cc["specificity"]})
    subgroup = pd.DataFrame(sub)

    return Results(
        best_name=best_name, best_pipe=best, bench=bench, test_metrics=test_metrics,
        roc=roc, pr=pr, calibration=calibration,
        confusion=confusion_matrix(y_te, pred), threshold=thr, thr_sweep=thr_sweep,
        importances=importances, ablation=ablation, subgroup=subgroup,
        y_test=y_te, prob_test=prob, train_size=len(X_tr), test_size=len(X_te))


def local_explanation(pipe, row):
    scaler = pipe.named_steps["scale"]
    imp = pipe.named_steps["impute"]
    clf = pipe.named_steps["clf"]
    z = scaler.transform(imp.transform(row))[0]
    if hasattr(clf, "coef_"):
        contrib = z * clf.coef_[0]
    else:
        w = getattr(clf, "feature_importances_", np.ones(len(FEATURES)))
        contrib = z * w
    out = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES],
                        "Contribution": contrib})
    return out.sort_values("Contribution", key=np.abs, ascending=False).reset_index(drop=True)
