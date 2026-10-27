"""Editor points 5 and 9 (tabular branch).

5  Independent diagnostic value: does the eight-variable Random Forest add discrimination beyond troponin and
   CK-MB? Compares, on the same split (seed 42), the full model with (a) a Random Forest on troponin + CK-MB only,
   (b) a logistic regression on log-troponin + log-CK-MB, (c) the two-threshold rule, and (d) a Random Forest on
   the six non-biomarker variables. Paired bootstrap (2,000 resamples of the test set) for AUC differences;
   exact McNemar test for paired accuracy at thresholds fixed on training data.
9  Feature-importance evidence: permutation importance (ROC-AUC drop, 20 repeats) recomputed on each of the
   five cross-validation validation folds, drop-column importance (cross-validated AUC loss when a variable is
   removed and the model refitted), and the impurity importance; rank agreement across folds (Kendall W).
Writes results/tab_incremental.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import meare_model as M

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
BIO = ["troponin", "kcm"]
NONBIO = [f for f in M.FEATURES if f not in BIO]


def youden(y, p):
    return M.youden_threshold(y, p)


def fit_eval(model, cols, X_tr, y_tr, X_te, cv):
    oof = cross_val_predict(clone(model), X_tr[cols], y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr = youden(y_tr, oof)
    m = clone(model).fit(X_tr[cols], y_tr)
    return m, m.predict_proba(X_te[cols])[:, 1], thr, roc_auc_score(y_tr, oof)


def paired_auc_diff(y, pa, pb, n=2000):
    d = []
    for _ in range(n):
        b = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            d.append(roc_auc_score(y[b], pa[b]) - roc_auc_score(y[b], pb[b]))
    d = np.array(d)
    return dict(diff=float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def mcnemar(y, ca, cb):
    a, b = ca == y, cb == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(only_a_correct=n01, only_b_correct=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


def main():
    df = M.load_and_clean(HERE / "heart_attack.csv")
    X, y = df[M.FEATURES], df.y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=M.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=M.RANDOM_STATE)
    rf = M.build_models()["Random Forest"]
    lr_log = Pipeline([("log", FunctionTransformer(lambda a: np.log10(np.asarray(a, float) + 1e-3))),
                       ("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    res = {}
    models = {"RF, all 8 variables": (rf, M.FEATURES), "RF, troponin + CK-MB": (rf, BIO),
              "Logistic, log troponin + log CK-MB": (lr_log, BIO), "RF, 6 non-biomarker variables": (rf, NONBIO)}
    preds = {}
    for name, (mdl, cols) in models.items():
        m, p, thr, cvauc = fit_eval(mdl, cols, X_tr, y_tr, X_te, cv)
        preds[name] = (p, (p >= thr).astype(int))
        res[name] = dict(cv_auc=float(cvauc), test_auc=float(roc_auc_score(y_te, p)), threshold=float(thr),
                         test_acc=float(((p >= thr).astype(int) == y_te).mean()))
    rule = ((X_te.troponin > 0.014) | (X_te.kcm > 6.2524)).astype(int).values
    res["Two-threshold rule"] = dict(test_acc=float((rule == y_te).mean()))
    full_p, full_c = preds["RF, all 8 variables"]
    cmp = {}
    for name in ("RF, troponin + CK-MB", "Logistic, log troponin + log CK-MB", "RF, 6 non-biomarker variables"):
        cmp[name] = dict(auc=paired_auc_diff(y_te, full_p, preds[name][0]), mcnemar=mcnemar(y_te, full_c, preds[name][1]))
    cmp["Two-threshold rule"] = dict(mcnemar=mcnemar(y_te, full_c, rule))
    res["full_vs"] = cmp

    # ---- feature-importance evidence
    perm = []
    for tr_i, va_i in cv.split(X_tr, y_tr):
        m = clone(rf).fit(X_tr.iloc[tr_i], y_tr[tr_i])
        pi = permutation_importance(m, X_tr.iloc[va_i], y_tr[va_i], scoring="roc_auc", n_repeats=20,
                                    random_state=M.RANDOM_STATE, n_jobs=-1)
        perm.append(pi.importances_mean)
    perm = np.array(perm)                                  # folds x features
    ranks = np.array([rankdata(-r) for r in perm])         # 1 = most important
    k, n = ranks.shape
    R = ranks.sum(0); S = ((R - R.mean()) ** 2).sum()
    kendall_w = float(12 * S / (k ** 2 * (n ** 3 - n)))
    base = cross_val_predict(clone(rf), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    base_auc = roc_auc_score(y_tr, base)
    drop = {}
    for f in M.FEATURES:
        cols = [c for c in M.FEATURES if c != f]
        o = cross_val_predict(clone(rf), X_tr[cols], y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        drop[f] = float(base_auc - roc_auc_score(y_tr, o))
    imp = clone(rf).fit(X_tr, y_tr).named_steps["clf"].feature_importances_
    res["importance"] = dict(
        features=[M.FEATURE_LABELS[f] for f in M.FEATURES],
        perm_mean=perm.mean(0).round(4).tolist(), perm_sd=perm.std(0, ddof=1).round(4).tolist(),
        perm_rank_per_fold=ranks.astype(int).tolist(), kendall_w=kendall_w,
        top2_every_fold=bool(all(set(np.argsort(-r)[:2]) == {M.FEATURES.index("troponin"), M.FEATURES.index("kcm")} for r in perm)),
        drop_column_auc_loss=drop, impurity=imp.round(4).tolist())
    (HERE / "results" / "tab_incremental.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:5000])


if __name__ == "__main__":
    main()
