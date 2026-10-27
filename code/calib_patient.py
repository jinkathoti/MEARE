# -*- coding: utf-8 -*-
"""R2 additions:
 (A) Tabular calibration: ECE + reliability curve for the selected Random Forest (same split, seed 42).
 (B) Patient-level cluster bootstrap for the controlled ECG comparison (image vs raw).
Outputs: results/calibration.json, figures fig_calib.png; results/patient_bootstrap.json."""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent
RES=HERE/"results"; RES.mkdir(exist_ok=True)
FIGDIR=HERE.parent/"manuscript"/"figures"
rng=np.random.default_rng(42)

# ---------- (A) tabular calibration ----------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

df=pd.read_csv(HERE/"heart_attack.csv")
FE=["age","gender","impluse","pressurehight","pressurelow","glucose","kcm","troponin"]
X=df[FE].astype(float).copy()
y=(df["class"].astype(str).str.lower().str.startswith("pos")).astype(int).values
# implausible heart rate -> NaN (imputed in pipeline)
X.loc[(X["impluse"]<20)|(X["impluse"]>300),"impluse"]=np.nan
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
               ("rf",RandomForestClassifier(n_estimators=400,max_features="sqrt",
                     class_weight="balanced",random_state=42))])
pipe.fit(Xtr,ytr)
p=pipe.predict_proba(Xte)[:,1]
auc=roc_auc_score(yte,p); brier=brier_score_loss(yte,p)
thr=0.65; pred=(p>=thr).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
# ECE (15 equal-width bins) + reliability curve
B=15; edges=np.linspace(0,1,B+1); ece=0.0; rel=[]
for b in range(B):
    m=(p>=edges[b])&(p<edges[b+1]) if b<B-1 else (p>=edges[b])&(p<=edges[b+1])
    nb=m.sum()
    if nb>0:
        conf=p[m].mean(); acc=yte[m].mean(); ece+=nb/len(p)*abs(acc-conf)
        rel.append({"bin_lo":float(edges[b]),"bin_hi":float(edges[b+1]),"n":int(nb),
                    "conf":float(conf),"acc":float(acc)})
calib={"n_test":int(len(yte)),"auc":float(auc),"brier":float(brier),"ece_15bin":float(ece),
       "confusion":{"tp":int(tp),"fn":int(fn),"tn":int(tn),"fp":int(fp)},
       "acc":float((tp+tn)/len(yte)),"reliability":rel}
json.dump(calib,open(RES/"calibration.json","w"),indent=1)
print("CALIB auc=%.4f acc=%.4f brier=%.4f ece=%.4f (tp%d fn%d tn%d fp%d)"%(
      auc,calib["acc"],brier,ece,tp,fn,tn,fp))

# reliability figure
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
confs=[r["conf"] for r in rel]; accs=[r["acc"] for r in rel]; ns=[r["n"] for r in rel]
ax[0].plot([0,1],[0,1],"--",color="grey",lw=1,label="perfect")
ax[0].plot(confs,accs,"o-",color="#b00020",lw=1.5,ms=5,label="Random Forest")
ax[0].set_xlabel("Mean predicted probability"); ax[0].set_ylabel("Observed frequency (positive)")
ax[0].set_title("Reliability diagram (test set)"); ax[0].legend(frameon=False); ax[0].set_xlim(0,1); ax[0].set_ylim(0,1)
ax[0].text(0.05,0.9,"ECE=%.3f\nBrier=%.3f"%(ece,brier),transform=ax[0].transAxes,va="top",fontsize=9)
ax[1].bar([(e0+e1)/2 for e0,e1 in zip(edges[:-1],edges[1:])],
          [next((r["n"] for r in rel if abs((r["bin_lo"]+r["bin_hi"])/2-((edges[b]+edges[b+1])/2))<1e-6),0) for b in range(B)],
          width=1/B*0.9,color="#4169e1")
ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Count"); ax[1].set_title("Prediction histogram"); ax[1].set_xlim(0,1)
plt.tight_layout(); plt.savefig(FIGDIR/"fig_calib.png",dpi=200); print("wrote",FIGDIR/"fig_calib.png")

# ---------- (B) patient-level cluster bootstrap, controlled ECG ----------
lab=pd.read_csv("D:/CDSIMER-Lastday/cvd_app/data_ecg/labels.csv")
pid=dict(zip(lab["ecg_id"].astype(int), lab["patient_id"].astype(float)))
def load_seedavg(kind):
    dfs=[pd.read_csv(RES/f"ecg_ctrl_{kind}_s{s}.csv") for s in (0,1,2)]
    base=dfs[0][["ecg_id","label"]].copy()
    base["p"]=np.mean([d["p"].values for d in dfs],axis=0)
    base["pid"]=base["ecg_id"].astype(int).map(pid)
    return base
img=load_seedavg("image"); raw=load_seedavg("raw")
assert (img["ecg_id"].values==raw["ecg_id"].values).all()
y=img["label"].values; pi=img["p"].values; pr=raw["p"].values; pids=img["pid"].values
uniq=np.unique(pids); groups={u:np.where(pids==u)[0] for u in uniq}
def auc_safe(yy,pp):
    return roc_auc_score(yy,pp) if len(np.unique(yy))>1 else np.nan
N=2000; diffs=[]; aimg=[]; araw=[]
for _ in range(N):
    samp=rng.choice(uniq,size=len(uniq),replace=True)
    idx=np.concatenate([groups[u] for u in samp])
    yy=y[idx]
    if len(np.unique(yy))<2: continue
    ai=auc_safe(yy,pi[idx]); ar=auc_safe(yy,pr[idx])
    aimg.append(ai); araw.append(ar); diffs.append(ar-ai)
diffs=np.array(diffs); aimg=np.array(aimg); araw=np.array(araw)
pb={"n_patients":int(len(uniq)),"n_records":int(len(y)),
    "image_auc_mean":float(auc_safe(y,pi)),"raw_auc_mean":float(auc_safe(y,pr)),
    "image_auc_ci_patient":[float(np.percentile(aimg,2.5)),float(np.percentile(aimg,97.5))],
    "raw_auc_ci_patient":[float(np.percentile(araw,2.5)),float(np.percentile(araw,97.5))],
    "diff_raw_minus_image":float(auc_safe(y,pr)-auc_safe(y,pi)),
    "diff_ci_patient":[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))],
    "p_two_sided_patient":float(2*min((diffs>0).mean(),(diffs<0).mean()))}
json.dump(pb,open(RES/"patient_bootstrap.json","w"),indent=1)
print("PATIENT-BOOT n_pat=%d img=%.4f raw=%.4f diff=%.4f CI[%.4f,%.4f] p=%.3f"%(
      pb["n_patients"],pb["image_auc_mean"],pb["raw_auc_mean"],pb["diff_raw_minus_image"],
      pb["diff_ci_patient"][0],pb["diff_ci_patient"][1],pb["p_two_sided_patient"]))
