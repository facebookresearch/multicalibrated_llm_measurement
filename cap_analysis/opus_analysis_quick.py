"""Quick analysis of Claude Opus results: binary labels vs P(Y/N) scores."""
import pandas as pd
import numpy as np
import glob, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from mcgrad import methods as mcgrad_methods

# Load data
sample = pd.read_csv('data/opus_30k_sample.csv')
sample['party'] = sample['party'].fillna('unknown')
sample['text_len'] = sample['text'].str.len()

binary_files = sorted(glob.glob('data/inference_output/claude-opus-30k-binary/shard_*.csv'))
binary = pd.concat([pd.read_csv(f) for f in binary_files]).drop_duplicates(subset='id', keep='first')
binary['llm_yes'] = (binary['answer'].str.lower() == 'yes').astype(int)

pyn_files = sorted(glob.glob('data/inference_output/claude-opus-30k-pyn/shard_*.csv'))
pyn = pd.concat([pd.read_csv(f) for f in pyn_files])

data = sample.merge(binary[['id', 'llm_yes']], on='id', how='left')
data = data.merge(pyn[['id', 'score']], on='id', how='left')
data.rename(columns={'score': 'pyn_score'}, inplace=True)

# Split
CAL_SUBPOPS = ['Denmark_parliamentary_question', 'Spain_parliamentary_question',
               'United States_bill', 'Belgium_newspaper']
cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = data[data['subpop'] == key].copy()
    cal, test = train_test_split(sub, test_size=0.33, random_state=42, stratify=sub['law_crime'])
    cal_parts.append(cal)
    test_parts.append(test)
cal_df = pd.concat(cal_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

ood_spain = data[data['subpop'] == 'Spain_media'].copy()
ood_belgium = data[data['subpop'] == 'Belgium_tv_news'].copy()

print(f"Cal: {len(cal_df):,}, Test: {len(test_df):,}")
print(f"Cal prevalence: {cal_df['law_crime'].mean():.3f}")

# === Condition 1: Binary + MCGrad ===
BASE_RATE = cal_df['law_crime'].mean()
for df in [cal_df, test_df, ood_spain, ood_belgium]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

print("\nFitting MCGrad (binary)...")
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(
    cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=['country', 'doc_type', 'party', 'llm_label'],
    numerical_feature_column_names=['decade', 'text_len'],
)
for df in [test_df, ood_spain, ood_belgium]:
    df['mc_bin'] = mcgrad_bin.predict(df, 'binary_init',
        categorical_feature_column_names=['country', 'doc_type', 'party', 'llm_label'],
        numerical_feature_column_names=['decade', 'text_len'])

# === Condition 2: P(Y/N) + MCGrad ===
print("Fitting MCGrad (P(Y/N))...")
isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime')
mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(
    cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=['country', 'doc_type', 'party'],
    numerical_feature_column_names=['decade', 'text_len'],
)
for df in [test_df, ood_spain, ood_belgium]:
    df['iso_pred'] = isotonic.predict(df, 'pyn_score')
    df['mc_pyn'] = mcgrad_pyn.predict(df, 'pyn_score',
        categorical_feature_column_names=['country', 'doc_type', 'party'],
        numerical_feature_column_names=['decade', 'text_len'])

# === IPW ===
def ipw_estimate(cal, target):
    feats = ['country', 'doc_type', 'decade', 'text_len']
    combined = pd.concat([cal[feats].assign(_t=0), target[feats].assign(_t=1)], ignore_index=True)
    X = pd.get_dummies(combined[feats], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal['law_crime'].values, weights=w), 0, 1)

# === Calibration params ===
fpr_arr, tpr_arr, thresholds = roc_curve(cal_df['law_crime'], cal_df['pyn_score'])
THRESHOLD = float(thresholds[np.argmax(tpr_arr - fpr_arr)])
bc = (cal_df['pyn_score'] >= THRESHOLD).astype(int)
cl = cal_df['law_crime'].astype(int)
cal_tpr = ((bc==1)&(cl==1)).sum() / cl.sum()
cal_fpr = ((bc==1)&(cl==0)).sum() / (1-cl).sum()
pacc_pos = cal_df[cal_df['law_crime']==1]['pyn_score'].mean()
pacc_neg = cal_df[cal_df['law_crime']==0]['pyn_score'].mean()
src_prev = cal_df['law_crime'].mean()

def sld(scores, sp=src_prev):
    p = sp
    for _ in range(100):
        rp = p/sp; rn = (1-p)/(1-sp)
        adj = (rp*scores)/(rp*scores+rn*(1-scores))
        pn = adj.mean()
        if abs(pn-p)<1e-6: break
        p = pn
    return p

# === Scenarios ===
N = 5000
def rs_country(df, t='Belgium', f=5.0, rs=42):
    w = np.where(df['country']==t, f, 1.0); w/=w.sum()
    return df.sample(n=min(N,len(df)), weights=w, replace=True, random_state=rs)
def rs_doctype(df, t='bill', f=5.0, rs=42):
    w = np.where(df['doc_type']==t, f, 1.0); w/=w.sum()
    return df.sample(n=min(N,len(df)), weights=w, replace=True, random_state=rs)

scenarios = {
    'Baseline':          test_df.sample(n=min(N, len(test_df)), replace=True, random_state=42),
    'Country shift':     rs_country(test_df),
    'Doc-type shift':    rs_doctype(test_df),
    'Spain media (OOD)': ood_spain,
    'Belgium TV (OOD)':  ood_belgium,
}

# === Results ===
print(f"\n{'Scenario':<22} {'True':>6}  {'CC':>7} {'Raw':>7} {'RG':>7} {'PACC':>7} {'SLD':>7} {'IPW':>7} {'Iso':>7} {'MC+Bin':>7} {'MC+PYN':>7}")
print("-"*102)

for name, t in scenarios.items():
    tp = t['law_crime'].mean()
    cc = t['llm_yes'].mean()
    raw = t['pyn_score'].mean()
    ap = (t['pyn_score'] >= THRESHOLD).mean()
    d = cal_tpr - cal_fpr
    rg = np.clip((ap - cal_fpr)/d, 0, 1) if abs(d)>1e-10 else ap
    pacc = np.clip((raw - pacc_neg)/(pacc_pos - pacc_neg), 0, 1)
    s = sld(t['pyn_score'].values)
    ipw = ipw_estimate(cal_df, t)
    iso = t['iso_pred'].mean()
    mcb = t['mc_bin'].mean()
    mcp = t['mc_pyn'].mean()

    def fb(e): return f"{(e-tp)*100:+.1f}"
    print(f"{name:<22} {tp:>5.1%}  {fb(cc):>7} {fb(raw):>7} {fb(rg):>7} {fb(pacc):>7} {fb(s):>7} {fb(ipw):>7} {fb(iso):>7} {fb(mcb):>7} {fb(mcp):>7}")

print(f"\nCC = Classify & Count (binary Yes/No)")
print(f"Raw = mean P(Y/N) scores")
print(f"MC+Bin = MCGrad on binary labels (base rate init, LLM label as feature)")
print(f"MC+PYN = MCGrad on P(Y/N) probability scores")

# === AUC comparison ===
print(f"\n=== Discriminative performance (test set) ===")
valid_bin = test_df.dropna(subset=['llm_yes'])
print(f"Binary labels AUC:  {roc_auc_score(valid_bin['law_crime'], valid_bin['llm_yes']):.4f}")
print(f"P(Y/N) scores AUC:  {roc_auc_score(test_df['law_crime'], test_df['pyn_score']):.4f}")
print(f"MCGrad(binary) AUC: {roc_auc_score(test_df['law_crime'], test_df['mc_bin']):.4f}")
print(f"MCGrad(P(Y/N)) AUC: {roc_auc_score(test_df['law_crime'], test_df['mc_pyn']):.4f}")
