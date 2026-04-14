"""MCGrad diagnostics: compare binary vs P(Y/N) model fit."""
import pandas as pd, numpy as np, glob, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from mcgrad import methods as mcgrad_methods, metrics as mcgrad_metrics

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
data = data.dropna(subset=['llm_yes'])
data['llm_yes'] = data['llm_yes'].astype(int)

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

BASE_RATE = cal_df['law_crime'].mean()
for df in [cal_df, test_df]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']

print("Fitting MCGrad (binary)...")
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)

print("Fitting MCGrad (P(Y/N))...")
mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)

test_df['mc_bin'] = mcgrad_bin.predict(test_df, 'binary_init',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)
test_df['mc_pyn'] = mcgrad_pyn.predict(test_df, 'pyn_score',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)

# === Model fit details ===
print(f"\n=== Model Fit ===")
print(f"Binary: {len(mcgrad_bin.mr)} rounds")
print(f"  Unshrink factors: {[f'{u:.3f}' for u in mcgrad_bin.unshrink_factors]}")
print(f"P(Y/N): {len(mcgrad_pyn.mr)} rounds")
print(f"  Unshrink factors: {[f'{u:.3f}' for u in mcgrad_pyn.unshrink_factors]}")

# === Test set diagnostics ===
print(f"\n=== Test Set Diagnostics ===")
configs = [
    ('Raw P(Y/N)', 'pyn_score', CAT_PYN),
    ('MCGrad (binary)', 'mc_bin', CAT_BIN),
    ('MCGrad (P(Y/N))', 'mc_pyn', CAT_PYN),
]

print(f"{'Method':<22} {'AUC':>7} {'LogLoss':>8} {'ECCE':>7} {'ECCE_s':>7} {'MCE':>7} {'MCE_s':>7} {'Mean':>7} {'Std':>7}")
print("-" * 85)
for name, col, cat in configs:
    v = test_df[col].values
    y = test_df['law_crime'].values
    auc = roc_auc_score(y, v)
    ll = log_loss(y, np.clip(v, 1e-10, 1-1e-10))
    ecce = mcgrad_metrics.ecce(y, v)
    ecce_s = mcgrad_metrics.ecce_sigma(y, v)
    mce = mcgrad_metrics.MulticalibrationError(df=test_df, label_column='law_crime',
        score_column=col, categorical_segment_columns=cat, numerical_segment_columns=NUM)
    print(f"{name:<22} {auc:>7.4f} {ll:>8.4f} {ecce:>7.4f} {ecce_s:>7.2f} {mce.mce:>7.4f} {mce.mce_sigma:>7.2f} {v.mean():>7.4f} {v.std():>7.4f}")

# === Score distribution ===
print(f"\n=== Score Distribution ===")
for name, col in [('MCGrad (binary)', 'mc_bin'), ('MCGrad (P(Y/N))', 'mc_pyn')]:
    v = test_df[col].values
    pos = v[test_df['law_crime']==1]
    neg = v[test_df['law_crime']==0]
    print(f"\n{name}: unique={len(np.unique(v))}")
    print(f"  Quantiles: 5%={np.quantile(v,.05):.4f} 25%={np.quantile(v,.25):.4f} "
          f"50%={np.quantile(v,.50):.4f} 75%={np.quantile(v,.75):.4f} 95%={np.quantile(v,.95):.4f}")
    print(f"  Pos mean={pos.mean():.4f} Neg mean={neg.mean():.4f}")
    print(f"  Range: [{v.min():.4f}, {v.max():.4f}]")

# === Per-subpop diagnostics ===
print(f"\n=== Per-Subpop Prevalence Bias (test set) ===")
print(f"{'Subpop':<40} {'True':>6} {'MC_bin':>7} {'MC_pyn':>7}")
print("-" * 65)
for subpop in sorted(test_df['subpop'].unique()):
    sub = test_df[test_df['subpop'] == subpop]
    tp = sub['law_crime'].mean()
    mb = sub['mc_bin'].mean()
    mp = sub['mc_pyn'].mean()
    print(f"{subpop:<40} {tp:>5.1%} {(mb-tp)*100:>+6.1f}pp {(mp-tp)*100:>+6.1f}pp")
