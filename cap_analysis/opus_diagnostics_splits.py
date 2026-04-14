"""Check what features MCGrad's LightGBM trees actually split on."""
import pandas as pd, numpy as np, glob, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

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
cal_parts = []
for key in CAL_SUBPOPS:
    sub = data[data['subpop'] == key].copy()
    cal, _ = train_test_split(sub, test_size=0.33, random_state=42, stratify=sub['law_crime'])
    cal_parts.append(cal)
cal_df = pd.concat(cal_parts, ignore_index=True)

BASE_RATE = cal_df['law_crime'].mean()
cal_df['binary_init'] = BASE_RATE
cal_df['llm_label'] = cal_df['llm_yes'].map({1: 'yes', 0: 'no'})

CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']

# Fit both
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)

mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)

# Feature names for each model
# MCGrad feature order: [categorical] + [numerical] + [prediction_logit]
bin_feature_names = CAT_BIN + NUM + ['prediction_logit']
pyn_feature_names = CAT_PYN + NUM + ['prediction_logit']

print("=== Feature Importance (LightGBM split counts) ===\n")

for name, model, feat_names in [
    ('MCGrad (binary)', mcgrad_bin, bin_feature_names),
    ('MCGrad (P(Y/N))', mcgrad_pyn, pyn_feature_names),
]:
    print(f"{name}: {len(model.mr)} round(s)")
    for round_idx, booster in enumerate(model.mr):
        # Get feature importance
        importance = booster.feature_importance(importance_type='split')
        importance_gain = booster.feature_importance(importance_type='gain')

        print(f"  Round {round_idx}:")
        print(f"    {'Feature':<20} {'Splits':>7} {'Gain':>12} {'% Splits':>10}")
        print(f"    {'-'*52}")

        total_splits = importance.sum()
        for feat, splits, gain in sorted(zip(feat_names, importance, importance_gain),
                                          key=lambda x: -x[1]):
            pct = splits / total_splits * 100 if total_splits > 0 else 0
            print(f"    {feat:<20} {splits:>7} {gain:>12.1f} {pct:>9.1f}%")

        # Also dump tree structure for first few trees
        tree_df = booster.trees_to_dataframe()
        split_features = tree_df[tree_df['split_feature'].notna()]['split_feature'].value_counts()
        print(f"\n    Tree split feature counts:")
        for feat, count in split_features.items():
            print(f"      {feat}: {count}")
    print()
