"""Test effect of feature_fraction on MCGrad P(Y/N) model."""
import pandas as pd, numpy as np, glob, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

ood_spain = data[data['subpop'] == 'Spain_media'].copy()
ood_belgium = data[data['subpop'] == 'Belgium_tv_news'].copy()

CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']

# Feature names: [categorical] + [numerical] + [prediction_logit]
feat_names = CAT_PYN + NUM + ['prediction_logit']

print(f"{'ff':>5} {'Rounds':>6}  {'AUC_test':>8} {'MCE_test':>8} {'Bias_base':>9} {'Bias_cntry':>10} {'Bias_doc':>8} {'Bias_ES':>7} {'Bias_BE':>7}  {'pred_logit%':>11} {'doc_type%':>9} {'country%':>9}")
print("-" * 135)

for ff in [1.0, 0.8, 0.6, 0.5, 0.4]:
    lgb_params = {'feature_fraction': ff, 'verbose': -1}

    model = mcgrad_methods.MCGrad(
        save_training_performance=True,
        lightgbm_params=lgb_params,
    )
    model = model.fit(cal_df, 'pyn_score', 'law_crime',
        categorical_feature_column_names=CAT_PYN,
        numerical_feature_column_names=NUM)

    for df in [test_df, ood_spain, ood_belgium]:
        df['mc_pyn'] = model.predict(df, 'pyn_score',
            categorical_feature_column_names=CAT_PYN,
            numerical_feature_column_names=NUM)

    # Metrics
    auc = roc_auc_score(test_df['law_crime'], test_df['mc_pyn'])
    mce = mcgrad_metrics.MulticalibrationError(df=test_df, label_column='law_crime',
        score_column='mc_pyn', categorical_segment_columns=CAT_PYN,
        numerical_segment_columns=NUM)

    # Prevalence bias
    N = 5000
    def rs_country(df, t='Belgium', f=5.0, rs=42):
        w = np.where(df['country']==t, f, 1.0); w/=w.sum()
        return df.sample(n=min(N,len(df)), weights=w, replace=True, random_state=rs)
    def rs_doctype(df, t='bill', f=5.0, rs=42):
        w = np.where(df['doc_type']==t, f, 1.0); w/=w.sum()
        return df.sample(n=min(N,len(df)), weights=w, replace=True, random_state=rs)

    baseline = test_df.sample(n=min(N,len(test_df)), replace=True, random_state=42)
    bias_base = (baseline['mc_pyn'].mean() - baseline['law_crime'].mean()) * 100
    bias_cntry = (rs_country(test_df)['mc_pyn'].mean() - rs_country(test_df)['law_crime'].mean()) * 100
    bias_doc = (rs_doctype(test_df)['mc_pyn'].mean() - rs_doctype(test_df)['law_crime'].mean()) * 100
    bias_es = (ood_spain['mc_pyn'].mean() - ood_spain['law_crime'].mean()) * 100
    bias_be = (ood_belgium['mc_pyn'].mean() - ood_belgium['law_crime'].mean()) * 100

    # Feature splits
    total_splits = 0
    pred_splits = 0
    doc_splits = 0
    country_splits = 0
    for booster in model.mr:
        imp = booster.feature_importance(importance_type='split')
        total_splits += imp.sum()
        pred_splits += imp[-1]  # prediction_logit is last
        # doc_type and country indices
        doc_idx = feat_names.index('doc_type')
        country_idx = feat_names.index('country')
        doc_splits += imp[doc_idx]
        country_splits += imp[country_idx]

    pred_pct = pred_splits / total_splits * 100 if total_splits > 0 else 0
    doc_pct = doc_splits / total_splits * 100 if total_splits > 0 else 0
    country_pct = country_splits / total_splits * 100 if total_splits > 0 else 0

    print(f"{ff:>5.1f} {len(model.mr):>6}  {auc:>8.4f} {mce.mce:>8.4f} {bias_base:>+8.1f}pp {bias_cntry:>+9.1f}pp {bias_doc:>+7.1f}pp {bias_es:>+6.1f}pp {bias_be:>+6.1f}pp  {pred_pct:>10.1f}% {doc_pct:>8.1f}% {country_pct:>8.1f}%")
