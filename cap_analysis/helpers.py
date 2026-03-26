import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def calibrate_threshold_youden(labels, predictions):
    """Find threshold that maximizes Youden's J = TPR - FPR."""
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, predictions)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def estimate_classifier_error_rates(labels, predictions, threshold):
    """Estimate TPR and FPR from calibration data at given threshold."""
    binary_preds = (predictions >= threshold).astype(int)
    labels_arr = labels.astype(int)
    tp = ((binary_preds == 1) & (labels_arr == 1)).sum()
    fp = ((binary_preds == 1) & (labels_arr == 0)).sum()
    tn = ((binary_preds == 0) & (labels_arr == 0)).sum()
    fn = ((binary_preds == 0) & (labels_arr == 1)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (EMQ) prevalence estimator.

    EM algorithm that iteratively re-estimates prevalence by adjusting
    posteriors for a new prior. Assumes label shift (P(X|Y) stable).
    """
    p_hat = source_prevalence
    for _ in range(max_iter):
        ratio_pos = p_hat / source_prevalence
        ratio_neg = (1 - p_hat) / (1 - source_prevalence)
        adjusted = (ratio_pos * scores) / (ratio_pos * scores + ratio_neg * (1 - scores))
        p_new = adjusted.mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


def pacc_estimate(scores, pos_mean, neg_mean):
    """Probabilistic Adjusted Classify & Count.

    Soft-score generalization of Rogan-Gladen: uses E[h(X)|Y=1] and E[h(X)|Y=0]
    instead of binary TPR/FPR.
    """
    pcc = scores.mean()
    denom = pos_mean - neg_mean
    if abs(denom) < 1e-10:
        return pcc
    return np.clip((pcc - neg_mean) / denom, 0.0, 1.0)


def compute_rogan_gladen_estimate(apparent_prevalence, tpr, fpr):
    """Rogan-Gladen adjusted prevalence estimate."""
    denominator = tpr - fpr
    if abs(denominator) < 1e-10:
        return apparent_prevalence
    adjusted = (apparent_prevalence - fpr) / denominator
    return max(0.0, min(1.0, adjusted))


def compute_all_prevalence_estimates(
    target_df,
    score_col,
    ir_col,
    mcgrad_col,
    label_col,
    cal_tpr,
    cal_fpr,
    pacc_pos_mean,
    pacc_neg_mean,
    source_prevalence,
    threshold=0.5,
    mcgrad_emb_col=None,
    sld_score_col=None,
):
    """Compute prevalence estimates using all methods."""
    true_prevalence = target_df[label_col].mean()

    # 1. Raw LLM scores
    raw_estimate = target_df[score_col].mean()

    # 2. Classify & Count
    binary_preds = (target_df[score_col] >= threshold).astype(int)
    classify_count = binary_preds.mean()

    # 3. Rogan-Gladen
    rogan_gladen = compute_rogan_gladen_estimate(classify_count, cal_tpr, cal_fpr)

    # 4. PACC
    pacc = pacc_estimate(target_df[score_col].values, pacc_pos_mean, pacc_neg_mean)

    # 5. SLD (EMQ) — use squashed scores if provided (same preprocessing as MCGrad)
    _sld_scores = target_df[sld_score_col].values if sld_score_col else target_df[score_col].values
    sld = sld_estimate(_sld_scores, source_prevalence)

    # 6. Isotonic Regression
    isotonic_estimate = target_df[ir_col].mean()

    # 7. MCGrad
    mcgrad_estimate = target_df[mcgrad_col].mean()

    results = {
        'True Prevalence': true_prevalence,
        'Raw Scores': raw_estimate,
        'Classify & Count': classify_count,
        'Rogan-Gladen': rogan_gladen,
        'PACC': pacc,
        'SLD (EMQ)': sld,
        'Isotonic Regression': isotonic_estimate,
        'MCGrad': mcgrad_estimate,
    }

    # 8. MCGrad + Embeddings (if available)
    if mcgrad_emb_col and mcgrad_emb_col in target_df.columns:
        results['MCGrad + Emb.'] = target_df[mcgrad_emb_col].mean()

    return results


def compute_bias_table(estimates):
    """Create a DataFrame showing estimates and bias for each method."""
    true_prev = estimates['True Prevalence']
    rows = []
    for method, estimate in estimates.items():
        if method == 'True Prevalence':
            continue
        bias = estimate - true_prev
        rows.append({
            'Method': method,
            'Estimate': estimate,
            'Bias': bias,
            'Relative Bias (%)': 100 * bias / true_prev if true_prev > 0 else 0,
        })
    return pd.DataFrame(rows).set_index('Method')


def resample_with_party_shift(
    df,
    label_column,
    shift='original',
    n_samples=20_000,
    random_state=42,
):
    actual_n = min(n_samples, len(df))
    if shift == 'original':
        weights = np.ones(len(df))
    else:
        party_topic_rate = df.groupby('party')[label_column].mean()
        party_rank = party_topic_rate.rank(pct=True)
        rank_values = df['party'].map(party_rank).values

        if shift == 'left_heavy':
            weights = np.exp(-2.0 * rank_values)
        elif shift == 'right_heavy':
            weights = np.exp(2.0 * rank_values)
        elif shift == 'polarized':
            weights = np.exp(2.0 * np.abs(rank_values - 0.5))
        else:
            raise ValueError(f"Unknown shift: {shift}")

    weights = weights / weights.sum()
    return df.sample(
        n=actual_n,
        weights=weights,
        replace=True,
        random_state=random_state,
    )


def resample_with_doctype_shift(
    df,
    target_doctype='bill',
    overweight_factor=5.0,
    n_samples=20_000,
    random_state=42,
):
    actual_n = min(n_samples, len(df))
    weights = np.where(
        df['doc_type'] == target_doctype,
        overweight_factor,
        1.0,
    )
    weights = weights / weights.sum()
    return df.sample(
        n=actual_n,
        weights=weights,
        replace=True,
        random_state=random_state,
    )


def resample_with_country_shift(
    df,
    target_country='Belgium',
    overweight_factor=5.0,
    n_samples=20_000,
    random_state=42,
):
    actual_n = min(n_samples, len(df))
    weights = np.where(
        df['country'] == target_country,
        overweight_factor,
        1.0,
    )
    weights = weights / weights.sum()
    return df.sample(
        n=actual_n,
        weights=weights,
        replace=True,
        random_state=random_state,
    )


def compute_bootstrap_rmse(
    source_df,
    score_col,
    ir_col,
    mcgrad_col,
    label_col,
    cal_tpr,
    cal_fpr,
    pacc_pos_mean,
    pacc_neg_mean,
    source_prevalence,
    threshold,
    mcgrad_emb_col=None,
    sld_score_col=None,
    n_bootstrap=200,
    n_samples=20_000,
):
    """Bootstrap resampling to compute bias, variance, and RMSE.

    Resamples uniformly from the source population (no additional shift).
    Sample size is min(n_samples, len(source_df)).
    """
    actual_n = min(n_samples, len(source_df))
    methods_list = [
        'Raw Scores', 'Classify & Count', 'Rogan-Gladen', 'PACC',
        'SLD (EMQ)', 'Isotonic Regression', 'MCGrad',
    ]
    if mcgrad_emb_col:
        methods_list.append('MCGrad + Emb.')
    estimates_by_method = {m: [] for m in methods_list}
    true_prevs = []

    for b in range(n_bootstrap):
        syn_df = source_df.sample(
            n=actual_n,
            replace=True,
            random_state=b,
        )
        est = compute_all_prevalence_estimates(
            syn_df, score_col, ir_col, mcgrad_col, label_col,
            cal_tpr, cal_fpr, pacc_pos_mean, pacc_neg_mean,
            source_prevalence, threshold,
            mcgrad_emb_col=mcgrad_emb_col,
            sld_score_col=sld_score_col,
        )
        true_prevs.append(est['True Prevalence'])
        for m in methods_list:
            estimates_by_method[m].append(est[m])

    results = {}
    for m in methods_list:
        ests = np.array(estimates_by_method[m])
        trues = np.array(true_prevs)
        errors = ests - trues
        results[m] = {
            'bias': np.mean(errors) * 100,       # in percentage points
            'variance': np.var(errors) * 100**2,  # in pp^2
            'rmse': np.sqrt(np.mean(errors**2)) * 100,  # in pp
        }
    results['True Prevalence'] = np.mean(true_prevs)
    return results


def compute_bootstrap_rmse_shifted(
    source_df,
    shift_fn,
    shift_kwargs,
    score_col,
    ir_col,
    mcgrad_col,
    label_col,
    cal_tpr,
    cal_fpr,
    pacc_pos_mean,
    pacc_neg_mean,
    source_prevalence,
    threshold,
    mcgrad_emb_col=None,
    sld_score_col=None,
    n_bootstrap=200,
):
    """Bootstrap with a shift resampling function (for within-calibration scenarios)."""
    methods_list = [
        'Raw Scores', 'Classify & Count', 'Rogan-Gladen', 'PACC',
        'SLD (EMQ)', 'Isotonic Regression', 'MCGrad',
    ]
    if mcgrad_emb_col:
        methods_list.append('MCGrad + Emb.')
    estimates_by_method = {m: [] for m in methods_list}
    true_prevs = []

    for b in range(n_bootstrap):
        syn_df = shift_fn(source_df, random_state=b, **shift_kwargs)
        est = compute_all_prevalence_estimates(
            syn_df, score_col, ir_col, mcgrad_col, label_col,
            cal_tpr, cal_fpr, pacc_pos_mean, pacc_neg_mean,
            source_prevalence, threshold,
            mcgrad_emb_col=mcgrad_emb_col,
            sld_score_col=sld_score_col,
        )
        true_prevs.append(est['True Prevalence'])
        for m in methods_list:
            estimates_by_method[m].append(est[m])

    results = {}
    for m in methods_list:
        ests = np.array(estimates_by_method[m])
        trues = np.array(true_prevs)
        errors = ests - trues
        results[m] = {
            'bias': np.mean(errors) * 100,
            'variance': np.var(errors) * 100**2,
            'rmse': np.sqrt(np.mean(errors**2)) * 100,
        }
    results['True Prevalence'] = np.mean(true_prevs)
    return results
