import numpy as np
import pandas as pd


def generate_data(n_samples, p_x0, p_y_given_x1, bias_b, bias_c):
    X = np.random.choice([0, 1], size=n_samples, p=[p_x0, 1 - p_x0])

    Y = np.array(
        [np.random.binomial(1, p_y_given_x1 if x == 1 else 1 - p_y_given_x1) for x in X]
    )

    p_estimated = np.array(
        [(1 - p_y_given_x1) * bias_b if x == 0 else p_y_given_x1 * bias_c for x in X]
    )

    assert p_estimated.min() >= 0 and p_estimated.max() <= 1, "Invalid probabilities"

    data = pd.DataFrame({"X": X, "Y": Y, "p_estimated": p_estimated})

    return data


def apply_rogan_gladen(apparent_prevalence, tpr, fpr):
    """Binary Rogan-Gladen: adjusts classify-and-count proportion using binary TPR/FPR."""
    return (apparent_prevalence - fpr) / (tpr - fpr)


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


def compute_multicalibration_factors(data):
    calibration_adjustments = {}
    for x_value in [0, 1]:
        subset = data[data["X"] == x_value]
        observed_prevalence = subset["Y"].mean()
        estimated_prevalence = subset["p_estimated"].mean()
        calibration_adjustments[x_value] = observed_prevalence - estimated_prevalence
    return calibration_adjustments


def learn_cc_threshold(calib_data):
    true_prev = calib_data['Y'].mean()
    p_values = np.sort(calib_data['p_estimated'].unique())

    best_threshold = 0.5
    best_error = float('inf')

    for threshold in p_values:
        cc_prev = (calib_data['p_estimated'] >= threshold).mean()
        error = abs(cc_prev - true_prev)
        if error < best_error:
            best_error = error
            best_threshold = threshold

    return best_threshold


def analyze_shifted_distribution(
    shifted_p_x0, calibration_factor, multicalibration_adjustments,
    rg_tpr, rg_fpr, cc_threshold,
    pacc_pos_mean, pacc_neg_mean, source_prevalence,
    n_samples, p_y_given_x1, bias_b, bias_c,
):
    shifted_sample = generate_data(
        n_samples=n_samples,
        p_x0=shifted_p_x0,
        p_y_given_x1=p_y_given_x1,
        bias_b=bias_b,
        bias_c=bias_c,
    )

    uncalibrated_prevalence_shifted = shifted_sample["p_estimated"].mean()

    cc_prevalence_shifted = (shifted_sample["p_estimated"] >= cc_threshold).mean()

    # Binary Rogan-Gladen: adjust CC proportion with binary TPR/FPR
    rg_prevalence_shifted = apply_rogan_gladen(cc_prevalence_shifted, rg_tpr, rg_fpr)

    # PACC: adjust continuous score mean with E[h(X)|Y=1] and E[h(X)|Y=0]
    pacc_prevalence_shifted = pacc_estimate(
        shifted_sample["p_estimated"].values, pacc_pos_mean, pacc_neg_mean
    )

    sld_prevalence_shifted = sld_estimate(shifted_sample["p_estimated"].values, source_prevalence)

    shifted_sample["p_estimated_calibrated"] = (
        shifted_sample["p_estimated"] * calibration_factor
    )
    calibrated_prevalence_shifted = shifted_sample["p_estimated_calibrated"].mean()

    shifted_sample["p_estimated_multicalibrated"] = shifted_sample.apply(
        lambda row: row["p_estimated"] + multicalibration_adjustments[row["X"]], axis=1
    )
    multicalibrated_prevalence_shifted = shifted_sample[
        "p_estimated_multicalibrated"
    ].mean()

    true_prevalence_shifted = (shifted_p_x0 * (1 - p_y_given_x1)) + (
        (1 - shifted_p_x0) * p_y_given_x1
    )

    return (
        shifted_sample,
        true_prevalence_shifted,
        uncalibrated_prevalence_shifted,
        cc_prevalence_shifted,
        rg_prevalence_shifted,
        pacc_prevalence_shifted,
        sld_prevalence_shifted,
        calibrated_prevalence_shifted,
        multicalibrated_prevalence_shifted,
    )


def compute_bias_curve(
    n_samples,
    p_y_given_x1,
    bias_b,
    bias_c,
    calibration_factor,
    multicalibration_adjustments,
    rg_tpr,
    rg_fpr,
    cc_threshold,
    pacc_pos_mean,
    pacc_neg_mean,
    source_prevalence,
    original_p_x0=0.5,
    shift_range=(0.01, 0.99),
    n_points=20,
):
    distribution_shifts = np.linspace(shift_range[0], shift_range[1], n_points)
    deltas = distribution_shifts - original_p_x0

    bias_percentage_uncalibrated = []
    bias_percentage_cc = []
    bias_percentage_rg = []
    bias_percentage_pacc = []
    bias_percentage_sld = []
    bias_percentage_global = []
    bias_percentage_multi = []

    se_uncalibrated = []
    se_cc = []
    se_rg = []
    se_pacc = []
    se_sld = []
    se_global = []
    se_multi = []

    for p_x0_shifted in distribution_shifts:

        (
            shifted_sample,
            true_prevalence_shifted,
            uncalibrated_prevalence_shifted,
            cc_prevalence_shifted,
            rg_prevalence_shifted,
            pacc_prevalence_shifted,
            sld_prevalence_shifted,
            calibrated_prevalence_shifted,
            multicalibrated_prevalence_shifted
        ) = analyze_shifted_distribution(
            shifted_p_x0=p_x0_shifted,
            calibration_factor=calibration_factor,
            multicalibration_adjustments=multicalibration_adjustments,
            rg_tpr=rg_tpr,
            rg_fpr=rg_fpr,
            cc_threshold=cc_threshold,
            pacc_pos_mean=pacc_pos_mean,
            pacc_neg_mean=pacc_neg_mean,
            source_prevalence=source_prevalence,
            n_samples=n_samples,
            p_y_given_x1=p_y_given_x1,
            bias_b=bias_b,
            bias_c=bias_c,
        )

        true_mean_y = shifted_sample["Y"].mean()

        bias_pct_uncalibrated = 100 * (uncalibrated_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_cc = 100 * (cc_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_rg = 100 * (rg_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_pacc = 100 * (pacc_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_sld = 100 * (sld_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_global = 100 * (calibrated_prevalence_shifted - true_mean_y) / true_mean_y
        bias_pct_multi = 100 * (multicalibrated_prevalence_shifted - true_mean_y) / true_mean_y

        bias_percentage_uncalibrated.append(bias_pct_uncalibrated)
        bias_percentage_cc.append(bias_pct_cc)
        bias_percentage_rg.append(bias_pct_rg)
        bias_percentage_pacc.append(bias_pct_pacc)
        bias_percentage_sld.append(bias_pct_sld)
        bias_percentage_global.append(bias_pct_global)
        bias_percentage_multi.append(bias_pct_multi)

        # Squared error (in percentage points squared) for MSE
        se_uncalibrated.append((100 * (uncalibrated_prevalence_shifted - true_mean_y)) ** 2)
        se_cc.append((100 * (cc_prevalence_shifted - true_mean_y)) ** 2)
        se_rg.append((100 * (rg_prevalence_shifted - true_mean_y)) ** 2)
        se_pacc.append((100 * (pacc_prevalence_shifted - true_mean_y)) ** 2)
        se_sld.append((100 * (sld_prevalence_shifted - true_mean_y)) ** 2)
        se_global.append((100 * (calibrated_prevalence_shifted - true_mean_y)) ** 2)
        se_multi.append((100 * (multicalibrated_prevalence_shifted - true_mean_y)) ** 2)

    return {
        "deltas": np.array(deltas),
        "bias_uncalibrated": np.array(bias_percentage_uncalibrated),
        "bias_cc": np.array(bias_percentage_cc),
        "bias_rg": np.array(bias_percentage_rg),
        "bias_pacc": np.array(bias_percentage_pacc),
        "bias_sld": np.array(bias_percentage_sld),
        "bias_global": np.array(bias_percentage_global),
        "bias_multi": np.array(bias_percentage_multi),
        "se_uncalibrated": np.array(se_uncalibrated),
        "se_cc": np.array(se_cc),
        "se_rg": np.array(se_rg),
        "se_pacc": np.array(se_pacc),
        "se_sld": np.array(se_sld),
        "se_global": np.array(se_global),
        "se_multi": np.array(se_multi),
    }


def compute_bias_curves_bootstrap(
    B,
    n_samples,
    n_calibration,
    p_x0,
    p_y_given_x1,
    bias_b,
    bias_c,
    original_p_x0=0.5,
    shift_range=(0.01, 0.99),
    n_points=20,
):
    all_uncalibrated_curves = []
    all_cc_curves = []
    all_rg_curves = []
    all_pacc_curves = []
    all_sld_curves = []
    all_calibrated_curves = []
    all_multicalibrated_curves = []

    all_se_uncalibrated = []
    all_se_cc = []
    all_se_rg = []
    all_se_pacc = []
    all_se_sld = []
    all_se_calibrated = []
    all_se_multicalibrated = []

    deltas = None

    for b in range(B):
        calib_data = generate_data(n_calibration, p_x0, p_y_given_x1, bias_b, bias_c)

        iter_cc_threshold = learn_cc_threshold(calib_data)

        # Binary TPR/FPR for Rogan-Gladen
        iter_rg_tpr = (calib_data[calib_data['Y'] == 1]['p_estimated'] >= iter_cc_threshold).mean()
        iter_rg_fpr = (calib_data[calib_data['Y'] == 0]['p_estimated'] >= iter_cc_threshold).mean()

        # Continuous score means for PACC
        iter_pacc_pos_mean = calib_data[calib_data['Y'] == 1]['p_estimated'].mean()
        iter_pacc_neg_mean = calib_data[calib_data['Y'] == 0]['p_estimated'].mean()

        iter_calibration_factor = calib_data['Y'].mean() / calib_data['p_estimated'].mean()
        iter_multicalibration_adjustments = compute_multicalibration_factors(calib_data)
        iter_source_prevalence = calib_data['Y'].mean()

        result = compute_bias_curve(
            n_samples=n_samples,
            p_y_given_x1=p_y_given_x1,
            bias_b=bias_b,
            bias_c=bias_c,
            calibration_factor=iter_calibration_factor,
            multicalibration_adjustments=iter_multicalibration_adjustments,
            rg_tpr=iter_rg_tpr,
            rg_fpr=iter_rg_fpr,
            cc_threshold=iter_cc_threshold,
            pacc_pos_mean=iter_pacc_pos_mean,
            pacc_neg_mean=iter_pacc_neg_mean,
            source_prevalence=iter_source_prevalence,
            original_p_x0=original_p_x0,
            shift_range=shift_range,
            n_points=n_points,
        )

        if deltas is None:
            deltas = result["deltas"]

        all_uncalibrated_curves.append(result["bias_uncalibrated"])
        all_cc_curves.append(result["bias_cc"])
        all_rg_curves.append(result["bias_rg"])
        all_pacc_curves.append(result["bias_pacc"])
        all_sld_curves.append(result["bias_sld"])
        all_calibrated_curves.append(result["bias_global"])
        all_multicalibrated_curves.append(result["bias_multi"])

        all_se_uncalibrated.append(result["se_uncalibrated"])
        all_se_cc.append(result["se_cc"])
        all_se_rg.append(result["se_rg"])
        all_se_pacc.append(result["se_pacc"])
        all_se_sld.append(result["se_sld"])
        all_se_calibrated.append(result["se_global"])
        all_se_multicalibrated.append(result["se_multi"])

    all_uncalibrated_curves = np.array(all_uncalibrated_curves)
    all_cc_curves = np.array(all_cc_curves)
    all_rg_curves = np.array(all_rg_curves)
    all_pacc_curves = np.array(all_pacc_curves)
    all_sld_curves = np.array(all_sld_curves)
    all_calibrated_curves = np.array(all_calibrated_curves)
    all_multicalibrated_curves = np.array(all_multicalibrated_curves)

    all_se_uncalibrated = np.array(all_se_uncalibrated)
    all_se_cc = np.array(all_se_cc)
    all_se_rg = np.array(all_se_rg)
    all_se_pacc = np.array(all_se_pacc)
    all_se_sld = np.array(all_se_sld)
    all_se_calibrated = np.array(all_se_calibrated)
    all_se_multicalibrated = np.array(all_se_multicalibrated)

    return {
        "deltas": deltas,
        "all_uncalibrated": all_uncalibrated_curves,
        "all_cc": all_cc_curves,
        "all_rg": all_rg_curves,
        "all_pacc": all_pacc_curves,
        "all_sld": all_sld_curves,
        "all_calibrated": all_calibrated_curves,
        "all_multicalibrated": all_multicalibrated_curves,
        "avg_uncalibrated": np.mean(all_uncalibrated_curves, axis=0),
        "avg_cc": np.mean(all_cc_curves, axis=0),
        "avg_rg": np.mean(all_rg_curves, axis=0),
        "avg_pacc": np.mean(all_pacc_curves, axis=0),
        "avg_sld": np.mean(all_sld_curves, axis=0),
        "avg_calibrated": np.mean(all_calibrated_curves, axis=0),
        "avg_multicalibrated": np.mean(all_multicalibrated_curves, axis=0),
        # MSE = mean of squared errors across bootstrap iterations
        "mse_uncalibrated": np.mean(all_se_uncalibrated, axis=0),
        "mse_cc": np.mean(all_se_cc, axis=0),
        "mse_rg": np.mean(all_se_rg, axis=0),
        "mse_pacc": np.mean(all_se_pacc, axis=0),
        "mse_sld": np.mean(all_se_sld, axis=0),
        "mse_calibrated": np.mean(all_se_calibrated, axis=0),
        "mse_multicalibrated": np.mean(all_se_multicalibrated, axis=0),
    }
