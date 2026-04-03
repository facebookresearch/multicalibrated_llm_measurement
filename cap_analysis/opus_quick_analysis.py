"""Quick analysis of Opus 4.6 inference results — matching feasibility study metrics."""

import csv
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score

BINARY_DIR = "cap_analysis/data/inference_output/claude-opus-30k-binary"
PYN_DIR = "cap_analysis/data/inference_output/claude-opus-30k-pyn"


def load_all_shards(directory, has_scores=False):
    rows = []
    for path in sorted(glob.glob(f"{directory}/shard_*.csv")):
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def analyze_binary(rows):
    """Analyze binary Yes/No campaign."""
    answers = [1 if r["answer"].strip() == "Yes" else 0 for r in rows]
    labels = [int(r["label"]) for r in rows]

    answers = np.array(answers)
    labels = np.array(labels)

    prevalence = labels.mean()
    pred_prev = answers.mean()
    acc = accuracy_score(labels, answers)
    prec = precision_score(labels, answers, zero_division=0)
    rec = recall_score(labels, answers, zero_division=0)
    auc = roc_auc_score(labels, answers)

    print("=" * 60)
    print("CAMPAIGN 1: Binary Yes/No (Opus 4.6)")
    print("=" * 60)
    print(f"  N:            {len(labels):,}")
    print(f"  Prevalence:   {prevalence:.1%}")
    print(f"  Pred rate:    {pred_prev:.1%}")
    print(f"  AUC:          {auc:.3f}")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  Precision:    {prec:.3f}")
    print(f"  Recall:       {rec:.3f}")


def analyze_pyn(rows):
    """Analyze P(Y/N) probability campaign — full feasibility metrics."""
    scores = np.array([float(r["score"]) for r in rows])
    labels = np.array([int(r["label"]) for r in rows])

    prevalence = labels.mean()
    preds = (scores >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores)

    eps = 1e-7
    clipped = np.clip(scores, eps, 1 - eps)
    ll = log_loss(labels, clipped)

    # ECE (10 equal-width bins)
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = scores[mask].mean()
        ece += mask.sum() / len(scores) * abs(bin_acc - bin_conf)

    # Distribution metrics
    exact0 = (scores < 0.005).mean()
    exact1 = (scores > 0.995).mean()
    mid = ((scores >= 0.1) & (scores <= 0.9)).mean()
    unique = len(set(f"{s:.4f}" for s in scores))
    pos_mean = scores[labels == 1].mean()
    neg_mean = scores[labels == 0].mean()

    print()
    print("=" * 60)
    print("CAMPAIGN 2: P(Y/N) Probabilities (Opus 4.6)")
    print("=" * 60)
    print(f"  N:            {len(labels):,}")
    print(f"  Prevalence:   {prevalence:.1%}")
    print()
    print("  --- Discrimination ---")
    print(f"  AUC:          {auc:.3f}")
    print(f"  Acc (t=.5):   {acc:.3f}")
    print(f"  Precision:    {prec:.3f}")
    print(f"  Recall:       {rec:.3f}")
    print()
    print("  --- Calibration ---")
    print(f"  Log loss:     {ll:.3f}")
    print(f"  ECE:          {ece:.3f}")
    print()
    print("  --- Distribution ---")
    print(f"  Pos mean:     {pos_mean:.3f}")
    print(f"  Neg mean:     {neg_mean:.3f}")
    print(f"  Exact 0.0%:   {exact0:.1%}")
    print(f"  Exact 1.0%:   {exact1:.1%}")
    print(f"  In [.1,.9]:   {mid:.1%}")
    print(f"  Unique scores:{unique}")

    # Per-language breakdown
    languages = sorted(set(r["language"] for r in rows))
    print()
    print("  --- Per-language AUC ---")
    for lang in languages:
        lang_scores = np.array([float(r["score"]) for r in rows if r["language"] == lang])
        lang_labels = np.array([int(r["label"]) for r in rows if r["language"] == lang])
        if len(set(lang_labels)) < 2:
            print(f"  {lang:12s}: N={len(lang_labels):,}  (single class)")
            continue
        lang_auc = roc_auc_score(lang_labels, lang_scores)
        lang_prev = lang_labels.mean()
        print(f"  {lang:12s}: AUC={lang_auc:.3f}  N={len(lang_labels):,}  prev={lang_prev:.1%}")


def main():
    print("Loading binary shards...")
    binary_rows = load_all_shards(BINARY_DIR)
    print(f"  Loaded {len(binary_rows):,} rows")

    print("Loading P(Y/N) shards...")
    pyn_rows = load_all_shards(PYN_DIR)
    print(f"  Loaded {len(pyn_rows):,} rows")

    analyze_binary(binary_rows)
    analyze_pyn(pyn_rows)


if __name__ == "__main__":
    main()
