# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare OASST1 data for the LLM-as-judge prevalence experiment.

Estimand: fraction of assistant responses that are "helpful" (human gold),
measured by an LLM judge, under cross-LANGUAGE covariate shift.

The human `helpfulness` label (0-1, averaged over >=3 volunteer reviewers) is
the gold standard. We binarize at >=0.6 -> helpful (label=1). The base rate
varies strongly by language (~0.32 zh .. ~0.71 de), so reweighting the language
mix genuinely shifts the target prevalence -- the covariate-shift regime.

Six languages, pilot size 1,000 docs each:
  calibration: en, es, ru, de   (4 langs, mirrors CAP's 4 calibration subpops)
  OOD:         fr, zh           (held-out languages = novel feature values)

Outputs (under judge_analysis/data/):
  oasst_pilot.csv            master table: id, lang, gold label, response len, split
  shards/shard_<i>.json      100 docs each, for the sub-agent judge campaigns

Usage:
    python judge_analysis/prepare_data.py
"""

import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HELPFUL_THRESHOLD = 0.6          # helpfulness >= this -> label 1 (helpful)
PER_LANG_N = 1000                # pilot size per language
SHARD_SIZE = 100
SEED = 42

CAL_LANGS = ["en", "es", "ru", "de"]
OOD_LANGS = ["fr", "zh"]
ALL_LANGS = CAL_LANGS + OOD_LANGS

# For the judge prompt (so it knows the language of the text it rates).
LANG_NAMES = {
    "en": "English", "es": "Spanish", "ru": "Russian",
    "de": "German", "fr": "French", "zh": "Chinese",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SHARD_DIR = os.path.join(DATA_DIR, "shards")
MASTER_CSV = os.path.join(DATA_DIR, "oasst_pilot.csv")

MAX_PROMPT_CHARS = 3000          # truncate long context / responses for the judge
MAX_RESP_CHARS = 4000


def get_label(labels, name):
    """Pull one attribute value out of OASST's {'name':[...], 'value':[...]}."""
    if labels is None:
        return None
    try:
        names = list(labels["name"])
        vals = list(labels["value"])
        return vals[names.index(name)] if name in names else None
    except Exception:
        return None


def main():
    os.makedirs(SHARD_DIR, exist_ok=True)

    print("Downloading OASST1 train split...")
    path = hf_hub_download(
        "OpenAssistant/oasst1",
        "data/train-00000-of-00001-b42a775f407cee45.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    print(f"  total messages: {len(df):,}")

    # Index every message by id so we can walk the tree upward for context.
    by_id = {r.message_id: r for r in df.itertuples(index=False)}

    def context_to_root(msg_row):
        """Reconstruct the dialogue from root down to (but excluding) this msg."""
        chain = []
        parent_id = msg_row.parent_id
        while parent_id is not None and parent_id in by_id:
            p = by_id[parent_id]
            role = "User" if p.role == "prompter" else "Assistant"
            chain.append(f"{role}: {p.text}")
            parent_id = p.parent_id
        return "\n\n".join(reversed(chain))

    # Assistant messages with a human helpfulness label.
    asst = df[(df.role == "assistant") & (~df.deleted)].copy()
    asst["helpfulness"] = asst["labels"].apply(lambda l: get_label(l, "helpfulness"))
    asst = asst[asst["helpfulness"].notna() & asst["lang"].isin(ALL_LANGS)].copy()
    print(f"  assistant msgs w/ helpfulness in target langs: {len(asst):,}")

    # Binarize gold, compute response length, reconstruct context.
    asst["label"] = (asst["helpfulness"] >= HELPFUL_THRESHOLD).astype(int)
    asst["resp_len"] = asst["text"].str.len()
    asst["context"] = [context_to_root(r) for r in asst.itertuples(index=False)]
    # Keep only rows that actually have a preceding user turn.
    asst = asst[asst["context"].str.len() > 0].copy()

    # Sample per language.
    parts = []
    for lg in ALL_LANGS:
        sub = asst[asst.lang == lg]
        n = min(PER_LANG_N, len(sub))
        parts.append(sub.sample(n=n, random_state=SEED))
        print(f"  {lg}: available={len(sub):5d}  sampled={n:5d}  "
              f"base_rate={sub.sample(n=n, random_state=SEED)['label'].mean():.3f}")
    sample = pd.concat(parts, ignore_index=True)
    sample["doc_id"] = [f"{r.lang}_{i}" for i, r in enumerate(sample.itertuples(index=False))]
    sample["split"] = sample["lang"].apply(
        lambda l: "cal" if l in CAL_LANGS else "ood")

    # ---- Master CSV (gold + metadata; no judge outputs yet) ----
    master = sample[[
        "doc_id", "lang", "label", "helpfulness", "resp_len", "split", "message_id",
    ]].copy()
    master.to_csv(MASTER_CSV, index=False)
    print(f"\nWrote master table: {MASTER_CSV}  ({len(master):,} rows)")

    # ---- Shards for the judge campaigns ----
    docs = []
    for r in sample.itertuples(index=False):
        prompt = r.context[-MAX_PROMPT_CHARS:]          # keep the most recent context
        response = r.text[:MAX_RESP_CHARS]
        docs.append({
            "id": r.doc_id,
            "language": LANG_NAMES[r.lang],
            "prompt": prompt,
            "response": response,
            # NOTE: gold label deliberately NOT included -- the judge must never
            # see it. Gold lives in oasst_pilot.csv, joined by id at analysis.
        })

    n_shards = 0
    for i in range(0, len(docs), SHARD_SIZE):
        shard = docs[i:i + SHARD_SIZE]
        with open(os.path.join(SHARD_DIR, f"shard_{n_shards}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False)
        n_shards += 1
    print(f"Wrote {n_shards} shards of up to {SHARD_SIZE} docs to {SHARD_DIR}")

    print("\nSummary:")
    print(f"  docs: {len(docs):,}  |  shards: {n_shards}  |  "
          f"threshold: helpfulness>={HELPFUL_THRESHOLD}")
    print(f"  calibration langs: {CAL_LANGS}  |  OOD langs: {OOD_LANGS}")
    print("  overall gold base rate: %.3f" % master["label"].mean())


if __name__ == "__main__":
    main()
