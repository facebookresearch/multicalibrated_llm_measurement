# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Full-scale data prep: extend the pilot to a balanced cap.

Caps per language: en/es -> 4000, ru/de/fr/zh -> all available. Reuses the
pilot's already-judged docs (preserving their doc_ids and existing judge
outputs) and samples only the ADDITIONAL docs, sharded separately (ids 100+,
250 docs/shard) so we judge only the new material.

Outputs:
  data/oasst_full.csv               master for the full run (pilot + new)
  data/shards/shard_<>=100>.json    NEW docs only, 250/shard, for both campaigns

Usage:
    python judge_analysis/prepare_data_full.py
"""

import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

CAPS = {"en": 4000, "es": 4000, "ru": 10**9, "de": 10**9, "fr": 10**9, "zh": 10**9}
CAL_LANGS = ["en", "es", "ru", "de"]
OOD_LANGS = ["fr", "zh"]
ALL_LANGS = CAL_LANGS + OOD_LANGS
HELPFUL_THRESHOLD = 0.6
NEW_SHARD_SIZE = 100  # match pilot exactly -> uniform judge instrument, genuine judging
NEW_SHARD_START = 100
SEED = 42
MAX_PROMPT_CHARS, MAX_RESP_CHARS = 3000, 4000
LANG_NAMES = {"en": "English", "es": "Spanish", "ru": "Russian",
              "de": "German", "fr": "French", "zh": "Chinese"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SHARD_DIR = os.path.join(DATA_DIR, "shards")
PILOT_CSV = os.path.join(DATA_DIR, "oasst_pilot.csv")
FULL_CSV = os.path.join(DATA_DIR, "oasst_full.csv")


def get_label(labels, name):
    if labels is None:
        return None
    try:
        n = list(labels["name"]); v = list(labels["value"])
        return v[n.index(name)] if name in n else None
    except Exception:
        return None


def main():
    path = hf_hub_download(
        "OpenAssistant/oasst1",
        "data/train-00000-of-00001-b42a775f407cee45.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    by_id = {r.message_id: r for r in df.itertuples(index=False)}

    def context_to_root(row):
        chain = []; pid = row.parent_id
        while pid is not None and pid in by_id:
            p = by_id[pid]
            chain.append(f"{'User' if p.role == 'prompter' else 'Assistant'}: {p.text}")
            pid = p.parent_id
        return "\n\n".join(reversed(chain))

    a = df[(df.role == "assistant") & (~df.deleted)].copy()
    a["helpfulness"] = a["labels"].apply(lambda l: get_label(l, "helpfulness"))
    a = a[a["helpfulness"].notna() & a["lang"].isin(ALL_LANGS)].copy()
    a["resp_len"] = a["text"].str.len()
    a["context"] = [context_to_root(r) for r in a.itertuples(index=False)]
    a = a[a["context"].str.len() > 0].copy()

    pilot = pd.read_csv(PILOT_CSV)
    pilot_mids = set(pilot["message_id"])
    print(f"Pilot docs: {len(pilot)} ({len(pilot_mids)} unique message_ids)")

    new_rows = []
    for lg in ALL_LANGS:
        pool = a[a.lang == lg]
        cap = min(CAPS[lg], len(pool))
        pilot_n = pilot[pilot.lang == lg].shape[0]
        n_new = max(0, cap - pilot_n)
        remaining = pool[~pool.message_id.isin(pilot_mids)]
        take = remaining.sample(n=min(n_new, len(remaining)), random_state=SEED)
        for i, r in enumerate(take.itertuples(index=False)):
            new_rows.append({
                "doc_id": f"{lg}_n{i}", "lang": lg,
                "label": int(r.helpfulness >= HELPFUL_THRESHOLD),
                "helpfulness": r.helpfulness, "resp_len": r.resp_len,
                "split": "cal" if lg in CAL_LANGS else "ood",
                "message_id": r.message_id,
                "context": r.context, "text": r.text,
            })
        print(f"  {lg}: cap={cap:5d}  pilot={pilot_n:4d}  new={len(take):4d}")

    new_df = pd.DataFrame(new_rows)

    # --- master_full = pilot rows + new rows (drop helper text cols from master) ---
    keep = ["doc_id", "lang", "label", "helpfulness", "resp_len", "split", "message_id"]
    full = pd.concat([pilot[keep], new_df[keep]], ignore_index=True)
    full.to_csv(FULL_CSV, index=False)
    print(f"\nWrote {FULL_CSV}: {len(full)} docs "
          f"({len(pilot)} pilot + {len(new_df)} new)")
    print("Full base rate: %.3f" % full.label.mean())

    # --- shards for NEW docs only ---
    docs = [{
        "id": r.doc_id, "language": LANG_NAMES[r.lang],
        "prompt": r.context[-MAX_PROMPT_CHARS:], "response": r.text[:MAX_RESP_CHARS],
    } for r in new_df.itertuples(index=False)]

    sid = NEW_SHARD_START
    for i in range(0, len(docs), NEW_SHARD_SIZE):
        with open(os.path.join(SHARD_DIR, f"shard_{sid}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(docs[i:i + NEW_SHARD_SIZE], f, ensure_ascii=False)
        sid += 1
    n_new_shards = sid - NEW_SHARD_START
    print(f"Wrote {n_new_shards} new shards ({NEW_SHARD_START}..{sid-1}), "
          f"{NEW_SHARD_SIZE} docs each, for {len(docs)} new docs")
    print(f"Judge these per campaign: shards {NEW_SHARD_START}..{sid-1}")


if __name__ == "__main__":
    main()
