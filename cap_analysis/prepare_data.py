"""
Download and prepare CAP datasets for LLM inference.

Downloads raw data from comparativeagendas.net, standardizes column names,
applies filtering, and creates inference samples.

Usage:
    pip install pandas tqdm requests
    python prepare_data.py [--output-dir data/] [--skip-download]

Output:
    data/*_final.csv           — 7 standardized sub-population files
    data/feasibility_sample_70b.csv — 7,000-doc sample (1K per sub-pop)
    data/full_sample.csv       — 105,000-doc sample (15K per sub-pop)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Download URLs (comparativeagendas.net, hosted on MinIO at UT Austin) ──

DOWNLOAD_URLS = {
    "denmark_questions": "https://minio.la.utexas.edu/compagendas/datasetfiles/questions_01092019_no_names.csv",
    "spain_questions": "https://minio.la.utexas.edu/compagendas/datasetfiles/Spain_OralQuestions19772019_19.1.csv",
    "spain_media_elpais": "https://minio.la.utexas.edu/compagendas/datasetfiles/Media_El_Pas_Web_CAP_csv.csv",
    "spain_media_elmundo": "https://minio.la.utexas.edu/compagendas/datasetfiles/Media_El_Mundo_Web_CAP_csv.csv",
    "us_bills": "https://minio.la.utexas.edu/compagendas/datasetfiles/US-Legislative-congressional_bills_19.3_3_3%20%281%29.csv",
    "belgium_tv": "https://minio.la.utexas.edu/compagendas/datasetfiles/belgium_television_new.csv",
    "belgium_newspaper": "https://minio.la.utexas.edu/compagendas/datasetfiles/belgium_newspaper_new.csv",
}

# ── Danish party code mapping (var4 -> party name, party family) ──

DENMARK_PARTIES = {
    1: ("Social Democratic Party", "Social Democracy"),
    2: ("Social Liberal Party", "Liberal"),
    3: ("Conservative Party", "Conservative"),
    4: ("Centre Democrats", "Liberal"),
    5: ("Justice Party", "Other"),
    6: ("Socialist People's Party", "Left"),
    7: ("Communist Party", "Left"),
    8: ("Danish People's Party", "Right-wing populist"),
    9: ("Common Course", "Other"),
    10: ("Christian Democrats", "Christian Democracy"),
    11: ("Liberal Party", "Liberal"),
    12: ("Red-Green Alliance", "Left"),
    13: ("Left Socialists", "Left"),
    14: ("Progress Party", "Right-wing populist"),
    15: ("Greenland/Faroe Islands", "Other"),
    16: ("No party/small parties", "Other"),
    17: ("Liberal Alliance", "Liberal"),
    21: ("The Alternative", "Green/Left"),
}


def download_file(url, output_path, proxies=None):
    """Download a file with progress bar."""
    if os.path.exists(output_path):
        print(f"  Already exists: {output_path}")
        return
    print(f"  Downloading {os.path.basename(output_path)}...")
    response = requests.get(url, stream=True, proxies=proxies)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="    "
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_all(raw_dir, proxies=None):
    """Download all raw CAP datasets."""
    os.makedirs(raw_dir, exist_ok=True)
    for name, url in DOWNLOAD_URLS.items():
        output_path = os.path.join(raw_dir, f"{name}_raw.csv")
        download_file(url, output_path, proxies=proxies)


def make_standard_df(df, text_col, country, language, doc_type,
                     party_col=None, party_family_col=None, party_code_col=None):
    """Create standardized DataFrame with consistent columns."""
    out = pd.DataFrame()
    out["id_original"] = df["id"].astype(str)
    out["country"] = country
    out["language"] = language
    out["doc_type"] = doc_type
    out["year"] = df["year"].astype(int)
    out["decade"] = (out["year"] // 10) * 10
    out["text"] = df[text_col].astype(str)
    out["majortopic"] = df["majortopic"]
    out["subtopic"] = df.get("subtopic", pd.NA)
    out["party"] = df[party_col] if party_col and party_col in df.columns else pd.NA
    out["party_family"] = df[party_family_col] if party_family_col and party_family_col in df.columns else pd.NA
    out["party_code"] = df[party_code_col] if party_code_col and party_code_col in df.columns else pd.NA
    out["law_crime"] = (df["majortopic"] == 12).astype(int)
    out["immigration"] = (df["majortopic"] == 9).astype(int)
    return out


def prepare_denmark(raw_dir):
    """Prepare Denmark parliamentary questions."""
    df = pd.read_csv(os.path.join(raw_dir, "denmark_questions_raw.csv"))
    n_raw = len(df)

    # Filter: drop missing descriptions and corrupted rows
    df = df.dropna(subset=["description"])
    df = df.dropna(subset=["id"])

    # Map party codes
    df["party_name"] = df["var4"].map(lambda x: DENMARK_PARTIES.get(int(x), (None, None))[0] if pd.notna(x) else None)
    df["party_fam"] = df["var4"].map(lambda x: DENMARK_PARTIES.get(int(x), (None, None))[1] if pd.notna(x) else None)

    out = make_standard_df(
        df, text_col="description",
        country="Denmark", language="Danish", doc_type="parliamentary_question",
        party_col="party_name", party_family_col="party_fam", party_code_col="var4",
    )
    print(f"  Denmark questions: {n_raw} -> {len(out)} rows")
    return out


def prepare_spain_questions(raw_dir):
    """Prepare Spain oral questions."""
    df = pd.read_csv(os.path.join(raw_dir, "spain_questions_raw.csv"))
    n_raw = len(df)

    # Filter: drop uncoded (majortopic == 0)
    df = df[df["majortopic"] != 0]

    # Party family mapping from abbreviations
    party_family_map = {
        "GS": "Social Democracy", "GUCD": "Conservative", "GP": "Conservative",
        "GIU-ICV": "Left", "GCDS": "Liberal", "GCP": "Left",
        "GMx": "Regionalist", "GER-IU": "Left", "GV": "Regionalist",
    }
    df["party_fam"] = df["parliamentary_group_asking_the_question"].map(
        lambda x: party_family_map.get(x, "Other") if pd.notna(x) else None
    )

    out = make_standard_df(
        df, text_col="description",
        country="Spain", language="Spanish", doc_type="parliamentary_question",
        party_col="parliamentary_group_asking_the_question",
        party_family_col="party_fam",
        party_code_col="parliamentary_group_code",
    )
    print(f"  Spain questions: {n_raw} -> {len(out)} rows")
    return out


def prepare_spain_media(raw_dir, name, filename):
    """Prepare Spain media (El Pais or El Mundo)."""
    df = pd.read_csv(os.path.join(raw_dir, filename))
    n_raw = len(df)

    # Filter: drop short titles (< 20 chars, section headers)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.len() >= 20]

    out = make_standard_df(
        df, text_col="title",
        country="Spain", language="Spanish", doc_type="media",
    )
    print(f"  Spain {name}: {n_raw} -> {len(out)} rows")
    return out


def prepare_us_bills(raw_dir):
    """Prepare US Congressional Bills."""
    df = pd.read_csv(os.path.join(raw_dir, "us_bills_raw.csv"), low_memory=False)
    n_raw = len(df)

    # Use bill_id as the document ID (more informative than numeric id)
    df["id"] = df["bill_id"]

    # Filter: drop uncoded (NaN majortopic, all from congress 114)
    df = df.dropna(subset=["majortopic"])
    df["majortopic"] = df["majortopic"].astype(int)

    # Map party codes: 100=Republican, 200=Democrat
    party_map = {100: "Republican", 200: "Democrat"}
    df["party_name"] = df["party"].map(party_map)
    df["party_fam"] = df["party_name"]  # Same for US two-party system

    out = make_standard_df(
        df, text_col="description",
        country="United States", language="English", doc_type="bill",
        party_col="party_name", party_family_col="party_fam", party_code_col="party",
    )
    print(f"  US bills: {n_raw} -> {len(out)} rows")
    return out


def prepare_belgium_tv(raw_dir):
    """Prepare Belgium TV news."""
    df = pd.read_csv(os.path.join(raw_dir, "belgium_tv_raw.csv"))
    n_raw = len(df)

    # Filter: drop missing descriptions
    df = df.dropna(subset=["description"])

    out = make_standard_df(
        df, text_col="description",
        country="Belgium", language="Dutch", doc_type="tv_news",
    )
    print(f"  Belgium TV: {n_raw} -> {len(out)} rows")
    return out


def prepare_belgium_newspaper(raw_dir):
    """Prepare Belgium newspaper (De Standaard)."""
    df = pd.read_csv(os.path.join(raw_dir, "belgium_newspaper_raw.csv"), low_memory=False)
    n_raw = len(df)

    # Filter: drop missing descriptions
    df = df.dropna(subset=["description"])

    # Normalize whitespace (articles contain embedded newlines)
    df["description"] = df["description"].apply(
        lambda x: re.sub(r"\s+", " ", str(x).strip()) if pd.notna(x) else x
    )

    out = make_standard_df(
        df, text_col="description",
        country="Belgium", language="Dutch", doc_type="newspaper",
    )
    print(f"  Belgium newspaper: {n_raw} -> {len(out)} rows")
    return out


def _assign_unique_ids(df):
    """Assign globally unique IDs based on row index."""
    df = df.copy()
    df["id"] = range(len(df))
    return df


def create_samples(output_dir, datasets):
    """Create feasibility and full inference samples."""
    import numpy as np

    # Feasibility sample: 1K per sub-population (7K total)
    print("\nCreating feasibility sample (1K per sub-pop)...")
    feasibility_frames = []
    for name, df in datasets.items():
        sampled = df.sample(n=min(1000, len(df)), random_state=42)
        feasibility_frames.append(sampled)
        print(f"  {name}: {len(sampled)} sampled, {int(sampled.law_crime.sum())} positives")
    feasibility = pd.concat(feasibility_frames, ignore_index=True)
    feasibility = _assign_unique_ids(feasibility)
    feas_path = os.path.join(output_dir, "feasibility_sample_70b.csv")
    feasibility.to_csv(feas_path, index=False)
    print(f"  Total: {len(feasibility)} docs, {int(feasibility.law_crime.sum())} positives")
    print(f"  Saved to {feas_path}")

    # Full sample: 15K per sub-population (105K total)
    print("\nCreating full sample (15K per sub-pop)...")
    full_frames = []
    for name, df in datasets.items():
        sampled = df.sample(n=min(15000, len(df)), random_state=42)
        full_frames.append(sampled)
        print(f"  {name}: {len(sampled)} sampled, {int(sampled.law_crime.sum())} positives")
    full = pd.concat(full_frames, ignore_index=True)
    full = _assign_unique_ids(full)
    full_path = os.path.join(output_dir, "full_sample.csv")
    full.to_csv(full_path, index=False)
    print(f"  Total: {len(full)} docs, {int(full.law_crime.sum())} positives")
    print(f"  Saved to {full_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare CAP datasets for LLM inference."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cap_analysis/data",
        help="Output directory for prepared data files (default: cap_analysis/data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw files (use existing files in output-dir/raw/)",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g., http://fwdproxy:8080 for Meta devservers)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(output_dir, exist_ok=True)

    proxies = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}

    # Step 1: Download raw data
    if not args.skip_download:
        print("=== Downloading raw datasets from comparativeagendas.net ===")
        download_all(raw_dir, proxies=proxies)
    else:
        print("=== Skipping download (using existing raw files) ===")

    # Step 2: Prepare each dataset
    print("\n=== Preparing standardized datasets ===")
    datasets = {}

    datasets["denmark_questions"] = prepare_denmark(raw_dir)
    datasets["spain_questions"] = prepare_spain_questions(raw_dir)
    datasets["spain_media_elpais"] = prepare_spain_media(
        raw_dir, "El Pais", "spain_media_elpais_raw.csv"
    )
    datasets["spain_media_elmundo"] = prepare_spain_media(
        raw_dir, "El Mundo", "spain_media_elmundo_raw.csv"
    )
    datasets["us_bills"] = prepare_us_bills(raw_dir)
    datasets["belgium_tv"] = prepare_belgium_tv(raw_dir)
    datasets["belgium_newspaper"] = prepare_belgium_newspaper(raw_dir)

    # Save final datasets
    print("\n=== Saving standardized datasets ===")
    for name, df in datasets.items():
        path = os.path.join(output_dir, f"{name}_final.csv")
        df.to_csv(path, index=False)
        prev = df.law_crime.mean()
        print(f"  {path}: {len(df)} rows, {prev:.1%} law/crime")

    # Step 3: Create samples
    create_samples(output_dir, datasets)

    # Summary
    total = sum(len(df) for df in datasets.values())
    print(f"\n=== Done ===")
    print(f"Total: {total:,} documents across {len(datasets)} sub-populations")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
