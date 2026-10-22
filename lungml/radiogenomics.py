"""Prepare richer, independently sourced radiomics with a locked patient split."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_COMMIT = "272c1de21a1e8831e3505733b9d673cfe1e14d29"
FEATURE_URL = f"https://raw.githubusercontent.com/lawrence-chillrud/ece475/{SOURCE_COMMIT}/data/generated/NSCLC_features.csv"
CLINICAL_URL = "https://wiki.cancerimagingarchive.net/download/attachments/28672347/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv?api=v2&modificationDate=1531967714295&version=1"


def download_radiogenomics(raw="data/raw/nsclc_radiogenomics", output="data/radiogenomics_subtype.csv", prepare_only=False):
    from .datasets import fetch_file
    raw = Path(raw)
    sources = {"features.csv": FEATURE_URL, "clinical.csv": CLINICAL_URL,
               "extraction_log.txt": f"https://raw.githubusercontent.com/lawrence-chillrud/ece475/{SOURCE_COMMIT}/data/generated/feature_logs/NSCLC_features_README.txt"}
    for filename, url in sources.items():
        if prepare_only and not (raw / filename).exists():
            raise FileNotFoundError(raw / filename)
        if not prepare_only:
            fetch_file(url, raw / filename)
    return prepare_radiogenomics(raw, output)


def prepare_radiogenomics(raw="data/raw/nsclc_radiogenomics", output="data/radiogenomics_subtype.csv"):
    raw, output = Path(raw), Path(output)
    features = pd.read_csv(raw / "features.csv")
    clinical = pd.read_csv(raw / "clinical.csv")
    clinical.columns = clinical.columns.str.strip()
    labels = clinical[["Case ID", "Histology"]]
    merged = features.merge(labels, on="Case ID", validate="one_to_one", how="left")
    selected = merged.Histology.isin(["Adenocarcinoma", "Squamous cell carcinoma"])
    cohort = merged.loc[selected].drop(columns="Histology").rename(columns={"Case ID": "patient_id"})
    cohort.insert(1, "sample_id", cohort.patient_id + "-tumour")
    cohort.insert(2, "label", merged.loc[selected, "Histology"].map({
        "Adenocarcinoma": "adenocarcinoma", "Squamous cell carcinoma": "squamous_cell_carcinoma"}).to_numpy())
    cohort = cohort.sort_values("patient_id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    provenance = {
        "collection": "TCIA NSCLC Radiogenomics",
        "collection_doi": "https://doi.org/10.7937/K9/TCIA.2017.7hs46erv",
        "source": "lawrence-chillrud/ece475: third-party PyRadiomics feature extraction",
        "source_commit": SOURCE_COMMIT,
        "features_url": FEATURE_URL, "clinical_url": CLINICAL_URL,
        "features_sha256": hashlib.sha256((raw / "features.csv").read_bytes()).hexdigest(),
        "clinical_sha256": hashlib.sha256((raw / "clinical.csv").read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "patients": len(cohort), "class_counts": cohort.label.value_counts().to_dict(),
        "original_features": len([c for c in cohort if c.startswith("original_")]),
        "total_features": len(cohort.columns) - 3,
        "extraction_log": (raw / "extraction_log.txt").read_text(),
        "limitations": [
            "Different cohort from the supplied paper.",
            "Third-party extraction; raw CT voxels and converted masks were not independently checked.",
            "Native spacing; no resampling; binWidth uses PyRadiomics defaults unless source log overrides it.",
            "Source conversion used November 2022 data and correctMask=True.",
            "139 usable AC/SCC patients, with only 28 SCC patients.",
            "No benign or normal controls.",
        ],
        "attribution": "Bakr et al. (2018), TCIA; Lawrence Chillrud, Christian Duffee and Charilaos Apostolidis",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    # This manifest is created once, before any modelling, and must not change.
    split_path = output.with_suffix(".splits.csv")
    if not split_path.exists():
        development, test = train_test_split(cohort.patient_id, test_size=0.25,
                                             stratify=cohort.label, random_state=42)
        splits = cohort[["patient_id", "sample_id", "label"]].copy()
        splits["split"] = "development"
        splits.loc[splits.patient_id.isin(test), "split"] = "test"
        splits.to_csv(split_path, index=False)
    print(f"Prepared {len(cohort)} patients with {len(cohort.columns)-3} features; locked 75/25 patient split.", flush=True)
    return provenance


if __name__ == "__main__":
    prepare_radiogenomics()
