"""Download and audit the public Lung1 radiomics feature cohort."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import pandas as pd

COMMIT = "841592dc855172061d42b4dfd47f831fee12d44d"
BASE = f"https://raw.githubusercontent.com/hhfchoi/Radiomics/{COMMIT}"
SOURCES = {
    "clinical.csv": "https://wiki.cancerimagingarchive.net/download/attachments/16056854/NSCLC%20Radiomics%20Lung1.clinical-version3-Oct%202019.csv?api=v2&modificationDate=1572013183040&version=1",
    "features_unpadded.csv": f"{BASE}/nsclc-radiomics.csv",
    "features.csv": f"{BASE}/NSCLC-Radiomics-agg-padded.csv",
    "feature_source_readme.md": f"{BASE}/README.md",
}


def fetch_file(url, target, md5=None, sha256=None):
    """Stage a download, verify it, then atomically replace the target."""
    target = Path(target)
    def valid(path):
        if not path.exists() or path.stat().st_size == 0:
            return False
        contents = path.read_bytes()
        return ((md5 is None or hashlib.md5(contents).hexdigest() == md5) and
                (sha256 is None or hashlib.sha256(contents).hexdigest() == sha256))
    if valid(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "lung-radiomics-research/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, staged.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if not valid(staged):
        raise ValueError(f"Downloaded file failed integrity verification: {target.name}")
    staged.replace(target)
    return target


def download_lung1(raw="data/raw/nsclc_radiomics", output="data/nsclc_subtype.csv", download=True):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = raw / name
        if not target.exists():
            if not download:
                raise FileNotFoundError(target)
            print(f"Downloading {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "lung-radiomics-research/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                target.write_bytes(response.read())
    clinical = pd.read_csv(raw / "clinical.csv")
    features = pd.read_csv(raw / "features_unpadded.csv")
    schema = pd.read_csv(raw / "features.csv", nrows=0)
    families = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
    named = [c for c in schema if c.startswith(tuple(f + "_" for f in families))]
    raw_names = features.columns[2:2 + len(named)].tolist()
    mapping = {}
    # Source aggregate table supplies family-qualified headers only. Its values
    # are never used: features come strictly from the primary GTV-1 mask.
    for old, qualified in zip(raw_names, named):
        family, feature, suffix = qualified.split("_")
        if suffix != "0" or re.sub(r"\.\d+$", "", old) != feature:
            raise ValueError(f"Feature header ordering mismatch: {old}, {qualified}")
        mapping[old] = f"original_{family}_{feature}"
    if len(mapping) != 107:
        raise ValueError("Unexpected source feature schema.")
    primary = features.loc[features.MaskID.str.contains(r"(?i)_segment_gtv-1$", na=False)].copy()
    if primary.PatientID.duplicated().any() or clinical.PatientID.duplicated().any():
        raise ValueError("Duplicate primary tumour or clinical patient IDs.")
    joined = primary[["PatientID", "MaskID", *raw_names]].merge(
        clinical[["PatientID", "Histology"]], on="PatientID", validate="one_to_one", how="left")
    selected = joined.Histology.isin(["adenocarcinoma", "squamous cell carcinoma"])
    cohort = joined.loc[selected].rename(columns={
        "PatientID": "patient_id", "MaskID": "source_mask", "Histology": "label", **mapping})
    cohort["label"] = cohort.label.replace({"squamous cell carcinoma": "squamous_cell_carcinoma"})
    cohort.insert(1, "sample_id", cohort.patient_id + "-GTV1")
    cohort = cohort[["patient_id", "sample_id", "label", "source_mask", *mapping.values()]]
    cohort = cohort.sort_values("patient_id").reset_index(drop=True)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(target, index=False)
    metadata = {
        "collection": "TCIA NSCLC-Radiomics (Lung1)",
        "collection_doi": "https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI",
        "clinical_source": "TCIA official clinical release, October 2019",
        "feature_source": "hhfchoi/Radiomics: third-party Slicer-derived tumour features",
        "feature_source_commit": COMMIT,
        "download_audit_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {name: {"url": url, "sha256": hashlib.sha256((raw / name).read_bytes()).hexdigest(),
                           "bytes": (raw / name).stat().st_size} for name, url in SOURCES.items()},
        "primary_tumour_patients": len(primary), "clinical_patients": len(clinical),
        "included_patients": len(cohort), "class_counts": cohort.label.value_counts().to_dict(),
        "features": len(mapping),
        "excluded_primary_patients": int((~selected).sum()),
        "clinical_histology_counts": clinical.Histology.fillna("missing").value_counts().to_dict(),
        "missing_primary_ids": sorted(set(clinical.PatientID) - set(primary.PatientID)),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "limitations": [
            "This is a different cohort from the supplied Chennai paper.",
            "Radiotherapy tumours include lesions larger than the paper's 3–30 mm solid nodules.",
            "107 available original features, versus 101 reported in the paper.",
            "Third-party feature extraction settings and software version are not fully documented.",
            "Official TCIA histology labels replace all clinical fields from the feature mirror.",
            "No benign or normal controls: supports subtype classification only.",
            "CT voxels are not downloaded by this lightweight feature-data command.",
        ],
        "license": "TCIA underlying collection: CC BY-NC 3.0; feature repository has no explicit license",
        "attribution": "Aerts et al. (2014), TCIA NSCLC-Radiomics; Clark et al. (2013); hhfchoi/Radiomics",
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {len(cohort)} real patients, {len(mapping)} features: {metadata['class_counts']}", flush=True)
    return metadata
