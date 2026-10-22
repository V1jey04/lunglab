"""Verified public CT crops, fixed CNN embeddings, and crop texture features."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ct_crops")
CT_FILES = {
    "TCGA-LUAD_CT_images.pkl": ("7051d5adc4f325abc990238b8c6a8a3e", "adenocarcinoma", 28, 47043934),
    "TCGA-LUSC_CT_images.pkl": ("74446785e9167e3c50008198c1aeed6d", "squamous_cell_carcinoma", 34, 47043937),
}


def download_crops():
    from .datasets import fetch_file
    for filename, (md5, _, _, file_id) in CT_FILES.items():
        fetch_file(f"https://ndownloader.figshare.com/files/{file_id}", RAW / filename, md5=md5)
    fetch_file("https://download.pytorch.org/models/resnet18-f37072fd.pth", RAW / "resnet18-f37072fd.pth")


class ArrayUnpickler(pickle.Unpickler):
    """Accept NumPy arrays only; reject arbitrary globals from downloaded pickle."""
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")
        return allowed[(module, name)]


def read_crops(path, md5, expected_patients):
    path = Path(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Incomplete or modified download: {path.name}")
    with path.open("rb") as f:
        values = ArrayUnpickler(f).load()
    if not isinstance(values, np.ndarray) or values.shape != (expected_patients, 224, 224, 5):
        raise ValueError(f"Unexpected patient crop shape: {getattr(values, 'shape', None)}")
    if values.dtype.kind not in "fui" or not np.isfinite(values).all():
        raise ValueError("Crop arrays must be finite numeric images.")
    if any(np.ptp(v) <= 0 for v in values):
        raise ValueError("A patient crop is empty or constant.")
    return values.astype(np.float32)


def normalize_crop(crop):
    """Per-case scaling only, without cohort or label statistics."""
    low, high = np.percentile(crop, [0.5, 99.5])
    if high <= low:
        raise ValueError("Crop has no usable intensity range.")
    return np.clip((crop - low) / (high - low), 0, 1).astype(np.float32)


def cnn_features(crops, weights=RAW / "resnet18-f37072fd.pth"):
    import torch
    from torchvision.models import resnet18
    path = Path(weights)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("ResNet18 weights do not match the official pretrained checkpoint.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    network = resnet18(weights=None)
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    network.fc = torch.nn.Identity()
    network.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    features = []
    with torch.inference_mode():
        for i, crop in enumerate(crops):
            crop = normalize_crop(crop)
            windows = np.stack([crop[:, :, offset:offset + 3].transpose(2, 0, 1)
                                for offset in range(3)])
            z = network((torch.from_numpy(windows) - mean) / std).numpy()
            features.append(np.concatenate([z.mean(axis=0), z.max(axis=0)]))
            if (i + 1) % 10 == 0:
                print(f"Frozen CNN embeddings: {i+1}/{len(crops)} patient stacks", flush=True)
    return pd.DataFrame(features, columns=[f"deep_resnet18_{mode}_{j:03d}"
                        for mode in ["mean", "max"] for j in range(512)])


def texture_features(crops):
    import SimpleITK as sitk
    from radiomics import featureextractor
    # These five slices were resized by their publisher. Physical CT spacing and
    # HU are unavailable. Shape measurements are excluded, and textures are
    # computed in each 2D plane with a fixed 0..255 per-case intensity range.
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=16, normalize=False, force2D=True, force2Ddimension=0,
        additionalInfo=False, minimumROIDimensions=2)
    extractor.disableAllFeatures()
    for family in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(family)
    rows = []
    for i, crop in enumerate(crops):
        pixels = normalize_crop(crop).transpose(2, 0, 1) * 255
        # Measure the resized bounding-box crop, excluding its one-pixel border.
        # The publisher provides no aligned mask; nonzero pixels are not a
        # reliable tumour mask in these mean-normalized crops.
        mask = np.zeros_like(pixels, dtype=np.uint8)
        mask[:, 1:-1, 1:-1] = 1
        image, labels = sitk.GetImageFromArray(pixels), sitk.GetImageFromArray(mask)
        result = extractor.execute(image, labels, label=1)
        rows.append({k: float(v) for k, v in result.items() if k.startswith("original_")})
        if (i + 1) % 10 == 0:
            print(f"Crop textures: {i+1}/{len(crops)} patient stacks", flush=True)
    return pd.DataFrame(rows)


def prepare(output="data/tcga_crops_features.csv"):
    output = Path(output)
    if output.exists():
        raise ValueError("CT crop features already exist; preserve the original preparation.")
    stacks, metadata = [], []
    for filename, (md5, label, count, file_id) in CT_FILES.items():
        images = read_crops(RAW / filename, md5, count)
        stacks.append(images)
        collection = filename.split("_")[0]
        for i, image in enumerate(images):
            # Identifiers denote one published patient array, not a recovered
            # clinical ID. Array ordering is not documented by the publisher.
            metadata.append({"patient_id": f"{collection}-array-{i:03d}",
                             "sample_id": f"{collection}-array-{i:03d}", "label": label,
                             "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    images = np.concatenate(stacks)
    seen, keep, duplicate_rows = {}, [], []
    for i, row in enumerate(metadata):
        digest = row["image_sha256"]
        if digest in seen:
            prior = metadata[seen[digest]]
            if row["label"] != prior["label"]:
                raise ValueError("Identical CT arrays have conflicting class labels.")
            duplicate_rows.append({"excluded": row["sample_id"], "identical_to": prior["sample_id"]})
        else:
            seen[digest] = i
            keep.append(i)
    images = images[keep]
    metadata = [metadata[i] for i in keep]
    textures, embeddings = texture_features(images), cnn_features(images)
    frame = pd.concat([pd.DataFrame(metadata), textures, embeddings], axis=1)
    frame.to_csv(output, index=False)
    provenance = {
        "collection": "TCGA-LUAD and TCGA-LUSC tumour CT crops",
        "source_doi": "https://doi.org/10.6084/m9.figshare.26037169",
        "publication": "https://doi.org/10.1016/j.crmeth.2024.100817",
        "license": "CC BY 4.0", "attribution": "Verma et al., 2024, Angione Lab; original TCIA/TCGA contributors",
        "patients": len(frame), "class_counts": frame.label.value_counts().to_dict(),
        "supplied_arrays": 62, "duplicate_arrays_excluded": duplicate_rows,
        "texture_features": len(textures.columns), "deep_features": len(embeddings.columns),
        "original_patient_ids_available": False,
        "patient_unit": "one 224x224x5 patient stack as supplied by the publisher",
        "label_source": "homogeneous TCGA-LUAD and TCGA-LUSC collection membership, not guessed row order",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": {name: {"md5": item[0], "url": f"https://ndownloader.figshare.com/files/{item[3]}"}
                  for name, item in CT_FILES.items()},
        "resnet18_weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "resnet18_sha256": hashlib.sha256((RAW / "resnet18-f37072fd.pth").read_bytes()).hexdigest(),
        "limitations": [
            "A new, small crop classification benchmark; not a reproduction of the paper's 3D HU radiomics.",
            "Collection membership determines class, so acquisition/source confounding must be investigated.",
            "One published array per patient; clinical ID order cannot be independently verified.",
            "Resized 5-slice crops lack physical spacing and HU; shape features are excluded.",
            "Textures measure bounding-box crop interiors, not segmented tumour masks.",
            "CNN is frozen ImageNet ResNet18, not pretrained on lung cancer or fine-tuned using test patients.",
            "R01 crop arrays were excluded: source generation uses unordered set-derived patient order, with no saved mapping.",
            "Neither benign controls nor a full-scan tumour detection task is included.",
        ],
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    from sklearn.model_selection import train_test_split
    split_path = output.with_suffix(".splits.csv")
    if split_path.exists():
        raise ValueError("A split manifest already exists; do not replace a locked split.")
    development, test = train_test_split(frame.patient_id, test_size=0.25,
                                         stratify=frame.label, random_state=42)
    splits = frame[["patient_id", "sample_id", "label"]].copy()
    splits["split"] = np.where(splits.patient_id.isin(test), "test", "development")
    splits.to_csv(split_path, index=False)
    print(f"Prepared {len(frame)} patient stacks: {len(textures.columns)} crop textures and {len(embeddings.columns)} CNN features.", flush=True)
    return provenance


if __name__ == "__main__":
    prepare()
