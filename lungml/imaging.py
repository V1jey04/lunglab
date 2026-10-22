"""3D CT and tumour mask loading, geometry checks, radiomic extraction."""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RADIOMIC_PREFIXES


def parameter_path(params):
    path = Path(params)
    if not path.is_file() and str(path) in {"configs/radiomics.yaml", "configs/radiomics-rich.yaml"}:
        path = Path(__file__).with_name("configs") / path.name
    return path


def imaging_dependencies():
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:
        raise RuntimeError("Install imaging dependencies using the README instructions.") from exc
    return sitk, featureextractor


def read_volume(path):
    sitk, _ = imaging_dependencies()
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path)) or []
        if len(series) != 1:
            raise ValueError("DICOM directory must contain exactly one CT series.")
        filenames = reader.GetGDCMSeriesFileNames(str(path), series[0])
        header = sitk.ImageFileReader()
        header.SetFileName(filenames[0])
        header.ReadImageInformation()
        if not header.HasMetaDataKey("0008|0060") or header.GetMetaData("0008|0060").strip() != "CT":
            raise ValueError("DICOM input must be a CT series, not MR, PET or segmentation objects.")
        reader.SetFileNames(filenames)
        return reader.Execute()
    if not path.is_file():
        raise FileNotFoundError(path)
    return sitk.ReadImage(str(path))


def validate_geometry(image, mask, label=1):
    sitk, _ = imaging_dependencies()
    if not isinstance(label, (int, np.integer)) or label <= 0:
        raise ValueError("Mask label must be a positive integer.")
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("A 3D CT volume and 3D tumour mask are required.")
    if image.GetNumberOfComponentsPerPixel() != 1 or mask.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("CT and mask must have scalar pixels.")
    if image.GetSize() != mask.GetSize():
        raise ValueError("CT and mask size mismatch.")
    for name in ("Spacing", "Origin", "Direction"):
        if not np.allclose(getattr(image, f"Get{name}")(), getattr(mask, f"Get{name}")(),
                           atol=1e-5, rtol=0):
            raise ValueError(f"CT and mask {name.lower()} mismatch. Align them before extraction.")
    voxels = sitk.GetArrayFromImage(image)
    segmentation = sitk.GetArrayFromImage(mask)
    if not np.isfinite(voxels).all() or not np.isfinite(segmentation).all():
        raise ValueError("CT or mask contains nonfinite voxels.")
    if not np.all(segmentation == np.floor(segmentation)):
        raise ValueError("Mask must contain integer segment labels.")
    if (segmentation < 0).any() or (segmentation > np.iinfo(np.uint32).max).any():
        raise ValueError("Mask labels must fit the nonnegative uint32 range.")
    coordinates = np.argwhere(segmentation == label)
    if len(coordinates) < 8 or np.any(np.ptp(coordinates, axis=0) < 1):
        raise ValueError("Mask must select a nonempty 3D tumour spanning at least two voxels in each axis.")
    if len(coordinates) == segmentation.size:
        raise ValueError("Mask labels the entire volume. Supply a tumour segmentation.")
    if min(image.GetSpacing()) <= 0:
        raise ValueError("Voxel spacing must be positive.")
    geometry = np.asarray([*image.GetSpacing(), *image.GetOrigin(), *image.GetDirection()])
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.isfinite(geometry).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Image geometry must be finite with orthonormal direction cosines.")


def extract_case(image_path, mask_path, params="configs/radiomics.yaml", label=1):
    image, mask = read_volume(image_path), read_volume(mask_path)
    return extract_images(image, mask, params, label)


def extract_images(image, mask, params="configs/radiomics.yaml", label=1):
    sitk, featureextractor = imaging_dependencies()
    validate_geometry(image, mask, label)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(parameter_path(params)))
    result = extractor.execute(sitk.Cast(image, sitk.sitkFloat32),
                               sitk.Cast(mask, sitk.sitkUInt32), label=label)
    features = {k: float(v) for k, v in result.items() if k.startswith(RADIOMIC_PREFIXES)}
    if not features:
        raise ValueError("Extraction produced no allowed radiomic features.")
    if not all(np.isfinite(v) for v in features.values()):
        raise ValueError("Radiomic extraction produced nonfinite features.")
    return features


def extract_manifest(manifest_path, params, output):
    manifest_path, output = Path(manifest_path), Path(output)
    params = parameter_path(params)
    with manifest_path.open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header):
        raise ValueError("Extraction manifest needs nonempty, unique raw column names.")
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "sample_id": str})
    required = {"patient_id", "sample_id", "label", "image_path", "mask_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest requires {sorted(required)}")
    if manifest[list(required)].isna().any().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Manifest fields must be nonmissing and sample IDs unique.")
    protected = {manifest_path.resolve(), params.resolve()}
    for item in manifest.to_dict("records"):
        for key in ("image_path", "mask_path"):
            source = Path(item[key])
            protected.add((source if source.is_absolute() else manifest_path.parent / source).resolve())
    if {output.resolve(), output.with_suffix(".provenance.json").resolve()} & protected:
        raise ValueError("Extraction outputs must not overwrite the manifest, parameters or source volumes.")
    rows = []
    for item in manifest.to_dict("records"):
        def resolve(name):
            path = Path(item[name])
            return path if path.is_absolute() else manifest_path.parent / path
        raw_label = float(item.get("mask_label", 1))
        if not np.isfinite(raw_label) or raw_label != int(raw_label):
            raise ValueError("mask_label must be a positive integer.")
        label = int(raw_label)
        features = extract_case(resolve("image_path"), resolve("mask_path"), params, label)
        rows.append({k: item[k] for k in ("patient_id", "sample_id", "label")} | features)
        print(f"Extracted {item['sample_id']}: {len(features)} radiomic features", flush=True)
    if not rows:
        raise ValueError("Manifest is empty.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "source": "local CT and tumour-mask extraction",
        "parameters_sha256": hashlib.sha256(Path(params).read_bytes()).hexdigest(),
        "parameters": Path(params).read_text(),
        "pyradiomics_version": version("pyradiomics"),
        "SimpleITK_version": version("SimpleITK"),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2))
