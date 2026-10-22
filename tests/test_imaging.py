import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("radiomics")

from lungml.imaging import extract_case, extract_manifest, read_volume, validate_geometry


def volumes():
    image = sitk.GetImageFromArray(np.random.default_rng(4).normal(-400, 150, (12, 14, 16)).astype("float32"))
    image.SetSpacing((0.8, 0.8, 1.5))
    pixels = np.zeros((12, 14, 16), dtype="uint8")
    pixels[3:8, 4:10, 5:11] = 1
    mask = sitk.GetImageFromArray(pixels)
    mask.CopyInformation(image)
    return image, mask


def test_misaligned_masks_are_rejected():
    image, mask = volumes()
    mask.SetOrigin((5, 0, 0))
    with pytest.raises(ValueError, match="origin mismatch"):
        validate_geometry(image, mask)


def test_empty_masks_are_rejected():
    image, mask = volumes()
    mask = mask * 0
    with pytest.raises(ValueError, match="nonempty 3D"):
        validate_geometry(image, mask)


def test_extracts_real_3d_feature_families(tmp_path):
    image, mask = volumes()
    image_path, mask_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    features = extract_case(image_path, mask_path)
    assert len(features) == 107
    assert features["original_shape_VoxelVolume"] > 0
    assert all(np.isfinite(v) for v in features.values())
    assert "original_glcm_Imc2" in features


def test_rich_extraction_preserves_wavelet_and_log_features(tmp_path):
    image, mask = volumes()
    image_path, mask_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    features = extract_case(image_path, mask_path, "configs/radiomics-rich.yaml")
    assert sum(k.startswith("original_") for k in features) == 107
    assert sum(k.startswith("wavelet-") for k in features) == 8 * 93
    assert sum(k.startswith("log-sigma-") for k in features) == 3 * 93
    assert all(np.isfinite(v) for v in features.values())


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_dicom_loader_requires_ct_modality(tmp_path, modality):
    from uuid import uuid4
    image, _ = volumes()
    image = sitk.Cast(image, sitk.sitkInt16)
    series_uid = "2.25." + str(uuid4().int)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for z in range(image.GetDepth()):
        slice_image = image[:, :, z]
        for key, value in {"0008|0060": modality, "0020|000e": series_uid,
                           "0020|0013": str(z + 1), "0020|0037": "1\\0\\0\\0\\1\\0",
                           "0020|0032": "\\".join(map(str, image.TransformIndexToPhysicalPoint((0, 0, z))))}.items():
            slice_image.SetMetaData(key, value)
        writer.SetFileName(str(tmp_path / f"slice-{z:03}.dcm"))
        writer.Execute(slice_image)
    if modality == "CT":
        loaded = read_volume(tmp_path)
        assert loaded.GetSize() == image.GetSize()
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded), sitk.GetArrayFromImage(image))
    else:
        with pytest.raises(ValueError, match="must be a CT series"):
            read_volume(tmp_path)


def test_extraction_cannot_overwrite_source_volumes(tmp_path):
    import pandas as pd
    image, mask = volumes()
    ct_path, mask_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(ct_path))
    sitk.WriteImage(mask, str(mask_path))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"patient_id": "001", "sample_id": "001-ct", "label": "adenocarcinoma",
                   "image_path": "ct.nrrd", "mask_path": "mask.nrrd"}]).to_csv(manifest, index=False)
    before = ct_path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        extract_manifest(manifest, "configs/radiomics.yaml", ct_path)
    assert ct_path.read_bytes() == before
