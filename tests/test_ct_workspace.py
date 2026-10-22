import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("radiomics")

from lungml.ct_workspace import (inspect_volume, oriented_arrays, perturb_mask, slice_figure,
                                 segmentation_stability, tumour_surface, figure_data_uri)
from lungml.dashboard_inspection import synthetic_geometry_example


def test_physical_volume_and_three_plane_slice_geometry():
    image, mask = synthetic_geometry_example()
    quality = inspect_volume(image, mask)
    selected = sitk.GetArrayFromImage(mask) == 1
    assert quality["tumour_volume_mm3"] == pytest.approx(selected.sum() * .9 * .9 * 1.8)
    assert quality["connected_components"] == 1
    for plane in ["Axial", "Coronal", "Sagittal"]:
        figure = slice_figure(image, mask, plane)
        assert figure_data_uri(figure).startswith("data:image/png;base64,")
        assert len(figure.axes[0].images) == 2
    with pytest.raises(ValueError, match="outside"):
        slice_figure(image, mask, index=10000)


def test_display_orientation_preserves_image_mask_alignment_and_physical_coordinates():
    image, mask = synthetic_geometry_example()
    image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    mask.CopyInformation(image)
    oriented, pixels, selected = oriented_arrays(image, mask)
    assert oriented.GetDirection() == (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    for zyx in np.argwhere(selected)[::50]:
        xyz = tuple(int(v) for v in zyx[::-1])
        physical = oriented.TransformIndexToPhysicalPoint(xyz)
        original_index = image.TransformPhysicalPointToIndex(physical)
        assert mask[original_index] == 1
        assert pixels[tuple(zyx)] == image[original_index]
    figure = tumour_surface(image, mask)
    assert len(figure.data[0].x) <= 5000
    assert max(figure.data[0].x) < 0  # Reflected physical LPS x coordinate.


def test_perturbation_uses_millimetres_and_preserves_geometry():
    image, mask = synthetic_geometry_example()
    original = sitk.GetArrayFromImage(mask).sum()
    eroded, dilated = perturb_mask(image, mask, -1), perturb_mask(image, mask, 1)
    assert sitk.GetArrayFromImage(eroded).sum() < original < sitk.GetArrayFromImage(dilated).sum()
    assert eroded.GetSpacing() == dilated.GetSpacing() == image.GetSpacing()
    assert eroded.GetDirection() == image.GetDirection()
    with pytest.raises(ValueError, match="within"):
        perturb_mask(image, mask, 10)


def test_stability_exports_all_actual_feature_measurements():
    image, mask = synthetic_geometry_example()
    summary, features = segmentation_stability(image, mask)
    assert summary.radius_mm.tolist() == [0, -1, 1]
    assert summary.status.tolist() == ["extracted"] * 3
    assert len(features.columns) == 108
    assert summary.median_absolute_relative_feature_change.iloc[0] == 0
    assert summary.volume_mm3.iloc[1] < summary.volume_mm3.iloc[0] < summary.volume_mm3.iloc[2]
