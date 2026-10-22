"""Physical CT/mask inspection and controlled segmentation perturbation."""
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .imaging import imaging_dependencies, validate_geometry, extract_images


def inspect_volume(image, mask, label=1):
    sitk, _ = imaging_dependencies()
    validate_geometry(image, mask, label)
    pixels = sitk.GetArrayFromImage(image)
    selected = sitk.GetArrayFromImage(mask) == label
    coordinates = np.argwhere(selected)
    _, components = ndimage.label(selected)
    touches_edge = bool(np.any(coordinates.min(axis=0) == 0) or
                        np.any(coordinates.max(axis=0) == np.asarray(selected.shape) - 1))
    flags = []
    if touches_edge:
        flags.append("Tumour mask touches the volume boundary; inspect for a cropped tumour.")
    if components > 1:
        flags.append(f"Mask contains {components} disconnected components; confirm the intended tumour.")
    if pixels.min() >= 0 and pixels.max() <= 255:
        flags.append("Intensity range resembles an 8-bit image. Confirm original CT Hounsfield units.")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    axis_aligned = np.allclose(np.max(np.abs(direction), axis=0), 1, atol=1e-5)
    if not axis_aligned:
        flags.append("Oblique acquisition. Viewer planes follow the image grid, not exact anatomical planes.")
    return {"size_xyz": list(image.GetSize()), "spacing_mm_xyz": list(image.GetSpacing()),
            "origin_mm_lps": list(image.GetOrigin()), "direction_cosines": list(image.GetDirection()),
            "mask_label": int(label), "tumour_voxels": int(selected.sum()),
            "tumour_volume_mm3": float(selected.sum() * np.prod(image.GetSpacing())),
            "connected_components": int(components), "mask_touches_edge": touches_edge,
            "intensity_range": [float(pixels.min()), float(pixels.max())],
            "axis_aligned": bool(axis_aligned), "review_flags": flags,
            "hu_note": "CT units require source verification; an intensity-range check cannot prove HU calibration."}


def oriented_arrays(image, mask, label=1):
    sitk, _ = imaging_dependencies()
    validate_geometry(image, mask, label)
    # Permute/flip both together. No interpolation or changes to extraction inputs.
    canonical_image = sitk.DICOMOrient(image, "LPS")
    canonical_mask = sitk.DICOMOrient(mask, "LPS")
    return canonical_image, sitk.GetArrayFromImage(canonical_image), sitk.GetArrayFromImage(canonical_mask) == label


def slice_figure(image, mask, plane="Axial", index=None, center=-600, width=1500, opacity=.35, label=1):
    from matplotlib.figure import Figure
    if plane not in {"Axial", "Coronal", "Sagittal"} or width <= 0 or not 0 <= opacity <= 1:
        raise ValueError("Choose a valid plane, positive window width and overlay opacity from 0 to 1.")
    oriented, pixels, selected = oriented_arrays(image, mask, label)
    axis = {"Axial": 0, "Coronal": 1, "Sagittal": 2}[plane]
    if index is None:
        index = int(np.median(np.argwhere(selected)[:, axis]))
    if not 0 <= index < pixels.shape[axis]:
        raise ValueError("Slice index is outside the CT volume.")
    ct_slice, mask_slice = np.take(pixels, index, axis=axis), np.take(selected, index, axis=axis)
    sx, sy, sz = oriented.GetSpacing()
    spacing = {"Axial": (sx, sy), "Coronal": (sx, sz), "Sagittal": (sy, sz)}[plane]
    figure = Figure(figsize=(7, 5), facecolor="#102b40", layout="constrained")
    ax = figure.subplots()
    extent = (0, ct_slice.shape[1] * spacing[0], 0, ct_slice.shape[0] * spacing[1])
    ax.imshow(ct_slice, cmap="gray", vmin=center - width / 2, vmax=center + width / 2, origin="lower", extent=extent)
    overlay = np.zeros((*mask_slice.shape, 4))
    overlay[mask_slice] = [0.12, .91, .70, opacity]
    ax.imshow(overlay, origin="lower", extent=extent)
    ax.set_title(f"{plane} grid · slice {index + 1}/{pixels.shape[axis]}", color="white", fontsize=11)
    ax.set_xlabel("Image grid distance (mm)", color="#adc5d5")
    ax.set_ylabel("Image grid distance (mm)", color="#adc5d5")
    ax.tick_params(colors="#adc5d5")
    return figure


def figure_data_uri(figure):
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=120, facecolor=figure.get_facecolor())
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def tumour_surface(image, mask, label=1):
    import plotly.graph_objects as go
    sitk, _ = imaging_dependencies()
    validate_geometry(image, mask, label)
    selected = sitk.GetArrayFromImage(mask) == label
    boundary = selected & ~ndimage.binary_erosion(selected)
    coords_zyx = np.argwhere(boundary)
    coords_zyx = coords_zyx[::max(1, int(np.ceil(len(coords_zyx) / 5000)))]
    physical = (coords_zyx[:, ::-1] * np.asarray(image.GetSpacing())) @ np.asarray(image.GetDirection()).reshape(3, 3).T
    physical += np.asarray(image.GetOrigin())
    fig = go.Figure(go.Scatter3d(x=physical[:, 0], y=physical[:, 1], z=physical[:, 2], mode="markers",
        marker={"size": 2, "color": physical[:, 2], "colorscale": "Teal", "opacity": .7},
        hovertemplate="LPS coordinates (mm)<br>x %{x:.1f}<br>y %{y:.1f}<br>z %{z:.1f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, b=0, t=15), paper_bgcolor="rgba(0,0,0,0)",
        scene={"aspectmode": "data", "xaxis_title": "LPS X (mm)", "yaxis_title": "LPS Y (mm)", "zaxis_title": "LPS Z (mm)"})
    return fig


def perturb_mask(image, mask, radius_mm, label=1):
    sitk, _ = imaging_dependencies()
    validate_geometry(image, mask, label)
    if not np.isfinite(radius_mm) or abs(radius_mm) > 5:
        raise ValueError("Mask perturbation radius must be finite and within ±5 mm.")
    selected = sitk.GetArrayFromImage(mask) == label
    spacing = image.GetSpacing()[::-1]
    if radius_mm < 0:
        result = ndimage.distance_transform_edt(selected, sampling=spacing) > abs(radius_mm)
    elif radius_mm > 0:
        result = ndimage.distance_transform_edt(~selected, sampling=spacing) <= radius_mm
    else:
        result = selected.copy()
    perturbed = sitk.GetImageFromArray(result.astype("uint32") * label)
    perturbed.CopyInformation(mask)
    return perturbed


def segmentation_stability(image, mask, params="configs/radiomics.yaml", label=1, bundle=None):
    """Stress test one case. Not ICC, performance validation or annotation truth."""
    sitk, _ = imaging_dependencies()
    baseline = extract_images(image, mask, params, label)
    rows, feature_rows = [], []
    for radius in (0.0, -1.0, 1.0):
        variant = perturb_mask(image, mask, radius, label)
        try:
            validate_geometry(image, variant, label)
            features = baseline if radius == 0 else extract_images(image, variant, params, label)
            relative = [abs(features[k] - v) / abs(v) for k, v in baseline.items() if abs(v) > 1e-8]
            row = {"radius_mm": radius, "status": "extracted", "volume_mm3": float(
                (sitk.GetArrayFromImage(variant) == label).sum() * np.prod(image.GetSpacing())),
                "median_absolute_relative_feature_change": float(np.median(relative)),
                "features_compared": len(relative)}
            feature_rows.append({"radius_mm": radius, **features})
            if bundle is not None:
                from .predict import predict_frame
                if set(bundle["feature_names"]).issubset(features):
                    prediction = predict_frame(bundle, pd.DataFrame([features])).iloc[0]
                    row.update({"prediction": prediction.prediction,
                                "positive_probability": float(prediction[f"p_{bundle['labels'][1]}"])})
                else:
                    row["prediction_note"] = "Extraction schema does not match this model; no probability computed."
            rows.append(row)
        except ValueError as error:
            rows.append({"radius_mm": radius, "status": "invalid_mask", "reason": str(error)})
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)
