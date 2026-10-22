"""Case-level evidence and interactive CT workbench for the research dashboard."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import streamlit as st

from .case_report import build_case_report
from .explanations import explain_case
from .quality import assess_inputs, load_profile, sha256


def render_case_inspection(model_path, bundle, frame, predictions, key="case", overlay=None):
    from .dashboard import chart, panel, TEAL, AMBER
    with st.expander("Case explanation, input quality and research report", expanded=True):
        panel("Inspect the evidence for this case", "04 / CASE REVIEW")
        try:
            profile = load_profile(model_path, bundle)
            quality = assess_inputs(bundle, frame, profile)
            item = quality[0]
            cols = st.columns(3)
            cols[0].metric("Missing inputs", item["missing_feature_count"])
            cols[1].metric("Outside training range", item["outside_reference_count"] if profile else "Unavailable")
            cols[2].metric("Input review", "Suggested" if item["input_review_suggested"] else ("No flag" if profile else "Incomplete checks"))
            st.caption("Range checks use the 1st–99th percentiles of fitted training patients. A flag does not measure clinical uncertainty.")
            if item["outside_reference_features"]:
                st.write("Features outside the reference range: " + ", ".join(item["outside_reference_features"][:8]))
            if profile is None:
                st.info("This model has no training-only input profile. Its cohort-range check is unavailable.")
            explanation = explain_case(bundle, frame, profile)
            if explanation["contributions"]:
                rows = pd.DataFrame(explanation["contributions"])
                rows["direction"] = np.where(rows.contribution >= 0, "Increases positive-class score", "Decreases positive-class score")
                chart(rows, {"mark": {"type": "bar", "cornerRadiusEnd": 3}, "encoding": {
                    "y": {"field": "feature", "type": "nominal", "sort": "-x", "title": None,
                          "axis": {"labelLimit": 280}},
                    "x": {"field": "contribution", "type": "quantitative", "title": explanation["units"]},
                    "color": {"field": "direction", "type": "nominal", "scale": {"range": [AMBER, TEAL]}},
                    "tooltip": [{"field": "feature"}, {"field": "contribution", "format": ".5f"}]}}, 360)
                st.caption("Positive class: " + explanation["positive_class"].replace("_", " ") + ". " + explanation["interpretation"])
                if explanation["method"] == "linear_log_odds":
                    st.caption(f"All contributions plus the intercept reconstruct P(positive) = {explanation['reconstructed_probability']:.4f}.")
            else:
                st.caption(explanation["interpretation"])
            report = build_case_report(bundle, frame, predictions, sha256(model_path), quality, explanation, overlay)
            st.download_button("Download case report · HTML / printable PDF", report,
                "lunglab_case_report.html", "text/html", key=f"{key}_report")
        except (ValueError, OSError, KeyError) as error:
            st.error(f"Case inspection unavailable: {error}")


def synthetic_geometry_example():
    """Explicit teaching phantom; never presented as a patient scan."""
    from .imaging import imaging_dependencies
    sitk, _ = imaging_dependencies()
    z, y, x = np.indices((32, 64, 64))
    lungs = (((x - 21) / 13) ** 2 + ((y - 33) / 23) ** 2 < 1) | (((x - 44) / 13) ** 2 + ((y - 33) / 23) ** 2 < 1)
    pixels = np.where(lungs, -800., 35.) + np.random.default_rng(42).normal(0, 25, z.shape)
    tumour = ((x - 23) / 7) ** 2 + ((y - 31) / 8) ** 2 + ((z - 17) / 5) ** 2 <= 1
    pixels[tumour] = 70 + np.random.default_rng(12).normal(0, 18, tumour.sum())
    image, mask = sitk.GetImageFromArray(pixels.astype("float32")), sitk.GetImageFromArray(tumour.astype("uint8"))
    image.SetSpacing((.9, .9, 1.8))
    mask.CopyInformation(image)
    return image, mask


def render_ct_workspace():
    from .dashboard import cards, panel
    from .imaging import read_volume, extract_images
    from .ct_workspace import inspect_volume, oriented_arrays, slice_figure, tumour_surface, segmentation_stability
    panel("Inspect the scan. Check the segmentation.", "CT / MASK WORKBENCH", "3D geometry · physical units · local extraction")
    st.caption("Supply a known tumour mask and original CT units. CT inspection and extraction are separate from the trained feature-table experiments.")
    source = st.radio("CT input", ["Synthetic geometry example", "Local CT and mask", "Upload volumes"], horizontal=True)
    image_path, mask_path, ct_upload, mask_upload = "", "", None, None
    if source == "Local CT and mask":
        a, b = st.columns(2)
        image_path = a.text_input("CT path or single-series DICOM directory", placeholder="/absolute/path/ct.nii.gz")
        mask_path = b.text_input("Aligned tumour mask path", placeholder="/absolute/path/mask.nii.gz")
    elif source == "Upload volumes":
        a, b = st.columns(2)
        ct_upload = a.file_uploader("CT volume · NIfTI or NRRD", type=["nii", "gz", "nrrd"], key="ct_volume")
        mask_upload = b.file_uploader("Tumour mask · NIfTI or NRRD", type=["nii", "gz", "nrrd"], key="ct_mask")
    label = int(st.number_input("Tumour mask label", min_value=1, max_value=65535, value=1, step=1))
    content = f"{source}|{image_path}|{mask_path}|{label}".encode()
    for upload in (ct_upload, mask_upload):
        if upload is not None:
            content += upload.getvalue()
    signature = hashlib.sha256(content).hexdigest()
    if st.session_state.get("ct_signature") != signature:
        for name in ("ct_loaded", "ct_extracted", "ct_stability"):
            st.session_state.pop(name, None)
        st.session_state["ct_signature"] = signature
    if source == "Synthetic geometry example":
        st.info("Synthetic geometry example. This is not a patient scan and produces no cancer predictions.")
        if "ct_loaded" not in st.session_state:
            st.session_state["ct_loaded"] = synthetic_geometry_example()
    elif st.button("Load and check CT / mask", type="primary", disabled=(not image_path or not mask_path) if source == "Local CT and mask" else (ct_upload is None or mask_upload is None)):
        try:
            if source == "Local CT and mask":
                image, mask = read_volume(image_path), read_volume(mask_path)
            else:
                with TemporaryDirectory(prefix="lunglab-ct-") as directory:
                    paths = []
                    for prefix, upload in [("ct-", ct_upload), ("mask-", mask_upload)]:
                        if not upload.name.lower().endswith((".nii", ".nii.gz", ".nrrd")):
                            raise ValueError("Use a self-contained .nii, .nii.gz or .nrrd volume.")
                        path = Path(directory) / (prefix + Path(upload.name).name)
                        path.write_bytes(upload.getvalue())
                        paths.append(path)
                    image, mask = read_volume(paths[0]), read_volume(paths[1])
            inspect_volume(image, mask, label)
            st.session_state["ct_loaded"] = image, mask
            st.session_state.pop("ct_extracted", None)
            st.session_state.pop("ct_stability", None)
        except (ValueError, OSError, RuntimeError) as error:
            st.session_state.pop("ct_loaded", None)
            st.error(str(error))
    if "ct_loaded" not in st.session_state:
        return
    image, mask = st.session_state["ct_loaded"]
    try:
        quality = inspect_volume(image, mask, label)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
        return
    cards([("Tumour volume", f"{quality['tumour_volume_mm3']:,.0f} mm³", "Voxel count × physical spacing"),
           ("CT grid", " × ".join(map(str, quality["size_xyz"])), "X × Y × Z voxels"),
           ("Voxel spacing", " / ".join(f"{x:g}" for x in quality["spacing_mm_xyz"]), "X / Y / Z in millimetres"),
           ("Mask components", quality["connected_components"], "Inspect disconnected regions")])
    for flag in quality["review_flags"]:
        st.warning(flag)
    controls = st.columns(3)
    with controls[0]:
        plane = st.selectbox("Slice plane", ["Axial", "Coronal", "Sagittal"])
        _, pixels, selected = oriented_arrays(image, mask, label)
        axis = {"Axial": 0, "Coronal": 1, "Sagittal": 2}[plane]
        default = int(np.median(np.argwhere(selected)[:, axis]))
        index = st.slider("Slice", 0, pixels.shape[axis] - 1, default, key=f"ct_slice_{signature[:8]}_{plane}")
    with controls[1]:
        preset = st.selectbox("CT window", ["Lung", "Soft tissue", "Custom"])
        center, width = (-600, 1500) if preset == "Lung" else (40, 400)
        if preset == "Custom":
            center = st.slider("Window centre (HU, if verified)", -1200, 1200, -600)
            width = st.slider("Window width", 1, 3000, 1500)
    with controls[2]:
        opacity = st.slider("Tumour overlay opacity", 0., 1., .35)
        st.caption("Overlay and CT use the same physical grid. Adjust the view without changing extraction inputs.")
    left, right = st.columns([1.4, 1])
    with left:
        figure = slice_figure(image, mask, plane, index, center, width, opacity, label)
        st.pyplot(figure, width="stretch")
        st.caption("Both volumes are permuted/flipped together toward LPS for display. Axes show image-grid distance. Extraction uses the original volumes.")
    with right:
        st.plotly_chart(tumour_surface(image, mask, label), width="stretch")
        st.caption("Drag to rotate the tumour boundary. Coordinates use the CT's physical LPS frame in millimetres.")
        with st.expander("Geometry and quality details"):
            st.json(quality)
    params = st.selectbox("Extraction protocol", ["configs/radiomics.yaml", "configs/radiomics-rich.yaml"])
    if st.session_state.get("ct_params") != params:
        st.session_state.pop("ct_extracted", None)
        st.session_state.pop("ct_stability", None)
        st.session_state["ct_params"] = params
    patient_id = st.text_input("Research case ID", value="synthetic-geometry" if source == "Synthetic geometry example" else "local-case")
    st.caption("These explicit protocols use 1 mm resampling. They do not reproduce the third-party native-spacing extraction behind the default model. Export for a matching new-cohort experiment.")
    a, b = st.columns(2)
    if a.button("Extract radiomic features", disabled=not patient_id.strip()):
        try:
            with st.spinner("Extracting measurements from the CT and mask…"):
                st.session_state["ct_extracted"] = extract_images(image, mask, params, label)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
    if b.button("Check segmentation stability · ±1 mm"):
        try:
            with st.spinner("Re-extracting the original, eroded and dilated masks…"):
                st.session_state["ct_stability"] = segmentation_stability(image, mask, params, label)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
    if "ct_extracted" in st.session_state:
        features = st.session_state["ct_extracted"]
        frame = pd.DataFrame([{"patient_id": patient_id.strip(), "sample_id": patient_id.strip(), **features}])
        st.success(f"Extracted {len(features):,} features. No subtype prediction has been made from this extraction.")
        st.download_button("Download extracted features", frame.to_csv(index=False), "ct_features.csv", "text/csv")
        import json
        provenance = {"context": "synthetic geometry example" if source == "Synthetic geometry example" else "local CT research extraction",
                      "parameters_sha256": sha256(params), "parameters": Path(params).read_text(), "geometry": quality,
                      "case_id": patient_id.strip(), "features_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()}
        st.download_button("Download extraction provenance", json.dumps(provenance, indent=2), "ct_provenance.json", "application/json")
    if "ct_stability" in st.session_state:
        summary, features = st.session_state["ct_stability"]
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption("One-case perturbation stress test. Relative changes exclude baseline features near zero. This is not an ICC estimate or clinical validation.")
        st.download_button("Download stability measurements", features.to_csv(index=False), "mask_stability_features.csv", "text/csv")
