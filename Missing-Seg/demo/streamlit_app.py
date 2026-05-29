from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Ours"

PLANES = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}


@st.cache_data
def load_volume(path: str) -> np.ndarray:
    volume = np.asarray(nib.load(path).get_fdata())
    return np.squeeze(volume)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lower, upper = np.percentile(image, [1, 99])
    image = np.clip(image, lower, upper)
    span = image.max() - image.min()
    if span == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image.min()) / span


def slice_volume(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    image = np.take(volume, index, axis=axis)
    return np.rot90(image)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    base_rgb = np.dstack([base, base, base])
    mask_bool = mask > 0
    overlay = base_rgb.copy()
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    return overlay


def dice_score(prediction: np.ndarray, label: np.ndarray) -> float:
    pred_mask = prediction > 0
    label_mask = label > 0
    denominator = pred_mask.sum() + label_mask.sum()
    if denominator == 0:
        return 1.0
    return 2 * np.logical_and(pred_mask, label_mask).sum() / denominator


def find_cases() -> list[dict[str, Path]]:
    image_files = sorted(DATA_DIR.glob("*_img.nii.gz"))
    cases = []
    for image_path in image_files:
        prefix = image_path.name.replace("_img.nii.gz", "")
        label_path = DATA_DIR / f"{prefix}_label.nii.gz"
        prediction_path = DATA_DIR / f"{prefix}_pre.nii.gz"
        if label_path.exists() and prediction_path.exists():
            cases.append({
                "name": prefix,
                "image": image_path,
                "label": label_path,
                "prediction": prediction_path,
            })
    return cases


st.set_page_config(page_title="Missing-Seg Demo", layout="wide")

st.title("Missing-Modality Brain Tumor Segmentation Demo")
st.write(
    "A lightweight visualization demo for one prepared Missing-Seg case. "
    "It compares the input MRI volume, ground-truth label, model prediction, "
    "and mask overlays across anatomical slices."
)

cases = find_cases()
if not cases:
    st.error("No demo case found. Expected files under Missing-Seg/Ours ending in _img, _label, and _pre.")
    st.stop()

case = st.sidebar.selectbox("Case", cases, format_func=lambda item: item["name"])
plane_name = st.sidebar.radio("View plane", list(PLANES.keys()), horizontal=True)
opacity = st.sidebar.slider("Overlay opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

image_volume = load_volume(str(case["image"]))
label_volume = load_volume(str(case["label"]))
prediction_volume = load_volume(str(case["prediction"]))

if image_volume.shape != label_volume.shape or image_volume.shape != prediction_volume.shape:
    st.error(
        "The image, label, and prediction volumes have different shapes. "
        f"Image: {image_volume.shape}, label: {label_volume.shape}, prediction: {prediction_volume.shape}"
    )
    st.stop()

axis = PLANES[plane_name]
slice_index = st.sidebar.slider("Slice", min_value=0, max_value=image_volume.shape[axis] - 1, value=image_volume.shape[axis] // 2)

image_slice = normalize_image(slice_volume(image_volume, axis, slice_index))
label_slice = slice_volume(label_volume, axis, slice_index)
prediction_slice = slice_volume(prediction_volume, axis, slice_index)

label_overlay = overlay_mask(image_slice, label_slice, color=(0.85, 0.1, 0.16), alpha=opacity)
prediction_overlay = overlay_mask(image_slice, prediction_slice, color=(0.0, 0.55, 0.58), alpha=opacity)

dice = dice_score(prediction_volume, label_volume)
st.metric("Volume Dice score", f"{dice:.3f}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image(image_slice, caption="Input MRI", clamp=True)
with col2:
    st.image(label_slice > 0, caption="Ground truth label", clamp=True)
with col3:
    st.image(prediction_slice > 0, caption="Prediction", clamp=True)
with col4:
    st.image(prediction_overlay, caption="Prediction overlay", clamp=True)

st.subheader("Overlay Comparison")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(label_overlay)
axes[0].set_title("Ground truth overlay")
axes[0].axis("off")
axes[1].imshow(prediction_overlay)
axes[1].set_title("Prediction overlay")
axes[1].axis("off")
st.pyplot(fig)

