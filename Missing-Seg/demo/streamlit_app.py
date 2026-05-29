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


def align_image_volume(image_volume: np.ndarray, reference_shape: tuple[int, int, int], modality: int) -> np.ndarray:
    if image_volume.shape == reference_shape:
        return image_volume

    if image_volume.ndim == 4:
        modality_axis = next((axis for axis, size in enumerate(image_volume.shape) if size <= 8), None)
        if modality_axis is None:
            st.error(f"Could not identify the modality axis in image shape {image_volume.shape}.")
            st.stop()

        modality_count = image_volume.shape[modality_axis]
        modality = min(modality, modality_count - 1)
        image_volume = np.take(image_volume, modality, axis=modality_axis)

    if image_volume.shape == reference_shape:
        return image_volume

    for axes in (
        (2, 0, 1),
        (1, 2, 0),
        (0, 2, 1),
        (1, 0, 2),
        (2, 1, 0),
    ):
        transposed = np.transpose(image_volume, axes)
        if transposed.shape == reference_shape:
            return transposed

    st.error(
        "Could not align the image volume with the label/prediction volume. "
        f"Image shape after modality selection: {image_volume.shape}; reference shape: {reference_shape}."
    )
    st.stop()


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


def best_slice(mask_volume: np.ndarray, axis: int) -> int:
    axes = tuple(dim for dim in range(mask_volume.ndim) if dim != axis)
    mask_area = (mask_volume > 0).sum(axis=axes)
    if mask_area.max() == 0:
        return mask_volume.shape[axis] // 2
    return int(mask_area.argmax())


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


def plot_panel(title: str, image: np.ndarray, cmap: str | None = None):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def projection(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.rot90(np.max(volume, axis=axis))


st.set_page_config(page_title="Missing-Seg Demo", layout="wide")

st.title("Missing-Modality Brain Tumor Segmentation Demo")
st.write(
    "This lightweight demo visualizes one prepared Missing-Seg case. "
    "Because NIfTI files are 3D medical volumes rather than single images, "
    "the app shows both a whole-volume projection and selectable 2D slices."
)

cases = find_cases()
if not cases:
    st.error("No demo case found. Expected files under Missing-Seg/Ours ending in _img, _label, and _pre.")
    st.stop()

case = st.sidebar.selectbox("Case", cases, format_func=lambda item: item["name"])
plane_name = st.sidebar.radio("View plane", list(PLANES.keys()), horizontal=True)
opacity = st.sidebar.slider("Overlay opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

raw_image_volume = load_volume(str(case["image"]))
label_volume = load_volume(str(case["label"]))
prediction_volume = load_volume(str(case["prediction"]))

modality_count = 1
if raw_image_volume.ndim == 4:
    modality_axis = next((axis for axis, size in enumerate(raw_image_volume.shape) if size <= 8), None)
    if modality_axis is not None:
        modality_count = raw_image_volume.shape[modality_axis]

modality = 0
if modality_count > 1:
    modality = st.sidebar.selectbox(
        "Input modality",
        list(range(modality_count)),
        format_func=lambda value: f"Modality {value + 1}",
    )

image_volume = align_image_volume(raw_image_volume, label_volume.shape, modality)

if image_volume.shape != label_volume.shape or image_volume.shape != prediction_volume.shape:
    st.error(
        "The image, label, and prediction volumes have different shapes. "
        f"Image: {image_volume.shape}, label: {label_volume.shape}, prediction: {prediction_volume.shape}"
    )
    st.stop()

axis = PLANES[plane_name]
default_slice = best_slice(label_volume + prediction_volume, axis)
slice_index = st.sidebar.slider(
    "Slice",
    min_value=0,
    max_value=image_volume.shape[axis] - 1,
    value=default_slice,
    help="The default slice is chosen automatically from the largest label/prediction area.",
)

st.caption(
    f"Volume shape: {image_volume.shape}. Showing {plane_name.lower()} slice {slice_index}. "
    "Use the sidebar slider to inspect the full volume."
)

image_slice = normalize_image(slice_volume(image_volume, axis, slice_index))
label_slice = slice_volume(label_volume, axis, slice_index)
prediction_slice = slice_volume(prediction_volume, axis, slice_index)

label_overlay = overlay_mask(image_slice, label_slice, color=(0.85, 0.1, 0.16), alpha=opacity)
prediction_overlay = overlay_mask(image_slice, prediction_slice, color=(0.0, 0.55, 0.58), alpha=opacity)

dice = dice_score(prediction_volume, label_volume)
st.metric("Volume Dice score", f"{dice:.3f}")

st.subheader("Whole-Volume Overview")
overview_cols = st.columns(3)
with overview_cols[0]:
    st.pyplot(plot_panel("MRI maximum-intensity projection", normalize_image(projection(image_volume, axis)), cmap="gray"))
with overview_cols[1]:
    st.pyplot(plot_panel("Ground truth projection", projection(label_volume > 0, axis), cmap="gray"))
with overview_cols[2]:
    st.pyplot(plot_panel("Prediction projection", projection(prediction_volume > 0, axis), cmap="gray"))

st.subheader("Selected Slice")
slice_cols = st.columns(4)
with slice_cols[0]:
    st.pyplot(plot_panel("Input MRI", image_slice, cmap="gray"))
with slice_cols[1]:
    st.pyplot(plot_panel("Ground truth label", label_slice > 0, cmap="gray"))
with slice_cols[2]:
    st.pyplot(plot_panel("Prediction", prediction_slice > 0, cmap="gray"))
with slice_cols[3]:
    st.pyplot(plot_panel("Prediction overlay", prediction_overlay))

st.subheader("Overlay Comparison")
comparison_cols = st.columns(2)
with comparison_cols[0]:
    st.pyplot(plot_panel("Ground truth overlay", label_overlay))
with comparison_cols[1]:
    st.pyplot(plot_panel("Prediction overlay", prediction_overlay))
