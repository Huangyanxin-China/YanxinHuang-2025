# Missing-Seg

`Ours` is one case from the test set. For full training and evaluation details,
see the documentation under `code/`.

## Lightweight Streamlit Demo

Live demo: https://missing-seg.streamlit.app/

This repository includes a small visualization demo based on the prepared
`Ours` case:

- input MRI volume
- ground-truth label volume
- predicted segmentation volume

The demo does not require model weights. It visualizes the existing prediction
file and is intended for portfolio and research presentation use.

### Run Locally

```powershell
pip install -r Missing-Seg/requirements-demo.txt
streamlit run Missing-Seg/demo/streamlit_app.py
```

### Demo Features

- choose axial, coronal, or sagittal view
- move through slices with a slider
- compare input MRI, ground truth, prediction, and overlay
- report volume-level Dice score for the included case

For a live online demo, deploy `Missing-Seg/demo/streamlit_app.py` with the
demo requirements on Streamlit Community Cloud or Hugging Face Spaces.
