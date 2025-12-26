# CLEAR-FLoc

<div align="center">
  <img src="assets/framework.png" width="100%">
</div>

This repository contains the implementation for **CLEAR-FLoc**.

## Environment Setup

1.  **Prerequisites**: Ensure you have Python installed (recommended version >= 3.8).
2.  **Install Dependencies**: Run the following command to install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Prerequisites

Before running the training scripts, ensure you have the following data and checkpoints in place:

### 1. Dataset
The project expects the **Structured3D(Full)** dataset.
*   **Location**: `datasets_s3d/Structured3D`
*   **Split Config**: `datasets_s3d/Structured3D/split.yaml`

### 2. Pretrained Checkpoints
You need the **Depth Anything V2** checkpoint (ViT-S version).
*   **Location**: `checkpoints/depth_anything_v2_vits.pth`
*   **Download**: Download the `depth_anything_v2_vits.pth` [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth).

## Training
### Train CLEAR Model

To train the CLEAR model, run:

```bash
python training/train_clear_model.py --config CLEAR_FLoc.yaml
```

### Train RRP Model

To train the RRP model, run:

```bash
python training/train_rrp_model.py --config RRP.yaml
```


## Evaluation
```bash
python eval/eval_clear_model_s3d.py
```
