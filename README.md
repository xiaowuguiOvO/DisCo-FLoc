# CLEAR-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization

<p align="center">
    <a href="https://arxiv.org/abs/YOUR_ARXIV_ID"><img src="https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b.svg"></a>
    <a href="https://xiaowuguiovo.github.io/CLEAR-FLoc_Project_Website/"><img src="https://img.shields.io/badge/Project-Website-blue.svg"></a>
    <a href="https://your-paper-link.pdf"><img src="https://img.shields.io/badge/Paper-PDF-green.svg"></a>
</p>

<div align="center">
  <img src="assets/framework.png" width="100%">
</div>

This repository contains the implementation for paper **CLEAR-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization**.

## Environment Setup

1.  **Prerequisites**: Ensure you have Python installed (recommended version >= 3.8).
2.  **Install Dependencies**: Run the following command to install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Prerequisites

Before running the training scripts, you need to prepare the dataset and checkpoints.

### 1. Dataset Preparation

We provide a **Metadata Pack** containing processed labels, poses, and DESDF features. You need to combine this with the raw images from the official Structured3D dataset.

**Step 1: Download Metadata**
*   Download our processed metadata (`datasets_s3d_metadata_only.zip`) from [**[HERE]**](https://drive.google.com/file/d/1Uyl_VoYHTyMi3he5jCuKLNOvgvQMUfYE/view?usp=sharing).
*   Unzip it to the project root. You will get a folder structure like `datasets_s3d/Structured3D/...`.

**Step 2: Download Raw Data**
*   Go to the [**Structured3D Official Website**](https://structured3d-dataset.org/) or [**GitHub**](https://github.com/bertjiazheng/Structured3D) to request access and download the **Full** dataset.
*   You specifically need the **RGB images** and **Perspective maps** (if not included in our metadata).

**Step 3: Merge Data**
*   Place the downloaded RGB images into the corresponding `imgs/` folders in our directory structure.
*   Ensure `map.png` (floorplan) exists in each scene folder (copy from official data if needed).

**Final Structure:**
```text
datasets_s3d/
└── Structured3D/
    ├── split.yaml
    ├── desdf/
    ├── scene_00000/
    │   ├── poses_map.txt   <-- Included in Metadata
    │   ├── depth40.txt     <-- Included in Metadata
    │   ├── map.png         <-- Included in Metadata or Copy from Official
    │   └── imgs/
    │       ├── 000.png     <-- PLACE OFFICIAL IMAGES HERE
    │       └── ...
    └── ...
```

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
