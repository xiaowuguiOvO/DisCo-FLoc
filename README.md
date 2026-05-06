# DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization

<div align="center">
  <img src="assets/framework.png" width="100%">
</div>

This repository contains the implementation for paper **DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization**.

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

* **Location**: `checkpoints/depth_anything_v2_vits.pth`
* **Download**: Download the `depth_anything_v2_vits.pth` from  
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

If you need the **trained DisCo-FLoc checkpoints**, you can download them [here](https://drive.google.com/drive/folders/1NmdI9edHnXZbCO11bcKiTgt3bDg1y_c_?usp=sharing).

## Training
### Train DisCo Model

To train the DisCo model, run:

```bash
python training/train_disco_model.py --config DisCo_FLoc.yaml
```

### Train RRP Model

To train the RRP model, run:

```bash
python training/train_rrp_model.py --config RRP.yaml
```


## Evaluation
```bash
python eval/eval_disco_model_s3d.py
```

## Gibson Version
If you want to train on Gibson datasets, you can
    ```
    git checkout main_gibson
    ```
to switch to Gibson version.
