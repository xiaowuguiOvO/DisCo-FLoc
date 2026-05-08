# DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization

<p align="center">
    <a href="https://arxiv.org/abs/2601.01822"><img src="https://img.shields.io/badge/arXiv-2601.01822-b31b1b.svg"></a>
    <a href="https://xiaowuguiovo.github.io/DisCo-FLoc_Project_Website/"><img src="https://img.shields.io/badge/Project-Website-blue.svg"></a>
    <a href="https://arxiv.org/pdf/2601.01822.pdf"><img src="https://img.shields.io/badge/Paper-PDF-green.svg"></a>
</p>

<p align="center">
    <strong>Ping Zhong<sup>1</sup>, Shiyong Meng<sup>1</sup>, Bolei Chen<sup>1,*</sup>, Tao Zou<sup>1</sup>, Chaoxu Mu<sup>2</sup>, Jianxin Wang<sup>1</sup></strong>
    <br>
</p>

<div align="center">
  <img src="assets/framework.png" width="100%">
</div>

This repository contains the implementation of DisCo-FLoc, a visual floorplan localization system that combines depth-aware geometric localization with visual-geometric DisCo reranking. The current codebase supports both Structured3D and Gibson training/evaluation in one branch.

## Environment

The code is tested with Python 3.8+ and PyTorch.

```bash
pip install -r requirements.txt
```

The image/depth backbone uses Depth Anything V2. Download the ViT-S checkpoint from [HERE](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth) and place it at:

```text
checkpoints/depth_anything_v2_vits.pth
```

The trained DisCo-FLoc checkpoints can be downloaded from [HERE](https://drive.google.com/drive/folders/1EbVorZNfjDQ6zmYISy_2Xy7rCLOmivXM?usp=sharing).

## Data Layout

Datasets and checkpoints are not included in this repository.

For Structured3D, download the processed metadata pack from [HERE](https://drive.google.com/file/d/1Uyl_VoYHTyMi3he5jCuKLNOvgvQMUfYE/view?usp=sharing), then merge it with the raw RGB images from the official Structured3D release. Request and download the original Structured3D data from the [official website](https://structured3d-dataset.org/) or [GitHub repository](https://github.com/bertjiazheng/Structured3D).

Structured3D should be arranged as:

```text
datasets_s3d/
├── Structured3D/
│   ├── split.yaml
│   ├── scene_00000/
│   │   ├── imgs/
│   │   ├── map.png
│   │   ├── poses_map.txt
│   │   └── depth40.txt
│   └── ...
└── desdf/
    ├── scene_00000/
    │   └── desdf.npy
    └── ...
```

Gibson-F should be arranged as:

```text
datasets_gibson/
└── gibson_f/
    ├── split.yaml
    ├── <scene_name>/
    │   ├── imgs/
    │   ├── map.png
    │   ├── poses_map.txt
    │   ├── depth40.txt
    │   └── desdf.npy
    └── ...
```

Update the paths in `configs/paper/*.yaml` if your local data layout is different.

## Training

Train RRP on Structured3D:

```bash
python training/train_rrp_model.py --config configs/paper/rrp_s3d.yaml
```

Train DisCo on Structured3D:

```bash
python training/train_disco_model.py --config configs/paper/disco_s3d.yaml
```

Train RRP on Gibson-F:

```bash
python training/train_rrp_model.py --config configs/paper/rrp_gibson.yaml
```

Train DisCo on Gibson-F:

```bash
python training/train_disco_model.py --config configs/paper/disco_gibson.yaml
```

## Evaluation

Evaluate Structured3D with RRP + DisCo:

```bash
python eval/eval_disco_model_s3d.py \
  --rrp_model_ckpt checkpoints/RRP_s3d_best.ckpt \
  --disco_model_ckpt checkpoints/DisCo_s3d_best.ckpt
```

Evaluate Gibson-F with RRP + DisCo:

```bash
python eval/eval_disco_model_gibson.py \
  --rrp_model_ckpt checkpoints/RRP_gibson_f_best.ckpt \
  --disco_model_ckpt checkpoints/DisCo_gibson_f_best.ckpt
```
