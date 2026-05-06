# Paper Release Commands

These commands assume the Python environment is already active and the dataset/checkpoint paths in the configs are available from the repository root.

## Training

Train RRP on Structured3D:

```bash
python training/train_rrp_model.py --config configs/paper/rrp_s3d.yaml
```

Train RRP on Gibson-F:

```bash
python training/train_rrp_model.py --config configs/paper/rrp_gibson.yaml
```

Train DisCo on Structured3D:

```bash
python training/train_disco_model.py --config configs/paper/disco_s3d.yaml
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
  --disco_model_ckpt checkpoints/Disco_s3d_best.ckpt
```

Evaluate Gibson-F with RRP + DisCo:

```bash
python eval/eval_disco_model_gibson.py \
  --rrp_model_ckpt checkpoints/RRP_gibson_f_best.ckpt \
  --disco_model_ckpt checkpoints/DisCo_gibson_f_best.ckpt
```

Both DisCo evaluation scripts use SE(2)-aware mode consolidation by default:

```text
mode_source_top_k = 1000
sigma_t = 0.6 m
sigma_theta = 30 deg
lambda_theta = 1.0
rho = 1.0
alpha = 0.5
```
