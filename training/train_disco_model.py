import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from training.DisCo_lightning_module import DisCoLocModel
from DisCo_model.disco_dataset import DisCo_Dataset
from datetime import datetime

def main(config):
    # Setup Dataset
    dataset_cfg = config['datasets']
    fp_size = tuple(dataset_cfg['floorplan_img_size'])
    
    # Ensure num_workers matches system capability
    num_workers = config.get('num_workers', 4)
    
    print("Loading Datasets...")
    # Default pose aug if not in config
    pose_aug_cfg = dataset_cfg.get('pose_aug', {'enable': True, 'trans_range': 25, 'rot_range': 0.26})
    
    train_dataset = DisCo_Dataset(
        data_folder=dataset_cfg['data_folder'],
        data_splits_path=dataset_cfg['data_splits'],
        split="train",
        floorplan_img_size=fp_size,
        pose_aug_params=pose_aug_cfg,
        dataset_cfg = dataset_cfg,
    )
    
    val_split = dataset_cfg.get("val_split", "val")
    with open(dataset_cfg["data_splits"], "r", encoding="utf-8") as f:
        available_splits = yaml.safe_load(f) or {}
    if val_split not in available_splits:
        if "test" in available_splits:
            print(f"[WARN] Requested val split '{val_split}' not found; fallback to 'test'.")
            val_split = "test"
        else:
            raise ValueError(
                f"Validation split '{val_split}' not found in {dataset_cfg['data_splits']}. "
                f"Available splits: {list(available_splits.keys())}"
            )
    print(f"Using validation split: {val_split}")

    val_dataset = DisCo_Dataset(
        data_folder=dataset_cfg['data_folder'],
        data_splits_path=dataset_cfg['data_splits'],
        split=val_split, 
        floorplan_img_size=fp_size,

        pose_aug_params=None, # No aug for validation
        dataset_cfg=dataset_cfg
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Important for contrastive loss stability
    )
    
    # Dummy validation loader (optional)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=True
    )
    
    # Model
    model = DisCoLocModel(config)
    
    # Project folder setup
    project_folder = os.path.join("logs", "disco_runs", config['run_name'])
    os.makedirs(project_folder, exist_ok=True)

    # Logger
    logger = True
    if config.get("use_wandb", False):
        try:
            logger = WandbLogger(project="disco_model", name=config['run_name'])
        except:
            pass
    
    # Checkpoint
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_folder, "checkpoints"),
        filename='{epoch:02d}-{val_acc:.2f}_' + timestamp, # Add timestamp to filename
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )
    
    # Trainer
    # Allow specifying devices from config, e.g., [0], [1], or "auto"
    devices = config.get('gpu_ids', 1)
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        max_epochs=config.get("epochs", 50),
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=project_folder,
        log_every_n_steps=10
    )
    
    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/disco_s3d.yaml", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--run_name", default=None, type=str, help="Experiment name. Defaults to run_name in the config.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override settings for this specific task
    if args.run_name:
        config['run_name'] = args.run_name
    elif "run_name" not in config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['run_name'] = f"disco_model_{timestamp}"

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    
    # Ensure dptv2 path is correct
    if 'dptv2_ckpt_path' not in config:
        config['dptv2_ckpt_path'] = 'checkpoints/depth_anything_v2_vits.pth'
    
    main(config)
