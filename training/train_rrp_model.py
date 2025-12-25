import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from training.RRP_lightning_module import FlonaLightningModule
from model.lightning_datamodule import FlonaDataModule
from training.callbacks import ImageLoggerCallback

def main(config, logger=True):
    # ==============================Data Module==============================
    data_module = FlonaDataModule(
        data_config=config["datasets"],
        batch_size=config["batch_size"],
        eval_batch_size=config.get("eval_batch_size"),
        num_workers=config["num_workers"],
    )

    model = FlonaLightningModule(
        config=config,
    )

    # ==============================Training==============================
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["project_folder"], "checkpoints"),
        filename='{epoch:02d}-{val_action_loss:.2f}',
        save_top_k=3,
        monitor='val_action_loss',
        mode='min',
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-4)

    image_log_callback = ImageLoggerCallback(
        num_images_log=config.get("num_images_log", 8),
        image_log_freq=config.get("image_log_freq", 10)
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.get("gpu_ids", [0]) if torch.cuda.is_available() else 1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.is_available() and len(config.get("gpu_ids", [0])) > 1 else 'auto',
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, swa_callback, image_log_callback],
        logger=logger,
        default_root_dir=config["project_folder"],
        log_every_n_steps=config["wandb_log_freq"],
    )

    # Find checkpoint to resume from
    ckpt_path = None
    if "load_run" in config and "load_checkpoint" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        ckpt_path = os.path.join(load_project_folder, "checkpoints", config["load_checkpoint"])
        
        if os.path.exists(ckpt_path):
            print(f"✅ Checkpoint found! Resuming training from: {ckpt_path}")
        else:
            print(f"❌ Warning: Checkpoint path given but not found: {ckpt_path}")
            print("   Starting training from scratch.")
            ckpt_path = None
    else:
        print("ℹ️ No 'load_run' or 'load_checkpoint' found in config. Starting training from scratch.")


    # Start training
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    print("Done!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    parser.add_argument(
        "--config",
        "-c",
        default="flona.yaml",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Project folder setup
    run_name = config.get("run_name", "flona_run") + "_" + time.strftime("%Y%m%d_%H%M%S")
    config["run_name"] = run_name
    project_folder = os.path.join(
        "logs", config.get("project_name", "flona_project"), run_name
    )
    config["project_folder"] = project_folder
    os.makedirs(project_folder, exist_ok=True)

    # Logger
    logger = True # Default logger (CSV/TensorBoard)
    if config.get("use_wandb", False):
        try:
            logger = WandbLogger(
                project=config.get("project_name", "flona_project"),
                name=run_name,
                config=config,
            )
            # The WandbLogger will handle saving the config
        except ImportError:
            print("Warning: 'use_wandb' is set to True in config, but the 'wandb' library is not installed. Please install it with 'pip install wandb' to enable logging.")
            logger = True # Fallback to default logger

    main(config, logger=logger)
