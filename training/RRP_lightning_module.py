import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler

from model.depth_models import DepthPredModels

# Normalization stats, assuming these are constant and can be moved here
RAY_STATS = {"min": torch.tensor([0.0]), "max": torch.tensor([20.0])}

def normalize_data(data, stats):
    stats['min'] = stats['min'].to(data.device)
    stats['max'] = stats['max'].to(data.device)
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata * 2 - 1

def unnormalize_data(ndata, stats):
    stats['min'] = stats['min'].to(ndata.device)
    stats['max'] = stats['max'].to(ndata.device)
    ndata = (ndata + 1) / 2
    return ndata * (stats['max'] - stats['min']) + stats['min']


class FlonaLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.image_log_freq = self.config['image_log_freq']

        self.model = DepthPredModels(config=self.config, encoder_type=self.config["encoder_type"], decoder_type=self.config["decoder_type"])

    def forward(self, func_name, **kwargs):
        return self.model(func_name, **kwargs)

    def training_step(self, batch, batch_idx):
        (
            batch_obs_image,
            pose,
            ray,
            floorplan_img,
            wh,
            *_ # Ignore local_map and neg_local_map if present
        ) = batch
        
        # Get vision features
        features = self.model("encode", obs_img=batch_obs_image)
        output = self.model("decoder_train", depth_cond=features, gt_ray=ray)
        
        pred_d = output["pred"]
        loss = output["loss"]
        # log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.trainer.is_global_zero:
            if (self.global_step + 1) % self.image_log_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    features = self.model("encode", obs_img=batch_obs_image)
                    pred_d = self.model("decoder_inference", depth_cond=features, num_samples=1)
                    
                    action_loss = F.mse_loss(pred_d, ray.squeeze(-1))
                    self.logger.log_metrics({'train_action_loss': action_loss}, step=self.global_step)
                self.model.train()
        return loss

    def validation_step(self, batch, batch_idx):
            (
                batch_obs_image,
                pose,
                ray,
                floorplan_img,
                wh,
                *_
            ) = batch

            features = self.model("encode", obs_img=batch_obs_image)
            pred_d = self.model("decoder_inference", depth_cond=features, num_samples=1)

            val_loss = F.mse_loss(pred_d, ray)
            self.log('val_action_loss', val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.config["lr"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        if self.config.get("warmup", False):
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=self.config["warmup_epochs"],
                after_scheduler=scheduler,
            )
            
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_action_loss",  # 监控 'val_loss' 指标
            "interval": "epoch",  # 每个epoch进行调整
            "frequency": 1,
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero: # 确保只从主进程记录，避免重复
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', current_lr, on_epoch=True, logger=True)

