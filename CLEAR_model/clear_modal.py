import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import numpy as np
import cv2
import wandb
from CLEAR_model.RRP import RRPFeatureExtractor
from CLEAR_model.map_encoder import MapEncoder
from CLEAR_model.viz_utils import visualize_cross_modal_batch

class ClearLocModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.use_cls_token = config.get('use_cls_token', True)
        
        # 1. Image Encoder (Frozen DINOv2 )
        self.image_encoder = RRPFeatureExtractor(
            encoder="vits", 
            checkpoint_path=config.get("dptv2_ckpt_path", "checkpoints/depth_anything_v2_vits.pth")
        )
        
        # 冻结 Image Encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # Image Projection Head (Trainable)
        # CLS token dim for DINOv2-vits is 384
        self.cls_proj = nn.Linear(384, 128)
        
        # Combined projection: Concat(Avg_Seq(128), Proj_CLS(128)) -> 256 -> 128
        # If use_cls_token is False, input dim is 128
        proj_input_dim = 256 if self.use_cls_token else 128
        self.img_proj = nn.Linear(proj_input_dim, 128)
        
        # 2. Map Encoder (Trainable)
        self.map_encoder = MapEncoder(input_channels=1, feature_dim=128)
        
        # Cross Attention: Image Query -> Map Key/Value
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Temperature for InfoNCE (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, obs_img, local_map):
        # 1. Image Features
        # (B, 40, 128), attn, (B, 384)
        img_feats_seq, _, cls_token = self.image_encoder(obs_img)
        
        # Global Pooling: (B, 128)
        img_feat_avg = img_feats_seq.mean(dim=1) 
        
        if self.use_cls_token:
            # Process CLS token
            img_feat_cls = self.cls_proj(cls_token) # (B, 384) -> (B, 128)
            img_feat_cls = F.relu(img_feat_cls)
            
            # Fuse: Concat -> (B, 256)
            img_feat_combined = torch.cat([img_feat_avg, img_feat_cls], dim=1)
        else:
            # Only use average pooling features
            img_feat_combined = img_feat_avg
        
        # Project and Normalize Image
        img_emb = self.img_proj(img_feat_combined)
        img_emb = F.normalize(img_emb, p=2, dim=1)
        
        # 2. Map Features (Spatial Feature Map)
        # (B, 128, H', W')
        map_feat = self.map_encoder(local_map) 
        B, C, H, W = map_feat.shape
        
        # Flatten for Attention: (B, H*W, C)
        map_seq = map_feat.view(B, C, -1).permute(0, 2, 1)
        
        # 3. Cross Attention
        # Query: Image (B, 1, 128)
        img_query = img_emb.unsqueeze(1)
        
        # attn_output: (B, 1, 128)
        map_aligned, _ = self.cross_attn(query=img_query, key=map_seq, value=map_seq)
        
        map_emb = map_aligned.squeeze(1)
        map_emb = F.normalize(map_emb, p=2, dim=1)
        
        return img_emb, map_emb

    def score_candidates(self, img_emb, candidate_maps):
        """
        Compute similarity scores between one image embedding and multiple candidate maps.
        img_emb: (1, 128) or (128,) - Normalized image embedding (Query)
        candidate_maps: (K, C, H, W) - Batch of local map crops
        Returns: (K,) similarity scores
        """
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
            
        K = candidate_maps.shape[0]
        
        # 1. Encode Maps
        map_feats = self.map_encoder(candidate_maps) # (K, 128, H, W)
        B, C, H, W = map_feats.shape
        
        # Flatten: (K, HW, 128)
        map_seq = map_feats.view(K, C, -1).permute(0, 2, 1)
        
        # 2. Expand Image Query
        # img_emb is (1, 128). Need (K, 1, 128)
        img_query = img_emb.unsqueeze(0).expand(K, -1, -1)
        
        # 3. Attention
        map_aligned, _ = self.cross_attn(query=img_query, key=map_seq, value=map_seq)
        
        # 4. Normalize
        map_emb = map_aligned.squeeze(1) # (K, 128)
        map_emb = F.normalize(map_emb, p=2, dim=1)
        
        # 5. Cosine Similarity
        # (1, 128) @ (128, K) -> (1, K)
        scores = (img_emb @ map_emb.t()).squeeze(0)
        
        return scores

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            obs_img = batch[0]
            pose = batch[1]
            floorplan_img = batch[3]
            wh = batch[4]
            local_map = batch[5]
            neg_local_map = batch[6]
            neg_pose = batch[7]
        elif isinstance(batch, dict):
            obs_img = batch['rgb']
            pose = batch['pose']
            floorplan_img = batch['floorplan_image']
            wh = batch['wh']
            local_map = batch['local_map']
            neg_local_map = batch['neg_local_map']
            neg_pose = batch['neg_pose']
        else:
            raise ValueError("Unknown batch type")
        
        batch_size = obs_img.shape[0]
        
        # 1. Image Embeddings (Queries)
        img_feats_seq, _, cls_token = self.image_encoder(obs_img)
        img_feat_avg = img_feats_seq.mean(dim=1)
        
        if self.use_cls_token:
            img_feat_cls = F.relu(self.cls_proj(cls_token))
            img_feat_combined = torch.cat([img_feat_avg, img_feat_cls], dim=1)
        else:
            img_feat_combined = img_feat_avg

        img_emb_all = self.img_proj(img_feat_combined)
        img_emb_all = F.normalize(img_emb_all, p=2, dim=1) # (B, 128)
        
        # 2. Map Features (Keys/Values)
        map_all_input = torch.cat([local_map, neg_local_map], dim=0)
        map_feat_all = self.map_encoder(map_all_input)
        
        B2, C, H, W = map_feat_all.shape
        map_seq_all = map_feat_all.view(B2, C, -1).permute(0, 2, 1)
        
        # 3. Compute Similarity Matrix
        sim_rows = []
        logit_scale = self.logit_scale.exp()
        
        for i in range(batch_size):
            q = img_emb_all[i].view(1, 1, -1).expand(B2, -1, -1)
            map_aligned, _ = self.cross_attn(query=q, key=map_seq_all, value=map_seq_all)
            map_aligned = F.normalize(map_aligned.squeeze(1), p=2, dim=1)
            sim = (img_emb_all[i] * map_aligned).sum(dim=1)
            sim_rows.append(sim)
            
        logits_matrix = torch.stack(sim_rows) * logit_scale
        
        targets = torch.arange(batch_size, device=self.device).long()
        loss = F.cross_entropy(logits_matrix, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('scale', logit_scale)
        
        with torch.no_grad():
            pred = torch.argmax(logits_matrix, dim=1)
            correct = (pred == targets).float().sum()
            acc = correct / batch_size
            self.log('train_acc', acc, prog_bar=True)

            # --- Visualization (Configurable Step) ---
            viz_step = self.config.get('train_viz_step', 1000)
            if batch_idx % viz_step == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
                viz_images = []
                num_viz = min(batch_size, 4)
                
                crop_size_meters = self.config['datasets'].get('local_map_crop_size_meters', 5.0)

                for i in range(num_viz):
                    # Prepare Pred Map info
                    pred_idx = pred[i].item()
                    pred_map_viz = map_all_input[pred_idx] # Tensor
                    
                    # Re-run attention for weights
                    q = img_emb_all[i].view(1, 1, -1).expand(B2, -1, -1)
                    _, attn_weights = self.cross_attn(query=q, key=map_seq_all, value=map_seq_all)
                    viz_attn = attn_weights[i]
                    
                    # Determine Pred Type & Pose
                    pred_type = "unknown"
                    pose_pred_viz = None
                    
                    if pred_idx < batch_size:
                        pred_type = "easy" if pred_idx != i else "correct"
                        # Don't visualize Easy Neg pose on global map
                    else:
                        pred_type = "hard"
                        hn_idx = pred_idx - batch_size
                        pose_pred_viz = neg_pose[hn_idx]
                        
                    img = visualize_cross_modal_batch(
                        logger=self.logger,
                        tag="train_viz",
                        step=self.global_step,
                        obs_img=obs_img[i],
                        floorplan_img=floorplan_img[i],
                        wh=wh[i],
                        local_map=local_map[i],
                        attn_weights=viz_attn,
                        pred_map=pred_map_viz,
                        pose_gt=pose[i],
                        pose_pred=pose_pred_viz,
                        pred_type=pred_type,
                        pred_idx=pred_idx,
                        sample_idx=i,
                        crop_size_meters=crop_size_meters
                    )
                    viz_images.append(img)
                    
                self.logger.experiment.log({"train_viz": viz_images})
            
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            obs_img = batch[0]
            pose = batch[1]
            floorplan_img = batch[3]
            wh = batch[4]
            local_map = batch[5]
        elif isinstance(batch, dict):
            obs_img = batch['rgb']
            pose = batch['pose']
            floorplan_img = batch['floorplan_image']
            wh = batch['wh']
            local_map = batch['local_map']
        else:
            raise ValueError("Unknown batch type")
        
        batch_size = obs_img.shape[0]
        
        # 1. Image Embeddings
        img_feats_seq, _, cls_token = self.image_encoder(obs_img)
        img_feat_avg = img_feats_seq.mean(dim=1)
        
        if self.use_cls_token:
            img_feat_cls = F.relu(self.cls_proj(cls_token))
            img_feat_combined = torch.cat([img_feat_avg, img_feat_cls], dim=1)
        else:
            img_feat_combined = img_feat_avg
            
        img_emb_all = self.img_proj(img_feat_combined)
        img_emb_all = F.normalize(img_emb_all, p=2, dim=1) # (B, 128)
        
        # 2. Map Features (Spatial)
        map_feat_all = self.map_encoder(local_map) # (B, C, H, W)
        B, C, H, W = map_feat_all.shape
        map_seq_all = map_feat_all.view(B, C, -1).permute(0, 2, 1) # (B, HW, 128)
        
        # 3. Compute Similarity Matrix (B, B)
        sim_rows = []
        logit_scale = self.logit_scale.exp()
        
        for i in range(batch_size):
            q = img_emb_all[i].view(1, 1, -1).expand(B, -1, -1)
            map_aligned, attn_weights = self.cross_attn(query=q, key=map_seq_all, value=map_seq_all)
            
            map_aligned = F.normalize(map_aligned.squeeze(1), p=2, dim=1)
            sim = (img_emb_all[i] * map_aligned).sum(dim=1)
            sim_rows.append(sim)
            
        logits_matrix = torch.stack(sim_rows) * logit_scale
        
        targets = torch.arange(batch_size, device=self.device).long()
        loss_val = F.cross_entropy(logits_matrix, targets)
        
        self.log('val_loss', loss_val, prog_bar=True)
        
        with torch.no_grad():
            pred_indices = torch.argmax(logits_matrix, dim=1)
            correct = (pred_indices == targets).float().sum()
            acc_val = correct / batch_size
            self.log('val_acc', acc_val, prog_bar=True)
            
            # --- Visualization ---
            if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
                viz_images = []
                num_viz = min(batch_size, 4)
                
                crop_size_meters = self.config['datasets'].get('local_map_crop_size_meters', 5.0)

                for i in range(num_viz):
                    # Re-run for viz data
                    q = img_emb_all[i].view(1, 1, -1).expand(B, -1, -1)
                    _, attn_weights = self.cross_attn(query=q, key=map_seq_all, value=map_seq_all)
                    viz_attn = attn_weights[i]
                    
                    pred_idx = pred_indices[i].item()
                    pred_map_viz = local_map[pred_idx]
                    
                    pred_type = "easy" if pred_idx != i else "correct"
                    pose_pred_viz = pose[pred_idx]
                    
                    img = visualize_cross_modal_batch(
                        logger=self.logger,
                        tag="val_viz",
                        step=self.current_epoch, 
                        obs_img=obs_img[i],
                        floorplan_img=floorplan_img[i],
                        wh=wh[i],
                        local_map=local_map[i],
                        attn_weights=viz_attn,
                        pred_map=pred_map_viz,
                        pose_gt=pose[i],
                        pose_pred=pose_pred_viz, # Here we use Easy Neg pose if wrong
                        pred_type=pred_type,
                        pred_idx=pred_idx,
                        sample_idx=i,
                        crop_size_meters=crop_size_meters
                    )
                    viz_images.append(img)
                
                self.logger.experiment.log({"val_viz": viz_images})
            
        return loss_val
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)