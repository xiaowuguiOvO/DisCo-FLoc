import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb

def visualize_cross_modal_batch(
    logger,
    tag,
    step,
    obs_img,      # Tensor (C, H, W) normalized
    floorplan_img, # Tensor (1, H, W)
    wh,           # Tensor (2,)
    local_map,    # Tensor (1, H, W) GT
    attn_weights, # Tensor (1, HW)
    pred_map,     # Tensor (1, H, W)
    pose_gt,      # Tensor (3,)
    pose_pred=None, # Tensor (3,) optional - for Pred Box/Arrow
    pred_type="unknown",
    pred_idx=0,
    sample_idx=0,
    crop_size_meters=5.0
):
    """
    Helper to visualize a single cross-modal sample.
    Uses standard image coordinate system (origin='upper').
    """
    device = obs_img.device
    
    # 1. RGB Image (Denormalize)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    rgb_viz = obs_img * std + mean
    rgb_viz = torch.clamp(rgb_viz, 0, 1).cpu().permute(1, 2, 0).numpy()
    
    # 2. Maps (CPU numpy)
    fp_viz = floorplan_img[0].cpu().numpy()
    gt_map_viz = local_map[0].cpu().numpy()
    pred_map_viz = pred_map[0].cpu().numpy()
    
    # 3. Attention
    viz_attn = attn_weights.detach().cpu()
    H_feat = int(np.sqrt(viz_attn.shape[1]))
    attn_map = viz_attn.view(H_feat, H_feat).numpy()
    attn_map_resized = cv2.resize(attn_map, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # 4. Pose on Global Map
    fp_h, fp_w = fp_viz.shape
    orig_w, orig_h = wh[0].item(), wh[1].item()
    scale_x = fp_w / orig_w
    scale_y = fp_h / orig_h
    
    pose_x = pose_gt[0].item() * scale_x
    pose_y = pose_gt[1].item() * scale_y
    pose_th = pose_gt[2].item()
    
    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    # --- Global Context ---
    axs[0].imshow(fp_viz, cmap='gray', origin='lower') # origin='upper' default
    axs[0].set_title("Global Context")
    
    arrow_len = 20
    dx = arrow_len * np.cos(pose_th)
    dy = arrow_len * np.sin(pose_th)
    
    # GT Arrow
    axs[0].arrow(pose_x, pose_y, dx, dy, color='lime', width=2, head_width=5)
    
    # Pred Arrow (if pose_pred provided)
    if pose_pred is not None:
        p_x = pose_pred[0].item() * scale_x
        p_y = pose_pred[1].item() * scale_y
        p_th = pose_pred[2].item()
        p_dx = arrow_len * np.cos(p_th)
        p_dy = arrow_len * np.sin(p_th)
        col = 'orange' if pred_type == "hard" else 'cyan'
        axs[0].arrow(p_x, p_y, p_dx, p_dy, color=col, width=2, head_width=5)
    
    # GT Box
    box_size_orig = int(crop_size_meters / 0.02)
    bw = box_size_orig * scale_x
    bh = box_size_orig * scale_y
    rect_gt = patches.Rectangle((pose_x - bw/2, pose_y - bh/2), bw, bh, linewidth=2, edgecolor='lime', facecolor='none')
    axs[0].add_patch(rect_gt)
    
    axs[0].axis('off')

    # --- Query Image ---
    axs[1].imshow(rgb_viz)
    axs[1].set_title("Query Image")
    axs[1].axis('off')
    
    # --- Helper for Local Maps ---
    def draw_ego_arrow(ax):
        cx, cy = 64, 64
        ax.plot(cx, cy, 'ro')
        # With origin='lower', pixel (0,0) is bottom-left.
        # Our crop logic rotates robot front to pixel (W/2, 0) [Top in image coords].
        # In 'lower' view, pixel (W/2, 0) is at the BOTTOM.
        # So Front is Down. Arrow points (0, -20).
        ax.arrow(cx, cy, 0, -20, color='red', width=2, head_width=5)

    # --- GT Crop ---
    axs[2].imshow(gt_map_viz, cmap='gray', origin='lower')
    axs[2].set_title("GT Map Crop")
    draw_ego_arrow(axs[2])
    axs[2].axis('off')
    
    # --- Attention ---
    axs[3].imshow(gt_map_viz, cmap='gray', origin='lower')
    axs[3].imshow(attn_map_resized, cmap='jet', alpha=0.5, origin='lower')
    axs[3].set_title("Attn on GT")
    draw_ego_arrow(axs[3])
    axs[3].axis('off')
    
    # --- Pred Map ---
    axs[4].imshow(pred_map_viz, cmap='gray', origin='lower')
    
    # Title Logic
    title_col = 'red'
    txt = pred_type
    if pred_type == "correct":
        title_col = 'green'
    elif pred_type == "hard":
        title_col = 'orange'
    elif pred_type == "easy":
        title_col = 'red'
        
    axs[4].set_title(f"Pred: {txt} ({pred_idx})", color=title_col)
    draw_ego_arrow(axs[4])
    axs[4].axis('off')
    
    # Log
    img = wandb.Image(fig, caption=f"Step {step} Sample {sample_idx}")
    plt.close(fig)
    return img