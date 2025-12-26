import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import tqdm
import yaml
from attrdict import AttrDict
import numpy as np
import cv2
from utils.data_utils import *
from utils.localization_utils import *

from training.RRP_lightning_module import RRPLightningModule
from training.CLEAR_lightning_module import ClearLocModel

# Global Args
parser = argparse.ArgumentParser(description="Eval with Clear Model")
parser.add_argument("--config", "-c", default="CLEAR_FLoc.yaml", type=str)
parser.add_argument("--net_type", type=str, default="rrp")
parser.add_argument("--dataset", type=str, default="Structured3D")
parser.add_argument("--dataset_path", type=str, default="./datasets_s3d/Structured3D/")
parser.add_argument("--desdf_path", type=str, default="./datasets_s3d/desdf/")
parser.add_argument("--ckpt_path", type=str, default="./eval/logs")
parser.add_argument("--visualize", action="store_true")

# New Args for CrossModal
parser.add_argument("--rrp_model_ckpt", type=str, default='logs\\rrp_runs\\rrp_model_20251226_013633\\checkpoints\\epoch=37-val_action_loss=0.74.ckpt', help="Path to RRP checkpoint")
parser.add_argument("--clear_model_ckpt", type=str, default='logs\clear_runs\clear_model_20251225_195749\checkpoints\epoch=15-val_acc=0.75_20251225_195753.ckpt', help="Path to CLEAR checkpoint")
parser.add_argument("--top_k", type=int, default=100, help="Number of candidates to re-rank")
parser.add_argument("--alpha", type=float, default=0.5, help="Weight of semantic score")
parser.add_argument("--clear_only", action="store_true", help="If True, ignore geometric probability and only use cross-modal score")
parser.add_argument("--all_imgs", default=True, help="If True, evaluate all images as reference frames in a sliding window manner (dense evaluation). Default to False (sparse evaluation).")

# Single Image Debugging
parser.add_argument("--scene_name", type=str, default=None, help="Debug: Specific scene name")
parser.add_argument("--img_id", type=int, default=None, help="Debug: Specific image ID within the scene")

args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
data_config = config["datasets"]

def crop_local_map(map_img, x, y, theta, crop_size_meters, res=0.02, output_size=128):
    """
    Standalone function to crop local map.
    x, y: pixels
    theta: radians
    """
    x = float(x)
    y = float(y)
    crop_size_px = int(crop_size_meters / res)
    pad = crop_size_px    
    if torch.is_tensor(map_img):
        map_img = map_img.cpu().numpy()
        
    H, W = map_img.shape[:2]
    # Ensure grayscale
    if len(map_img.shape) == 3:
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        
    map_padded = cv2.copyMakeBorder(map_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    center = (x + pad, y + pad)
    angle_deg = np.degrees(theta)
    
    # Rotate so robot heading points UP (-y)
    # S3D: 0 is Right (+x). Up is -90 deg relative to Right.
    # Rotation angle for warpAffine (CCW): angle + 90
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg + 90, 1.0)
    
    rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
    rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]
    
    local_map = cv2.warpAffine(
        map_padded, rot_matrix, (crop_size_px, crop_size_px), 
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    
    if crop_size_px != output_size:
        local_map = cv2.resize(local_map, (output_size, output_size), interpolation=cv2.INTER_AREA)
        
    return local_map

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # --- 1. Load RRP Model ---
    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt , map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()
    # --- 2. Load CLEAR Model ---
    cl_model = ClearLocModel.load_from_checkpoint(args.clear_model_ckpt, config=config, map_location=device)  
    cl_model.to(device)
    cl_model.eval()

    # --- 3. Setup Dataset ---
    L = 3
    dataset_dir = args.dataset_path
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
        
    test_set = GridSeqDataset(
        dataset_dir,
        split.test,
        L=L,
        depth_dir=dataset_dir,
        depth_suffix="depth40", # Assuming S3D
        add_rp=False,
        net_type="rrp",
        all_imgs=args.all_imgs, # s3d evalutate all images
    )

    # Load desdfs and maps
    desdf_path = args.desdf_path
    print("Loading desdfs...")
    desdfs = {}
    maps = {}
    gt_poses = {} # Map coordinates

    # S3D Params
    F_W = 1 / np.tan(0.698132) / 2
    map_res = 0.02
    desdf_stride = 5 # 0.1 / 0.02

    for scene in tqdm.tqdm(test_set.scene_names):
        # DESDF
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 20] = 20
        
        # MAP
        maps[scene] = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        
        # POSES
        with open(os.path.join(dataset_dir, scene, "poses_map.txt"), "r") as f:
            poses = []
            for line in f:
                parts = list(map(float, line.strip().split()))
                poses.append(parts[:3]) # x, y, th
            gt_poses[scene] = np.array(poses, dtype=np.float32)

    import matplotlib.pyplot as plt

    # --- Evaluation Loop ---
    acc_record = []
    acc_orn_record = []
    
    # Stats for semantic improvement
    improved_count = 0
    worsened_count = 0

    # Create visualization directories
    if args.visualize:
        viz_dir = os.path.join(args.ckpt_path, "visualizations")
        os.makedirs(os.path.join(viz_dir, "improved"), exist_ok=True)
        os.makedirs(os.path.join(viz_dir, "degraded"), exist_ok=True)
        # Add a folder for single image debug
        if args.scene_name:
             os.makedirs(os.path.join(viz_dir, "debug"), exist_ok=True)
        print(f"Saving visualizations to {viz_dir}")

    print("Starting Evaluation...")
    
    # Determine Loop Range
    if args.scene_name is not None and args.img_id is not None:
        try:
            scene_idx = test_set.scene_names.index(args.scene_name)
            start_idx = test_set.scene_start_idx[scene_idx]
            
            # Adjust for sparse/dense evaluation mapping if necessary
            # The test_set itself usually maps index 0..N to specific frames
            # Assuming img_id passed is the index relative to the scene in the dataset object
            
            target_idx = start_idx + args.img_id
            
            if target_idx >= len(test_set) or (scene_idx + 1 < len(test_set.scene_start_idx) and target_idx >= test_set.scene_start_idx[scene_idx+1]):
                print(f"Error: img_id {args.img_id} out of bounds for scene {args.scene_name}")
                return
                
            loop_range = [target_idx]
            print(f"Debug Mode: Evaluating only {args.scene_name}, Image {args.img_id} (Global Index {target_idx})")
        except ValueError:
            print(f"Error: Scene {args.scene_name} not found in test set.")
            return
    else:
        loop_range = range(len(test_set))

    for data_idx in tqdm.tqdm(loop_range):
        # Get data
        data = test_set[data_idx]
        
        # Meta info
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
        
        # GT Pose
        # Handle index mapping based on sampling strategy
        if args.all_imgs:
             ref_idx = idx_within_scene
        else:
             ref_idx = idx_within_scene * (L + 1) + L
             
        ref_pose_map = gt_poses[scene][ref_idx, :]
        
        # Transform GT to DESDF frame
        desdf_data = desdfs[scene]
        gt_pose_desdf = ref_pose_map.copy()
        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf_data["l"]) / desdf_stride
        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf_data["t"]) / desdf_stride

        # Prepare Input
        ref_img = data["ref_img"] # (C, H, W)
        obs_img_tensor = data["obs_tensor"].unsqueeze(0).to(device) # (1, C, H, W) 
        
        # --- 1. Geometric Prediction (RRP) ---
        with torch.no_grad():
            feat = rrp_model("encode", obs_img=obs_img_tensor)
            pred_depths_tensor = rrp_model("decoder_inference", depth_cond=feat, return_uncertainty=True)
            pred_depths = pred_depths_tensor.squeeze(0).detach().cpu().numpy()
            
            # Get Rays
            pred_rays = get_ray_from_depth(pred_depths, V=9, F_W=F_W)
            pred_rays = torch.tensor(pred_rays, device="cpu")
            
            # Localize (Get Probability Volume)
            # We need prob_vol, not just the single best prediction
            prob_vol, prob_dist, orientations, _ = localize(
                torch.tensor(desdf_data["desdf"]), pred_rays, return_np=False
            )
            # prob_dist: (H_desdf, W_desdf)
            
        # Get UnLoc (Geo-only) Prediction first for comparison
        geo_pred_y, geo_pred_x = torch.where(prob_dist == prob_dist.max())
        if geo_pred_y.numel() > 0:
            geo_pred = np.array([geo_pred_x[0].item(), geo_pred_y[0].item()])
            geo_error = np.linalg.norm(geo_pred - gt_pose_desdf[:2]) * 0.1 # 0.1m per desdf unit
        else:
            geo_error = 999.0
            geo_pred = np.array([0, 0])

        # --- 2. Semantic Re-ranking ---
        
        # Get image embedding (only once)
        with torch.no_grad():
            img_emb, _ = cl_model(obs_img_tensor, torch.zeros(1, 1, 128, 128).to(device)) # Dummy map
            # We only need img_emb
        
        # Flatten prob_dist to find Top-K candidates
        flat_probs = prob_dist.flatten()
        topk_vals, topk_indices = torch.topk(flat_probs, k=min(args.top_k, flat_probs.numel()))
        
        # Convert indices back to (y, x) in desdf frame
        H_d, W_d = prob_dist.shape
        topk_y = topk_indices // W_d
        topk_x = topk_indices % W_d
        
        # Prepare batch for Map Encoder
        local_maps = []
        valid_indices = []
        
        scene_map = maps[scene]
        
        for i in range(len(topk_indices)):
            py, px = topk_y[i].item(), topk_x[i].item()
            
            # Get orientation from UnLoc result for this cell
            # orientations is (H, W), stores index 0..35
            orn_idx = orientations[py, px].item()
            theta = (orn_idx / 36) * 2 * np.pi
            
            # Convert desdf (px, py) back to map (map_x, map_y)
            map_x = px * desdf_stride + desdf_data["l"]
            map_y = py * desdf_stride + desdf_data["t"]
            
            # Crop
            crop_local_map_size = data_config.get("local_map_crop_size_meters", 5.0) # meters
            lmap = crop_local_map(scene_map, map_x, map_y, theta, crop_size_meters=crop_local_map_size)
            lmap_tensor = torch.from_numpy(lmap).float() / 255.0
            local_maps.append(lmap_tensor.unsqueeze(0)) # (1, H, W)
            valid_indices.append(i)
            
        if local_maps:
            local_maps_batch = torch.stack(local_maps).to(device) # (K, 1, 128, 128)
            
            with torch.no_grad():
                # Use model's internal attention logic to score candidates
                sim_scores = cl_model.score_candidates(img_emb, local_maps_batch)
                
                # Fusion
                geo_probs = topk_vals.to(device)
                
                semantic_weight = torch.exp(sim_scores * args.alpha)
                
                if args.clear_only:
                    final_scores = semantic_weight
                else:
                    final_scores = geo_probs * semantic_weight
                
                # Find best in Top-K
                best_idx_in_k = torch.argmax(final_scores).item()
                
                # Retrieve original desdf coordinates
                final_i = valid_indices[best_idx_in_k]
                final_y = topk_y[final_i].item()
                final_x = topk_x[final_i].item()
                
                # Get Pose
                final_orn_idx = orientations[final_y, final_x].item()
                final_orn = (final_orn_idx / 36) * 2 * np.pi
                pose_pred = np.array([final_x, final_y, final_orn])
        else:
            # Fallback
            pose_pred = np.array([geo_pred_x[0].item(), geo_pred_y[0].item(), 0.0])
            final_scores = torch.tensor([], device=device)
            semantic_weight = torch.tensor([], device=device)

        # --- Accuracy ---
        acc = np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.1
        acc_record.append(acc)
        
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)
        
        # Compare (Critical changes crossing 1m threshold)
        is_improved = (geo_error > 1.0 and acc < 1.0)
        is_degraded = (geo_error < 1.0 and acc > 1.0)
        
        if is_improved:
            improved_count += 1
        elif is_degraded:
            worsened_count += 1

        # Visualization
        # Force viz if debug mode (scene_name is set)
        should_viz = args.visualize and (is_improved or is_degraded or args.scene_name is not None)
        
        if should_viz:
            if args.scene_name:
                save_folder = "debug"
            else:
                save_folder = "improved" if is_improved else "degraded"
            
            # Prepare Maps
            H, W = prob_dist.shape
            
            # 1. RRP Map (Dense)
            map_rrp = prob_dist.cpu().numpy()
            
            # 2. CLEAR Map (Sparse)
            map_cl = np.zeros((H, W), dtype=np.float32)
            cl_vals = semantic_weight.cpu().numpy() if len(semantic_weight) > 0 else []
            
            # Find max CL score point
            cl_pred_pt = None
            if len(cl_vals) > 0:
                best_cl_idx = np.argmax(cl_vals)
                # Map back to valid_indices -> topk_indices -> (y, x)
                # Note: valid_indices maps 0..K_valid to index in topk_indices
                final_i_cl = valid_indices[best_cl_idx]
                flat_idx_cl = topk_indices[final_i_cl].item()
                y_cl, x_cl = flat_idx_cl // W, flat_idx_cl % W
                cl_pred_pt = np.array([x_cl, y_cl])
            
            for k_idx, val_idx in enumerate(valid_indices):
                flat_idx = topk_indices[val_idx].item()
                y, x = flat_idx // W, flat_idx % W
                map_cl[y, x] = cl_vals[k_idx]
            
            # 3. Combined Map (Dense with Top-K modified)
            map_final = map_rrp.copy()
            final_vals = final_scores.cpu().numpy() if len(final_scores) > 0 else []
            
            for k_idx, val_idx in enumerate(valid_indices):
                flat_idx = topk_indices[val_idx].item()
                y, x = flat_idx // W, flat_idx % W
                map_final[y, x] = final_vals[k_idx]

            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Helper
            def viz_map(ax, m, title, pred_pt=None, pred_orn=None, vmin=None, vmax=None):
                im = ax.imshow(m, origin='lower', cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # GT Arrow
                ax.quiver(gt_pose_desdf[0], gt_pose_desdf[1], np.cos(gt_pose_desdf[2]), np.sin(gt_pose_desdf[2]),
                          color='lime', width=0.02, scale_units='xy', scale=0.1,
                          headwidth=10, headlength=10, label='GT', zorder=3)
                
                # Pred Arrow
                if pred_pt is not None and pred_orn is not None:
                     ax.quiver(pred_pt[0], pred_pt[1], np.cos(pred_orn), np.sin(pred_orn),
                               color='Cyan', width=0.02, scale_units='xy', scale=0.1,
                               headwidth=10, headlength=10, label='Pred', zorder=3)
                # ax.legend()
            
            # Prepare orientations for Viz
            # 1. CL Pred Orn
            cl_pred_orn = None
            if cl_pred_pt is not None:
                # Need to look up orientation for the CL point
                # Since cl_pred_pt came from flat_idx_cl
                # flat_idx_cl = y * W + x
                y_c, x_c = int(cl_pred_pt[1]), int(cl_pred_pt[0])
                cl_orn_idx = orientations[y_c, x_c].item()
                cl_pred_orn = (cl_orn_idx / 36) * 2 * np.pi

            # 2. RRP Pred Orn
            rrp_pred_orn = None
            if geo_error < 900: # If valid
                y_g, x_g = int(geo_pred[1]), int(geo_pred[0])
                rrp_orn_idx = orientations[y_g, x_g].item()
                rrp_pred_orn = (rrp_orn_idx / 36) * 2 * np.pi
            
            # 3. Final Pose Orn is already in pose_pred[2]

            # Calculate vmin/vmax for each map for better contrast
            # Ensure vmax is at least a small positive number to avoid div by zero if map is all zeros
            cl_vmax = np.max(map_cl) if np.max(map_cl) > 0 else 1e-6
            rrp_vmax = np.max(map_rrp) if np.max(map_rrp) > 0 else 1e-6
            final_vmax = np.max(map_final) if np.max(map_final) > 0 else 1e-6

            # 1. CLEAR Score Map (Semantic Weights)
            cl_error_str = ""
            if cl_pred_pt is not None:
                cl_err = np.linalg.norm(cl_pred_pt - gt_pose_desdf[:2]) * 0.1
                cl_error_str = f" (Err: {cl_err:.2f}m)"
            
            viz_map(axs[0], map_cl, f"Disambiguation Score Map{cl_error_str}", pred_pt=cl_pred_pt, pred_orn=cl_pred_orn, vmin=1, vmax=cl_vmax)
            
            # 2. RRP
            viz_map(axs[1], map_rrp, f"RRP Prob (Err: {geo_error:.2f}m)", pred_pt=geo_pred, pred_orn=rrp_pred_orn)
            
            # 3. Combined
            viz_map(axs[2], map_final, f"CLEAR Combined Prob (Err: {acc:.2f}m)", pred_pt=pose_pred[:2], pred_orn=pose_pred[2], vmin=None, vmax=final_vmax)
            
            plt.suptitle(f"{scene} - Frame {idx_within_scene} ({save_folder.upper()})")
            plt.tight_layout()
            
            out_path = os.path.join(viz_dir, save_folder, f"{scene}_{idx_within_scene}.png")
            plt.savefig(out_path)
            plt.close()

    # Summary
    acc_record = np.array(acc_record)
    acc_orn_record = np.array(acc_orn_record)
    total_samples = len(acc_record)
    
    print("\n" + "="*30)
    print(f"Results with CLEAR (k={args.top_k}, alpha={args.alpha})")
    print(f"1m recall = {np.sum(acc_record < 1) / total_samples:.4f}")
    print(f"0.5m recall = {np.sum(acc_record < 0.5) / total_samples:.4f}")
    print(f"0.1m recall = {np.sum(acc_record < 0.1) / total_samples:.4f}")
    print(f"1m 30 deg recall = {np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / total_samples:.4f}")
    print(f"1m 10 deg recall = {np.sum(np.logical_and(acc_record < 1, acc_orn_record < 10)) / total_samples:.4f}")
    print("-" * 20)
    
    imp_pct = improved_count / total_samples * 100
    wor_pct = worsened_count / total_samples * 100
    print(f"Improved samples: {improved_count} ({imp_pct:.2f}%)")
    print(f"Worsened samples: {worsened_count} ({wor_pct:.2f}%)")
    print("="*30)

    # --- Log Results to File ---
    import datetime
    import json
    
    log_file = "eval/eval_history.txt"
    
    # 1. Current Result
    current_result = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": args.clear_model_ckpt,
        "k": args.top_k,
        "alpha": args.alpha,
        "1m_recall": np.sum(acc_record < 1) / total_samples,
        "0.5m_recall": np.sum(acc_record < 0.5) / total_samples,
        "0.1m_recall": np.sum(acc_record < 0.1) / total_samples,
        "1m_30deg_recall": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / total_samples,
        "1m_10deg_recall": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 10)) / total_samples,
        "improved": f"{improved_count} ({imp_pct:.2f}%)",
        "worsened": f"{worsened_count} ({wor_pct:.2f}%)"
    }
    
    # 2. Read Existing
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            # Attempt to parse line as JSON
                            entry = json.loads(line)
                            history.append(entry)
                        except json.JSONDecodeError:
                            pass # Skip malformed lines
        except Exception as e:
            print(f"Warning: Could not read history file: {e}")

    # 3. Append and Sort
    history.append(current_result)
    # Sort by 1m_recall descending
    history.sort(key=lambda x: x.get("1m_recall", 0), reverse=True)
    
    # 4. Write Back
    with open(log_file, "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Result logged to {log_file} (Sorted by 1m Recall)")

if __name__ == "__main__":
    evaluate()
