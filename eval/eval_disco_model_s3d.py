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
from training.DisCo_lightning_module import DisCoLocModel

# Global Args
parser = argparse.ArgumentParser(description="Eval with DisCo Model")
parser.add_argument("--config", "-c", default="configs/paper/disco_s3d.yaml", type=str)
parser.add_argument("--net_type", type=str, default="rrp")
parser.add_argument("--dataset", type=str, default="Structured3D")
parser.add_argument("--dataset_path", type=str, default="./datasets_s3d/Structured3D/")
parser.add_argument("--desdf_path", type=str, default="./datasets_s3d/desdf/")
parser.add_argument("--ckpt_path", type=str, default="./eval/logs")
parser.add_argument("--visualize", action="store_true")

# New Args for CrossModal
parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_s3d_best.ckpt", help="Path to RRP checkpoint")
parser.add_argument("--disco_model_ckpt", type=str, default="checkpoints/DisCo_s3d_best.ckpt", help="Path to DisCo checkpoint")
parser.add_argument("--alpha", type=float, default=0.5, help="Weight of semantic score")
parser.add_argument("--all_imgs", default=True, help="If True, evaluate all images as reference frames in a sliding window manner (dense evaluation). Default to False (sparse evaluation).")
parser.add_argument("--fov", type=float, default=80.0, help="Horizontal field of view used to convert depth40 to rays.")
parser.add_argument(
    "--mode_source_top_k",
    type=int,
    default=1000,
    help="Number of RRP candidates used to extract SE(2) representative mode hypotheses.",
)
parser.add_argument(
    "--se2_sigma_t_m",
    type=float,
    default=0.6,
    help="Translation scale in meters for SE(2)-aware mode consolidation.",
)
parser.add_argument(
    "--se2_sigma_theta_deg",
    type=float,
    default=30.0,
    help="Angular scale in degrees for SE(2)-aware mode consolidation.",
)
parser.add_argument(
    "--se2_angle_weight",
    type=float,
    default=1.0,
    help="Angular term weight for SE(2)-aware mode consolidation.",
)
parser.add_argument(
    "--se2_mode_radius",
    type=float,
    default=1.0,
    help="Threshold on normalized SE(2) distance for assigning candidates to a mode basin.",
)
parser.add_argument(
    "--gpu_localize",
    action="store_true",
    default=True,
    help="Run DESDF localization on GPU with localize_fast. Falls back to CPU if CUDA is unavailable.",
)
parser.add_argument(
    "--cpu_localize",
    dest="gpu_localize",
    action="store_false",
    help="Disable GPU DESDF localization and use the original CPU localize path.",
)

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


def wrap_angle_rad(angle):
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def consolidate_se2_modes(
    topk_vals,
    topk_indices,
    orientations,
    width,
    meters_per_cell,
    sigma_t_m,
    sigma_theta_deg,
    angle_weight,
    mode_radius,
):
    if topk_indices.numel() == 0:
        keep_positions = torch.empty(0, dtype=torch.long)
        basin_sizes = torch.empty(0, dtype=torch.long)
        return topk_vals, topk_indices, keep_positions, basin_sizes

    if mode_radius <= 0 or sigma_t_m <= 0 or sigma_theta_deg <= 0:
        keep_positions = torch.arange(topk_indices.shape[0], dtype=torch.long)
        basin_sizes = torch.ones(topk_indices.shape[0], dtype=torch.long)
        return topk_vals, topk_indices, keep_positions, basin_sizes

    sigma_theta = np.deg2rad(sigma_theta_deg)
    topk_y = (topk_indices // width).to(torch.float32)
    topk_x = (topk_indices % width).to(torch.float32)
    topk_theta_idx = orientations[topk_y.long(), topk_x.long()].to(torch.float32)
    topk_theta = topk_theta_idx / 36.0 * 2.0 * torch.pi

    assigned = torch.zeros(topk_indices.shape[0], dtype=torch.bool)
    keep_positions = []
    basin_sizes = []

    for idx in range(topk_indices.shape[0]):
        if assigned[idx]:
            continue

        dx_m = (topk_x - topk_x[idx]) * meters_per_cell
        dy_m = (topk_y - topk_y[idx]) * meters_per_cell
        dtheta = wrap_angle_rad(topk_theta - topk_theta[idx])
        spatial_term = (dx_m * dx_m + dy_m * dy_m) / (sigma_t_m * sigma_t_m)
        angular_term = angle_weight * (dtheta * dtheta) / (sigma_theta * sigma_theta)
        se2_dist = torch.sqrt(spatial_term + angular_term)
        basin_mask = (~assigned) & (se2_dist <= mode_radius)

        keep_positions.append(idx)
        basin_sizes.append(int(basin_mask.sum().item()))
        assigned |= basin_mask

    keep_positions = torch.tensor(keep_positions, dtype=torch.long)
    basin_sizes = torch.tensor(basin_sizes, dtype=torch.long)

    return (
        topk_vals[keep_positions],
        topk_indices[keep_positions],
        keep_positions,
        basin_sizes,
    )

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # --- 1. Load RRP Model ---
    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt , map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()
    # --- 2. Load DisCo Model ---
    cl_model = DisCoLocModel.load_from_checkpoint(args.disco_model_ckpt, config=config, map_location=device)  
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
    desdf_tensors = {}
    maps = {}
    gt_poses = {} # Map coordinates

    # S3D Params
    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    map_res = 0.02
    desdf_stride = 5 # 0.1 / 0.02

    for scene in tqdm.tqdm(test_set.scene_names):
        # DESDF
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 20] = 20
        if args.gpu_localize and device == "cuda":
            desdf_tensors[scene] = torch.tensor(
                desdfs[scene]["desdf"], dtype=torch.float32, device=device
            )
        
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
    recall_1m_hits = 0
    
    # Stats for semantic improvement
    improved_count = 0
    worsened_count = 0
    meters_per_desdf_cell = desdf_stride * map_res
    mode_pool_sizes = []
    mode_rep_sizes = []
    mode_basin_sizes = []

    # Create visualization directories
    if args.visualize:
        viz_dir = os.path.join(args.ckpt_path, "visualizations")
        os.makedirs(os.path.join(viz_dir, "improved"), exist_ok=True)
        os.makedirs(os.path.join(viz_dir, "degraded"), exist_ok=True)
        # Add a folder for single image debug
        if args.scene_name:
             os.makedirs(os.path.join(viz_dir, "debug"), exist_ok=True)
        print(f"Saving visualizations to {viz_dir}")

    print(
        "Starting Evaluation with SE(2)-Aware Mode Consolidation "
        f"(source_top_k={args.mode_source_top_k}, sigma_t={args.se2_sigma_t_m:.2f}m, "
        f"sigma_theta={args.se2_sigma_theta_deg:.1f}deg, rho={args.se2_mode_radius:.2f})..."
    )
    print(
        "DESDF localize: "
        f"{'gpu_fast' if args.gpu_localize and device == 'cuda' else 'cpu_original'}"
    )
    
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

    eval_pbar = tqdm.tqdm(loop_range, desc="Evaluating")
    for data_idx in eval_pbar:
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
            if args.gpu_localize and device == "cuda":
                prob_dist, orientations, _ = localize_fast(
                    desdf_tensors[scene],
                    torch.tensor(pred_rays, dtype=torch.float32, device=device),
                    return_np=False,
                )
                prob_dist = prob_dist.cpu()
                orientations = orientations.cpu()
            else:
                pred_rays = torch.tensor(pred_rays, device="cpu")
                _, prob_dist, orientations, _ = localize(
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
            img_tokens = cl_model.encode_image(obs_img_tensor)
        
        # Flatten prob_dist to find Top-K candidates
        flat_probs = prob_dist.flatten()
        topk_vals, topk_indices = torch.topk(
            flat_probs, k=min(args.mode_source_top_k, flat_probs.numel())
        )
        
        # Convert indices back to (y, x) in desdf frame
        H_d, W_d = prob_dist.shape
        mode_pool_sizes.append(len(topk_indices))
        topk_vals, topk_indices, _, basin_sizes = consolidate_se2_modes(
            topk_vals,
            topk_indices,
            orientations,
            width=W_d,
            meters_per_cell=meters_per_desdf_cell,
            sigma_t_m=args.se2_sigma_t_m,
            sigma_theta_deg=args.se2_sigma_theta_deg,
            angle_weight=args.se2_angle_weight,
            mode_radius=args.se2_mode_radius,
        )
        mode_rep_sizes.append(len(topk_indices))
        if len(basin_sizes) > 0:
            mode_basin_sizes.append(float(basin_sizes.float().mean().item()))

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
                sim_scores = cl_model.score_candidates(img_tokens, local_maps_batch)
                
                # Fusion
                geo_probs = topk_vals.to(device)
                
                semantic_weight = torch.exp(sim_scores * args.alpha)
                
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
        if acc < 1.0:
            recall_1m_hits += 1
         
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)

        current_1m_recall = recall_1m_hits / len(acc_record)
        postfix = {"1m_recall": f"{current_1m_recall:.4f}"}
        if mode_rep_sizes:
            postfix["avg_rep_k"] = f"{(sum(mode_rep_sizes) / len(mode_rep_sizes)):.1f}"
        eval_pbar.set_postfix(postfix)
        
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
            
            # 2. DisCo Map (Sparse)
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

            # 1. DisCo Score Map (Semantic Weights)
            cl_error_str = ""
            if cl_pred_pt is not None:
                cl_err = np.linalg.norm(cl_pred_pt - gt_pose_desdf[:2]) * 0.1
                cl_error_str = f" (Err: {cl_err:.2f}m)"
            
            viz_map(axs[0], map_cl, f"Disambiguation Score Map{cl_error_str}", pred_pt=cl_pred_pt, pred_orn=cl_pred_orn, vmin=1, vmax=cl_vmax)
            
            # 2. RRP
            viz_map(axs[1], map_rrp, f"RRP Prob (Err: {geo_error:.2f}m)", pred_pt=geo_pred, pred_orn=rrp_pred_orn)
            
            # 3. Combined
            viz_map(axs[2], map_final, f"DisCo Combined Prob (Err: {acc:.2f}m)", pred_pt=pose_pred[:2], pred_orn=pose_pred[2], vmin=None, vmax=final_vmax)
            
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
    avg_pool_k = np.mean(mode_pool_sizes) if mode_pool_sizes else 0.0
    avg_rep_k = np.mean(mode_rep_sizes) if mode_rep_sizes else 0.0
    avg_basin_size = np.mean(mode_basin_sizes) if mode_basin_sizes else 0.0
    print(
        "Results with DisCo SE(2)-Aware Mode Consolidation "
        f"(source_top_k={args.mode_source_top_k}, sigma_t={args.se2_sigma_t_m:.2f}m, "
        f"sigma_theta={args.se2_sigma_theta_deg:.1f}deg, rho={args.se2_mode_radius:.2f}, "
        f"avg_rep_k={avg_rep_k:.1f}, avg_basin_size={avg_basin_size:.2f}, "
        f"alpha={args.alpha})"
    )
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
        "ckpt": args.disco_model_ckpt,
        "candidate_consolidation": "se2_mode",
        "mode_source_top_k": args.mode_source_top_k,
        "gpu_localize": bool(args.gpu_localize and device == "cuda"),
        "alpha": args.alpha,
        "1m_recall": np.sum(acc_record < 1) / total_samples,
        "0.5m_recall": np.sum(acc_record < 0.5) / total_samples,
        "0.1m_recall": np.sum(acc_record < 0.1) / total_samples,
        "1m_30deg_recall": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / total_samples,
        "1m_10deg_recall": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 10)) / total_samples,
        "improved": f"{improved_count} ({imp_pct:.2f}%)",
        "worsened": f"{worsened_count} ({wor_pct:.2f}%)"
    }
    current_result["avg_mode_pool_k"] = float(np.mean(mode_pool_sizes)) if mode_pool_sizes else 0.0
    current_result["avg_mode_rep_k"] = float(np.mean(mode_rep_sizes)) if mode_rep_sizes else 0.0
    current_result["avg_basin_size"] = float(np.mean(mode_basin_sizes)) if mode_basin_sizes else 0.0
    current_result["se2_sigma_t_m"] = args.se2_sigma_t_m
    current_result["se2_sigma_theta_deg"] = args.se2_sigma_theta_deg
    current_result["se2_angle_weight"] = args.se2_angle_weight
    current_result["se2_mode_radius"] = args.se2_mode_radius
    
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
