import argparse
import os
import sys
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
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

# Define custom dataset class to handle Gibson's poses.txt
class GibsonGridSeqDataset(GridSeqDataset):
    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        
        # Gibson Map Params for loading
        map_res = 0.01
        
        for scene in self.scene_names:
            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose (Gibson uses 'poses.txt' in world coords)
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            
            # Load map dim for conversion
            map_path = os.path.join(self.dataset_dir, scene, "map.png")
            map_img = cv2.imread(map_path)[:, :, 0]
            h_map, w_map = map_img.shape

            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose_str = poses_txt[state_id].split(" ")
                x_world = float(pose_str[0])
                y_world = float(pose_str[1])
                th = float(pose_str[2])
                
                # Convert to map coords
                x_map = x_world / map_res + w_map / 2
                y_map = y_world / map_res + h_map / 2
                
                pose = np.array([x_map, y_map, th], dtype=np.float32)
                scene_poses.append(pose)

            if self.all_imgs:
                start_idx += traj_len
            else:
                start_idx += traj_len // (self.L + 1)
            
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        # Override to handle 'rgb' folder and filename format
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        if self.all_imgs:
            ref_idx = idx_within_scene
            history_start_raw = ref_idx - self.L
        else:
            ref_idx = idx_within_scene * (self.L + 1) + self.L
            history_start_raw = idx_within_scene * (self.L + 1)

        data_dict = {}
        
        # 1. Depths
        ref_depth = self.gt_depth[scene_idx][ref_idx]
        data_dict["ref_depth"] = ref_depth
        
        src_depth_list = []
        for l in range(self.L):
            h_idx = history_start_raw + l
            valid_h_idx = max(0, h_idx)
            src_depth_list.append(self.gt_depth[scene_idx][valid_h_idx])
        data_dict["src_depth"] = np.stack(src_depth_list, axis=0)

        # 2. Poses
        ref_pose = self.gt_pose[scene_idx][ref_idx]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose
        
        src_pose_list = []
        for l in range(self.L):
            h_idx = history_start_raw + l
            valid_h_idx = max(0, h_idx)
            src_pose_list.append(self.gt_pose[scene_idx][valid_h_idx])
        data_dict["src_pose"] = np.stack(src_pose_list, axis=0)
        data_dict["src_noise"] = 0

        # 3. Images (Custom Logic)
        # Format: rgb/XXXXX-0.png (Assuming view 0)
        
        def load_img(s_name, idx_val):
            # Gibson format: XXXXX-X.png (4 frames per major step)
            # idx_val corresponds to flattened index
            major_step = idx_val // 4
            minor_step = idx_val % 4
            fname = f"{str(major_step).zfill(5)}-{str(minor_step)}.png"
            p = os.path.join(self.dataset_dir, s_name, "rgb", fname)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                # If 4 minor steps is assumption, verify if it fails
                # Try simple format if fail
                raise FileNotFoundError(f"Image not found: {p}")
            return img

        # Source
        src_img = []
        for l in range(self.L):
            flat_idx = history_start_raw + l
            valid_flat_idx = max(0, flat_idx)
            src_img.append(load_img(scene_name, valid_flat_idx))
        src_img = np.stack(src_img, axis=0).astype(np.float32)

        # Ref
        ref_img = load_img(scene_name, ref_idx).astype(np.float32)

        # Normalize
        # RRP Normalization logic
        if self.net_type == 'rrp' or self.net_type == 'unloc':
            normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_tensor = torch.from_numpy(ref_img_rgb).permute(2, 0, 1) / 255.0
            ref_img_norm_tensor = normalizer(ref_tensor)
            ref_img = ref_img_norm_tensor.permute(1, 2, 0).numpy()
            
            src_img_list_norm = []
            for l in range(self.L):
                src_img_l_rgb = cv2.cvtColor(src_img[l], cv2.COLOR_BGR2RGB)
                src_tensor = torch.from_numpy(src_img_l_rgb).permute(2, 0, 1) / 255.0
                src_norm_tensor = normalizer(src_tensor)             
                src_img_list_norm.append(src_norm_tensor.permute(1, 2, 0).numpy())
            src_img = np.stack(src_img_list_norm, axis=0).astype(np.float32)
        else:
             # Basic
             ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
             ref_img -= (0.485, 0.456, 0.406)
             ref_img /= (0.229, 0.224, 0.225)
             for l in range(self.L):
                 src_img[l, :, :, :] = (cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0)
                 src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
                 src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
        data_dict["ref_img"] = ref_img
        data_dict["src_img"] = src_img
        data_dict["obs_tensor"] = ref_img_norm_tensor if (self.net_type=='rrp' or self.net_type=='unloc') else torch.from_numpy(ref_img)
        
        return data_dict


# Global Args
parser = argparse.ArgumentParser(description="Eval with DisCo Model on Gibson")
parser.add_argument("--config", "-c", default="configs/paper/disco_gibson.yaml", type=str)
parser.add_argument("--net_type", type=str, default="rrp")
parser.add_argument("--dataset", type=str, default="gibson_f")
parser.add_argument("--dataset_path", type=str, default="./datasets_gibson/gibson_f")
parser.add_argument("--desdf_path", type=str, default="./datasets_gibson/desdf/")
parser.add_argument("--ckpt_path", type=str, default="./eval/logs")
parser.add_argument("--visualize", action="store_true")

# New Args for CrossModal
parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_gibson_f_best.ckpt", help="Path to RRP checkpoint")
parser.add_argument("--disco_model_ckpt", type=str, default="checkpoints/DisCo_gibson_f_best.ckpt", help="Path to DisCo checkpoint. Use an empty string to run RRP only.")
parser.add_argument("--alpha", type=float, default=0.5, help="Weight of semantic score")
parser.add_argument("--all_imgs", action="store_true", help="If True, evaluate all images. Gibson script default was often sparsely sampled.")
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
    help="Run standard RRP DESDF localization on GPU with localize_fast. Falls back to CPU if CUDA is unavailable.",
)
parser.add_argument(
    "--cpu_localize",
    dest="gpu_localize",
    action="store_false",
    help="Disable GPU DESDF localization and use the original CPU localize path.",
)
# Gibson specific
parser.add_argument("--fov", type=float, default=106.2602, help="Horizontal field of view used to convert depth40 to rays.")
parser.add_argument("--V", type=int, default=11, help="Number of Rays")

# Single Image Debugging
parser.add_argument("--scene_name", type=str, default=None, help="Debug: Specific scene name")
parser.add_argument("--img_id", type=int, default=None, help="Debug: Specific image ID within the scene")

args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
data_config = config["datasets"]

def crop_local_map(map_img, x, y, theta, crop_size_meters, res=0.01, output_size=128):
    """
    Standalone function to crop local map.
    Gibson res is 0.01 usually.
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
    
    # --- 2. Load DisCo Model (Optional) ---
    cl_model = None
    if args.disco_model_ckpt:
        cl_model = DisCoLocModel.load_from_checkpoint(args.disco_model_ckpt, config=config, map_location=device)  
        cl_model.to(device)
        cl_model.eval()

    # --- 3. Setup Dataset ---
    L = 3
    dataset_dir = args.dataset_path
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
        
    test_set = GibsonGridSeqDataset(
        dataset_dir,
        split.test,
        L=L,
        depth_dir=dataset_dir,
        depth_suffix="depth40",
        add_rp=False, # Gibson script usually False or 0 roll/pitch
        net_type=args.net_type,
        all_imgs=args.all_imgs,
    )

    # Load desdfs and maps
    desdf_path = args.desdf_path
    print("Loading desdfs...")
    desdfs = {}
    desdf_tensors = {}
    maps = {}
    gt_poses = {} # Map coordinates

    # Gibson Params
    # V = 11
    # res = 0.01 (Gibson map resolution)
    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    map_res = 0.01
    desdf_stride = 10 # 0.1 / 0.01
    meters_per_desdf_cell = desdf_stride * map_res

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
        h_map, w_map = maps[scene].shape
        
        # POSES
        # Gibson poses usually: x (m), y (m), theta (rad). Need to convert to map px.
        # Format in poses.txt: "x y theta" in world coords?
        # Re-using logic from snippet:
        # x = float(pose[0]) / 0.01 + w / 2
        # y = float(pose[1]) / 0.01 + h / 2
        
        with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]
            traj_len = len(poses_txt)
            poses = np.zeros([traj_len, 3], dtype=np.float32)
            for state_id in range(traj_len):
                pose_str = poses_txt[state_id].split(" ")
                # from world coordinate to map coordinate
                x_world = float(pose_str[0])
                y_world = float(pose_str[1])
                th = float(pose_str[2])
                
                x_map = x_world / map_res + w_map / 2
                y_map = y_world / map_res + h_map / 2
                
                poses[state_id, :] = np.array((x_map, y_map, th), dtype=np.float32)

            gt_poses[scene] = poses

    import matplotlib.pyplot as plt

    # --- Evaluation Loop ---
    acc_record = []
    acc_orn_record = []
    
    # Stats 
    improved_count = 0
    worsened_count = 0
    use_mode_rerank = cl_model is not None
    mode_pool_sizes = []
    mode_rep_sizes = []
    mode_basin_sizes = []

    # Create visualization directories
    if args.visualize:
        viz_dir = os.path.join(args.ckpt_path, "visualizations_gibson")
        os.makedirs(os.path.join(viz_dir, "improved"), exist_ok=True)
        os.makedirs(os.path.join(viz_dir, "degraded"), exist_ok=True)
        # Add a folder for single image debug
        if args.scene_name:
             os.makedirs(os.path.join(viz_dir, "debug"), exist_ok=True)
        print(f"Saving visualizations to {viz_dir}")

    if use_mode_rerank:
        print(
            "Starting Evaluation with SE(2)-Aware Mode Consolidation "
            f"(source_top_k={args.mode_source_top_k}, sigma_t={args.se2_sigma_t_m:.2f}m, "
            f"sigma_theta={args.se2_sigma_theta_deg:.1f}deg, rho={args.se2_mode_radius:.2f})..."
        )
    else:
        print("Starting Evaluation...")
    print(
        "DESDF localize: "
        f"{'gpu_fast' if args.gpu_localize and device == 'cuda' and args.net_type != 'unloc' else 'cpu_original'}"
    )
    
    # Determine Loop Range
    if args.scene_name is not None and args.img_id is not None:
        try:
            scene_idx = test_set.scene_names.index(args.scene_name)
            start_idx = test_set.scene_start_idx[scene_idx]
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

    eval_pbar = tqdm.tqdm(loop_range)
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
        obs_img_tensor = data["obs_tensor"].unsqueeze(0).to(device) # (1, C, H, W) 
        
        # --- 1. Geometric Prediction ---
        with torch.no_grad():
            feat = rrp_model("encode", obs_img=obs_img_tensor)
            
            if args.net_type == "unloc":
                 pred_depths_tensor, b_tensor = rrp_model("decoder_inference", depth_cond=feat, return_uncertainty=True)
                 pred_depths = pred_depths_tensor.squeeze(0).detach().cpu().numpy()
                 b = b_tensor.squeeze(0).detach().cpu().numpy()
                 
                 pred_rays, b_rays = get_ray_from_depth_unloc(pred_depths, b, V=args.V, F_W=F_W)
                 
                 pred_rays = torch.tensor(pred_rays, device="cpu")
                 b_rays = torch.tensor(b_rays, device="cpu")
                 
                 prob_vol, prob_dist, orientations, _ = localize_unloc(
                    torch.tensor(desdf_data["desdf"]), pred_rays, b_rays, return_np=False
                 )
            else:
                # Standard RRP (L1)
                pred_depths_tensor = rrp_model("decoder_inference", depth_cond=feat)
                pred_depths = pred_depths_tensor.squeeze(0).detach().cpu().numpy()
                
                pred_rays = get_ray_from_depth(pred_depths, V=args.V, F_W=F_W)
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
            
            
        # Get Best Geo Prediction
        geo_pred_y, geo_pred_x = torch.where(prob_dist == prob_dist.max())
        if geo_pred_y.numel() > 0:
            geo_pred = np.array([geo_pred_x[0].item(), geo_pred_y[0].item()])
            geo_error = np.linalg.norm(geo_pred - gt_pose_desdf[:2]) * 0.1 # 0.1m per desdf unit
        else:
            geo_error = 999.0
            geo_pred = np.array([0, 0])

        # --- 2. Semantic Re-ranking (If DisCo model provided) ---
        final_scores = torch.tensor([], device=device)
        semantic_weight = torch.tensor([], device=device)
        
        if cl_model:
            # Get image embedding (only once)
            with torch.no_grad():
                img_emb = cl_model.encode_image(obs_img_tensor)
            
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
                
                # Get orientation
                orn_idx = orientations[py, px].item()
                theta = (orn_idx / 36) * 2 * np.pi
                
                # Convert desdf (px, py) back to map (map_x, map_y)
                map_x = px * desdf_stride + desdf_data["l"]
                map_y = py * desdf_stride + desdf_data["t"]
                
                # Crop
                # For Gibson, verify pixel vs meters. 
                # desdf_stride=10 (pixels), map_res=0.01
                # 1 desdf unit = 0.1m = 10 px
                
                crop_local_map_size = data_config.get("local_map_crop_size_meters", 5.0) 
                lmap = crop_local_map(scene_map, map_x, map_y, theta, crop_size_meters=crop_local_map_size, res=map_res)
                lmap_tensor = torch.from_numpy(lmap).float() / 255.0
                local_maps.append(lmap_tensor.unsqueeze(0)) # (1, H, W)
                valid_indices.append(i)
                
            if local_maps:
                local_maps_batch = torch.stack(local_maps).to(device)
                
                with torch.no_grad():
                    sim_scores = cl_model.score_candidates(img_emb, local_maps_batch)
                    
                    geo_probs = topk_vals.to(device)
                    semantic_weight = torch.exp(sim_scores * args.alpha)
                    
                    final_scores = geo_probs * semantic_weight
                    
                    # Find best in Top-K
                    best_idx_in_k = torch.argmax(final_scores).item()
                    
                    final_i = valid_indices[best_idx_in_k]
                    final_y = topk_y[final_i].item()
                    final_x = topk_x[final_i].item()
                    
                    final_orn_idx = orientations[final_y, final_x].item()
                    final_orn = (final_orn_idx / 36) * 2 * np.pi
                    pose_pred = np.array([final_x, final_y, final_orn])
            else:
                 pose_pred = np.array([geo_pred_x[0].item(), geo_pred_y[0].item(), 0.0])
        else:
            # Only RRP/Unloc
            # Reconstruct orn
            y_g, x_g = int(geo_pred[1]), int(geo_pred[0])
            orn_idx = orientations[y_g, x_g].item()
            orn = (orn_idx / 36) * 2 * np.pi
            pose_pred = np.array([geo_pred[0], geo_pred[1], orn])

        # --- Accuracy ---
        acc = np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.1
        acc_record.append(acc)
        
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)

        if len(acc_record) > 0:
            current_1m_recall = np.sum(np.array(acc_record) < 1) / len(acc_record)
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
        should_viz = args.visualize and (is_improved or is_degraded or args.scene_name is not None)
        
        if should_viz and cl_model: # Only robust viz if DisCo is involved, else just simple viz if needed
            if args.scene_name:
                save_folder = "debug"
            else:
                save_folder = "improved" if is_improved else "degraded"
            
            # Prepare Maps
            H, W = prob_dist.shape
            map_rrp = prob_dist.cpu().numpy()
            map_cl = np.zeros((H, W), dtype=np.float32)
            cl_vals = semantic_weight.cpu().numpy() if len(semantic_weight) > 0 else []
            
            # Find max CL score point
            cl_pred_pt = None
            if len(cl_vals) > 0:
                best_cl_idx = np.argmax(cl_vals)
                final_i_cl = valid_indices[best_cl_idx]
                flat_idx_cl = topk_indices[final_i_cl].item()
                y_cl, x_cl = flat_idx_cl // W, flat_idx_cl % W
                cl_pred_pt = np.array([x_cl, y_cl])
            
            for k_idx, val_idx in enumerate(valid_indices):
                flat_idx = topk_indices[val_idx].item()
                y, x = flat_idx // W, flat_idx % W
                map_cl[y, x] = cl_vals[k_idx]
            
            map_final = map_rrp.copy()
            final_vals = final_scores.cpu().numpy() if len(final_scores) > 0 else []
            
            for k_idx, val_idx in enumerate(valid_indices):
                flat_idx = topk_indices[val_idx].item()
                y, x = flat_idx // W, flat_idx % W
                map_final[y, x] = final_vals[k_idx]

            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
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
            
            cl_pred_orn = None
            if cl_pred_pt is not None:
                y_c, x_c = int(cl_pred_pt[1]), int(cl_pred_pt[0])
                cl_orn_idx = orientations[y_c, x_c].item()
                cl_pred_orn = (cl_orn_idx / 36) * 2 * np.pi

            rrp_pred_orn = None
            if geo_error < 900: 
                y_g, x_g = int(geo_pred[1]), int(geo_pred[0])
                rrp_orn_idx = orientations[y_g, x_g].item()
                rrp_pred_orn = (rrp_orn_idx / 36) * 2 * np.pi
            
            cl_vmax = np.max(map_cl) if np.max(map_cl) > 0 else 1e-6
            final_vmax = np.max(map_final) if np.max(map_final) > 0 else 1e-6

            viz_map(axs[0], map_cl, f"Disambiguation Score", pred_pt=cl_pred_pt, pred_orn=cl_pred_orn, vmin=1, vmax=cl_vmax)
            viz_map(axs[1], map_rrp, f"UnLoc Prob (Err: {geo_error:.2f}m)", pred_pt=geo_pred, pred_orn=rrp_pred_orn)
            viz_map(axs[2], map_final, f"Combined Prob (Err: {acc:.2f}m)", pred_pt=pose_pred[:2], pred_orn=pose_pred[2], vmin=None, vmax=final_vmax)
            
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
    if use_mode_rerank:
        avg_pool_k = np.mean(mode_pool_sizes) if mode_pool_sizes else 0.0
        avg_rep_k = np.mean(mode_rep_sizes) if mode_rep_sizes else 0.0
        avg_basin_size = np.mean(mode_basin_sizes) if mode_basin_sizes else 0.0
        print(
            "Results on Gibson with DisCo SE(2)-Aware Mode Consolidation "
            f"(source_top_k={args.mode_source_top_k}, sigma_t={args.se2_sigma_t_m:.2f}m, "
            f"sigma_theta={args.se2_sigma_theta_deg:.1f}deg, rho={args.se2_mode_radius:.2f}, "
            f"avg_rep_k={avg_rep_k:.1f}, avg_basin_size={avg_basin_size:.2f}, "
            f"alpha={args.alpha}, V={args.V}, FOV={args.fov}, Net={args.net_type})"
        )
    elif cl_model:
        print(f"Results on Gibson with DisCo (alpha={args.alpha}, V={args.V}, FOV={args.fov}, Net={args.net_type})")
    else:
        print(f"Results on Gibson RRP-only (V={args.V}, FOV={args.fov}, Net={args.net_type})")
    print(f"1m recall = {np.sum(acc_record < 1) / total_samples:.4f}")
    print(f"0.5m recall = {np.sum(acc_record < 0.5) / total_samples:.4f}")
    print(f"0.1m recall = {np.sum(acc_record < 0.1) / total_samples:.4f}")
    print(f"1m 30 deg recall = {np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / total_samples:.4f}")
    print("-" * 20)
    
    if cl_model:
        imp_pct = improved_count / total_samples * 100
        wor_pct = worsened_count / total_samples * 100
        print(f"Improved samples: {improved_count} ({imp_pct:.2f}%)")
        print(f"Worsened samples: {worsened_count} ({wor_pct:.2f}%)")
    print("="*30)

    # --- Log Results to File ---
    import datetime
    import json
    
    log_file = "eval/eval_history_gibson.txt"
    
    current_result = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rrp_ckpt": args.rrp_model_ckpt,
        "disco_ckpt": args.disco_model_ckpt,
        "candidate_consolidation": "se2_mode" if cl_model else "none",
        "mode_source_top_k": args.mode_source_top_k if cl_model else 0,
        "gpu_localize": bool(args.gpu_localize and device == "cuda" and args.net_type != "unloc"),
        "alpha": args.alpha,
        "1m_recall": np.sum(acc_record < 1) / total_samples,
        "0.5m_recall": np.sum(acc_record < 0.5) / total_samples,
        "0.1m_recall": np.sum(acc_record < 0.1) / total_samples,
        "1m_30deg_recall": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / total_samples,
    }
    if use_mode_rerank:
        current_result["avg_mode_pool_k"] = float(np.mean(mode_pool_sizes)) if mode_pool_sizes else 0.0
        current_result["avg_mode_rep_k"] = float(np.mean(mode_rep_sizes)) if mode_rep_sizes else 0.0
        current_result["avg_basin_size"] = float(np.mean(mode_basin_sizes)) if mode_basin_sizes else 0.0
        current_result["se2_sigma_t_m"] = args.se2_sigma_t_m
        current_result["se2_sigma_theta_deg"] = args.se2_sigma_theta_deg
        current_result["se2_angle_weight"] = args.se2_angle_weight
        current_result["se2_mode_radius"] = args.se2_mode_radius
    
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            history.append(entry)
                        except json.JSONDecodeError:
                            pass 
        except Exception as e:
            print(f"Warning: Could not read history file: {e}")

    history.append(current_result)
    history.sort(key=lambda x: x.get("1m_recall", 0), reverse=True)
    
    with open(log_file, "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Result logged to {log_file}")

if __name__ == "__main__":
    evaluate()
