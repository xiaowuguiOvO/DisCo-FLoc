import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
import cv2
import matplotlib.pyplot as plt
from attrdict import AttrDict

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.data_utils import *
from utils.localization_utils import *
from training.RRP_lightning_module import RRPLightningModule
from training.DisCo_lightning_module import DisCoLocModel

# Define custom dataset class to handle Gibson's poses.txt
# Reusing the class from eval_disco_model_gibson.py
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
            valid_h_idx = max(0, h_idx) # pad with 0
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
        def load_img(s_name, idx_val):
            # Gibson T format: XXXXX.png (Sequential)
            # Gibson F format: XXXXX-X.png (Multi-view)
            # Try Gibson T first
            fname_t = f"{str(idx_val).zfill(5)}.png"
            p_t = os.path.join(self.dataset_dir, s_name, "rgb", fname_t)
            
            if os.path.exists(p_t):
                img = cv2.imread(p_t, cv2.IMREAD_COLOR)
                return img
                
            # Fallback to Gibson F format if needed (or raise error)
            major_step = idx_val // 4
            minor_step = idx_val % 4
            fname_f = f"{str(major_step).zfill(5)}-{str(minor_step)}.png"
            p_f = os.path.join(self.dataset_dir, s_name, "rgb", fname_f)
            
            img = cv2.imread(p_f, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found at {p_t} OR {p_f}")
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

def get_rel_pose(ref_pose, src_pose):
    """
    Input:
        ref_pose: torch.tensor(N, 3)
        src_pose: torch.tensor(N, L, 3) or (N, 3)
    Output:
        rel_pose: torch.tensor(N, L, 3) or (N, 3)
    """
    # NOTE: the relative pose theta needs to be in -pi/pi
    if ref_pose.dim() == 1 and src_pose.dim() == 1:
        # only compute a single one
        rel_pose = src_pose - ref_pose  # (3)
        cr = torch.cos(ref_pose[-1])
        sr = torch.sin(ref_pose[-1])
        rel_x = cr * rel_pose[0] + sr * rel_pose[1]
        rel_y = -sr * rel_pose[0] + cr * rel_pose[1]
        rel_pose[0] = rel_x
        rel_pose[1] = rel_y
        rel_pose[-1] = (rel_pose[-1] + torch.pi) % (torch.pi * 2) - torch.pi
    else:
        # compute the source pose w.r.t. reference pose
        if src_pose.dim() == 2:
             # Add L dim if missing to match ref logic
             src_pose = src_pose.unsqueeze(1)
        
        rel_pose = src_pose - ref_pose.unsqueeze(1)  # (N, L, 3)
        cr = torch.cos(ref_pose[:, -1]).unsqueeze(-1)  # (N, 1)
        sr = torch.sin(ref_pose[:, -1]).unsqueeze(-1)  # (N, 1)
        rel_x = cr * rel_pose[:, :, 0] + sr * rel_pose[:, :, 1]  # (N, L)
        rel_y = -sr * rel_pose[:, :, 0] + cr * rel_pose[:, :, 1]  # (N, L)
        rel_pose[:, :, 0] = rel_x
        rel_pose[:, :, 1] = rel_y
        rel_pose[:, :, -1] = (rel_pose[:, :, -1] + torch.pi) % (torch.pi * 2) - torch.pi

    return rel_pose.squeeze() if (rel_pose.dim() > 1 and rel_pose.shape[1]==1) else rel_pose


def crop_local_map(map_img, x, y, theta, crop_size_meters, res=0.01, output_size=128):
    """
    Crop an oriented local map around a candidate pose.
    Coordinate convention matches eval_disco_model_gibson.py.
    """
    x = float(x)
    y = float(y)
    crop_size_px = int(crop_size_meters / res)
    pad = crop_size_px

    if torch.is_tensor(map_img):
        map_img = map_img.cpu().numpy()
    if len(map_img.shape) == 3:
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    map_padded = cv2.copyMakeBorder(
        map_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
    )

    center = (x + pad, y + pad)
    angle_deg = np.degrees(theta)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg + 90, 1.0)
    rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
    rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]

    local_map = cv2.warpAffine(
        map_padded,
        rot_matrix,
        (crop_size_px, crop_size_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if crop_size_px != output_size:
        local_map = cv2.resize(
            local_map, (output_size, output_size), interpolation=cv2.INTER_AREA
        )

    return local_map


def build_first_frame_disco_prior(
    cl_model,
    obs_img,
    scene_map,
    prob_dist,
    orientations,
    desdf_data,
    desdf_stride,
    map_res,
    crop_size_meters,
    top_k,
    alpha,
    prior_eps,
    mode,
    device,
):
    """
    Build a sparse pose prior from DisCo scores over RRP top-k candidates.
    The prior has the same (H, W, O) shape as the filtering posterior.
    """
    if cl_model is None:
        raise ValueError("DisCo prior requested but cl_model is None")

    prob_dist_cpu = prob_dist.detach().cpu()
    orientations_cpu = orientations.detach().cpu().long()
    H, W = prob_dist_cpu.shape
    O = int(desdf_data["desdf"].shape[2]) if "desdf" in desdf_data else 36

    flat_probs = prob_dist_cpu.flatten()
    k = min(top_k, flat_probs.numel())
    topk_vals, topk_indices = torch.topk(flat_probs, k=k)

    local_maps = []
    candidate_records = []
    for flat_idx, geo_val in zip(topk_indices.tolist(), topk_vals.tolist()):
        py = int(flat_idx // W)
        px = int(flat_idx % W)
        orn_idx = int(orientations_cpu[py, px].item())
        theta = (orn_idx / 36.0) * 2.0 * np.pi

        map_x = px * desdf_stride + desdf_data["l"]
        map_y = py * desdf_stride + desdf_data["t"]
        local_map = crop_local_map(
            scene_map,
            map_x,
            map_y,
            theta,
            crop_size_meters=crop_size_meters,
            res=map_res,
            output_size=128,
        )
        local_maps.append(torch.from_numpy(local_map).float().unsqueeze(0) / 255.0)
        candidate_records.append((py, px, orn_idx, float(geo_val)))

    if not local_maps:
        return None

    local_maps_batch = torch.stack(local_maps).to(device)
    with torch.no_grad():
        img_emb = cl_model.encode_image(obs_img)
        sim_scores = cl_model.score_candidates(img_emb, local_maps_batch)
        semantic_weight = torch.exp(sim_scores * alpha).detach().cpu()

    prior = torch.full((H, W, O), float(prior_eps), dtype=torch.float32)
    for i, (py, px, orn_idx, geo_val) in enumerate(candidate_records):
        weight = float(semantic_weight[i].item())
        if mode == "combined":
            weight *= max(geo_val, 0.0)
        prior[py, px, orn_idx] = max(weight, float(prior_eps))

    prior = prior / (prior.sum() + 1e-12)
    return prior

def evaluate_filtering():
    parser = argparse.ArgumentParser(description="Filtering evaluation.")
    parser.add_argument("--config", "-c", default="configs/paper/disco_gibson.yaml", type=str)
    parser.add_argument("--dataset_path", type=str, default="./datasets_gibson/gibson_t")
    parser.add_argument("--desdf_path", type=str, default="./datasets_gibson/desdf/")
    parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_gibson_best.ckpt", help="Path to RRP checkpoint")
    parser.add_argument("--net_type", type=str, default="rrp")
    parser.add_argument("--log_dir", type=str, default="eval/logs_filtering")
    parser.add_argument("--traj_len", type=int, default=100, help="Length of each evaluated trajectory chunk")
    parser.add_argument("--eval_last_n", type=int, default=10, help="Number of final frames used for trajectory success")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Optional cap on evaluated trajectory chunks for quick tests")
    parser.add_argument("--first_frame_disco_prior", action="store_true", help="Use DisCo reranking as the initial prior for each trajectory")
    parser.add_argument("--disco_model_ckpt", type=str, default=None, help="Path to DisCo checkpoint used by --first_frame_disco_prior")
    parser.add_argument("--disco_top_k", type=int, default=100, help="Number of RRP candidates reranked by DisCo for the initial prior")
    parser.add_argument("--disco_alpha", type=float, default=0.5, help="Scale applied to DisCo similarity before exponentiation")
    parser.add_argument("--disco_prior_eps", type=float, default=1e-12, help="Small prior mass assigned outside DisCo top-k candidates")
    parser.add_argument("--disco_prior_mode", type=str, default="semantic", choices=["semantic", "combined"], help="Use semantic-only or geometry*semantic weights for the initial prior")
    
    # Gibson specific
    parser.add_argument("--fov", type=float, default=106.2602, help="Horizontal field of view used to convert depth40 to rays.")
    parser.add_argument("--V", type=int, default=11, help="Number of Rays")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"======= USING DEVICE : {device} =======")
    
    # Create log dir
    os.makedirs(args.log_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    data_config = config.get("datasets", {})

    # 1. Load Model
    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt , map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()

    cl_model = None
    if args.first_frame_disco_prior:
        if not args.disco_model_ckpt:
            raise ValueError("--first_frame_disco_prior requires --disco_model_ckpt")
        print(f"Loading DisCo model for first-frame prior: {args.disco_model_ckpt}")
        cl_model = DisCoLocModel.load_from_checkpoint(
            args.disco_model_ckpt, config=config, map_location=device
        )
        cl_model.to(device)
        cl_model.eval()

    # 2. Setup Dataset
    dataset_dir = args.dataset_path
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
        
    # Using L=3 for model input history
    L = 3 
    
    # Filtering needs sequential access. 
    # GibsonGridSeqDataset with all_imgs=True gives us every frame index.
    test_set = GibsonGridSeqDataset(
        dataset_dir,
        split.test,
        L=L, 
        depth_dir=dataset_dir,
        depth_suffix="depth40",
        add_rp=False,
        net_type=args.net_type,
        all_imgs=True,
    )

    # 3. Load desdfs
    desdf_path = args.desdf_path
    print("Loading desdfs...")
    desdfs = {}
    maps = {}
    
    # Gibson parameters
    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    map_res = 0.01
    desdf_stride = 10 # 0.1m / 0.01m pixel
    
    for scene in tqdm.tqdm(test_set.scene_names):
        desdf_file = os.path.join(desdf_path, scene, "desdf.npy")
        if not os.path.exists(desdf_file):
            raise FileNotFoundError(
                f"Missing DESDF for scene '{scene}': {desdf_file}. "
                "Use --desdf_path pointing to a directory that contains all test scenes, e.g. ./datasets_gibson/desdf"
            )
        desdfs[scene] = np.load(
            desdf_file, allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 20] = 20
        if args.first_frame_disco_prior:
            map_file = os.path.join(dataset_dir, scene, "map.png")
            scene_map = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
            if scene_map is None:
                raise FileNotFoundError(f"Missing map image for scene '{scene}': {map_file}")
            maps[scene] = scene_map

    # 4. Filter Loop
    success_10_all = []
    success_5_all = []
    success_3_all = []
    success_2_all = []
    RMSEs = []
    
    print("Starting Filtering Evaluation...")
    traj_len = args.traj_len
    eval_last_n = args.eval_last_n
    if traj_len <= L:
        raise ValueError(f"--traj_len must be larger than history length L={L}, got {traj_len}")
    if eval_last_n <= 0:
        raise ValueError(f"--eval_last_n must be positive, got {eval_last_n}")

    scene_trajs = []
    dropped_frames = 0
    total_filter_steps = 0
    for scene_idx, scene_name in enumerate(test_set.scene_names):
        if scene_idx + 1 >= len(test_set.scene_start_idx):
             scene_len = len(test_set) - test_set.scene_start_idx[scene_idx]
        else:
             scene_len = test_set.scene_start_idx[scene_idx+1] - test_set.scene_start_idx[scene_idx]

        usable_scene_len = scene_len - (scene_len % traj_len)
        dropped_frames += scene_len - usable_scene_len
        for chunk_start in range(0, usable_scene_len, traj_len):
            scene_trajs.append((scene_idx, scene_name, chunk_start, traj_len))
            total_filter_steps += traj_len - L

    if args.max_trajectories is not None:
        scene_trajs = scene_trajs[: args.max_trajectories]
        total_filter_steps = sum(chunk_len - L for _, _, _, chunk_len in scene_trajs)

    print(f"Trajectory protocol: traj_len={traj_len}, eval_last_n={eval_last_n}, trajectories={len(scene_trajs)}, dropped_tail_frames={dropped_frames}")
    eval_pbar = tqdm.tqdm(total=total_filter_steps, desc="Filtering", dynamic_ncols=True)
    
    # Iterate trajectory chunks, matching the original f3loc filtering protocol.
    try:
        for traj_idx, (scene_idx, scene_name, chunk_start, chunk_len) in enumerate(scene_trajs):
            tqdm.tqdm.write(f"Evaluating Trajectory {traj_idx + 1}/{len(scene_trajs)}: {scene_name}[{chunk_start}:{chunk_start + chunk_len}]")
            
            desdf = desdfs[scene_name]
            desdf_tensor = torch.tensor(desdf["desdf"], device=device)
            
            # Reset prior for each trajectory chunk, as in the original f3loc eval.
            prior = torch.ones_like(desdf_tensor) / desdf_tensor.numel()
            
            last_pose = None
            last_pose_map = None
            traj_errors = []
            
            for t in range(chunk_len - L):
                frame_offset = chunk_start + t + L
                global_idx = test_set.scene_start_idx[scene_idx] + frame_offset
                
                data = test_set[global_idx]
                
                obs_img = data["obs_tensor"].unsqueeze(0).to(device)
                # GT Pose in map frame
                gt_pose_map = torch.tensor(data["ref_pose"], device=device)
                
                # Transform GT to DESDF frame
                gt_pose_desdf = gt_pose_map.clone()
                gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) / desdf_stride
                gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) / desdf_stride
                
                # --- Prediction Step (Transit) ---
                if last_pose is not None:
                    # Calculate odometry using simpler Map Pixel Poses
                    # last_pose_map and current_pose_map are in pixels (1cm resolution)
                    transition_px = get_rel_pose(last_pose_map, gt_pose_map)
                    
                    # Convert Pixels to Meters for transit function
                    # Gibson: 1 pixel = 0.01 meters
                    transition_meters = transition_px.clone()
                    transition_meters[0] *= 0.01 # dx (m)
                    transition_meters[1] *= 0.01 # dy (m)
                    # dtheta is radians, no scale needed 
                    
                    # Apply transition to prior
                    # transit expect tensor
                    prior = transit(
                        prior, transition_meters, 
                        sig_o=0.1, sig_x=0.1, sig_y=0.1, # Tighter noise for correct scale
                        tsize=11, rsize=11, resolution=0.1
                    ).to(device)
                
                # DEBUG: Force uniform prior to test Observation Model only
                # prior = torch.ones_like(desdf_tensor) / desdf_tensor.numel()
                
                last_pose_map = gt_pose_map
                last_pose = gt_pose_desdf
                
                # --- Update Step (Observation) ---
                with torch.no_grad():
                    feat = rrp_model("encode", obs_img=obs_img)
                    
                    if args.net_type == "unloc":
                         pred_depths, b = rrp_model("decoder_inference", depth_cond=feat, return_uncertainty=True)
                         d_hat = pred_depths.squeeze(0)
                         b_hat = b.squeeze(0)
                         
                         pred_rays, b_rays = get_ray_from_depth_unloc(d_hat.cpu().numpy(), b_hat.cpu().numpy(), V=args.V, F_W=F_W)
                         
                         pred_rays = torch.tensor(pred_rays, device=device)
                         b_rays = torch.tensor(b_rays, device=device)
                         
                         # localize_unloc adapted to return tensor if return_np=False
                         # We need to modify utils or adapt here.
                         # localize_unloc in utils usually runs on CPU if not adapted. 
                         # Assuming we have CPU/GPU tensor mix handling or pass return_np=False
                         
                         prob_vol, prob_dist_obs, orientations_obs, _ = localize_unloc(
                            desdf_tensor.cpu(), pred_rays.cpu(), b_rays.cpu(), return_np=False
                         )
                         likelihood = prob_vol.to(device)
                         
                    else:
                        # RRP
                        pred_depths = rrp_model("decoder_inference", depth_cond=feat).squeeze(0)
                        pred_rays = get_ray_from_depth(pred_depths.cpu().numpy(), V=args.V, F_W=F_W)
                        pred_rays = torch.tensor(pred_rays, device=device)
                        
                        prob_vol, prob_dist_obs, orientations_obs, _ = localize(
                            desdf_tensor.cpu(), pred_rays.cpu(), return_np=False
                        )
                        likelihood = prob_vol.to(device)

                # Posterior Update
                if args.first_frame_disco_prior and t == 0:
                    crop_size_meters = data_config.get("local_map_crop_size_meters", 5.0)
                    disco_prior = build_first_frame_disco_prior(
                        cl_model=cl_model,
                        obs_img=obs_img,
                        scene_map=maps[scene_name],
                        prob_dist=prob_dist_obs,
                        orientations=orientations_obs,
                        desdf_data=desdf,
                        desdf_stride=desdf_stride,
                        map_res=map_res,
                        crop_size_meters=crop_size_meters,
                        top_k=args.disco_top_k,
                        alpha=args.disco_alpha,
                        prior_eps=args.disco_prior_eps,
                        mode=args.disco_prior_mode,
                        device=device,
                    )
                    if disco_prior is not None:
                        prior = disco_prior.to(device)

                posterior = prior * likelihood + 1e-15
                if posterior.sum() == 0:
                    # print("Warning: Posterior vanished, resetting to uniform.")
                    posterior = torch.ones_like(prior) / prior.numel()
                else:
                    posterior = posterior / posterior.sum() # Normalize
                
                # Update Prior for next step
                prior = posterior
                
                # --- Estimation & Metric ---
                prob_dist, _ = torch.max(posterior, dim=2)
                pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
                
                if len(pred_y) > 0:
                    # Take mean if multiple peaks or just first
                    est_y = pred_y[0].float().item()
                    est_x = pred_x[0].float().item()
                    
                    est_pose = np.array([est_x, est_y])
                    gt_xy = gt_pose_desdf[:2].cpu().numpy()
                    
                    error = np.linalg.norm(est_pose - gt_xy) * 0.1 # 0.1m resolution
                else:
                    error = 100.0 # Penalty

                traj_errors.append(error)

                completed_samples = len(success_10_all)
                eval_pbar.set_postfix(
                    scene=scene_name,
                    traj=traj_idx + 1,
                    frame=frame_offset,
                    err_m=f"{error:.2f}",
                    done_samples=completed_samples,
                    sample_acc1m=f"{np.mean(success_10_all):.3f}" if completed_samples else "n/a",
                    sample_acc05m=f"{np.mean(success_5_all):.3f}" if completed_samples else "n/a",
                    sample_acc03m=f"{np.mean(success_3_all):.3f}" if completed_samples else "n/a",
                    sample_acc02m=f"{np.mean(success_2_all):.3f}" if completed_samples else "n/a",
                )
                eval_pbar.update(1)
                    
            if len(traj_errors) > 0:
                traj_errors = np.array(traj_errors)
                last_n = min(eval_last_n, len(traj_errors))
                last_errors = traj_errors[-last_n:]
                RMSE = np.sqrt(np.mean(last_errors ** 2))
                RMSEs.append(RMSE)

                success_10_all.append(bool(np.all(last_errors < 1.0)))
                success_5_all.append(bool(np.all(last_errors < 0.5)))
                success_3_all.append(bool(np.all(last_errors < 0.3)))
                success_2_all.append(bool(np.all(last_errors < 0.2)))

                eval_pbar.set_postfix(
                    scene=scene_name,
                    traj=traj_idx + 1,
                    done_samples=len(success_10_all),
                    sample_acc1m=f"{np.mean(success_10_all):.3f}",
                    sample_acc05m=f"{np.mean(success_5_all):.3f}",
                    sample_acc03m=f"{np.mean(success_3_all):.3f}",
                    sample_acc02m=f"{np.mean(success_2_all):.3f}",
                )
                
                tqdm.tqdm.write(
                    f"Trajectory {traj_idx + 1}/{len(scene_trajs)} {scene_name}[{chunk_start}:{chunk_start + chunk_len}]: "
                    f"last{last_n} 1m={success_10_all[-1]}, 0.5m={success_5_all[-1]}, 0.3m={success_3_all[-1]}, "
                    f"0.2m={success_2_all[-1]}, RMSE={RMSE:.3f}m"
                )
    finally:
        eval_pbar.close()

    # Final Summary
    success_10_all = np.array(success_10_all)
    success_5_all = np.array(success_5_all)
    success_3_all = np.array(success_3_all)
    success_2_all = np.array(success_2_all)
    RMSEs = np.array(RMSEs)

    print("\n" + "="*30)
    print(f"Overall Filtering Results ({len(success_10_all)} trajectories, traj_len={traj_len}, eval_last_n={eval_last_n})")
    print(f"1.0m Success Rate: {np.mean(success_10_all):.4f}")
    print(f"0.5m Success Rate: {np.mean(success_5_all):.4f}")
    print(f"0.3m Success Rate: {np.mean(success_3_all):.4f}")
    print(f"0.2m Success Rate: {np.mean(success_2_all):.4f}")
    print(f"Mean RMSE succeeded: {RMSEs[success_10_all].mean():.4f}" if np.any(success_10_all) else "Mean RMSE succeeded: nan")
    print(f"Mean RMSE all: {RMSEs.mean():.4f}" if len(RMSEs) else "Mean RMSE all: nan")
    print("="*30)

if __name__ == "__main__":
    evaluate_filtering()
