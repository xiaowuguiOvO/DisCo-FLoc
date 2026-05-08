import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.DisCo_lightning_module import DisCoLocModel
from training.RRP_lightning_module import RRPLightningModule
from utils.data_utils import GridSeqDataset
from utils.localization_utils import get_ray_from_depth, localize, localize_fast


def parse_int_values(raw_values):
    values = []
    for chunk in raw_values.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise ValueError("No valid top-N values were provided.")
    return values


def sync_if_needed(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_call(device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, time.perf_counter() - start


def crop_local_map(map_img, x, y, theta, crop_size_meters, res=0.02, output_size=128):
    x = float(x)
    y = float(y)
    crop_size_px = int(crop_size_meters / res)
    pad = crop_size_px

    height, width = map_img.shape[:2]
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
        local_map = cv2.resize(local_map, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return local_map


def cluster_topk_candidates(topk_vals, topk_indices, width, radius_m, meters_per_cell):
    if topk_indices.numel() == 0 or radius_m <= 0:
        keep_positions = torch.arange(topk_indices.shape[0], dtype=torch.long)
        cluster_sizes = torch.ones(topk_indices.shape[0], dtype=torch.long)
        return topk_vals, topk_indices, keep_positions, cluster_sizes

    radius_cells = radius_m / meters_per_cell
    radius_sq = radius_cells * radius_cells

    topk_y = (topk_indices // width).to(torch.float32)
    topk_x = (topk_indices % width).to(torch.float32)

    suppressed = torch.zeros(topk_indices.shape[0], dtype=torch.bool)
    keep_positions = []
    cluster_sizes = []

    for idx in range(topk_indices.shape[0]):
        if suppressed[idx]:
            continue

        dx = topk_x - topk_x[idx]
        dy = topk_y - topk_y[idx]
        cluster_mask = (~suppressed) & ((dx * dx + dy * dy) <= radius_sq)

        keep_positions.append(idx)
        cluster_sizes.append(int(cluster_mask.sum().item()))
        suppressed |= cluster_mask

    keep_positions = torch.tensor(keep_positions, dtype=torch.long)
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.long)
    return topk_vals[keep_positions], topk_indices[keep_positions], keep_positions, cluster_sizes


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0}
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "median_ms": float(np.median(arr) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
    }


def load_dataset(dataset_dir, all_imgs):
    with open(os.path.join(dataset_dir, "split.yaml"), "r", encoding="utf-8") as f:
        split = AttrDict(yaml.safe_load(f))

    return GridSeqDataset(
        dataset_dir,
        split.test,
        L=3,
        depth_dir=dataset_dir,
        depth_suffix="depth40",
        add_rp=False,
        net_type="rrp",
        all_imgs=all_imgs,
    )


def sample_indices(test_set, args):
    if args.scene_name is not None and args.img_id is not None:
        scene_idx = test_set.scene_names.index(args.scene_name)
        return [test_set.scene_start_idx[scene_idx] + args.img_id]

    indices = list(range(args.start_idx, len(test_set), args.stride))
    return indices[: args.num_samples]


def scene_for_index(test_set, data_idx):
    scene_starts = np.array(test_set.scene_start_idx)
    scene_idx = int(np.sum(data_idx >= scene_starts) - 1)
    scene = test_set.scene_names[scene_idx]
    idx_within_scene = int(data_idx - test_set.scene_start_idx[scene_idx])
    return scene, idx_within_scene


def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    topn_values = parse_int_values(args.topn_values)
    max_topn = max(topn_values)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_config = config["datasets"]

    print(f"Using device: {device}")
    print(f"Benchmark top-N clusters: {topn_values}")
    print(
        "Cluster settings: "
        f"source_top_k={args.cluster_source_top_k}, radius={args.cluster_radius_m:.2f}m"
    )
    print(f"Localize mode: {args.localize_mode}")

    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt, map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()

    disco_model = DisCoLocModel.load_from_checkpoint(
        args.disco_model_ckpt, config=config, map_location=device
    )
    disco_model.to(device)
    disco_model.eval()

    test_set = load_dataset(args.dataset_path, args.all_imgs)
    indices = sample_indices(test_set, args)
    if not indices:
        raise ValueError("No samples selected for benchmarking.")

    scenes_needed = sorted({scene_for_index(test_set, data_idx)[0] for data_idx in indices})
    desdfs = {}
    desdf_tensors = {}
    maps = {}
    for scene in tqdm.tqdm(scenes_needed, desc="Loading scenes"):
        desdf = np.load(os.path.join(args.desdf_path, scene, "desdf.npy"), allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 20] = 20
        desdfs[scene] = desdf
        if args.localize_mode == "gpu_fast":
            desdf_tensors[scene] = torch.tensor(
                desdf["desdf"], dtype=torch.float32, device=device
            )
        maps[scene] = cv2.imread(os.path.join(args.dataset_path, scene, "map.png"))[:, :, 0]

    fov_factor = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    desdf_stride = 5
    map_res = float(data_config.get("map_res", 0.02))
    meters_per_desdf_cell = desdf_stride * map_res
    crop_size_meters = float(data_config.get("local_map_crop_size_meters", 5.0))

    fixed_times = {
        "rrp_depth": [],
        "localize": [],
        "image_encode": [],
        "topk_cluster": [],
    }
    rerank_times = {
        topn: {
            "crop": [],
            "stack_to_device": [],
            "disco_score": [],
            "fusion_select": [],
            "rerank_total": [],
            "effective_candidates": [],
        }
        for topn in topn_values
    }
    rep_counts = []

    progress = tqdm.tqdm(indices, desc="Benchmarking")
    for sample_idx, data_idx in enumerate(progress):
        data = test_set[data_idx]
        scene, _ = scene_for_index(test_set, data_idx)
        desdf_data = desdfs[scene]
        scene_map = maps[scene]
        obs_img_tensor = data["obs_tensor"].unsqueeze(0).to(device)

        with torch.no_grad():
            features, elapsed = timed_call(
                device, lambda: rrp_model("encode", obs_img=obs_img_tensor)
            )
            pred_depths_tensor, decoder_elapsed = timed_call(
                device, lambda: rrp_model("decoder_inference", depth_cond=features)
            )
            fixed_times["rrp_depth"].append(elapsed + decoder_elapsed)

            pred_depths = pred_depths_tensor.squeeze(0).detach().cpu().numpy()
            pred_rays = torch.tensor(get_ray_from_depth(pred_depths, V=9, F_W=fov_factor))

            def run_localize():
                if args.localize_mode == "cpu_original":
                    _, prob_dist_out, orientations_out, _ = localize(
                        torch.tensor(desdf_data["desdf"]),
                        pred_rays,
                        return_np=False,
                    )
                    return prob_dist_out, orientations_out

                if args.localize_mode == "cpu_fast":
                    prob_dist_out, orientations_out, _ = localize_fast(
                        torch.tensor(desdf_data["desdf"], dtype=torch.float32),
                        pred_rays,
                        return_np=False,
                    )
                    return prob_dist_out.cpu(), orientations_out.cpu()

                prob_dist_out, orientations_out, _ = localize_fast(
                    desdf_tensors[scene],
                    pred_rays.to(device),
                    return_np=False,
                )
                return prob_dist_out.cpu(), orientations_out.cpu()

            prob_dist, orientations, elapsed = None, None, None
            (prob_dist, orientations), elapsed = timed_call(
                device if args.localize_mode == "gpu_fast" else "cpu",
                run_localize,
            )
            fixed_times["localize"].append(elapsed)

            img_tokens, elapsed = timed_call(
                device, lambda: disco_model.encode_image(obs_img_tensor)
            )
            fixed_times["image_encode"].append(elapsed)

        def do_topk_cluster():
            flat_probs = prob_dist.flatten()
            source_k = min(args.cluster_source_top_k, flat_probs.numel())
            topk_vals, topk_indices = torch.topk(flat_probs, k=source_k)
            height, width = prob_dist.shape
            rep_vals, rep_indices, _, cluster_sizes = cluster_topk_candidates(
                topk_vals,
                topk_indices,
                width=width,
                radius_m=args.cluster_radius_m,
                meters_per_cell=meters_per_desdf_cell,
            )
            return rep_vals, rep_indices, cluster_sizes, width

        (rep_vals, rep_indices, cluster_sizes, width), elapsed = timed_call("cpu", do_topk_cluster)
        fixed_times["topk_cluster"].append(elapsed)
        rep_counts.append(int(rep_indices.numel()))

        for topn in topn_values:
            take_n = min(topn, rep_indices.numel())
            current_vals = rep_vals[:take_n]
            current_indices = rep_indices[:take_n]
            topk_y = current_indices // width
            topk_x = current_indices % width

            rerank_start = time.perf_counter()
            local_maps = []
            crop_start = time.perf_counter()
            for py_tensor, px_tensor in zip(topk_y, topk_x):
                py = int(py_tensor.item())
                px = int(px_tensor.item())
                orn_idx = int(orientations[py, px].item())
                theta = (orn_idx / 36) * 2 * np.pi
                map_x = px * desdf_stride + desdf_data["l"]
                map_y = py * desdf_stride + desdf_data["t"]
                local_map = crop_local_map(
                    scene_map,
                    map_x,
                    map_y,
                    theta,
                    crop_size_meters=crop_size_meters,
                    res=map_res,
                )
                local_maps.append(torch.from_numpy(local_map).float().unsqueeze(0) / 255.0)
            crop_elapsed = time.perf_counter() - crop_start

            if local_maps:
                local_maps_batch, stack_elapsed = timed_call(
                    device, lambda: torch.stack(local_maps).to(device)
                )
                with torch.no_grad():
                    sim_scores, score_elapsed = timed_call(
                        device, lambda: disco_model.score_candidates(img_tokens, local_maps_batch)
                    )
                    _, fusion_elapsed = timed_call(
                        device,
                        lambda: torch.argmax(current_vals.to(device) * torch.exp(sim_scores * args.alpha)),
                    )
            else:
                stack_elapsed = 0.0
                score_elapsed = 0.0
                fusion_elapsed = 0.0

            rerank_elapsed = time.perf_counter() - rerank_start
            if sample_idx >= args.warmup_samples:
                rerank_times[topn]["crop"].append(crop_elapsed)
                rerank_times[topn]["stack_to_device"].append(stack_elapsed)
                rerank_times[topn]["disco_score"].append(score_elapsed)
                rerank_times[topn]["fusion_select"].append(fusion_elapsed)
                rerank_times[topn]["rerank_total"].append(rerank_elapsed)
                rerank_times[topn]["effective_candidates"].append(int(take_n))

        if sample_idx < args.warmup_samples:
            for key in fixed_times:
                fixed_times[key].pop()
            rep_counts.pop()

        measured = max(0, sample_idx + 1 - args.warmup_samples)
        progress.set_postfix({"measured": measured, "rep": f"{np.mean(rep_counts):.1f}" if rep_counts else "warmup"})

    fixed_summary = {name: summarize(values) for name, values in fixed_times.items()}
    rerank_summary = {}
    for topn, stats in rerank_times.items():
        rerank_summary[str(topn)] = {
            name: summarize(values)
            for name, values in stats.items()
            if name != "effective_candidates"
        }
        effective = np.array(stats["effective_candidates"], dtype=np.float64)
        rerank_summary[str(topn)]["effective_candidates_mean"] = (
            float(effective.mean()) if effective.size else 0.0
        )

    total_fixed = np.sum(
        [np.array(values, dtype=np.float64) for values in fixed_times.values()],
        axis=0,
    )
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": max(0, len(indices) - args.warmup_samples),
        "warmup_samples": args.warmup_samples,
        "topn_values": topn_values,
        "cluster_source_top_k": args.cluster_source_top_k,
        "cluster_radius_m": args.cluster_radius_m,
        "localize_mode": args.localize_mode,
        "avg_rep_count": float(np.mean(rep_counts)) if rep_counts else 0.0,
        "fixed_total": summarize(total_fixed.tolist() if total_fixed.size else []),
        "fixed_stages": fixed_summary,
        "rerank_by_topn": rerank_summary,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"disco_cluster_topn_speed_s3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nS3D DisCo clustered top-N speed benchmark")
    print("-" * 88)
    print(f"measured samples: {result['num_samples']} (warmup {args.warmup_samples})")
    print(f"localize mode: {args.localize_mode}")
    print(f"avg representative clusters: {result['avg_rep_count']:.1f}")
    print(
        "fixed pipeline median: "
        f"{result['fixed_total']['median_ms']:.1f} ms "
        "(RRP + localize + image encode + topk/cluster)"
    )
    print("\nTopN | eff cand | crop ms | score ms | rerank total ms | full sample ms")
    for topn in topn_values:
        row = rerank_summary[str(topn)]
        full_sample_ms = result["fixed_total"]["median_ms"] + row["rerank_total"]["median_ms"]
        print(
            f"{topn:>4} | "
            f"{row['effective_candidates_mean']:>8.1f} | "
            f"{row['crop']['median_ms']:>7.1f} | "
            f"{row['disco_score']['median_ms']:>8.1f} | "
            f"{row['rerank_total']['median_ms']:>15.1f} | "
            f"{full_sample_ms:>14.1f}"
        )
    print("-" * 88)
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark S3D DisCo clustered top-N inference speed")
    parser.add_argument("--config", "-c", default="configs/paper/disco_s3d.yaml", type=str)
    parser.add_argument("--dataset_path", type=str, default="./datasets_s3d/Structured3D")
    parser.add_argument("--desdf_path", type=str, default="./datasets_s3d/desdf")
    parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_s3d_best.ckpt")
    parser.add_argument("--disco_model_ckpt", type=str, default="checkpoints/DisCo_s3d_best.ckpt")
    parser.add_argument("--topn_values", type=str, default="1,2,5,10,20,50")
    parser.add_argument("--cluster_source_top_k", type=int, default=1000)
    parser.add_argument("--cluster_radius_m", type=float, default=0.6)
    parser.add_argument(
        "--localize_mode",
        choices=["cpu_original", "cpu_fast", "gpu_fast"],
        default="gpu_fast",
        help="cpu_original uses the legacy localize, cpu_fast/gpu_fast use localize_fast without full prob_vol.",
    )
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--warmup_samples", type=int, default=3)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=97)
    parser.add_argument("--all_imgs", action="store_true", default=True)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--img_id", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./eval/logs/rrp_cluster")
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
