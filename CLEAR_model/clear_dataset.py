import os
import sys
from typing import List, Tuple
import yaml
from pathlib import Path
import numpy as np
import tqdm
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

from .data_utils import (
    img_path_to_data,
    img_path_to_data_and_point_transfer,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class CLEAR_Dataset(Dataset):
    def __init__(self, data_folder: str, data_splits_path: str, split: str, floorplan_img_size: Tuple[int, int], pose_aug_params: dict = None, dataset_cfg: dict = None):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.floorplan_img_size = floorplan_img_size
        self.pose_aug_params = pose_aug_params if pose_aug_params else {'enable': False}
        self.dataset_cfg = dataset_cfg
        with open(self.data_splits_path, 'r', encoding='utf-8') as f:
            data_splits = yaml.safe_load(f)
        # Remove all whitespace characters from each scene name
        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)

    def _load_data(self, data_folder, data_split):
        data = []
        for i in range(len(data_split)):
            cur_dir = os.path.join(data_folder, data_split[i])
            # Check if directory exists
            if not os.path.exists(cur_dir):
                # print(f"Warning: {cur_dir} does not exist, skipping.")
                continue

            # 一行一个列表，元素为 float
            pose_data = [
                list(map(float, line.split()))[:3]  # 默认空格/Tab 分隔，只取前3个(x_pix, y_pix, yaw)
                for line in Path(cur_dir + "/poses_map.txt").read_text(encoding='utf-8').splitlines()
                if line.strip()  # 跳过空行
            ]
            ray_data = [
                list(map(float, line.split()))  # 默认空格/Tab 分隔
                for line in Path(cur_dir + "/depth40.txt").read_text(encoding='utf-8').splitlines()
                if line.strip()  # 跳过空行
            ]
            rgb_dir = os.path.join(cur_dir, "imgs")

            def sort_key(p: Path):
                return int(p.stem) # Images are named like '000.png'

            files = sorted((p for p in Path(rgb_dir).iterdir() if p.is_file() and (p.suffix == '.png' or p.suffix == '.jpg')),
                           key=sort_key)
            rgb_names = [f.name for f in files]

            for n in range(len(rgb_names)):
                metadata = {}
                metadata["rgb_image"] = rgb_dir + "/" + rgb_names[n]
                metadata["floorplan_image"] = cur_dir + "/map.png"
                metadata["pose"] = pose_data[n]
                metadata["ray"] = ray_data[n]
                data.append(metadata)
        return data

    def _load_floorplan(self, floorplan_path):
        # Load and resize without cropping to preserve full map context for visualization
        try:
            with Image.open(floorplan_path) as img:
                img = img.convert('RGB') # Standardize to 3 channels (even if map is BW)
                img = img.resize(self.floorplan_img_size) # Direct resize, might distort aspect ratio but keeps content
                return transforms.ToTensor()(img)
        except Exception as e:
            print(f"Failed to load floorplan {floorplan_path}: {e}")
            return torch.zeros((3, *self.floorplan_img_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        data = self.data[i]
        rgb_image = Image.open(data["rgb_image"])
        transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])  
        rgb_image = transfrom(rgb_image)
        
        # rgb_tensor = self.transform(rgb_image)
        
        pose = torch.tensor(data["pose"])
        ray = torch.tensor(data["ray"])

        w, h = Image.open(data["floorplan_image"]).size
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        
        # Use new _load_floorplan to avoid cropping
        floorplan_img = self._load_floorplan(data["floorplan_image"])
        
        # --- Local Map Cropping (Positive) ---
        raw_map = cv2.imread(data["floorplan_image"], 0) # (H, W)
        
        if raw_map is None:
            # Fallback
            local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            neg_local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            print(f"Warning: Failed to load floorplan image {data['floorplan_image']}. Using zero maps.")
        else:
            # Pose Augmentation for Positive Sample (Train only)
            pose_aug = pose.numpy().copy()
            if self.split == 'train' and self.pose_aug_params.get('enable', False):
                trans_range = self.pose_aug_params.get('trans_range', 25) # Default 25 pixels (~0.5m)
                rot_range = self.pose_aug_params.get('rot_range', 0.26) # Default 0.26 rad (~15 deg)
                
                # Shift +/- trans_range
                pose_aug[0] += np.random.uniform(-trans_range, trans_range)
                pose_aug[1] += np.random.uniform(-trans_range, trans_range)
                # Rotate +/- rot_range
                pose_aug[2] += np.random.uniform(-rot_range, rot_range)
            
            crop_size_meters = self.dataset_cfg.get('local_map_crop_size_meters', 5.0)
            local_map_np = self.crop_local_map(raw_map, pose_aug, crop_size_meters)
            local_map = torch.from_numpy(local_map_np).float().unsqueeze(0) / 255.0
            
            # Hard Negative
            neg_pose_list = self.get_hard_negative_pose(pose.numpy())
            neg_pose = torch.tensor(neg_pose_list, dtype=torch.float32)
            neg_local_map_np = self.crop_local_map(raw_map, neg_pose.numpy(), crop_size_meters)
            neg_local_map = torch.from_numpy(neg_local_map_np).float().unsqueeze(0) / 255.0
            
        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
            local_map,
            neg_local_map,
            torch.as_tensor(neg_pose, dtype=torch.float32) if raw_map is not None else torch.zeros(3, dtype=torch.float32)
        )
        
    def get_hard_negative_pose(self, pose):
        """
        Generate a hard negative pose by perturbing the ground truth pose.
        pose: [x, y, theta]
        """
        x, y, theta = pose
        
        if np.random.rand() < 0.5:
            # Rotate by 180 degrees (pi radians) + small noise
            theta_new = theta + np.pi + np.random.uniform(-0.2, 0.2)
            return [x, y, theta_new]
        else:
            # Shift by 1.5m to 3.0m (75 to 150 pixels)
            dist_px = np.random.uniform(75, 150)
            angle = np.random.uniform(0, 2 * np.pi)
            
            x_new = x + dist_px * np.cos(angle)
            y_new = y + dist_px * np.sin(angle)
            
            # Keep original theta or add small noise
            theta_new = theta + np.random.uniform(-0.2, 0.2)
            
            return [x_new, y_new, theta_new]

    def crop_local_map(self, map_img, pose, crop_size_meters, res=0.02, output_size=128):
        """
        Crop an ego-centric map patch.
        pose: [x (pixels), y (pixels), theta (radians)]
        """
        x, y, theta = pose
        
        # Pixels to crop
        crop_size_px = int(crop_size_meters / res) # e.g. 250
        
        # 1. Pad map to handle borders
        H, W = map_img.shape
        pad = crop_size_px
        map_padded = cv2.copyMakeBorder(map_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255) # 255 for white space (empty)
        
        # Adjust center for padding
        center = (x + pad, y + pad)
        
        angle_deg = np.degrees(theta) 

        rot_matrix = cv2.getRotationMatrix2D(center, angle_deg + 90, 1.0)

        rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
        rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]
        
        local_map = cv2.warpAffine(
            map_padded, rot_matrix, (crop_size_px, crop_size_px), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        
        # 4. Resize to output size (e.g. 128)
        if crop_size_px != output_size:
            local_map = cv2.resize(local_map, (output_size, output_size), interpolation=cv2.INTER_AREA)
            
        return local_map
        
if __name__ == "__main__":
    DATA_FOLDER = 'datasets_s3d/Structured3D'
    DATA_SPLITS_PATH = 'datasets_s3d/Structured3D/split.yaml'
    SPLIT = 'train'
    RGB_SIZE = (256, 256)
    
    dataset = CLEAR_Dataset(
            data_folder=DATA_FOLDER,
            data_splits_path=DATA_SPLITS_PATH,
            split=SPLIT,
            rgb_image_size=RGB_SIZE,
            floorplan_img_size=(256, 256), # Added missing arg
            pose_aug_params={'enable': True, 'trans_range': 25, 'rot_range': 0.26}
        )