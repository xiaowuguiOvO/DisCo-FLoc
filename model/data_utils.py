import os
import io
from typing import Union, Tuple

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

VISUALIZATION_IMAGE_SIZE = (160, 160)
IMAGE_ASPECT_RATIO = (
    1 / 1
)  

def get_data_path(data_folder: str, f: str, name):
    if type(name) == int or type(name) == np.int16:
        return os.path.join(data_folder, f, "{:05d}.png".format(name))
    elif type(name) == str:
        return os.path.join(data_folder, f, name + ".png") 
    else:
        raise ValueError("name should be either int or str") 


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(            
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def to_global_coords(           
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert     in local coordinates  (b,2) 
        curr_pos (np.ndarray): current position         in global coordinates  (2,)
        curr_yaw (float): current yaw                  in global coordinates   (2,)
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)  # R from local to global
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return curr_pos + positions.dot(rotmat.transpose())


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    img = img.convert('RGB')
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    

def img_path_to_data_and_point_transfer(path: Union[str, io.BytesIO], ori_size: float, image_resize_size: Tuple[int, int], cur_pos: np.ndarray, goal_pos: np.ndarray, cur_ori: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image and two points, transform the image and transfer the points to local coordinates
    Args:
        path (str): path to the image
        cur_pos (np.ndarray): current position in the pixel coordinate of original image [x,y]
        goal_pos (np.ndarray): goal position in the pixel coordinate of original image
        image_resize_size (Tuple[int, int]): size to resize the image to [width, height]
    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray]: resized image as tensor, current position in the transformed image size coordinate, goal position in the same coordinate
    """
    
    with Image.open(path) as img:
        w0 = ori_size
        h0 = ori_size
        w, h = img.size
        cur_pos = cur_pos * 100 + np.array([w0 / 2, h0 / 2])
        goal_pos = goal_pos * 100 + np.array([w0 / 2, h0 / 2])
        cur_ori = cur_ori * 100 + np.array([w0 / 2, h0 / 2])      
        aspect_ratio = IMAGE_ASPECT_RATIO
                  
        img = img.resize(image_resize_size)
        cur_pos_in_resizeSize = cur_pos * image_resize_size[0] / w0
        goal_pos_in_resizeSize = goal_pos * image_resize_size[0] / w0
        cur_ori_in_resizeSize = cur_ori * image_resize_size[0] / w0       
        resize_img = TF.to_tensor(img)
        
        return (resize_img, cur_pos_in_resizeSize, goal_pos_in_resizeSize, cur_ori_in_resizeSize)