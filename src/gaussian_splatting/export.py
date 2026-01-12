"""
Export utilities for 3D Gaussian Splatting.
Exports to PLY format compatible with Unity Gaussian Splatting plugins.
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Optional
import os


def export_ply(
    model: 'GaussianModel',
    output_path: str,
    include_sh: bool = True,
    max_sh_degree: int = 3
):
    """
    Export Gaussian model to PLY file.
    
    Format is compatible with:
    - UnityGaussianSplatting (https://github.com/aras-p/UnityGaussianSplatting)
    - Original 3DGS viewer
    
    Args:
        model: Trained GaussianModel
        output_path: Path to save PLY file
        include_sh: Whether to include spherical harmonics coefficients
        max_sh_degree: Maximum SH degree to export (0-3)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Get model parameters
    xyz = model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)  # Not used, but required for format
    
    # DC color (first SH coefficient)
    f_dc = model._features_dc.detach().cpu().numpy()  # [N, 1, 3]
    f_dc = f_dc.reshape(-1, 3)  # [N, 3]
    
    # Higher-order SH
    if include_sh and max_sh_degree > 0:
        f_rest = model._features_rest.detach().cpu().numpy()  # [N, 15, 3]
        # Limit to requested degree
        num_sh = min(f_rest.shape[1], (max_sh_degree + 1) ** 2 - 1)
        f_rest = f_rest[:, :num_sh, :]
        f_rest = f_rest.reshape(-1, num_sh * 3)  # [N, num_sh*3]
    else:
        f_rest = None
    
    # Opacity (as logit, inverse of sigmoid)
    opacities = model._opacity.detach().cpu().numpy()  # [N, 1]
    
    # Scale (as log-scale)
    scales = model._scaling.detach().cpu().numpy()  # [N, 3]
    
    # Rotation (quaternion, wxyz)
    rotations = model._rotation.detach().cpu().numpy()  # [N, 4]
    # Normalize quaternions
    rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-10)
    
    # Build PLY structure
    n = xyz.shape[0]
    print(f"Exporting {n} Gaussians to {output_path}")
    
    # Define dtype
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    
    # Add SH rest if included
    if f_rest is not None:
        for i in range(f_rest.shape[1]):
            dtype_list.append((f'f_rest_{i}', 'f4'))
    
    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    # Create structured array
    elements = np.empty(n, dtype=dtype_list)
    
    # Fill in values
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    if f_rest is not None:
        for i in range(f_rest.shape[1]):
            elements[f'f_rest_{i}'] = f_rest[:, i]
    
    elements['opacity'] = opacities.squeeze()
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rotations[:, 0]  # w
    elements['rot_1'] = rotations[:, 1]  # x
    elements['rot_2'] = rotations[:, 2]  # y
    elements['rot_3'] = rotations[:, 3]  # z
    
    # Create PLY
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"PLY export complete: {output_path}")
    print(f"  - Vertices: {n}")
    print(f"  - SH degree: {max_sh_degree if include_sh else 0}")


def load_ply(path: str, device: torch.device = None) -> dict:
    """
    Load Gaussian parameters from PLY file.
    
    Args:
        path: Path to PLY file
        device: Target device for tensors
    
    Returns:
        Dictionary with Gaussian parameters
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # Position
    xyz = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=1)
    
    # Normals (usually not used)
    if 'nx' in vertex:
        normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=1)
    else:
        normals = None
    
    # DC features
    f_dc = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=1)
    
    # Find number of SH rest features
    sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith('f_rest_')])
    if sh_rest_names:
        f_rest = np.stack([vertex[name] for name in sh_rest_names], axis=1)
    else:
        f_rest = None
    
    # Opacity
    opacity = vertex['opacity'][:, np.newaxis]
    
    # Scale
    scale = np.stack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ], axis=1)
    
    # Rotation
    rotation = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=1)
    
    return {
        'xyz': torch.tensor(xyz, dtype=torch.float32, device=device),
        'f_dc': torch.tensor(f_dc, dtype=torch.float32, device=device).unsqueeze(1),
        'f_rest': torch.tensor(f_rest, dtype=torch.float32, device=device).reshape(-1, len(sh_rest_names) // 3, 3) if f_rest is not None else None,
        'opacity': torch.tensor(opacity, dtype=torch.float32, device=device),
        'scaling': torch.tensor(scale, dtype=torch.float32, device=device),
        'rotation': torch.tensor(rotation, dtype=torch.float32, device=device),
    }


def export_cameras_json(cameras: list, output_path: str):
    """
    Export camera parameters to JSON for visualization.
    
    Args:
        cameras: List of Camera objects
        output_path: Path to save JSON file
    """
    import json
    
    camera_data = []
    for cam in cameras:
        camera_data.append({
            'id': cam.uid,
            'img_name': cam.image_name,
            'width': cam.image_width,
            'height': cam.image_height,
            'position': cam.camera_center.cpu().tolist(),
            'rotation': cam.R.cpu().tolist(),
            'fx': cam.focal_x,
            'fy': cam.focal_y,
        })
    
    with open(output_path, 'w') as f:
        json.dump({'cameras': camera_data}, f, indent=2)
    
    print(f"Exported {len(cameras)} cameras to {output_path}")
