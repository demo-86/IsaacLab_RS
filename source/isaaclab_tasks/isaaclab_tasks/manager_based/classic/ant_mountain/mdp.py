# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Local ground queries for the static Mountain terrain; no added policy inputs."""

from __future__ import annotations

import numpy as np
import torch
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.classic.humanoid.mdp import progress_reward
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


def ground_height(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local ground altitude and validity, including immediately after resets."""
    position = env.scene["robot"].data.root_pos_w
    if env.scene.terrain.cfg.terrain_type == "plane":
        return torch.zeros_like(position[:, 2]), torch.ones_like(position[:, 2], dtype=torch.bool)
    if env.scene.terrain.cfg.terrain_type != "generator":
        raise ValueError("Ant Mountain supports only plane or generated terrain.")
    # One static mesh per environment instance, not shared between simulator stages.
    if not hasattr(env, "_ant_mountain_ground_mesh"):
        prim = sim_utils.get_first_matching_child_prim(
            env.scene.terrain.cfg.prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if prim is None:
            raise RuntimeError("Generated Ant Mountain terrain has no mesh.")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        env._ant_mountain_ground_mesh = convert_to_warp_mesh(points, indices, device=env.device)
    starts = position.clone()
    starts[:, 2] += 20.0
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    hits, _, _, _ = raycast_mesh(starts, directions, env._ant_mountain_ground_mesh)
    valid = torch.isfinite(hits[:, 2])
    return torch.where(valid, hits[:, 2], env.scene.env_origins[:, 2]), valid


def base_height_above_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Replace just the scalar height input with local clearance."""
    height, _ = ground_height(env)
    return (env.scene["robot"].data.root_pos_w[:, 2] - height).unsqueeze(-1)


def fallen_below_ground_height(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """Apply the baseline fall threshold relative to local terrain."""
    height, valid = ground_height(env)
    return valid & (env.scene["robot"].data.root_pos_w[:, 2] - height < minimum_height)


def outside_terrain(env: ManagerBasedRLEnv, margin: float = 0.5) -> torch.Tensor:
    """Truncate at the generated grid edge or if a ground query misses."""
    if env.scene.terrain.cfg.terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    generator = env.scene.terrain.cfg.terrain_generator
    position = env.scene["robot"].data.root_pos_w
    _, valid = ground_height(env)
    return (
        ~valid
        | (position[:, 0].abs() >= generator.num_rows * generator.size[0] / 2 - margin)
        | (position[:, 1].abs() >= generator.num_cols * generator.size[1] / 2 - margin)
    )


def water_forces(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    drag: float | tuple[float, float],
    buoyancy: float | tuple[float, float],
    fraction: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Approximate a fully submerged robot with per-link buoyancy and linear drag (world frame).

    PhysX has no fluid, so each link gets ``buoyancy * m * g`` upwards and ``-drag * m * v``. With a
    ``(min, max)`` range, each environment keeps its own coefficients, sampled at the first call.
    Only a random ``fraction`` of the environments is under water; the rest get no force.
    Called every control step as an interval event with ``interval_range_s=(0.0, 0.0)``.
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_water_coeffs"):
        def sample(value):
            low, high = (value, value) if isinstance(value, (int, float)) else value
            return torch.empty(env.num_envs, 1, 1, device=env.device).uniform_(low, high)

        wet = (torch.rand(env.num_envs, 1, 1, device=env.device) < fraction).float()
        env._water_coeffs = (wet * sample(drag), wet * sample(buoyancy))
    drag_c, buoy_c = env._water_coeffs
    mass = asset.data.default_mass.to(env.device)[env_ids].unsqueeze(-1)
    gravity = abs(env.cfg.sim.gravity[2])
    forces = -drag_c[env_ids] * mass * asset.data.body_lin_vel_w[env_ids]
    forces[..., 2] += buoy_c[env_ids, :, 0] * mass[..., 0] * gravity
    asset.set_external_force_and_torque(forces, torch.zeros_like(forces), env_ids=env_ids, is_global=True)


def fallen_mixed(env: ManagerBasedRLEnv, minimum_height: float, world_fraction: float) -> torch.Tensor:
    """Ground-relative fall for every environment, plus the original world-height rule
    (``z < minimum_height``, as in Isaac-Ant-v0) for a fixed random ``world_fraction`` of them."""
    if not hasattr(env, "_world_height_termination"):
        env._world_height_termination = torch.rand(env.num_envs, device=env.device) < world_fraction
    world_fall = env.scene["robot"].data.root_pos_w[:, 2] < minimum_height
    return fallen_below_ground_height(env, minimum_height) | (env._world_height_termination & world_fall)


def terrain_levels_distance(
    env: ManagerBasedRLEnv, env_ids, up_distance: float = 4.0, down_distance: float = 2.0
) -> torch.Tensor:
    """Terrain curriculum for the Ant: at reset, promote robots that walked off their tile and demote
    robots that barely moved. Runs before the reset, so the robot is still where the episode ended."""
    terrain = env.scene.terrain
    position = env.scene["robot"].data.root_pos_w[env_ids, :2]
    distance = torch.norm(position - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > up_distance
    move_down = (distance < down_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def flat_ground_mask(env: ManagerBasedRLEnv, threshold: float = 0.03, sensor_name: str = "height_scanner") -> torch.Tensor:
    """True where the height scan sees flat ground (spread of hit heights below ``threshold`` metres)."""
    hits_z = env.scene.sensors[sensor_name].data.ray_hits_w[..., 2]
    return (hits_z.max(dim=1).values - hits_z.min(dim=1).values) < threshold


class flat_progress_bonus(progress_reward):
    """Training-only priority term: the progress reward again, paid only while the scan sees flat ground,
    so that keeping baseline-like speed on the original (flat) terrain is worth more than elsewhere."""

    def __call__(self, env, target_pos, threshold: float = 0.03, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        return super().__call__(env, target_pos, asset_cfg) * flat_ground_mask(env, threshold).float()
