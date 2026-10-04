# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Named stress conditions for evaluating (and later training) the 60-input Ant.

A condition string is one or more ``axis:value`` items joined by ``+``, e.g.
``slope:0.3``, ``friction:0.2``, ``friction:0.3+rough:0.03+push:1.0`` (beach-like).

Axes:
    flat            original infinite plane (reference, same as Isaac-Ant-v0)
    slope:g         constant uphill grade g (rise/run) along +x, starting 1 m ahead of spawn
    hills:g         sinusoidal hills along +x with maximum grade g (wavelength 8 m)
    rough:h         uniform random height noise in [-h, +h] m
    friction:mu     ground static/dynamic friction mu (combine mode "min", so mu is effective)
    push:v          random xy velocity kick of up to +-v m/s every 2-4 s
    mass:s          all robot link masses scaled by s (payload / weaker motors)
    water:c         submerged: linear drag -c*m*v on every link (c in 1/s)
    buoy:b          submerged: upward buoyancy b*m*g on every link (0 <= b < 1 keeps it on the bottom)

Terrain axes (slope, hills, rough) use a single long 400 m x 20 m track so a fast
policy (~7 m/s) does not reach the generated grid edge within the 16 s episode.
"""

from __future__ import annotations

import math

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .mdp import water_forces

TRACK_SIZE = (400.0, 20.0)
HILL_WAVELENGTH = 8.0
# flat run-up in front of the spawn point before the terrain feature starts
RUN_UP = 1.0


STAIR_WIDTH = 0.3
STAIRS_PER_FLIGHT = 5
BOX_WIDTH = 0.45


@height_field_to_mesh
def forward_profile_terrain(difficulty: float, cfg: ForwardProfileTerrainCfg) -> np.ndarray:
    """Terrain that starts 1 m in front of the tile centre (the spawn point) and continues along +x.

    ``grade`` (or ``grade_range`` sampled with the generator's difficulty) is the profile parameter:
    rise/run for "slope", maximum grade for "hills", step height in m for "stairs" (5 steps up, 5 down,
    repeated) and maximum block height in m for "boxes" (random 0.45 m blocks).
    """
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    x = np.arange(width) * cfg.horizontal_scale - cfg.size[0] / 2 - RUN_UP
    ahead = x >= 0.0
    x = np.clip(x, 0.0, None)
    grade = cfg.grade if cfg.grade_range is None else cfg.grade_range[0] + difficulty * (
        cfg.grade_range[1] - cfg.grade_range[0]
    )
    if cfg.profile == "slope":
        z = grade * x
    elif cfg.profile == "hills":
        amplitude = grade * HILL_WAVELENGTH / math.pi
        z = 0.5 * amplitude * (1.0 - np.cos(2.0 * math.pi * x / HILL_WAVELENGTH))
    elif cfg.profile == "stairs":
        phase = np.floor(x / STAIR_WIDTH).astype(int) % (2 * STAIRS_PER_FLIGHT)
        z = grade * np.minimum(phase, 2 * STAIRS_PER_FLIGHT - phase) * ahead
    elif cfg.profile == "boxes":
        cell = int(BOX_WIDTH / cfg.horizontal_scale)
        blocks = np.random.uniform(0.0, grade, size=(width // cell + 1, length // cell + 1))
        z2d = np.kron(blocks, np.ones((cell, cell)))[:width, :length] * ahead[:, None]
        return np.rint(z2d / cfg.vertical_scale).astype(np.int16)
    else:
        raise ValueError(f"Unknown profile: {cfg.profile}")
    heights = np.repeat((z / cfg.vertical_scale)[:, None], length, axis=1)
    return np.rint(heights).astype(np.int16)


@configclass
class ForwardProfileTerrainCfg(HfTerrainBaseCfg):
    function = forward_profile_terrain
    profile: str = "slope"
    grade: float = 0.0
    grade_range: tuple[float, float] | None = None


def _track(sub_terrain) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
        size=TRACK_SIZE,
        border_width=5.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        curriculum=False,
        use_cache=False,
        sub_terrains={"track": sub_terrain},
    )


def parse_condition(spec: str) -> dict[str, float]:
    """Parse ``"a:1+b:2"`` into ``{"a": 1.0, "b": 2.0}``; ``"flat"`` gives ``{}``."""
    items = {}
    for item in spec.split("+"):
        item = item.strip()
        if item in ("", "flat"):
            continue
        axis, _, value = item.partition(":")
        if axis not in ("slope", "hills", "rough", "friction", "push", "mass", "water", "buoy") or not value:
            raise ValueError(f"Bad condition item '{item}'. See conditions.py for the syntax.")
        items[axis] = float(value)
    if len({"slope", "hills", "rough"} & items.keys()) > 1:
        raise ValueError("Use at most one terrain axis (slope, hills, rough) per condition.")
    return items


def apply_condition(env_cfg, spec: str):
    """Modify a Mountain/Flat Ant env config in place for the given condition."""
    items = parse_condition(spec)
    terrain = env_cfg.scene.terrain

    if "slope" in items or "hills" in items:
        profile = "slope" if "slope" in items else "hills"
        sub = ForwardProfileTerrainCfg(profile=profile, grade=items[profile], border_width=0.0)
        terrain.terrain_type = "generator"
        terrain.terrain_generator = _track(sub)
    elif "rough" in items:
        h = items["rough"]
        sub = terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(-h, h), noise_step=0.005, downsampled_scale=0.2, border_width=0.0
        )
        terrain.terrain_type = "generator"
        terrain.terrain_generator = _track(sub)

    if "friction" in items:
        mu = items["friction"]
        terrain.physics_material.static_friction = mu
        terrain.physics_material.dynamic_friction = mu
        # "min" has priority over the robot's "average", so the contact uses mu
        terrain.physics_material.friction_combine_mode = "min"

    if "push" in items:
        v = items["push"]
        env_cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
        )

    if "mass" in items:
        s = items["mass"]
        env_cfg.events.scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (s, s),
                "operation": "scale",
            },
        )
    if "water" in items or "buoy" in items:
        env_cfg.events.water = EventTerm(
            func=water_forces,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"drag": items.get("water", 0.0), "buoyancy": items.get("buoy", 0.0)},
        )
    return env_cfg
