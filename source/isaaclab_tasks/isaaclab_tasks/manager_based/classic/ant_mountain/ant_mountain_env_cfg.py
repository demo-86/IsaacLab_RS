# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
from typing import Literal

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from .terrains import (
    ANT_LANES_TERRAINS_CFG,
    ANT_MOUNTAIN_TERRAINS_CFG,
    ANT_ROBUST2_TERRAINS_CFG,
    ANT_ROBUST_TERRAINS_CFG,
)
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, TerminationsCfg

from . import mdp
from .conditions import apply_condition


@configclass
class MountainTerminationsCfg(TerminationsCfg):
    torso_height = DoneTerm(func=mdp.fallen_below_ground_height, params={"minimum_height": 0.31})
    terrain_boundary = DoneTerm(func=mdp.outside_terrain, time_out=True)


@configclass
class AntMountainEnvCfg(AntEnvCfg):
    """60 observations and baseline actions/rewards, with gentle generated terrain.

    Default world-height input preserves the trained policy's observation meaning.
    Ground-relative preprocessing is an explicit optional evaluation condition.
    """

    height_observation: Literal["world", "ground"] = "world"
    terminations: MountainTerminationsCfg = MountainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = self.scene.terrain.replace(
            terrain_type="generator", terrain_generator=ANT_MOUNTAIN_TERRAINS_CFG.copy()
        )
        if self.height_observation not in ("world", "ground"):
            raise ValueError("height_observation must be 'world' or 'ground'.")
        # Resolve mode at call time too, so Hydra CLI overrides work after construction.
        self.observations.policy.base_height.func = mountain_base_height
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


def mountain_base_height(env):
    """Preserve baseline height by default; allow explicitly requested preprocessing."""
    if env.cfg.height_observation == "ground":
        return mdp.base_height_above_ground(env)
    if env.cfg.height_observation == "world":
        return env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(-1)
    raise ValueError("height_observation must be 'world' or 'ground'.")


@configclass
class AntMountainFlatEnvCfg(AntMountainEnvCfg):
    """Flat control to compare the same checkpoint against the baseline task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = AntEnvCfg().scene.terrain.copy()


@configclass
class AntEvalEnvCfg(AntMountainFlatEnvCfg):
    """Flat Ant plus the stress condition named in the ``ANT_CONDITION`` environment variable.

    Example: ``ANT_CONDITION=slope:0.3`` (see conditions.py). Unset or ``flat`` gives the plain plane.
    Falls are ground-relative unless ``ANT_FALL=world``.
    """

    def __post_init__(self):
        super().__post_init__()
        apply_condition(self, os.environ.get("ANT_CONDITION", "flat"))
        # ANT_FALL=world restores Isaac-Ant-v0's rule (world z < 0.31 m), as the TA's environment uses
        if os.environ.get("ANT_FALL", "ground") == "world":
            self.terminations.torso_height = DoneTerm(
                func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.31}
            )


@configclass
class AntRobustEnvCfg(AntMountainEnvCfg):
    """Experiment B: baseline observations, actions and rewards; randomized terrain, friction and mass.

    Ranges stay below the held-out evaluation levels so those remain unseen during training.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ROBUST_TERRAINS_CFG.copy()
        # ~170 robots share each lane origin: spread them over the flat run-up behind it and across
        # the 8 m lane (forward-profile lanes are flat for x < origin), and enlarge the contact buffer
        self.events.reset_base.params["pose_range"] = {"x": (-20.0, 0.0), "y": (-3.0, 3.0)}
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        # ground "min" combine makes each robot's sampled friction the effective contact friction
        self.scene.terrain.physics_material.friction_combine_mode = "min"
        self.events.robot_friction = EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.events.robot_mass = EventTerm(
            func=base_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.3),
                "operation": "scale",
            },
        )


@configclass
class AntRobust2EnvCfg(AntRobustEnvCfg):
    """Experiment B2: B's friction/mass randomization on stairs/boxes/rough/slope tiles, 30% under water.

    Observations, actions and rewards stay those of Isaac-Ant-v0. Water drag <= 3 1/s and buoyancy <= 0.6
    so stronger water remains unseen during training.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ROBUST2_TERRAINS_CFG.copy()
        # ~10 robots per tile origin here, so spawn at the tile centre like Isaac Lab's rough tasks
        self.events.reset_base.params["pose_range"] = {}
        self.events.water = EventTerm(
            func=mdp.water_forces,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"drag": (0.0, 3.0), "buoyancy": (0.0, 0.6), "fraction": 0.3},
        )


@configclass
class AntRobust3EnvCfg(AntRobust2EnvCfg):
    """Experiment B3: B2 plus, for half of the environments, the original world-height fall rule
    (z < 0.31 m) that Isaac-Ant-v0 uses; the policy observes world height, so it can learn to avoid it."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.torso_height = DoneTerm(
            func=mdp.fallen_mixed, params={"minimum_height": 0.31, "world_fraction": 0.5}
        )


@configclass
class AntRobust4EnvCfg(AntRobustEnvCfg):
    """Experiment B4: B2's terrain types, water, friction and mass, but on single-type 300 m lanes
    (30% flat) instead of mixed 8 m tiles. Observations, actions and rewards are unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_LANES_TERRAINS_CFG.copy()
        self.events.water = EventTerm(
            func=mdp.water_forces,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"drag": (0.0, 3.0), "buoyancy": (0.0, 0.6), "fraction": 0.3},
        )


@configclass
class AntRobust5NoDrEnvCfg(AntRobust2EnvCfg):
    """B5-noDR: B2 without friction and mass randomization (terrain and water unchanged)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_friction = None
        self.events.robot_mass = None


@configclass
class AntRobust5NarrowEnvCfg(AntRobust2EnvCfg):
    """B5-narrow: B2 with narrower friction (0.6-1.0) and mass (0.9-1.1x) randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_friction.params["static_friction_range"] = (0.6, 1.0)
        self.events.robot_friction.params["dynamic_friction_range"] = (0.6, 1.0)
        self.events.robot_mass.params["mass_distribution_params"] = (0.9, 1.1)


@configclass
class AntRobust5Flat40EnvCfg(AntRobust2EnvCfg):
    """B5-flat40: B2 with 40% flat tiles; the other tile types keep their relative shares."""

    def __post_init__(self):
        super().__post_init__()
        tiles = self.scene.terrain.terrain_generator.sub_terrains
        rest = sum(t.proportion for name, t in tiles.items() if name != "flat")
        for name, t in tiles.items():
            t.proportion = 0.4 if name == "flat" else t.proportion * 0.6 / rest


@configclass
class TerrainCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_distance)


@configclass
class AntTerrainCurriculumEnvCfg(AntRobust5NoDrEnvCfg):
    """Experiment T-curriculum: T's tiles and water, with difficulty rising by grid row; robots start
    in the four easiest rows and move up after walking off their 8 m tile, down after < 2 m."""

    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 3


@configclass
class AntTerrainHardEnvCfg(AntRobust5NoDrEnvCfg):
    """Experiment T-hard: T with obstacle limits raised to Isaac Lab's ROUGH_TERRAINS_CFG ranges
    (stairs 5-23 cm, boxes 5-20 cm, rough 2-10 cm, slopes 0-0.4); tile shares and water unchanged."""

    def __post_init__(self):
        super().__post_init__()
        tiles = self.scene.terrain.terrain_generator.sub_terrains
        tiles["pyramid_stairs"].step_height_range = (0.05, 0.23)
        tiles["pyramid_stairs_inv"].step_height_range = (0.05, 0.23)
        tiles["boxes"].grid_height_range = (0.05, 0.20)
        tiles["random_rough"].noise_range = (0.02, 0.10)
        tiles["random_rough"].noise_step = 0.02
        tiles["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        tiles["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)


@configclass
class AntR1EnvCfg(AntEnvCfg):
    """R1 evaluation: the original Isaac-Ant-v0 (observations, rewards, world-height fall rule) on
    Isaac Lab's ROUGH_TERRAINS_CFG. Reproduces the lecture's unseen-terrain example (baseline ~5-6)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = None


# --- Observation-history variants (analysis only: the policy input becomes 60 * k) -------------
# Same environments as T / Eval / R1; the policy group keeps the last k observations, flattened.


def _stack_history(cfg, k: int):
    cfg.observations.policy.history_length = k
    cfg.observations.policy.flatten_history_dim = True


@configclass
class AntTHist8EnvCfg(AntRobust5NoDrEnvCfg):
    """T with the last 8 observations stacked (480 inputs)."""

    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 8)


@configclass
class AntTHist4EnvCfg(AntRobust5NoDrEnvCfg):
    """T with the last 4 observations stacked (240 inputs)."""

    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 4)


@configclass
class AntEvalHist8EnvCfg(AntEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 8)


@configclass
class AntEvalHist4EnvCfg(AntEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 4)


@configclass
class AntR1Hist8EnvCfg(AntR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 8)


@configclass
class AntR1Hist4EnvCfg(AntR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _stack_history(self, 4)


@configclass
class AntDrHardEnvCfg(AntRobust2EnvCfg):
    """DR_HARD: T-hard obstacle limits (stairs 5-23 cm, boxes 5-20 cm, rough 2-10 cm, slopes 0-0.4) with
    B2's friction (0.3-1.0) and mass (0.8-1.3x) randomization; tile shares and water as in T."""

    def __post_init__(self):
        super().__post_init__()
        tiles = self.scene.terrain.terrain_generator.sub_terrains
        tiles["pyramid_stairs"].step_height_range = (0.05, 0.23)
        tiles["pyramid_stairs_inv"].step_height_range = (0.05, 0.23)
        tiles["boxes"].grid_height_range = (0.05, 0.20)
        tiles["random_rough"].noise_range = (0.02, 0.10)
        tiles["random_rough"].noise_step = 0.02
        tiles["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        tiles["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)


# --- Height-scan sensor variants: the policy also sees the terrain around the torso ------------
# A yaw-aligned grid of downward rays (1.6 m x 1.2 m, 0.2 m spacing = 63 points) attached to the
# torso; the 60 original observations are kept in order and the 63 heights are appended (123 inputs).


def _add_height_scan(cfg):
    # the ray caster looks robots up on the USD stage, which fabric cloning leaves empty beyond env_0
    cfg.scene.clone_in_fabric = False
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.2]),
        mesh_prim_paths=["/World/ground"],
        update_period=cfg.decimation * cfg.sim.dt,
        debug_vis=False,
    )
    cfg.observations.policy.height_scan = ObsTerm(
        func=base_mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
    )


@configclass
class AntTScanEnvCfg(AntRobust5NoDrEnvCfg):
    """T with the height-scan sensor."""

    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)


@configclass
class AntEvalScanEnvCfg(AntEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)


@configclass
class AntR1ScanEnvCfg(AntR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)


@configclass
class AntTScanCurEnvCfg(AntTerrainCurriculumEnvCfg):
    """T_SCAN_CUR: height scan + terrain curriculum (difficulty by row; promote after > 4 m, demote after < 2 m)."""

    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)


@configclass
class AntTScanCurDrEnvCfg(AntTScanCurEnvCfg):
    """T_SCAN_CUR_DR: Rudin et al. (2021)-style recipe for the Ant: height scan + terrain curriculum + domain
    randomization of friction (0.3-1.0), link masses (0.8-1.3x) and random pushes (+-1 m/s every 10-15 s)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_friction = EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.events.robot_mass = EventTerm(
            func=base_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.3),
                "operation": "scale",
            },
        )
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )


@configclass
class AntTScanF40EnvCfg(AntTScanEnvCfg):
    """T_SCAN_F40: T_SCAN with 40% flat tiles (other tile types keep their relative shares)."""

    def __post_init__(self):
        super().__post_init__()
        tiles = self.scene.terrain.terrain_generator.sub_terrains
        rest = sum(t.proportion for name, t in tiles.items() if name != "flat")
        for name, t in tiles.items():
            t.proportion = 0.4 if name == "flat" else t.proportion * 0.6 / rest


@configclass
class AntScanLanesEnvCfg(AntRobust4EnvCfg):
    """SCAN_LANES: height scan on single-type 300 m lanes (30% flat, stairs, boxes, rough, slopes, hills) with
    water as in T and no friction/mass randomization, so long flat stretches exist during training."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_friction = None
        self.events.robot_mass = None
        _add_height_scan(self)


def _center_friction_on_one(cfg):
    """Effective contact friction 0.5-1.5 (centre 1.0, the evaluation plane's value): the ground keeps the "min"
    combine mode but is raised to 1.5, so each robot's sampled friction is the effective one."""
    cfg.scene.terrain.physics_material.static_friction = 1.5
    cfg.scene.terrain.physics_material.dynamic_friction = 1.5
    cfg.events.robot_friction.params["static_friction_range"] = (0.5, 1.5)
    cfg.events.robot_friction.params["dynamic_friction_range"] = (0.5, 1.5)


@configclass
class AntTScanCurDrMuEnvCfg(AntTScanCurDrEnvCfg):
    """T_SCAN_CUR_DR_MU: T_SCAN_CUR_DR with effective friction 0.5-1.5 instead of 0.3-1.0."""

    def __post_init__(self):
        super().__post_init__()
        _center_friction_on_one(self)


@configclass
class AntTScanDrMuEnvCfg(AntTScanCurDrMuEnvCfg):
    """T_SCAN_DR_MU: as T_SCAN_CUR_DR_MU but without the terrain curriculum (random tile difficulty)."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None


@configclass
class AntTScanDrMuProg2EnvCfg(AntTScanDrMuEnvCfg):
    """Training-reward variant: progress weight 2.0 instead of 1.0. Evaluation still uses the original
    Isaac-Ant-v0 reward (Isaac-Ant-Eval-Scan-v0 / Isaac-Ant-R1-Scan-v0)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.progress.weight = 2.0


# --- Compatible variants: the policy keeps the original 60 observations and the baseline MLP --------


def _add_friction_mass_push_dr(cfg):
    """Friction 0.5-1.5 (centred on the evaluation value 1.0), link masses 0.8-1.3x, pushes +-1 m/s every 10-15 s."""
    cfg.events.robot_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    cfg.scene.terrain.physics_material.static_friction = 1.5
    cfg.scene.terrain.physics_material.dynamic_friction = 1.5
    cfg.events.robot_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.3),
            "operation": "scale",
        },
    )
    cfg.events.push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class AntTDrMuEnvCfg(AntRobust5NoDrEnvCfg):
    """T_DR_MU: T + friction (0.5-1.5) / mass / push randomization. Policy observations unchanged (60)."""

    def __post_init__(self):
        super().__post_init__()
        _add_friction_mass_push_dr(self)


@configclass
class AntTDrMuAsymEnvCfg(AntTDrMuEnvCfg):
    """T_DR_MU_ASYM: asymmetric actor-critic. The policy (actor) keeps the original 60 observations; a "critic"
    observation group adds the 63-point height scan, which RSL-RL feeds to the critic only (training-time info)."""

    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)
        # move the scan from the policy group to a critic-only group
        scan_term = self.observations.policy.height_scan
        self.observations.policy.height_scan = None
        self.observations.critic = copy.deepcopy(self.observations.policy)
        self.observations.critic.height_scan = scan_term


@configclass
class AntTScanDrMuLongEnvCfg(AntTScanDrMuEnvCfg):
    """T_SCAN_DR_MU_LONG: T_SCAN_DR_MU with a forward-biased scan, 0.4 m behind to 2.0 m ahead of the torso
    (2.4 m x 1.2 m, 0.2 m spacing = 91 points), so the policy can see far enough ahead to run fast."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.2, size=[2.4, 1.2])
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0))


@configclass
class AntEvalScanLongEnvCfg(AntEvalScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.2, size=[2.4, 1.2])
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0))


@configclass
class AntR1ScanLongEnvCfg(AntR1ScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.2, size=[2.4, 1.2])
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0))


# --- "Keep baseline-level performance on the original terrain" as a training objective -------------


@configclass
class AntTScanDrMuFlatPriorityEnvCfg(AntTScanDrMuEnvCfg):
    """P1: T_SCAN_DR_MU plus a training-only reward term that pays progress again while the scan sees flat ground."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.flat_progress = RewTerm(
            func=mdp.flat_progress_bonus, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0), "threshold": 0.03}
        )


# --- Control-parameter variants (allowed: software control elements such as torque limits) -------------
# ANT_ACTION_SCALE sets the joint-effort scale (torque per unit action; Isaac-Ant-v0 uses 7.5). Evaluation
# variants read the same variable so a model is always evaluated with the torque it was trained with.


def _action_scale(cfg):
    cfg.actions.joint_effort.scale = float(os.environ.get("ANT_ACTION_SCALE", "7.5"))


@configclass
class AntTScanDrMuTqEnvCfg(AntTScanDrMuEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


@configclass
class AntFlatTqEnvCfg(AntEnvCfg):
    """Baseline (flat Isaac-Ant-v0) trained with a different torque scale, for a fair comparison."""

    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


@configclass
class AntEvalTqEnvCfg(AntEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


@configclass
class AntR1TqEnvCfg(AntR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


@configclass
class AntEvalScanTqEnvCfg(AntEvalScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


@configclass
class AntR1ScanTqEnvCfg(AntR1ScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _action_scale(self)


# --- Training-reward variants aligned with the evaluation criterion (distance in 16 s) -------------
# Evaluation still uses the original Isaac-Ant-v0 reward.


@configclass
class AntTScanDrMuDistanceOnlyEnvCfg(AntTScanDrMuEnvCfg):
    """D1: train on the progress (distance) term only."""

    def __post_init__(self):
        super().__post_init__()
        for name in ("alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits"):
            setattr(self.rewards, name, None)


@configclass
class AntTScanDrMuNoPenaltyEnvCfg(AntTScanDrMuEnvCfg):
    """D2: train without the effort penalties (action L2, energy, joint-limit); positive terms unchanged."""

    def __post_init__(self):
        super().__post_init__()
        for name in ("action_l2", "energy", "joint_pos_limits"):
            setattr(self.rewards, name, None)


# --- Submission ------------------------------------------------------------------------------------


@configclass
class AntSubmitEnvCfg(AntEnvCfg):
    """Evaluation environment for the submitted model: the original Isaac-Ant-v0 (flat plane, original
    observations, rewards and fall rule) plus the 63-point height scan appended to the observations.
    To evaluate on another terrain, replace ``scene.terrain`` only; the scan ray-casts against /World/ground."""

    def __post_init__(self):
        super().__post_init__()
        _add_height_scan(self)
