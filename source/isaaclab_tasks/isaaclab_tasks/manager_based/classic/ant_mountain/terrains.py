# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain presets for the Ant Mountain tasks; independent of the shared rough preset."""

import isaaclab.terrains as terrain_gen

from .conditions import ForwardProfileTerrainCfg

ANT_MOUNTAIN_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=5.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=False,
    use_cache=False,
    sub_terrains={
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.35, slope_range=(0.03, 0.12), platform_width=2.0, border_width=0.25
        ),
        "inverted_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.35, slope_range=(0.03, 0.12), platform_width=2.0, border_width=0.25
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(-0.02, 0.02),
            noise_step=0.01,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
    },
)


# Experiment B training distribution: 24 long lanes (300 m x 8 m) side by side. Upper limits are
# kept below the held-out evaluation levels (slope 0.3, hills 0.3/0.5, rough 0.08/0.12).
# curriculum=True with a single row only fixes the lane-type proportions; difficulty stays random.
ANT_ROBUST_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(300.0, 8.0),
    border_width=5.0,
    num_rows=1,
    num_cols=24,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": ForwardProfileTerrainCfg(profile="slope", grade=0.0, border_width=0.0, proportion=0.15),
        "slope": ForwardProfileTerrainCfg(profile="slope", grade_range=(0.0, 0.2), border_width=0.0, proportion=0.3),
        "hills": ForwardProfileTerrainCfg(profile="hills", grade_range=(0.0, 0.2), border_width=0.0, proportion=0.25),
        "rough_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.02), noise_step=0.005, downsampled_scale=0.2, border_width=0.0
        ),
        "rough_high": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.05, 0.05), noise_step=0.005, downsampled_scale=0.2, border_width=0.0
        ),
    },
)


# Experiment B2 training terrain: 8 m tiles like Isaac Lab's ROUGH_TERRAINS_CFG (stairs, boxes, rough,
# slopes), but with every height limit below that preset so it remains a held-out evaluation terrain.
ANT_ROBUST2_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.03, 0.12), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.03, 0.12), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.03, 0.10), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
    },
)


# Experiment B4 training terrain: like B2's mix, but every lane is one terrain type for its whole
# 300 m length, so what a robot feels in its first seconds stays true for the whole episode.
# 24 lanes like B: with 40 lanes (~19 M triangles) robots fell through the ground mesh.
ANT_LANES_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(300.0, 8.0),
    border_width=5.0,
    num_rows=1,
    num_cols=24,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": ForwardProfileTerrainCfg(profile="slope", grade=0.0, border_width=0.0, proportion=0.3),
        "slope": ForwardProfileTerrainCfg(profile="slope", grade_range=(0.0, 0.2), border_width=0.0, proportion=0.1),
        "hills": ForwardProfileTerrainCfg(profile="hills", grade_range=(0.0, 0.2), border_width=0.0, proportion=0.1),
        "stairs": ForwardProfileTerrainCfg(
            profile="stairs", grade_range=(0.03, 0.12), border_width=0.0, proportion=0.15
        ),
        "boxes": ForwardProfileTerrainCfg(profile="boxes", grade_range=(0.03, 0.10), border_width=0.0, proportion=0.15),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, downsampled_scale=0.2, border_width=0.0
        ),
    },
)
