# Ant Mountain — existing 60-input checkpoint

`Isaac-Ant-Mountain-v0` is a separate Manager-based task derived from `Isaac-Ant-v0`. It uses the existing 60-input policy, including the four foot wrenches. The original Ant task, robot asset and baseline PPO files are unchanged.

## Run the trained Ant on gentle terrain

From a terminal:

```bash
conda activate lerobot-arena
cd /home/siyeon/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Ant-Mountain-v0 \
  --num_envs 1 \
  --seed 42 \
  --checkpoint /home/siyeon/IsaacLab/logs/rsl_rl/ant/2026-09-21_18-20-29_ant_baseline/model_999.pt
```

This runs inference, without additional training. The camera follows the robot. Stop with Ctrl+C. The existing play script exports policy files to `exported/` beside the checkpoint; it does not modify the checkpoint itself.

For a flat control with the same new configuration, replace the task with `Isaac-Ant-Mountain-Flat-v0`.

## What changes

- Terrain uses the gentle preset in `terrains.py`: 4 x 4 tiles, each 12 m square, with slopes of ratio 0.03–0.12 and roughness samples between -2 and +2 cm.
- The standard scene terrain importer and reset event place robots at generated terrain origins. The original reset randomization is retained.
- Falls use the original 0.31 m threshold relative to local ground, queried using Isaac Lab's Warp mesh utilities. Exiting the generated grid or missing the ground truncates the episode.
- Actions, observation order, foot-force inputs, rewards, target definition, timestep, episode length and PPO parameters remain inherited. Only the runner experiment directory changes to `ant_mountain`.
- The default observation retains **world height**, exactly as in the trained baseline. Terrain altitude therefore causes a real input distribution shift. To try ground-relative preprocessing explicitly, append `env.height_observation=ground` to the play command. That changes only the height scalar and must be reported as a separate evaluation condition.
- There is no adaptive curriculum, stairs or deformable sand in this initial version. Robots can cross between adjacent terrain tiles; episode results are not isolated terrain-type benchmarks.

## Stress-condition evaluation (`Isaac-Ant-Eval-v0`)

`Isaac-Ant-Eval-v0` is the flat Ant plus one stress condition read from the `ANT_CONDITION`
environment variable (syntax and axes in `conditions.py`). It is evaluated with the course's
unmodified `play_one_episode.py` on 100 environments:

```bash
ANT_CONDITION=slope:0.3 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Eval-v0 --headless --num_envs 100 --seed 42 \
  --checkpoint logs/rsl_rl/ant/2026-09-21_18-20-29_ant_baseline/model_999.pt
```

Axes: `slope:g`, `hills:g`, `rough:h`, `friction:mu`, `push:v`, `mass:s`; combine with `+`
(e.g. `friction:0.3+rough:0.03+push:1.0`). Unset or `flat` gives the original plane. Terrain
axes use a single 400 m x 20 m track so the policy does not reach the grid edge within 16 s.

## Validation

GPU smoke runs loaded the existing `model_999.pt` without optimizer loading or training. With seed 42 and four environments, the original flat task and new flat control had identical 60-dimensional observation trajectories over 300 policy steps (maximum difference 0). The Mountain task completed 300 steps with finite observations, actions and rewards; two episode endings occurred. This is a compatibility smoke check, not a robustness benchmark.

Regression tests cover configuration isolation, unchanged observation order/actions/rewards/PPO, partial resets at terrain origins, both height input modes, ground-relative falls and missing-ground truncation:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_ant_manager_mountain.py -q -s
```
