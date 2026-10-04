# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep a PPO policy close to the flat-ground baseline while it learns new terrain.

RSL-RL's PPO adds ``mirror_loss_coeff * MSE(policy(aug_obs), aug_actions)`` when a symmetry
config with ``use_mirror_loss=True`` is given. This module reuses that hook without mirroring:
the "augmented" batch is a copy of the observations, and the target actions are the baseline
policy's mean actions for them. The loss therefore pulls the policy towards the baseline's
behaviour, and PPO moves it away only where the task reward pays for it.

The teacher checkpoint is read from the ``ANT_TEACHER`` environment variable (an Isaac-Ant-v0
RSL-RL checkpoint with the 60-400-200-100-8 actor).
"""

from __future__ import annotations

import os

import torch
from torch import nn

_teacher: nn.Module | None = None
_last_obs: torch.Tensor | None = None


def _load_teacher(device) -> nn.Module:
    path = os.environ.get("ANT_TEACHER")
    if not path:
        raise RuntimeError("Set ANT_TEACHER to the baseline checkpoint used as the imitation target.")
    state = torch.load(path, map_location=device)["model_state_dict"]
    actor = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    teacher = nn.Sequential(
        nn.Linear(60, 400), nn.ELU(), nn.Linear(400, 200), nn.ELU(), nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 8)
    ).to(device)
    teacher.load_state_dict(actor)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def baseline_imitation_targets(env, obs=None, actions=None):
    """PPO calls this twice per mini-batch: first with the observations, then with the policy's mean actions."""
    global _teacher, _last_obs
    if obs is not None:
        _last_obs = obs["policy"].detach()
        return torch.cat([obs, obs], dim=0), None
    if _teacher is None:
        _teacher = _load_teacher(actions.device)
    with torch.no_grad():
        target = _teacher(_last_obs)
    return None, torch.cat([actions, target], dim=0)


def baseline_imitation_targets_flat(env, obs=None, actions=None):
    """Same hook, but the baseline is imitated only where the policy's height scan (observations 60..122) shows
    flat ground; elsewhere the target is the policy's own action, so the loss there is zero and PPO is free."""
    global _teacher, _last_obs
    if obs is not None:
        _last_obs = obs["policy"].detach()
        return torch.cat([obs, obs], dim=0), None
    if _teacher is None:
        _teacher = _load_teacher(actions.device)
    with torch.no_grad():
        scan = _last_obs[:, 60:]
        flat = (scan.max(dim=1).values - scan.min(dim=1).values) < 0.03
        target = torch.where(flat[:, None], _teacher(_last_obs[:, :60]), actions.detach())
    return None, torch.cat([actions, target], dim=0)
