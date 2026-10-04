# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlSymmetryCfg
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg

from .imitation import baseline_imitation_targets, baseline_imitation_targets_flat


@configclass
class AntMountainPPORunnerCfg(AntPPORunnerCfg):
    """Keep baseline PPO parameters, with a separate experiment directory."""

    experiment_name = "ant_mountain"


@configclass
class AntRobustPPORunnerCfg(AntPPORunnerCfg):
    """Experiment B: baseline PPO parameters, separate experiment directory."""

    experiment_name = "ant_robust"


@configclass
class AntRobust2PPORunnerCfg(AntPPORunnerCfg):
    """Experiment B2 with the baseline MLP policy."""

    experiment_name = "ant_robust2"


@configclass
class AntRobust2RnnPPORunnerCfg(AntPPORunnerCfg):
    """Experiment B2 with an LSTM in front of the baseline MLP, so the policy can infer the environment
    (e.g. water vs. land) from recent motion. PPO hyperparameters are unchanged."""

    experiment_name = "ant_robust2_rnn"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class AntRobust2BigPPORunnerCfg(AntPPORunnerCfg):
    """Experiment B2 with a wider MLP (~0.50 M actor parameters) to match the LSTM policy (~0.53 M),
    separating the effect of memory from that of network size."""

    experiment_name = "ant_robust2_big"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[640, 512, 256],
        critic_hidden_dims=[640, 512, 256],
        activation="elu",
    )


@configclass
class AntRobust3RnnPPORunnerCfg(AntRobust2RnnPPORunnerCfg):
    """Experiment B3: same LSTM policy and PPO settings as B2-RNN, separate experiment directory."""

    experiment_name = "ant_robust3_rnn"


@configclass
class AntRobust4RnnPPORunnerCfg(AntRobust2RnnPPORunnerCfg):
    """Experiment B4: same LSTM policy and PPO settings as B2-RNN, separate experiment directory."""

    experiment_name = "ant_robust4_rnn"


@configclass
class AntRobust2FinetunePPORunnerCfg(AntPPORunnerCfg):
    """Experiment B2-FT: the baseline MLP, resumed from the baseline checkpoint and trained on B2."""

    experiment_name = "ant_robust2_ft"


@configclass
class AntRobust5RnnPPORunnerCfg(AntRobust2RnnPPORunnerCfg):
    """Experiment B5 variants: same LSTM policy and PPO settings as B2-RNN (run_name tells them apart)."""

    experiment_name = "ant_robust5_rnn"


@configclass
class AntRobust5PPORunnerCfg(AntPPORunnerCfg):
    """Experiment B5 with the baseline MLP policy (400-200-100), so the checkpoint loads with the
    original Isaac-Ant-v0 agent configuration. PPO hyperparameters are unchanged."""

    experiment_name = "ant_robust5"


@configclass
class AntTerrainPPORunnerCfg(AntPPORunnerCfg):
    """Experiment T: terrain tiles + water, no friction/mass randomization, baseline MLP and PPO."""

    experiment_name = "ant_T"


@configclass
class AntTerrainGuidedPPORunnerCfg(AntPPORunnerCfg):
    """Experiment T-guided: resume from a baseline checkpoint and train on T with PPO plus an auxiliary
    loss towards the baseline's actions (teacher = ``ANT_TEACHER``). Other PPO settings unchanged."""

    experiment_name = "ant_T_guided"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=baseline_imitation_targets,
            mirror_loss_coeff=1.0,
        )


@configclass
class AntTerrainCurriculumPPORunnerCfg(AntPPORunnerCfg):
    """Experiment T-curriculum: baseline MLP and PPO settings."""

    experiment_name = "ant_T_curriculum"


@configclass
class AntTerrainHardPPORunnerCfg(AntPPORunnerCfg):
    """Experiment T-hard: baseline MLP and PPO settings."""

    experiment_name = "ant_T_hard"


# --- Architecture comparison on fixed environments (T, T-hard) -------------------------------
# Only ``policy`` and ``experiment_name`` differ from the baseline AntPPORunnerCfg; PPO settings,
# rollout length and iterations are inherited unchanged. T_MLP = AntTerrainPPORunnerCfg ("ant_T"),
# T_HARD_MLP = AntTerrainHardPPORunnerCfg ("ant_T_hard").

_LSTM_POLICY = RslRlPpoActorCriticRecurrentCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[400, 200, 100],
    critic_hidden_dims=[400, 200, 100],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
)

# ~0.50 M actor parameters, to match the LSTM policy's ~0.53 M (baseline MLP: ~0.13 M)
_BIG_MLP_POLICY = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[640, 512, 256],
    critic_hidden_dims=[640, 512, 256],
    activation="elu",
)


@configclass
class AntTLstmPPORunnerCfg(AntPPORunnerCfg):
    """T_LSTM: T environment, LSTM(256) in front of the baseline MLP."""

    experiment_name = "ant_T_LSTM"
    policy = _LSTM_POLICY


@configclass
class AntTMlpBigPPORunnerCfg(AntPPORunnerCfg):
    """T_MLP_BIG: T environment, parameter-matched MLP without memory."""

    experiment_name = "ant_T_MLP_BIG"
    policy = _BIG_MLP_POLICY


@configclass
class AntTHardLstmPPORunnerCfg(AntPPORunnerCfg):
    """T_HARD_LSTM: T-hard environment, LSTM policy."""

    experiment_name = "ant_T_HARD_LSTM"
    policy = _LSTM_POLICY


@configclass
class AntTHardMlpBigPPORunnerCfg(AntPPORunnerCfg):
    """T_HARD_MLP_BIG: T-hard environment, parameter-matched MLP."""

    experiment_name = "ant_T_HARD_MLP_BIG"
    policy = _BIG_MLP_POLICY


@configclass
class AntTHist8MlpPPORunnerCfg(AntPPORunnerCfg):
    """T_HIST8_MLP: baseline MLP and PPO settings; the environment stacks the last 8 observations."""

    experiment_name = "ant_T_HIST8_MLP"


@configclass
class AntTHist4MlpPPORunnerCfg(AntPPORunnerCfg):
    """T_HIST4_MLP: baseline MLP and PPO settings; the environment stacks the last 4 observations."""

    experiment_name = "ant_T_HIST4_MLP"


@configclass
class AntDrHardLstmPPORunnerCfg(AntPPORunnerCfg):
    """DR_HARD_LSTM: T-hard terrain + friction/mass randomization, LSTM policy."""

    experiment_name = "ant_DR_HARD_LSTM"
    policy = _LSTM_POLICY


@configclass
class AntDrHardMlpPPORunnerCfg(AntPPORunnerCfg):
    """DR_HARD_MLP: T-hard terrain + friction/mass randomization, baseline MLP."""

    experiment_name = "ant_DR_HARD_MLP"


@configclass
class AntTScanMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_MLP: baseline MLP and PPO settings; the environment adds a 63-point height scan."""

    experiment_name = "ant_T_SCAN_MLP"


@configclass
class AntTScanCurMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_CUR_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_SCAN_CUR_MLP"


@configclass
class AntTScanCurDrMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_CUR_DR_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_SCAN_CUR_DR_MLP"


@configclass
class AntTScanF40MlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_F40_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_SCAN_F40_MLP"


@configclass
class AntScanLanesMlpPPORunnerCfg(AntPPORunnerCfg):
    """SCAN_LANES_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_SCAN_LANES_MLP"


@configclass
class AntTScanCurDrMuMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_CUR_DR_MU_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_SCAN_CUR_DR_MU_MLP"


@configclass
class AntTScanDrMuMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_DR_MU_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_SCAN_DR_MU_MLP"


@configclass
class AntTScanDrMuLstmPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_DR_MU_LSTM: LSTM policy, PPO settings unchanged."""

    experiment_name = "ant_T_SCAN_DR_MU_LSTM"
    policy = _LSTM_POLICY


@configclass
class AntTScanDrMuProg2MlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_DR_MU_PROG2_MLP: baseline MLP and PPO settings; training reward progress weight 2.0."""

    experiment_name = "ant_T_SCAN_DR_MU_PROG2_MLP"


@configclass
class AntTDrMuMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_DR_MU_MLP: baseline MLP and PPO settings."""

    experiment_name = "ant_T_DR_MU_MLP"


@configclass
class AntTDrMuAsymPPORunnerCfg(AntPPORunnerCfg):
    """T_DR_MU_ASYM_MLP: baseline MLP and PPO settings; the critic receives the environment's "critic" group."""

    experiment_name = "ant_T_DR_MU_ASYM_MLP"


@configclass
class AntTScanDrMuLongMlpPPORunnerCfg(AntPPORunnerCfg):
    """T_SCAN_DR_MU_LONG_MLP: baseline MLP and PPO settings (151 inputs: 60 + 91-point forward scan)."""

    experiment_name = "ant_T_SCAN_DR_MU_LONG_MLP"


@configclass
class AntTScanDrMuFlatPriorityPPORunnerCfg(AntPPORunnerCfg):
    """P1_FLAT_REWARD: baseline MLP and PPO settings; environment adds the flat-ground progress term."""

    experiment_name = "ant_P1_FLAT_REWARD"


@configclass
class AntTScanDrMuFlatImitationPPORunnerCfg(AntPPORunnerCfg):
    """P2_FLAT_IMITATION: PPO plus an auxiliary loss towards the baseline policy's actions only where the
    height scan shows flat ground (teacher = ANT_TEACHER, the Isaac-Ant-v0 baseline). PPO settings unchanged."""

    experiment_name = "ant_P2_FLAT_IMITATION"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=baseline_imitation_targets_flat,
            mirror_loss_coeff=1.0,
        )


@configclass
class AntTorqueScanPPORunnerCfg(AntPPORunnerCfg):
    """Torque-scale variants of T_SCAN_DR_MU (run_name records the scale)."""

    experiment_name = "ant_TORQUE_T_SCAN_DR_MU"


@configclass
class AntTorqueFlatPPORunnerCfg(AntPPORunnerCfg):
    """Torque-scale variants of the flat baseline (run_name records the scale)."""

    experiment_name = "ant_TORQUE_BASELINE"


@configclass
class AntDistanceOnlyPPORunnerCfg(AntPPORunnerCfg):
    """D1_DISTANCE_ONLY: baseline MLP and PPO settings."""

    experiment_name = "ant_D1_DISTANCE_ONLY"


@configclass
class AntNoPenaltyPPORunnerCfg(AntPPORunnerCfg):
    """D2_NO_PENALTY: baseline MLP and PPO settings."""

    experiment_name = "ant_D2_NO_PENALTY"


@configclass
class AntSubmitPPORunnerCfg(AntPPORunnerCfg):
    """Agent configuration of the submitted model (baseline MLP 400-200-100 and PPO settings)."""

    experiment_name = "ant_T_SCAN_DR_MU_MLP"
