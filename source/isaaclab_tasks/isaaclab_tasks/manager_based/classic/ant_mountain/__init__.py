# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mountain and flat control tasks compatible with the 60-input Ant policy."""

import gymnasium as gym

for task_id, cfg_name, runner_name in (
    ("Isaac-Ant-Mountain-v0", "AntMountainEnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-Mountain-Flat-v0", "AntMountainFlatEnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-Eval-v0", "AntEvalEnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-Robust-v0", "AntRobustEnvCfg", "AntRobustPPORunnerCfg"),
    ("Isaac-Ant-Robust2-v0", "AntRobust2EnvCfg", "AntRobust2PPORunnerCfg"),
    ("Isaac-Ant-Robust2-RNN-v0", "AntRobust2EnvCfg", "AntRobust2RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust2-Big-v0", "AntRobust2EnvCfg", "AntRobust2BigPPORunnerCfg"),
    ("Isaac-Ant-Robust3-RNN-v0", "AntRobust3EnvCfg", "AntRobust3RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust4-RNN-v0", "AntRobust4EnvCfg", "AntRobust4RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust2-FT-v0", "AntRobust2EnvCfg", "AntRobust2FinetunePPORunnerCfg"),
    ("Isaac-Ant-Robust5-NoDR-RNN-v0", "AntRobust5NoDrEnvCfg", "AntRobust5RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust5-Narrow-RNN-v0", "AntRobust5NarrowEnvCfg", "AntRobust5RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust5-Flat40-RNN-v0", "AntRobust5Flat40EnvCfg", "AntRobust5RnnPPORunnerCfg"),
    ("Isaac-Ant-Robust5-Narrow-v0", "AntRobust5NarrowEnvCfg", "AntRobust5PPORunnerCfg"),
    ("Isaac-Ant-T-v0", "AntRobust5NoDrEnvCfg", "AntTerrainPPORunnerCfg"),
    ("Isaac-Ant-T-Guided-v0", "AntRobust5NoDrEnvCfg", "AntTerrainGuidedPPORunnerCfg"),
    ("Isaac-Ant-T-Curriculum-v0", "AntTerrainCurriculumEnvCfg", "AntTerrainCurriculumPPORunnerCfg"),
    ("Isaac-Ant-T-Hard-v0", "AntTerrainHardEnvCfg", "AntTerrainHardPPORunnerCfg"),
    # architecture comparison: same environments as T / T-hard, only the policy differs
    ("Isaac-Ant-T-LSTM-v0", "AntRobust5NoDrEnvCfg", "AntTLstmPPORunnerCfg"),
    ("Isaac-Ant-T-MLP-Big-v0", "AntRobust5NoDrEnvCfg", "AntTMlpBigPPORunnerCfg"),
    ("Isaac-Ant-T-Hard-LSTM-v0", "AntTerrainHardEnvCfg", "AntTHardLstmPPORunnerCfg"),
    ("Isaac-Ant-T-Hard-MLP-Big-v0", "AntTerrainHardEnvCfg", "AntTHardMlpBigPPORunnerCfg"),
    ("Isaac-Ant-DR-Hard-LSTM-v0", "AntDrHardEnvCfg", "AntDrHardLstmPPORunnerCfg"),
    ("Isaac-Ant-DR-Hard-MLP-v0", "AntDrHardEnvCfg", "AntDrHardMlpPPORunnerCfg"),
    # height-scan sensor: training, stress-condition evaluation and R1 evaluation
    ("Isaac-Ant-T-Scan-MLP-v0", "AntTScanEnvCfg", "AntTScanMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-Cur-MLP-v0", "AntTScanCurEnvCfg", "AntTScanCurMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-Cur-DR-MLP-v0", "AntTScanCurDrEnvCfg", "AntTScanCurDrMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-F40-MLP-v0", "AntTScanF40EnvCfg", "AntTScanF40MlpPPORunnerCfg"),
    ("Isaac-Ant-Scan-Lanes-MLP-v0", "AntScanLanesEnvCfg", "AntScanLanesMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-Cur-DR-Mu-MLP-v0", "AntTScanCurDrMuEnvCfg", "AntTScanCurDrMuMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-DR-Mu-MLP-v0", "AntTScanDrMuEnvCfg", "AntTScanDrMuMlpPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-DR-Mu-LSTM-v0", "AntTScanDrMuEnvCfg", "AntTScanDrMuLstmPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-DR-Mu-Prog2-MLP-v0", "AntTScanDrMuProg2EnvCfg", "AntTScanDrMuProg2MlpPPORunnerCfg"),
    ("Isaac-Ant-T-DR-Mu-MLP-v0", "AntTDrMuEnvCfg", "AntTDrMuMlpPPORunnerCfg"),
    ("Isaac-Ant-T-DR-Mu-Asym-MLP-v0", "AntTDrMuAsymEnvCfg", "AntTDrMuAsymPPORunnerCfg"),
    ("Isaac-Ant-T-Scan-DR-Mu-Long-MLP-v0", "AntTScanDrMuLongEnvCfg", "AntTScanDrMuLongMlpPPORunnerCfg"),
    ("Isaac-Ant-Eval-Scan-Long-v0", "AntEvalScanLongEnvCfg", "AntTScanDrMuLongMlpPPORunnerCfg"),
    ("Isaac-Ant-R1-Scan-Long-v0", "AntR1ScanLongEnvCfg", "AntTScanDrMuLongMlpPPORunnerCfg"),
    ("Isaac-Ant-P1-Flat-Reward-v0", "AntTScanDrMuFlatPriorityEnvCfg", "AntTScanDrMuFlatPriorityPPORunnerCfg"),
    ("Isaac-Ant-P2-Flat-Imitation-v0", "AntTScanDrMuEnvCfg", "AntTScanDrMuFlatImitationPPORunnerCfg"),
    ("Isaac-Ant-Torque-T-Scan-DR-Mu-v0", "AntTScanDrMuTqEnvCfg", "AntTorqueScanPPORunnerCfg"),
    ("Isaac-Ant-Torque-Baseline-v0", "AntFlatTqEnvCfg", "AntTorqueFlatPPORunnerCfg"),
    ("Isaac-Ant-Eval-Tq-v0", "AntEvalTqEnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-R1-Tq-v0", "AntR1TqEnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-Eval-Scan-Tq-v0", "AntEvalScanTqEnvCfg", "AntTScanMlpPPORunnerCfg"),
    ("Isaac-Ant-R1-Scan-Tq-v0", "AntR1ScanTqEnvCfg", "AntTScanMlpPPORunnerCfg"),
    ("Isaac-Ant-D1-Distance-Only-v0", "AntTScanDrMuDistanceOnlyEnvCfg", "AntDistanceOnlyPPORunnerCfg"),
    ("Isaac-Ant-D2-No-Penalty-v0", "AntTScanDrMuNoPenaltyEnvCfg", "AntNoPenaltyPPORunnerCfg"),
    # submitted model: original Isaac-Ant-v0 + height scan (training task: Isaac-Ant-T-Scan-DR-Mu-MLP-v0)
    ("Isaac-Ant-Submit-v0", "AntSubmitEnvCfg", "AntSubmitPPORunnerCfg"),
    ("Isaac-Ant-Eval-Scan-v0", "AntEvalScanEnvCfg", "AntTScanMlpPPORunnerCfg"),
    ("Isaac-Ant-Eval-Scan-RNN-v0", "AntEvalScanEnvCfg", "AntTScanDrMuLstmPPORunnerCfg"),
    ("Isaac-Ant-R1-Scan-RNN-v0", "AntR1ScanEnvCfg", "AntTScanDrMuLstmPPORunnerCfg"),
    ("Isaac-Ant-R1-Scan-v0", "AntR1ScanEnvCfg", "AntTScanMlpPPORunnerCfg"),
    # observation-history MLPs (analysis): training, stress-condition evaluation and R1 evaluation
    ("Isaac-Ant-T-Hist8-MLP-v0", "AntTHist8EnvCfg", "AntTHist8MlpPPORunnerCfg"),
    ("Isaac-Ant-T-Hist4-MLP-v0", "AntTHist4EnvCfg", "AntTHist4MlpPPORunnerCfg"),
    ("Isaac-Ant-Eval-Hist8-v0", "AntEvalHist8EnvCfg", "AntTHist8MlpPPORunnerCfg"),
    ("Isaac-Ant-Eval-Hist4-v0", "AntEvalHist4EnvCfg", "AntTHist4MlpPPORunnerCfg"),
    ("Isaac-Ant-R1-Hist8-v0", "AntR1Hist8EnvCfg", "AntTHist8MlpPPORunnerCfg"),
    ("Isaac-Ant-R1-Hist4-v0", "AntR1Hist4EnvCfg", "AntTHist4MlpPPORunnerCfg"),
    # R1 evaluation (original Ant on Isaac Lab's rough terrain), one task per policy architecture
    ("Isaac-Ant-R1-v0", "AntR1EnvCfg", "AntMountainPPORunnerCfg"),
    ("Isaac-Ant-R1-RNN-v0", "AntR1EnvCfg", "AntRobust2RnnPPORunnerCfg"),
    ("Isaac-Ant-R1-Big-v0", "AntR1EnvCfg", "AntRobust2BigPPORunnerCfg"),
    # evaluation of recurrent checkpoints needs the recurrent runner config
    ("Isaac-Ant-Eval-RNN-v0", "AntEvalEnvCfg", "AntRobust2RnnPPORunnerCfg"),
    ("Isaac-Ant-Eval-Big-v0", "AntEvalEnvCfg", "AntRobust2BigPPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_mountain_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:{runner_name}",
        },
    )
