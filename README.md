# 실습과제 1: 처음 보는 지형에서도 걷는 Ant

## 제출 모델

| 항목 | 값 |
|---|---|
| 모델 | `T_SCAN_DR_MU` / MLP (400-200-100), PPO 기본 하이퍼파라미터 사용 |
| 관측 | 기존 `Isaac-Ant-v0` 60차원 관측 + 높이 센서 63차원 |
| 높이 센서 | 몸통 기준 1.6 m × 1.2 m 영역을 0.2 m 간격으로 측정함 |
| 학습 태스크 | `Isaac-Ant-T-Scan-DR-Mu-MLP-v0` |
| 학습 설정 | seed 42, 1000 iterations |
| 평가 태스크 | `Isaac-Ant-Submit-v0` |
| 체크포인트 | `logs/rsl_rl/ant_T_SCAN_DR_MU_MLP/2026-10-03_15-18-29_T_SCAN_DR_MU_MLP_seed42/model_999.pt` |

최종 제출 모델은 다양한 지형과 물리 조건에서 학습한 `T_SCAN_DR_MU`임.

기존 `Isaac-Ant-v0`의 60차원 관측에 몸통 주변의 지면 높이를 측정하는 63차원 높이 센서를 추가하여 총 123차원 관측을 사용함.

`Isaac-Ant-Submit-v0`는 기존 `Isaac-Ant-v0`의 평면 지형, 원본 보상 함수 및 넘어짐 판정을 유지하고 높이 센서만 추가한 제출용 평가 환경임.

로봇의 링크, 관절 및 USD asset은 변경하지 않음. 높이 센서는 Python 설정의 `RayCasterCfg`를 이용하여 추가함.

---

## 평가 명령어

공식 평가 조건인 `--seed 24 --num_envs 100`을 사용함.

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Submit-v0 \
  --num_envs 100 \
  --seed 24 \
  --checkpoint logs/rsl_rl/ant_T_SCAN_DR_MU_MLP/2026-10-03_15-18-29_T_SCAN_DR_MU_MLP_seed42/model_999.pt
```

평가 시 수업에서 제공한 `play_one_episode.py`를 수정하지 않고 사용함.

평가에서는 원본 `Isaac-Ant-v0`의 보상 함수와 넘어짐 판정을 사용함.

### 평지 평가 결과

| 항목 | 결과 |
|---|---:|
| Episode reward total mean | 91.36 |
| Episode reward total std | 19.46 |
| Episode steps mean | 916.64 |
| Episode steps std | 161.96 |

---

## 최종 모델 구성

`T_SCAN_DR_MU`의 주요 학습 설정은 다음과 같음.

- 다양한 지형으로 구성된 `T` 학습 환경을 사용함
- 전체 환경 중 약 30%에 물 환경을 적용함
- 마찰계수를 `0.5 ~ 1.5` 범위로 무작위화함
- 로봇 질량을 기본값의 `0.8 ~ 1.3배` 범위로 무작위화함
- 10 ~ 15초 간격으로 외력 조건을 무작위화함
- 기존 60차원 관측에 높이 센서 63차원을 추가함
- Policy는 MLP (400-200-100)를 사용함
- 원본 `Isaac-Ant-v0` 보상 함수를 사용함
- PPO 하이퍼파라미터는 `Isaac-Ant-v0` 기본 설정을 유지함
- 총 1000 iterations 동안 학습함

---

## 학습 명령어

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-T-Scan-DR-Mu-MLP-v0 \
  --headless \
  --seed 42
```

학습은 4096개의 병렬 환경에서 수행함.

최종 제출 체크포인트는 학습 seed 42의 `model_999.pt`임.

---

## 코드 위치

실험 관련 코드는 다음 경로에 정리되어 있음.

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_mountain/
```

주요 파일은 다음과 같음.

| 파일 | 내용 |
|---|---|
| `ant_mountain_env_cfg.py` | 학습·평가 환경, 지형, 물 환경, domain randomization, 높이 센서 및 제출용 환경을 정의함 |
| `terrains.py` | 학습에 사용하는 terrain 설정을 정의함 |
| `conditions.py` | 일반화 성능 평가에 사용한 35개 조건을 정의함 |
| `mdp.py` | 물 환경의 부력·항력, curriculum 및 추가 MDP 관련 기능을 정의함 |
| `rsl_rl_ppo_cfg.py` | 실험별 RSL-RL PPO runner 및 policy 설정을 정의함 |
| `__init__.py` | 학습 및 평가용 Gym task를 등록함 |

PPO 하이퍼파라미터는 최종 모델에서 `Isaac-Ant-v0` 기본값을 유지함.
