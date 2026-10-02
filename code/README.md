# MicroVLA training source

This archive contains the training code we used for PPO, OpenVLA-OFT and pi0.5,
plus our MicroEnv simulator and the 15 layout JSON files.

## Layout and source

- `MicroEnv/MicroEnv`: simulator, vessel configuration and layout definitions.
- `RL`: PPO trainer, demonstration collector, requirements and a physical copy of MicroEnv.
- `VLA/OpenVLA`: complete fine-tuning source, data converter, launcher, MicroEnv and dependencies.
- `VLA/pi`: complete OpenPI training source, data preparation and locked dependencies.

Each training module includes its local source dependencies and can be copied
independently. Third-party Python packages, training datasets and base model
weights must still be installed or supplied. Run each command from the module
directory indicated below. Select GPU indices explicitly.

## Vision

`Vision/` contains the complete physical vision source, YOLO training and live
localization, calibration, data preparation, SAM labeling, SAM3 source and bundled EdgeTAM
source. EdgeTAM is one of our on-device implementations, based on upstream
EdgeTAM, included to facilitate deployment across more devices and scenarios. See `Vision/README.md` for setup and external weights/data requirements.

## 1. PPO and demonstrations

We trained PPO with Python 3.9.23, PyTorch 2.8.0+cu128, NumPy 2.0.2,
Gymnasium 1.1.1 and stable-baselines3 2.7.1. Install `RL/requirements.txt` in a
compatible CUDA environment. Set the final training budget, seed, layout set and
output directory for the experiment being reproduced. For example:

```bash
cd RL
python train_ppo_torch_vessel_no_progress_reward.py \
  --mode mixed --total-timesteps 100000000 --seed 42 \
  --device cuda:0 --save-dir /path/to/ppo-run
```

The trainer defaults to the first ten numbered layouts. Its arguments control
the simulator, PPO settings, checkpoint schedule and built-in validation.
After training, supply the resulting PPO checkpoint to our demonstration
collector:

```bash
python collect_ppo_dataset.py \
  --checkpoint /path/to/ppo-model.zip \
  --output-dir /path/to/ppo-demos \
  --successful-episodes-per-layout 100 --GPU 0
```

## 2. OpenVLA-OFT training

We built this module on the OpenVLA-OFT source tree with local MicroEnv
modifications. We trained with Python 3.10.20, PyTorch 2.2.0+cu121,
Transformers 4.40.1 and PEFT 0.11.1. Install `VLA/OpenVLA/pyproject.toml` in a
compatible environment. `VLA/OpenVLA/environment-record.txt` lists the package
versions in our training environment, and the two Git dependencies in
`pyproject.toml` are pinned to the commits we used. Convert the successful PPO
demonstrations:

```bash
cd VLA/OpenVLA  # from the archive root
python -m pip install -e .
python MicroEnv/scripts/convert_ppo_demos_to_openvla_dataset.py \
  --input-dir /path/to/ppo-demos \
  --output-dir /path/to/openvla-data/microenv_vessel_goal6 \
  --dataset-name microenv_vessel_goal6 \
  --num-actions-chunk 8 --val-ratio 0.1 --seed 7
```

Provide a compatible OpenVLA base model in `VLA_PATH`, then run the archived
fine-tuning launcher. It defaults to offline W&B logging. Review its explicit
batch size, step count and other environment-variable settings before use.

```bash
CUDA_VISIBLE_DEVICES=0 VLA_PATH=/path/to/openvla-7b \
DATA_ROOT_DIR=/path/to/openvla-data DATASET_NAME=microenv_vessel_goal6 \
bash MicroEnv/scripts/train_openvla_microenv.sh
```

## 3. pi0.5 training

We built this module on the OpenPI source tree with local MicroEnv changes. We
trained with Python 3.11.15, JAX 0.5.3 and Flax 0.10.2. Use the bundled
`uv.lock` and set `HF_LEROBOT_HOME` to the intended local dataset root. The
default simulation config expects the dataset repo ID
`anonymous/microenv_ppo_no_distance_2000` and the base pi0.5 weights named in
`src/openpi/training/config.py`; substitute your own repo ID after conversion.

```bash
cd VLA/pi  # from the archive root
uv sync --locked
export HF_LEROBOT_HOME=/path/to/lerobot-data
uv run examples/microenv/convert_microenv_data_to_lerobot.py \
  /path/to/ppo-demos --repo-id anonymous/microenv_ppo_no_distance_2000 \
  --no-goal-distance
uv run scripts/compute_norm_stats.py pi05_microenv_lora
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  ./train_microenv.sh pi05_microenv_lora pi05_microenv_reproduction
```

We retain the real-data and clot-data conversion scripts and configs in
`VLA/pi` for their respective training cohorts.
