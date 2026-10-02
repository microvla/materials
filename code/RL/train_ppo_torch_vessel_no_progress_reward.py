import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from gymnasium import spaces
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from MicroEnv.scripts._bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(__file__)

from MicroEnv.MicroEnv.load_vessel_branches import get_branch_layout, list_branch_layouts
from MicroEnv.MicroEnv.torch_vessel_env import make_torch_vessel_batch_env, resolve_torch_device
from MicroEnv.MicroEnv.vessel_train_cfg import VesselTrainConfig


TRAIN_DEFAULTS = VesselTrainConfig.Train
ENV_DEFAULTS = VesselTrainConfig.Env


class _TeeStream:
    def __init__(self, terminal_stream, log_stream=None):
        self._terminal_stream = terminal_stream
        self._log_stream = log_stream

    def write(self, data):
        self._terminal_stream.write(data)
        if self._log_stream is not None and "\r" not in data:
            self._log_stream.write(data)
        return len(data)

    def flush(self):
        self._terminal_stream.flush()
        if self._log_stream is not None:
            self._log_stream.flush()

    def isatty(self):
        return getattr(self._terminal_stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._terminal_stream, name)


def _enable_terminal_log(save_dir: Path):
    log_path = save_dir / "terminal.log"
    log_fp = log_path.open("a", encoding="utf-8", buffering=1)
    log_fp.write(f"\n===== Training started at {datetime.now().isoformat(timespec='seconds')} =====\n")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    return log_path, log_fp, original_stdout, original_stderr


def _add_bool_pair(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{dest: default})


def _default_mixed_layouts() -> List[str]:
    layouts = list_branch_layouts()
    numbered = [name for name in layouts if name[:2].isdigit()]
    return numbered[:10] if numbered else layouts


def _parse_layouts(values: Sequence[str]) -> List[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list_branch_layouts()
    layouts: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                layouts.append(item)
    if not layouts:
        raise SystemExit("At least one branch layout is required.")
    return layouts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on TorchVesselBatchEnv with the goal-progress reward removed "
            "and a larger terminal goal bonus."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional python config file.")
    parser.add_argument("--mode", choices=("single", "mixed"), default="single", help="Training layout mode.")
    parser.add_argument(
        "--branch-layout",
        type=str,
        default=getattr(ENV_DEFAULTS, "branch_layout", "default_2"),
        help="Layout name for --mode single.",
    )
    parser.add_argument(
        "--branch-layouts",
        nargs="+",
        default=_default_mixed_layouts(),
        help="Layouts for --mode mixed, comma-separated values, or 'all'.",
    )
    parser.add_argument("--total-timesteps", type=int, default=TRAIN_DEFAULTS.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=TRAIN_DEFAULTS.num_envs, help="Single-layout training env count.")
    parser.add_argument(
        "--num-envs-per-layout",
        type=int,
        default=1024,
        help="Per-layout env count for mixed training.",
    )
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=ENV_DEFAULTS.seed)
    parser.add_argument("--sim-dt", type=float, default=ENV_DEFAULTS.sim_dt)
    parser.add_argument("--decimation", type=int, default=ENV_DEFAULTS.decimation)
    parser.add_argument("--episode-seconds", type=float, default=ENV_DEFAULTS.episode_seconds)
    parser.add_argument("--eval-episode-seconds", type=float, default=ENV_DEFAULTS.eval_episode_seconds)
    parser.add_argument("--action-noise-std", type=float, default=ENV_DEFAULTS.action_noise_std)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_DEFAULTS.learning_rate)
    parser.add_argument("--n-steps", type=int, default=TRAIN_DEFAULTS.n_steps)
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--n-epochs", type=int, default=TRAIN_DEFAULTS.n_epochs)
    parser.add_argument("--gamma", type=float, default=TRAIN_DEFAULTS.gamma)
    parser.add_argument("--gae-lambda", type=float, default=TRAIN_DEFAULTS.gae_lambda)
    parser.add_argument("--ent-coef", type=float, default=TRAIN_DEFAULTS.ent_coef)
    parser.add_argument("--save-dir", type=Path, default=Path("./checkpoints/ppo_torch_no_progress_reward"))
    parser.add_argument("--model-name", type=str, default=f"{TRAIN_DEFAULTS.model_name}_no_progress_reward")
    parser.add_argument(
        "--goal-reward-bonus",
        type=float,
        default=10.0,
        help="Terminal bonus when distance_to_goal <= goal_radius. Original environment uses 5.0.",
    )
    parser.add_argument(
        "--original-progress-weight",
        type=float,
        default=2.0,
        help="Progress coefficient used by the original environment reward; removed by this script.",
    )
    parser.add_argument("--device", type=str, default=TRAIN_DEFAULTS.device)
    parser.add_argument("--progress-bar", action="store_true", default=TRAIN_DEFAULTS.progress_bar)
    parser.add_argument("--eval-freq", type=int, default=10000, help="Success-rate eval frequency in total env steps.")
    parser.add_argument("--train-log-freq", type=int, default=10000, help="Training metric log frequency in total env steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=256)
    parser.add_argument("--checkpoint-every-updates", type=int, default=TRAIN_DEFAULTS.checkpoint_every_updates)
    parser.add_argument("--clot-probability", type=float, default=ENV_DEFAULTS.clot_probability)
    parser.add_argument("--clot-region-start", type=float, default=ENV_DEFAULTS.clot_region[0])
    parser.add_argument("--clot-region-end", type=float, default=ENV_DEFAULTS.clot_region[1])
    parser.add_argument("--clot-length-min", type=int, default=ENV_DEFAULTS.clot_length_range[0])
    parser.add_argument("--clot-length-max", type=int, default=ENV_DEFAULTS.clot_length_range[1])
    parser.add_argument("--clot-max-length", type=float, default=ENV_DEFAULTS.clot_max_length)
    parser.add_argument("--clot-junction-margin", type=float, default=ENV_DEFAULTS.clot_junction_margin)
    parser.add_argument("--clot-min-y-slice-paths", type=int, default=ENV_DEFAULTS.clot_min_y_slice_paths)
    parser.add_argument("--clot-y-slice-samples", type=int, default=ENV_DEFAULTS.clot_y_slice_samples)
    parser.add_argument("--clot-y-merge-tolerance", type=float, default=ENV_DEFAULTS.clot_y_merge_tolerance)
    parser.add_argument("--clot-radius-factor", type=float, default=ENV_DEFAULTS.clot_radius_factor)
    _add_bool_pair(parser, "randomize-start", ENV_DEFAULTS.randomize_start, "Randomize start position on reset.")
    _add_bool_pair(parser, "randomize-goal", ENV_DEFAULTS.randomize_goal, "Randomize goal position on reset.")
    _add_bool_pair(parser, "clot-event", ENV_DEFAULTS.clot_event, "Enable clot generation.")
    _add_bool_pair(parser, "clot-randomize-on-reset", ENV_DEFAULTS.clot_randomize_on_reset, "Randomize clot on reset.")
    _add_bool_pair(parser, "clot-branch-randomize", ENV_DEFAULTS.clot_branch_randomize, "Randomize clot branch.")
    return parser


def _load_config(path: Path):
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")
    spec = importlib.util.spec_from_file_location("vessel_train_cfg", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load config from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "CONFIG"):
        return module.CONFIG
    if hasattr(module, "get_config"):
        return module.get_config()
    raise SystemExit("Config file must define CONFIG class or get_config().")


def _apply_config(args: argparse.Namespace, cfg) -> None:
    if cfg is None:
        return

    def _apply_attrs(obj):
        if obj is None:
            return
        for key in dir(obj):
            if key.startswith("_"):
                continue
            value = getattr(obj, key)
            if callable(value):
                continue
            if hasattr(args, key):
                setattr(args, key, value)

    if hasattr(cfg, "Train") or hasattr(cfg, "Env"):
        _apply_attrs(getattr(cfg, "Train", None))
        _apply_attrs(getattr(cfg, "Env", None))
    else:
        _apply_attrs(cfg)


def _env_kwargs(args: argparse.Namespace, layout: str, seed: int, episode_seconds: float) -> Dict[str, Any]:
    return {
        "seed": seed,
        "sim_dt": args.sim_dt,
        "decimation": args.decimation,
        "episode_seconds": episode_seconds,
        "action_noise_std": args.action_noise_std,
        "clot_event": args.clot_event,
        "clot_probability": args.clot_probability,
        "clot_randomize_on_reset": args.clot_randomize_on_reset,
        "clot_region": (args.clot_region_start, args.clot_region_end),
        "clot_length_range": (args.clot_length_min, args.clot_length_max),
        "clot_max_length": args.clot_max_length,
        "clot_junction_margin": args.clot_junction_margin,
        "clot_min_y_slice_paths": args.clot_min_y_slice_paths,
        "clot_y_slice_samples": args.clot_y_slice_samples,
        "clot_y_merge_tolerance": args.clot_y_merge_tolerance,
        "clot_branch_randomize": args.clot_branch_randomize,
        "clot_radius_factor": args.clot_radius_factor,
        "branches": get_branch_layout(layout),
        "start": ENV_DEFAULTS.start,
        "goal": ENV_DEFAULTS.goal,
        "randomize_start": args.randomize_start,
        "randomize_goal": args.randomize_goal,
        "min_clot_distance": ENV_DEFAULTS.min_clot_distance,
        "goal_radius": ENV_DEFAULTS.goal_radius,
    }


def _make_batch_env(args: argparse.Namespace, layout: str, num_envs: int, seed: int, episode_seconds: float):
    batch_env = make_torch_vessel_batch_env(
        num_envs=num_envs,
        device=args.device,
        **_env_kwargs(args, layout, seed=seed, episode_seconds=episode_seconds),
    )
    _override_no_progress_reward(
        batch_env,
        original_progress_weight=args.original_progress_weight,
        goal_reward_bonus=args.goal_reward_bonus,
    )
    return batch_env


def _override_no_progress_reward(batch_env, original_progress_weight: float, goal_reward_bonus: float) -> None:
    original_step = batch_env.step
    original_goal_reward_bonus = 5.0

    def step_without_goal_progress(actions: torch.Tensor):
        previous_distance = batch_env.distance_to_goal.clone()
        obs, reward, terminated, truncated, info = original_step(actions)
        progress = previous_distance - batch_env.distance_to_goal
        reward = reward - float(original_progress_weight) * progress
        reward = torch.where(
            terminated,
            reward + (float(goal_reward_bonus) - original_goal_reward_bonus),
            reward,
        )
        return obs, reward, terminated, truncated, info

    batch_env.step = step_without_goal_progress


def _load_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install it first, for example:\n"
            "  pip install stable-baselines3[extra]"
        ) from exc
    return PPO, BaseCallback, CheckpointCallback, VecEnv


PPO, BaseCallback, CheckpointCallback, VecEnv = _load_sb3()


class TorchBranchVecEnv(VecEnv):
    def __init__(self, groups: Sequence[Tuple[str, Any]]):
        if not groups:
            raise ValueError("TorchBranchVecEnv requires at least one environment group.")
        self.groups = list(groups)
        self.actions = None
        self.group_offsets: List[int] = []
        offset = 0
        for _, batch_env in self.groups:
            self.group_offsets.append(offset)
            offset += batch_env.num_envs
        first_env = self.groups[0][1]
        observation_space = spaces.Box(low=-20.0, high=20.0, shape=(first_env.OBS_DIM,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(offset, observation_space, action_space)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self):
        obs_chunks = []
        self.reset_infos = []
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        for layout, batch_env in self.groups:
            obs, info = batch_env.reset()
            obs_chunks.append(obs.detach().cpu().numpy())
            for i in range(batch_env.num_envs):
                self.reset_infos.append(
                    {
                        "branch_layout": layout,
                        "start": info["start"][i].detach().cpu().numpy(),
                        "goal": info["goal"][i].detach().cpu().numpy(),
                        "clot_on": bool(info["clot_on"][i].item()),
                    }
                )
        return np.concatenate(obs_chunks, axis=0)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        obs_chunks = []
        reward_chunks = []
        done_chunks = []
        infos: List[dict] = []
        for group_idx, (layout, batch_env) in enumerate(self.groups):
            offset = self.group_offsets[group_idx]
            action_slice = self.actions[offset : offset + batch_env.num_envs]
            obs, reward, terminated, truncated, info = batch_env.step(
                torch.as_tensor(action_slice, device=batch_env.device, dtype=torch.float32)
            )
            done = terminated | truncated
            obs_np = obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy()
            done_np = done.detach().cpu().numpy()
            terminated_np = terminated.detach().cpu().numpy()
            truncated_np = truncated.detach().cpu().numpy()

            done_ids = []
            group_infos: List[dict] = []
            for i in range(batch_env.num_envs):
                global_i = offset + i
                self.episode_returns[global_i] += float(reward_np[i])
                self.episode_lengths[global_i] += 1
                item = {
                    "branch_layout": layout,
                    "accepted_move": bool(info["accepted_move"][i].item()),
                    "distance_to_goal": float(info["distance_to_goal"][i].item()),
                    "clearance": float(info["clearance"][i].item()),
                    "clot_on": bool(info["clot_on"][i].item()),
                    "is_success": bool(terminated_np[i]),
                    "TimeLimit.truncated": bool(truncated_np[i] and not terminated_np[i]),
                }
                if done_np[i]:
                    item["terminal_observation"] = obs_np[i].copy()
                    item["episode"] = {
                        "r": float(self.episode_returns[global_i]),
                        "l": int(self.episode_lengths[global_i]),
                        "is_success": bool(terminated_np[i]),
                    }
                    item["episode_success"] = bool(terminated_np[i])
                    done_ids.append(i)
                group_infos.append(item)

            if done_ids:
                reset_ids = torch.tensor(done_ids, device=batch_env.device, dtype=torch.long)
                reset_obs, reset_info = batch_env.reset(reset_ids)
                reset_obs_np = reset_obs.detach().cpu().numpy()
                for local_j, env_id in enumerate(done_ids):
                    obs_np[env_id] = reset_obs_np[env_id]
                    global_id = offset + env_id
                    self.episode_returns[global_id] = 0.0
                    self.episode_lengths[global_id] = 0
                    group_infos[env_id]["reset_start"] = reset_info["start"][local_j].detach().cpu().numpy()
                    group_infos[env_id]["reset_goal"] = reset_info["goal"][local_j].detach().cpu().numpy()
                    group_infos[env_id]["reset_clot_on"] = bool(reset_info["clot_on"][local_j].item())

            obs_chunks.append(obs_np)
            reward_chunks.append(reward_np)
            done_chunks.append(done_np)
            infos.extend(group_infos)
        return (
            np.concatenate(obs_chunks, axis=0),
            np.concatenate(reward_chunks, axis=0),
            np.concatenate(done_chunks, axis=0),
            infos,
        )

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None):
        values = []
        for _, batch_env in self.groups:
            value = getattr(batch_env, attr_name)
            values.extend([value for _ in range(batch_env.num_envs)])
        if indices is None:
            return values
        if isinstance(indices, int):
            return [values[indices]]
        return [values[i] for i in indices]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        if indices is not None:
            raise NotImplementedError("Indexed set_attr is not implemented for TorchBranchVecEnv.")
        for _, batch_env in self.groups:
            setattr(batch_env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        results = []
        for _, batch_env in self.groups:
            method = getattr(batch_env, method_name)
            result = method(*method_args, **method_kwargs)
            results.extend([result for _ in range(batch_env.num_envs)])
        if indices is None:
            return results
        if isinstance(indices, int):
            return [results[indices]]
        return [results[i] for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        count = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [False for _ in range(count)]


def _evaluate_success(model, eval_env: TorchBranchVecEnv, n_episodes: int) -> Dict[str, float]:
    if n_episodes > eval_env.num_envs:
        raise ValueError(
            f"n_eval_episodes ({n_episodes}) must be <= eval_num_envs ({eval_env.num_envs}) "
            "because online success eval counts each parallel env's first completed episode exactly once."
        )

    obs = eval_env.reset()
    completed = 0
    successes = 0
    rewards: List[float] = []
    episode_rewards = np.zeros(eval_env.num_envs, dtype=np.float64)
    completed_envs = set()
    while completed < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = eval_env.step(actions)
        episode_rewards += reward
        for idx, item in enumerate(infos):
            if not done[idx]:
                continue
            episode_reward = float(episode_rewards[idx])
            episode_rewards[idx] = 0.0
            if idx in completed_envs:
                continue
            completed_envs.add(idx)
            completed += 1
            successes += int(bool(item.get("is_success", False)))
            rewards.append(episode_reward)
            if completed >= n_episodes:
                break
    return {
        "episodes": float(completed),
        "successes": float(successes),
        "success_rate": float(successes / max(completed, 1)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_freq_total_steps: int, save_dir: Path, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.log_freq_total_steps = max(1, int(log_freq_total_steps))
        self.next_log_timestep = self.log_freq_total_steps
        self.save_dir = save_dir
        self.csv_path = save_dir / "train_metrics.csv"
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[float] = []

    def _init_callback(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=["timesteps", "episodes", "reward_mean", "length_mean", "success_rate"],
                )
                writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_rewards.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))
            self.episode_successes.append(float(bool(episode.get("is_success", False))))

        if self.num_timesteps < self.next_log_timestep:
            return True

        episodes = len(self.episode_rewards)
        if episodes == 0:
            while self.next_log_timestep <= self.num_timesteps:
                self.next_log_timestep += self.log_freq_total_steps
            return True

        reward_mean = float(np.mean(self.episode_rewards))
        length_mean = float(np.mean(self.episode_lengths))
        success_rate = float(np.mean(self.episode_successes))

        row = {
            "timesteps": int(self.num_timesteps),
            "episodes": episodes,
            "reward_mean": reward_mean,
            "length_mean": length_mean,
            "success_rate": success_rate,
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["timesteps", "episodes", "reward_mean", "length_mean", "success_rate"],
            )
            writer.writerow(row)

        self.logger.record("train_custom/episode_reward_mean", reward_mean)
        self.logger.record("train_custom/episode_length_mean", length_mean)
        self.logger.record("train_custom/success_rate", success_rate)
        self.logger.record("train_custom/episodes", episodes)
        self.logger.dump(self.num_timesteps)
        if self.verbose:
            print(
                f"[train_metrics] steps={self.num_timesteps} episodes={episodes} "
                f"reward_mean={reward_mean:.3f} success_rate={success_rate:.3f}"
            )

        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()
        while self.next_log_timestep <= self.num_timesteps:
            self.next_log_timestep += self.log_freq_total_steps
        return True


class SuccessRateEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_envs: Dict[str, TorchBranchVecEnv],
        eval_freq_total_steps: int,
        n_eval_episodes: int,
        save_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_envs = eval_envs
        self.eval_freq_total_steps = max(1, int(eval_freq_total_steps))
        self.next_eval_timestep = self.eval_freq_total_steps
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.save_dir = save_dir
        self.best_macro_success = -1.0
        self.csv_path = save_dir / "eval_success.csv"
        self.jsonl_path = save_dir / "eval_success.jsonl"

    def _init_callback(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=["timesteps", "layout", "episodes", "successes", "success_rate", "mean_reward"],
                )
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq_total_steps <= 0 or self.num_timesteps < self.next_eval_timestep:
            return True
        rows = []
        for layout, eval_env in self.eval_envs.items():
            metrics = _evaluate_success(self.model, eval_env, self.n_eval_episodes)
            row = {"timesteps": int(self.num_timesteps), "layout": layout, **metrics}
            rows.append(row)
        macro_success = float(np.mean([row["success_rate"] for row in rows])) if rows else 0.0
        with self.csv_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["timesteps", "layout", "episodes", "successes", "success_rate", "mean_reward"],
            )
            writer.writerows(rows)
        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"timesteps": int(self.num_timesteps), "macro_success_rate": macro_success, "rows": rows}) + "\n")
        self.logger.record("eval/macro_success_rate", macro_success)
        for row in rows:
            layout_tag = row["layout"].replace("/", "_")
            self.logger.record(f"eval_success/{layout_tag}", row["success_rate"])
            self.logger.record(f"eval_reward/{layout_tag}", row["mean_reward"])
        self.logger.dump(self.num_timesteps)
        if self.verbose:
            parts = ", ".join(f"{row['layout']}={row['success_rate']:.3f}" for row in rows)
            print(f"[eval_success] steps={self.num_timesteps} macro={macro_success:.3f} {parts}")
        if macro_success > self.best_macro_success:
            self.best_macro_success = macro_success
            best_path = self.save_dir / "best_success_model"
            self.model.save(str(best_path))
            if self.verbose:
                print(f"[eval_success] saved new best success model to: {best_path}.zip")
        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq_total_steps
        return True


def _make_train_env(args: argparse.Namespace, layouts: Sequence[str]) -> TorchBranchVecEnv:
    if args.mode == "single":
        groups = [(layouts[0], _make_batch_env(args, layouts[0], args.num_envs, args.seed, args.episode_seconds))]
        return TorchBranchVecEnv(groups)
    groups = []
    for i, layout in enumerate(layouts):
        groups.append(
            (
                layout,
                _make_batch_env(
                    args,
                    layout,
                    args.num_envs_per_layout,
                    args.seed + i * 1009,
                    args.episode_seconds,
                ),
            )
        )
    return TorchBranchVecEnv(groups)


def _make_eval_envs(args: argparse.Namespace, layouts: Sequence[str]) -> Dict[str, TorchBranchVecEnv]:
    eval_envs = {}
    for i, layout in enumerate(layouts):
        batch_env = _make_batch_env(
            args,
            layout,
            args.eval_num_envs,
            args.seed + 100_000 + i * 1009,
            args.eval_episode_seconds,
        )
        eval_envs[layout] = TorchBranchVecEnv([(layout, batch_env)])
    return eval_envs


def main() -> None:
    parser = build_arg_parser()
    initial_args = parser.parse_args()
    cfg = _load_config(initial_args.config)
    if cfg is not None:
        args = parser.parse_args(args=[], namespace=argparse.Namespace())
        _apply_config(args, cfg)
        args = parser.parse_args(namespace=args)
    else:
        args = initial_args

    if args.eval_num_envs <= 0:
        raise SystemExit("--eval-num-envs must be a positive integer.")
    if args.n_eval_episodes <= 0:
        raise SystemExit("--n-eval-episodes must be a positive integer.")
    if args.eval_freq > 0 and args.n_eval_episodes > args.eval_num_envs:
        raise SystemExit(
            "--n-eval-episodes must be <= --eval-num-envs because online success eval now "
            "counts each parallel env's first completed episode exactly once."
        )
    if args.mode == "single" and args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.mode == "mixed" and args.num_envs_per_layout <= 0:
        raise SystemExit("--num-envs-per-layout must be positive.")

    args.device = str(resolve_torch_device(args.device))
    layouts = [args.branch_layout] if args.mode == "single" else _parse_layouts(args.branch_layouts)
    for layout in layouts:
        get_branch_layout(layout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.branch_layout if args.mode == "single" else f"mixed_{len(layouts)}layouts"
    args.save_dir = args.save_dir / run_name / f"run_{timestamp}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path, log_fp, original_stdout, original_stderr = _enable_terminal_log(args.save_dir)
    print(f"Terminal log file: {log_path}")

    try:
        train_env = _make_train_env(args, layouts)
        eval_envs = _make_eval_envs(args, layouts) if args.eval_freq > 0 else {}
        checkpoint_callback = CheckpointCallback(
            save_freq=max((args.checkpoint_every_updates * args.n_steps) // max(train_env.num_envs, 1), 1),
            save_path=str(args.save_dir),
            name_prefix=args.model_name,
        )
        callbacks: List[Any] = [
            checkpoint_callback,
            TrainingMetricsCallback(log_freq_total_steps=args.train_log_freq, save_dir=args.save_dir, verbose=1),
        ]
        if args.eval_freq > 0:
            callbacks.append(
                SuccessRateEvalCallback(
                    eval_envs=eval_envs,
                    eval_freq_total_steps=args.eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    save_dir=args.save_dir,
                    verbose=1,
                )
            )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(args.save_dir / "tb"),
        )

        metadata = {
            "mode": args.mode,
            "layouts": layouts,
            "train_num_envs": train_env.num_envs,
            "eval_num_envs": args.eval_num_envs,
            "eval_freq_total_env_steps": args.eval_freq,
            "train_log_freq_total_env_steps": args.train_log_freq,
            "device": args.device,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        (args.save_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("Observation space:", train_env.observation_space)
        print("Action space:", train_env.action_space)
        print(f"Resolved device: {args.device}")
        print(f"Mode: {args.mode}")
        print(f"Layouts: {', '.join(layouts)}")
        print(f"Training envs: {train_env.num_envs}")
        print(f"Rollout horizon: {args.n_steps}")
        print(f"Training metric log frequency: every {args.train_log_freq} total env steps")
        if args.eval_freq > 0:
            print(f"Eval success frequency: every {args.eval_freq} total env steps")
            print(
                f"Eval setup: {args.n_eval_episodes} episodes per layout, "
                f"{args.eval_episode_seconds:.2f}s each, {args.eval_num_envs} eval envs per layout"
            )
        else:
            print("Eval success frequency: disabled")
        print(f"Training for {args.total_timesteps} timesteps...")

        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
        model_path = args.save_dir / f"{args.model_name}_final"
        model.save(str(model_path))
        print(f"Saved final model to: {model_path}.zip")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()


if __name__ == "__main__":
    main()
