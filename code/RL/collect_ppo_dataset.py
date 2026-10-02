"""
Collect successful PPO expert rollouts as MicroEnv VLA demonstrations.

Each episode is buffered in memory and is written only when the environment
reports goal termination. Failed and time-limit-truncated episodes are dropped.

This version filters start-goal pairs immediately after reset. Invalid pairs are
resampled before rollout starts, so too-short start-goal tasks are not executed
and not recorded.

python MicroEnv/scripts/collect_ppo_dataset.py --output-dir MicroEnv/vla_demos_ppo_goal6 --successful-episodes-per-layout 100 --num-envs 256 --GPU 0,1,2,3,4 --min-x-distance 1 --min-y-distance 2 --max-x-distance 6 --max-y-distance 6

"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MicroEnv.MicroEnv.load_vessel_branches import _layout_path, get_branch_layout
from MicroEnv.MicroEnv.torch_vessel_env import TorchVesselBatchEnv, resolve_torch_device
from MicroEnv.MicroEnv.vla_renderer import (
    add_episode_clot,
    build_static_layout_background,
    render_vla_frame,
)
from MicroEnv.MicroEnv.vessel_train_cfg import VesselTrainConfig


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "MicroEnv/checkpoints/ppo_mixed_layouts/20260614_160714/"
    "mixed_10layouts/run_20260614_160718"
)
DEFAULT_INSTRUCTION = "Move the blue agent to the yellow goal dot through the vessel."
VLA_STATE_FIELDS = ["x", "y", "velocity_x", "velocity_y", "goal_delta_x", "goal_delta_y"]


@dataclass
class EpisodeBuffer:
    layout: str
    started_at: str
    start: list[float]
    goal: list[float]
    clot: dict[str, Any] = field(default_factory=dict)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    reward: float = 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=("best", "final", "latest"),
        default="best",
        help=(
            "Which checkpoint to collect from when --checkpoint is not set: "
            "best_success_model.zip, <model_name>_final.zip, or the highest-step checkpoint."
        ),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "MicroEnv/vla_demos_ppo_goal6",
    )
    parser.add_argument("--successful-episodes-per-layout", type=int, default=500)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Batched environments per layout in each worker, matching PPO eval by default.",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Defaults to layouts stored in run_metadata.json.",
    )
    parser.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--GPU",
        "--gpu",
        dest="gpu",
        type=str,
        default="all",
        help="Comma-separated GPU IDs, e.g. --GPU 0,1,2,3,4. Default: all visible GPUs.",
    )
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--no-goal-distance",
        "--no-goal-delta",
        action="store_true",
        help=(
            "Deprecated for collection. Goal delta is always saved in states.npy; "
            "use the conversion script option with the same name to zero it for training."
        ),
    )
    parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override PPO run_metadata start randomization. Pass --no-randomize-start "
            "to keep VesselTrainConfig.Env.start fixed."
        ),
    )
    parser.add_argument(
        "--randomize-goal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override PPO run_metadata goal randomization. Pass --no-randomize-goal "
            "to keep VesselTrainConfig.Env.goal fixed."
        ),
    )
    parser.add_argument(
        "--ppo-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use deterministic PPO actions. Enabled by default; pass "
            "--no-ppo-deterministic to sample stochastic PPO actions."
        ),
    )
    parser.add_argument(
        "--min-episode-steps",
        type=int,
        default=8,
        help="Discard successful episodes shorter than this number of control steps.",
    )
    parser.add_argument(
        "--clot-event",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable clot episodes explicitly. Clots are disabled when this flag is omitted, "
            "regardless of run_metadata.json or VesselTrainConfig.Env."
        ),
    )
    parser.add_argument(
        "--use-current-clot-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use VesselTrainConfig.Env for all clot_* settings instead of values frozen "
            "in run_metadata.json. Enabled by default."
        ),
    )
    parser.add_argument(
        "--clot-probability",
        type=float,
        default=None,
        help="Override clot probability after selecting metadata/current config.",
    )
    parser.add_argument(
        "--clot-radius-factor",
        type=float,
        default=None,
        help="Override the physical clot-segment lumen-radius factor.",
    )
    parser.add_argument(
        "--clot-fixed-branch",
        "--clot-fixed-branch-idx",
        dest="clot_fixed_branch_idx",
        type=int,
        default=None,
        help="Fix clot to this branch index. Requires --clot-fixed-start-idx and --clot-fixed-end-idx.",
    )
    parser.add_argument(
        "--clot-fixed-start-idx",
        type=int,
        default=None,
        help="Fix clot start segment index within --clot-fixed-branch.",
    )
    parser.add_argument(
        "--clot-fixed-end-idx",
        type=int,
        default=None,
        help="Fix clot end segment index within --clot-fixed-branch.",
    )
    parser.add_argument(
        "--min-x-distance",
        "--min_x_distance",
        type=float,
        default=0.0,
        help="Minimum absolute x-distance between start and goal. Default: 0.0.",
    )
    parser.add_argument(
        "--min-y-distance",
        "--min_y_distance",
        type=float,
        default=0.0,
        help="Minimum absolute y-distance between start and goal. Default: 0.0.",
    )
    parser.add_argument(
        "--max-x-distance",
        "--max_x_distance",
        type=float,
        default=None,
        help="Maximum absolute x-distance between start and goal. Default: disabled.",
    )
    parser.add_argument(
        "--max-y-distance",
        "--max_y_distance",
        type=float,
        default=None,
        help="Maximum absolute y-distance between start and goal. Default: disabled.",
    )
    parser.add_argument(
        "--custom",
        "--自定义",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Direct constrained endpoint sampling mode. Build a valid start-goal pair pool "
            "once per layout and sample from it on reset instead of rejection-reset loops."
        ),
    )
    parser.add_argument(
        "--max-start-goal-resample-rounds",
        type=int,
        default=1000,
        help=(
            "Maximum reset-resampling rounds for invalid start-goal pairs. "
            "If this is exceeded, the script stops because the environment likely "
            "cannot sample valid start-goal pairs under the current distance filter."
        ),
    )
    parser.add_argument(
        "--max-attempts-per-layout",
        type=int,
        default=100_000,
        help="Stop with an error if this many completed episodes do not yield the requested successes.",
    )
    return parser


def parse_devices(args: argparse.Namespace) -> list[str]:
    """Parse GPU IDs and create one CUDA worker device per selected GPU."""
    if args.devices:
        return list(dict.fromkeys(args.devices))
    if args.device:
        return [args.device]

    visible_count = torch.cuda.device_count()
    if visible_count < 1:
        raise RuntimeError("No CUDA devices are visible to PyTorch.")

    spec = args.gpu.strip().lower()
    if spec in {"all", "auto"}:
        gpu_ids = list(range(visible_count))
    else:
        try:
            gpu_ids = [int(item.strip()) for item in spec.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(
                f"Invalid --GPU value {args.gpu!r}; expected comma-separated integers."
            ) from exc

        gpu_ids = list(dict.fromkeys(gpu_ids))
        if not gpu_ids:
            raise ValueError("--GPU did not contain any GPU IDs.")

        invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
        if invalid:
            raise ValueError(
                f"Invalid GPU IDs {invalid}; PyTorch sees {visible_count} GPUs with IDs "
                f"0..{visible_count - 1}."
            )

    return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]


def load_run_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any], list[str]]:
    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = (args.metadata or (run_dir / "run_metadata.json")).expanduser().resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    checkpoint = (args.checkpoint or select_model_path(run_dir, metadata, args.model)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")

    layouts = list(args.layouts or metadata.get("layouts") or [])

    if not layouts:
        raise ValueError(f"No layouts found in {metadata_path}; pass --layouts explicitly.")

    for layout in layouts:
        get_branch_layout(layout)

    return checkpoint, metadata, layouts


def checkpoint_step_number(path: Path) -> int:
    name = path.stem
    marker = "_steps"
    if not name.endswith(marker):
        return -1
    try:
        return int(name.rsplit("_", 2)[-2])
    except ValueError:
        return -1


def select_model_path(run_dir: Path, metadata: dict[str, Any], model: str) -> Path:
    model_name = str(metadata.get("args", {}).get("model_name", "ppo_vessel_torch_state_progress"))
    if model == "best":
        return run_dir / "best_success_model.zip"
    if model == "final":
        return run_dir / f"{model_name}_final.zip"
    if model == "latest":
        candidates = sorted(run_dir.glob(f"{model_name}_*_steps.zip"), key=checkpoint_step_number)
        if not candidates:
            raise FileNotFoundError(f"No step checkpoints found in: {run_dir}")
        return candidates[-1]
    raise ValueError(f"Unsupported model choice: {model}")


def write_collection_log(
    output_dir: Path,
    summary: dict[str, Any],
    summary_path: Path,
    collection_started_at: datetime,
) -> Path:
    finished_at = datetime.now()
    command_argv = [sys.executable, *sys.argv]
    command_line = " ".join(shlex.quote(part) for part in command_argv)
    layout_counts = {
        str(item.get("layout")): int(item.get("successful_episodes", 0))
        for item in summary.get("layouts", [])
    }
    total_successes = int(sum(layout_counts.values()))
    if "layout_checkpoints" in summary:
        policy_files = sorted({str(path) for path in summary["layout_checkpoints"].values()})
    else:
        policy_files = [str(summary["checkpoint"])] if summary.get("checkpoint") else []

    summary["collection_started_at"] = collection_started_at.isoformat(timespec="seconds")
    summary["collection_finished_at"] = finished_at.isoformat(timespec="seconds")
    summary["collection_seconds"] = (finished_at - collection_started_at).total_seconds()
    summary["collected_successful_episodes"] = total_successes

    record = {
        "command_line": command_line,
        "argv": command_argv,
        "cwd": str(Path.cwd()),
        "ppo_policy_files": policy_files,
        "collected_successful_episodes": total_successes,
        "successful_episodes_by_layout": layout_counts,
        "collection_started_at": summary["collection_started_at"],
        "collection_finished_at": summary["collection_finished_at"],
        "collection_seconds": summary["collection_seconds"],
        "output_dir": str(output_dir),
    }
    if summary.get("checkpoint_root") is not None:
        record["checkpoint_root"] = summary.get("checkpoint_root")
    if summary.get("model") is not None:
        record["model"] = summary.get("model")
    if summary.get("collector_version") is not None:
        record["collector_version"] = summary.get("collector_version")

    log_path = output_dir / "collection_log.json"
    log_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def value(config: dict[str, Any], name: str, default: Any) -> Any:
    stored = config.get(name)
    return default if stored is None else stored


def layout_endpoints(layout: str, defaults: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return per-layout start/goal when the layout JSON defines them."""
    start = defaults.start
    goal = defaults.goal
    try:
        data = json.loads(_layout_path(layout).read_text(encoding="utf-8"))
    except Exception:
        return start, goal

    def _point(name: str, fallback: Any) -> tuple[float, float]:
        point = data.get(name, fallback)
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return fallback
        return (float(point[0]), float(point[1]))

    return _point("start", start), _point("goal", goal)


def validate_fixed_clot_span_for_layouts(args: argparse.Namespace, layouts: list[str]) -> None:
    fixed_values = (args.clot_fixed_branch_idx, args.clot_fixed_start_idx, args.clot_fixed_end_idx)
    if any(value is not None for value in fixed_values) and not all(
        value is not None for value in fixed_values
    ):
        raise ValueError(
            "Fixed clot position requires --clot-fixed-branch, "
            "--clot-fixed-start-idx, and --clot-fixed-end-idx together."
        )
    if args.clot_fixed_branch_idx is None:
        return

    branch_idx = int(args.clot_fixed_branch_idx)
    start_idx = int(args.clot_fixed_start_idx)
    end_idx = int(args.clot_fixed_end_idx)
    invalid: list[str] = []
    for layout in layouts:
        branches = get_branch_layout(layout)
        if not (0 <= branch_idx < len(branches)):
            invalid.append(f"{layout}: branch {branch_idx} missing; valid branches 0..{len(branches) - 1}")
            continue
        segment_count = max(0, len(branches[branch_idx]["points"]) - 1)
        if not (0 <= start_idx <= end_idx < segment_count):
            branch_name = branches[branch_idx].get("name", f"branch_{branch_idx}")
            invalid.append(
                f"{layout}: branch {branch_idx} ({branch_name}) has {segment_count} segments; "
                f"valid span is 0 <= start_idx <= end_idx < {segment_count}"
            )
    if invalid:
        details = "\n  ".join(invalid)
        raise ValueError(
            f"Invalid fixed clot span ({branch_idx}, {start_idx}, {end_idx}) for selected layouts:\n  {details}"
        )


def make_env(
    layout: str,
    num_envs: int,
    device: str,
    seed: int,
    run_args: dict[str, Any],
) -> TorchVesselBatchEnv:
    defaults = VesselTrainConfig.Env
    layout_start, layout_goal = layout_endpoints(layout, defaults)
    return TorchVesselBatchEnv(
        num_envs=num_envs,
        device=device,
        seed=seed,
        sim_dt=value(run_args, "sim_dt", defaults.sim_dt),
        decimation=value(run_args, "decimation", defaults.decimation),
        episode_seconds=value(run_args, "episode_seconds", defaults.episode_seconds),
        step_scale=defaults.step_scale,
        action_noise_std=value(run_args, "action_noise_std", defaults.action_noise_std),
        drift_gain=defaults.drift_gain,
        inertia_decay=defaults.inertia_decay,
        max_speed=defaults.max_speed,
        wall_margin=defaults.wall_margin,
        agent_radius=defaults.agent_radius,
        clot_event=value(run_args, "clot_event", defaults.clot_event),
        clot_probability=value(run_args, "clot_probability", defaults.clot_probability),
        clot_randomize_on_reset=value(
            run_args, "clot_randomize_on_reset", defaults.clot_randomize_on_reset
        ),
        clot_region=(
            value(run_args, "clot_region_start", defaults.clot_region[0]),
            value(run_args, "clot_region_end", defaults.clot_region[1]),
        ),
        clot_length_range=(
            value(run_args, "clot_length_min", defaults.clot_length_range[0]),
            value(run_args, "clot_length_max", defaults.clot_length_range[1]),
        ),
        clot_max_length=value(run_args, "clot_max_length", defaults.clot_max_length),
        clot_junction_margin=value(run_args, "clot_junction_margin", defaults.clot_junction_margin),
        clot_min_y_slice_paths=value(
            run_args, "clot_min_y_slice_paths", defaults.clot_min_y_slice_paths
        ),
        clot_y_slice_samples=value(run_args, "clot_y_slice_samples", defaults.clot_y_slice_samples),
        clot_y_merge_tolerance=value(
            run_args, "clot_y_merge_tolerance", defaults.clot_y_merge_tolerance
        ),
        clot_branch_randomize=value(
            run_args, "clot_branch_randomize", defaults.clot_branch_randomize
        ),
        clot_radius_factor=value(run_args, "clot_radius_factor", defaults.clot_radius_factor),
        clot_fixed_branch_idx=value(run_args, "clot_fixed_branch_idx", None),
        clot_fixed_start_idx=value(run_args, "clot_fixed_start_idx", None),
        clot_fixed_end_idx=value(run_args, "clot_fixed_end_idx", None),
        branches=get_branch_layout(layout),
        start=layout_start,
        goal=layout_goal,
        randomize_start=value(run_args, "randomize_start", defaults.randomize_start),
        randomize_goal=value(run_args, "randomize_goal", defaults.randomize_goal),
        min_clot_distance=defaults.min_clot_distance,
        goal_radius=defaults.goal_radius,
    )


def vector(tensor: torch.Tensor, env_id: int) -> list[float]:
    return tensor[env_id].detach().cpu().numpy().astype(float).tolist()


def new_buffer(env: TorchVesselBatchEnv, env_id: int, layout: str) -> EpisodeBuffer:
    return EpisodeBuffer(
        layout=layout,
        started_at=datetime.now().isoformat(timespec="microseconds"),
        start=vector(env.start, env_id),
        goal=vector(env.goal, env_id),
        clot={
            "on": bool(env.clot_on[env_id].item()),
            "branch_idx": int(env.clot_branch_idx[env_id].item()),
            "start_idx": int(env.clot_start_idx[env_id].item()),
            "end_idx": int(env.clot_end_idx[env_id].item()),
            "radius_factor": float(env.clot_radius_factor),
        },
    )


def start_goal_distances(
    env: TorchVesselBatchEnv,
    env_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return Euclidean start-goal distances for selected envs."""
    distances = torch.norm(env.goal - env.start, dim=1)
    if env_ids is None:
        return distances
    return distances[env_ids]


def invalid_start_goal_mask(
    x_distances: torch.Tensor,
    y_distances: torch.Tensor,
    min_x_distance: float,
    min_y_distance: float,
    max_x_distance: Optional[float],
    max_y_distance: Optional[float],
) -> torch.Tensor:
    """Return mask of pairs outside the independent x/y distance bounds."""
    invalid = (x_distances < min_x_distance) | (y_distances < min_y_distance)
    if max_x_distance is not None:
        invalid = invalid | (x_distances > max_x_distance)
    if max_y_distance is not None:
        invalid = invalid | (y_distances > max_y_distance)
    return invalid


def build_custom_endpoint_pool(
    env: TorchVesselBatchEnv,
    min_x_distance: float,
    min_y_distance: float,
    max_x_distance: Optional[float],
    max_y_distance: Optional[float],
    candidate_pairs: int = 262_144,
    max_pool_size: int = 65_536,
) -> torch.Tensor:
    """Build reusable geometry-valid endpoint pairs for direct custom sampling."""
    if env.randomize_start:
        starts = env._sample_points_in_vessel(candidate_pairs)
    else:
        starts = env.start[0].unsqueeze(0).expand(candidate_pairs, -1).clone()
    if env.randomize_goal:
        goals = env._sample_points_in_vessel(candidate_pairs)
    else:
        goals = env.goal[0].unsqueeze(0).expand(candidate_pairs, -1).clone()
    deltas = torch.abs(goals - starts)
    invalid = invalid_start_goal_mask(
        deltas[:, 0],
        deltas[:, 1],
        min_x_distance,
        min_y_distance,
        max_x_distance,
        max_y_distance,
    )
    valid_ids = torch.nonzero(~invalid, as_tuple=False).flatten()
    if valid_ids.numel() == 0:
        raise RuntimeError(
            "Custom endpoint sampling found no valid start-goal pairs. "
            f"Required x=[{min_x_distance}, {max_x_distance}], "
            f"y=[{min_y_distance}, {max_y_distance}]."
        )
    if valid_ids.numel() > max_pool_size:
        valid_ids = valid_ids[:max_pool_size]
    return torch.stack([starts[valid_ids], goals[valid_ids]], dim=1).contiguous()


def apply_custom_endpoints(
    env: TorchVesselBatchEnv,
    env_ids: torch.Tensor,
    endpoint_pool: torch.Tensor,
) -> int:
    """Assign valid pooled endpoint pairs, resampling only clot-conflicting pairs."""
    remaining = env_ids
    clot_resamples = 0
    for _ in range(128):
        if remaining.numel() == 0:
            break
        pair_ids = env._randint(0, endpoint_pool.shape[0], (remaining.shape[0],))
        pairs = endpoint_pool[pair_ids]
        starts = pairs[:, 0]
        goals = pairs[:, 1]
        blocked = env._points_blocked_by_clot(starts, remaining) | env._points_blocked_by_clot(
            goals, remaining
        )
        good = ~blocked
        if torch.any(good):
            good_ids = remaining[good]
            env.start[good_ids] = starts[good]
            env.goal[good_ids] = goals[good]
        clot_resamples += int(blocked.sum().item())
        remaining = remaining[blocked]

    if remaining.numel() > 0:
        # Match the environment's existing endpoint-validity priority: in an
        # over-constrained rare case, keep endpoints and disable those clots.
        env.clot_on[remaining] = False
        pair_ids = env._randint(0, endpoint_pool.shape[0], (remaining.shape[0],))
        pairs = endpoint_pool[pair_ids]
        env.start[remaining] = pairs[:, 0]
        env.goal[remaining] = pairs[:, 1]

    env.pos[env_ids] = env.start[env_ids]
    env.velocity[env_ids] = 0.0
    env.steps[env_ids] = 0
    env.distance_to_goal[env_ids] = torch.linalg.norm(env.goal[env_ids] - env.pos[env_ids], dim=1)
    return clot_resamples


def reset_until_valid_start_goal(
    env: TorchVesselBatchEnv,
    reset_ids: Optional[torch.Tensor],
    min_x_distance: float,
    min_y_distance: float,
    max_x_distance: Optional[float],
    max_y_distance: Optional[float],
    max_resample_rounds: int,
    custom_endpoint_pool: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Any, int]:
    """
    Reset envs and immediately resample invalid start-goal pairs.

    This function is the key difference from post-filtering:
    invalid episodes never enter rollout and never get recorded into buffers.

    Returns:
        obs: observation returned by env.reset
        info: info returned by env.reset
        resampled_pairs: number of invalid start-goal samples discarded during reset
    """
    obs, info = env.reset(reset_ids)

    if reset_ids is None:
        candidate_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        candidate_ids = reset_ids.to(device=env.device, dtype=torch.long)

    if custom_endpoint_pool is not None:
        resampled_pairs = apply_custom_endpoints(env, candidate_ids, custom_endpoint_pool)
        obs = env._compute_obs()
        info = {
            "start": env.start[candidate_ids].clone(),
            "goal": env.goal[candidate_ids].clone(),
            "clot_on": env.clot_on[candidate_ids].clone(),
        }
        return obs, info, resampled_pairs

    resampled_pairs = 0

    for round_index in range(max_resample_rounds + 1):
        deltas = torch.abs(env.goal[candidate_ids] - env.start[candidate_ids])
        x_distances = deltas[:, 0]
        y_distances = deltas[:, 1]
        invalid = invalid_start_goal_mask(
            x_distances,
            y_distances,
            min_x_distance,
            min_y_distance,
            max_x_distance,
            max_y_distance,
        )

        if not bool(torch.any(invalid).item()):
            return obs, info, resampled_pairs

        if round_index >= max_resample_rounds:
            x_cpu = x_distances.detach().cpu().numpy()
            y_cpu = y_distances.detach().cpu().numpy()
            raise RuntimeError(
                "Could not sample valid start-goal pairs after "
                f"{max_resample_rounds} resampling rounds. "
                f"Current x stats: min={float(x_cpu.min()):.4f}, "
                f"mean={float(x_cpu.mean()):.4f}, max={float(x_cpu.max()):.4f}; "
                f"y stats: min={float(y_cpu.min()):.4f}, "
                f"mean={float(y_cpu.mean()):.4f}, max={float(y_cpu.max()):.4f}. "
                f"Required x=[{min_x_distance}, {max_x_distance}], "
                f"y=[{min_y_distance}, {max_y_distance}]. "
                "Check randomize_start/randomize_goal or reduce the distance filter."
            )

        bad_ids = candidate_ids[invalid]
        resampled_pairs += int(bad_ids.numel())

        obs, info = env.reset(bad_ids)

    raise RuntimeError("Unexpected reset_until_valid_start_goal control flow.")


def episode_start_goal_distance(buffer: EpisodeBuffer) -> float:
    start = np.asarray(buffer.start, dtype=np.float32)
    goal = np.asarray(buffer.goal, dtype=np.float32)
    return float(np.linalg.norm(goal - start))


def record_step(
    buffer: EpisodeBuffer,
    state: np.ndarray,
    action: np.ndarray,
) -> None:
    """Buffer only compact numeric data; rendering is deferred until success."""
    buffer.states.append(np.asarray(state, dtype=np.float32).reshape(6).copy())
    buffer.actions.append(np.asarray(action, dtype=np.float32).reshape(2).copy())


def save_success(
    buffer: EpisodeBuffer,
    env: TorchVesselBatchEnv,
    env_id: int,
    output_dir: Path,
    instruction: str,
    checkpoint: Path,
    image_size: int,
    background: str,
    static_background: np.ndarray,
    ppo_deterministic: bool,
    custom_endpoint_sampling: bool,
    no_goal_distance: bool,
) -> tuple[Path, float]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"episode_{timestamp}"
    clot_bucket = "clot_on" if bool(buffer.clot["on"]) else "no_clot"
    bucket_dir = output_dir / buffer.layout / clot_bucket
    partial_dir = bucket_dir / f".{name}.partial"
    episode_dir = bucket_dir / name

    if partial_dir.exists():
        shutil.rmtree(partial_dir)

    images_dir = partial_dir / "images"
    images_dir.mkdir(parents=True)

    states = np.stack(buffer.states).astype(np.float32)
    actions = np.stack(buffer.actions).astype(np.float32)

    render_started = time.perf_counter()

    start = np.asarray(buffer.start, dtype=np.float32)
    goal = np.asarray(buffer.goal, dtype=np.float32)
    start_goal_distance = float(np.linalg.norm(goal - start))
    episode_background = add_episode_clot(env, static_background, buffer.clot)

    for index, state in enumerate(states):
        frame = render_vla_frame(env, episode_background, start, goal, state[:2])
        Image.fromarray(frame).save(
            images_dir / f"{index:06d}.png",
            format="PNG",
            compress_level=1,
            optimize=False,
        )

    render_seconds = time.perf_counter() - render_started

    np.save(partial_dir / "states.npy", states)
    np.save(partial_dir / "actions.npy", actions)

    distance = float(env.distance_to_goal[env_id].item())

    metadata = {
        "instruction": instruction,
        "status": "success",
        "success": True,
        "terminated": True,
        "truncated": False,
        "reward": float(buffer.reward),
        "num_steps": int(len(actions)),
        "action_dim": 2,
        "state_dim": 6,
        "state_fields": VLA_STATE_FIELDS,
        "proprio_mode": "position_velocity_goal_delta",
        "goal_delta_recorded": True,
        "action_fields": ["x", "y"],
        "image_size": [image_size, image_size],
        "image_dir": "images",
        "image_background": background,
        "control_dt": float(env.control_dt),
        "sim_dt": float(env.sim_dt),
        "decimation": int(env.decimation),
        "episode_seconds": float(env.episode_seconds),
        "source": "ppo_expert",
        "source_checkpoint": str(checkpoint),
        "ppo_deterministic": bool(ppo_deterministic),
        "endpoint_sampling": "custom_pool" if custom_endpoint_sampling else "reset_rejection",
        "rendering": "deferred_success_only_snapshot_style_cached_background",
        "render_seconds": render_seconds,
        "branch_layout": buffer.layout,
        "started_at": buffer.started_at,
        "finished_at": datetime.now().isoformat(timespec="microseconds"),
        "initial_env": {
            "start": buffer.start,
            "goal": buffer.goal,
            "start_goal_distance": start_goal_distance,
            "start_goal_x_distance": float(abs(goal[0] - start[0])),
            "start_goal_y_distance": float(abs(goal[1] - start[1])),
        },
        "clot": buffer.clot,
        "final_env": {
            "pos": vector(env.pos, env_id),
            "goal": vector(env.goal, env_id),
            "distance_to_goal": distance,
            "clot_on": bool(env.clot_on[env_id].item()),
        },
    }

    (partial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    partial_dir.rename(episode_dir)

    return episode_dir, render_seconds


def collect_layout(
    model: PPO,
    checkpoint: Path,
    metadata: dict[str, Any],
    layout: str,
    args: argparse.Namespace,
    layout_index: int,
) -> dict[str, Any]:
    run_args = dict(metadata.get("args", {}))
    if args.use_current_clot_config:
        run_args = {key: item for key, item in run_args.items() if not key.startswith("clot_")}
    # Collection is clot-free unless the caller explicitly passes --clot-event.
    # This final override intentionally wins over metadata and current train config.
    run_args["clot_event"] = bool(args.clot_event)
    if args.clot_probability is not None:
        run_args["clot_probability"] = args.clot_probability
    if args.clot_radius_factor is not None:
        run_args["clot_radius_factor"] = args.clot_radius_factor
    if args.clot_fixed_branch_idx is not None:
        run_args["clot_fixed_branch_idx"] = args.clot_fixed_branch_idx
        run_args["clot_fixed_start_idx"] = args.clot_fixed_start_idx
        run_args["clot_fixed_end_idx"] = args.clot_fixed_end_idx
    if args.randomize_start is not None:
        run_args["randomize_start"] = args.randomize_start
    if args.randomize_goal is not None:
        run_args["randomize_goal"] = args.randomize_goal

    env = make_env(layout, args.num_envs, args.device, args.seed + layout_index * 10_000, run_args)
    effective_clot_event = bool(value(run_args, "clot_event", VesselTrainConfig.Env.clot_event))
    custom_endpoint_pool = None
    if args.custom:
        custom_endpoint_pool = build_custom_endpoint_pool(
            env,
            min_x_distance=args.min_x_distance,
            min_y_distance=args.min_y_distance,
            max_x_distance=args.max_x_distance,
            max_y_distance=args.max_y_distance,
        )
        print(
            f"[{layout}] custom endpoint pool size={custom_endpoint_pool.shape[0]}",
            flush=True,
        )
    obs, _, initial_resampled_pairs = reset_until_valid_start_goal(
        env=env,
        reset_ids=None,
        min_x_distance=args.min_x_distance,
        min_y_distance=args.min_y_distance,
        max_x_distance=args.max_x_distance,
        max_y_distance=args.max_y_distance,
        max_resample_rounds=args.max_start_goal_resample_rounds,
        custom_endpoint_pool=custom_endpoint_pool,
    )

    static_background = build_static_layout_background(env, args.image_size, args.background)
    buffers = [new_buffer(env, i, layout) for i in range(args.num_envs)]

    print(
        f"[{layout}] clot_event={effective_clot_event} "
        f"clot_probability={env.clot_probability} "
        f"clot_radius_factor={env.clot_radius_factor} "
        f"clot_on_after_reset={int(env.clot_on.sum().item())}/{args.num_envs}",
        flush=True,
    )

    initial_distances = start_goal_distances(env).detach().cpu().numpy()
    initial_x_distances = torch.abs(env.goal[:, 0] - env.start[:, 0]).detach().cpu().numpy()
    initial_y_distances = torch.abs(env.goal[:, 1] - env.start[:, 1]).detach().cpu().numpy()
    print(
        f"[{layout}] initial valid start-goal distance stats: "
        f"min={float(initial_distances.min()):.4f}, "
        f"mean={float(initial_distances.mean()):.4f}, "
        f"max={float(initial_distances.max()):.4f}, "
        f"x_min={float(initial_x_distances.min()):.4f}, "
        f"x_mean={float(initial_x_distances.mean()):.4f}, "
        f"x_max={float(initial_x_distances.max()):.4f}, "
        f"y_min={float(initial_y_distances.min()):.4f}, "
        f"y_mean={float(initial_y_distances.mean()):.4f}, "
        f"y_max={float(initial_y_distances.max()):.4f}, "
        f"resampled_pairs={initial_resampled_pairs}",
        flush=True,
    )

    successes = 0
    attempts = 0
    short_successes = 0
    reset_resampled_pairs_total = initial_resampled_pairs
    render_seconds_total = 0.0

    saved_distance_sum = 0.0
    saved_distance_min: Optional[float] = None
    saved_distance_max: Optional[float] = None

    while successes < args.successful_episodes_per_layout:
        obs_np = obs.detach().cpu().numpy().astype(np.float32)

        actions, _ = model.predict(obs_np, deterministic=args.ppo_deterministic)
        actions = np.asarray(actions, dtype=np.float32).reshape(args.num_envs, 2)

        vla_states = torch.cat([env.pos, env.velocity, env.goal - env.pos], dim=1)
        vla_states_np = vla_states.detach().cpu().numpy().astype(np.float32)

        for env_id in range(args.num_envs):
            record_step(buffers[env_id], vla_states_np[env_id], actions[env_id])

        obs, rewards, terminated, truncated, _ = env.step(
            torch.as_tensor(actions, device=env.device, dtype=torch.float32)
        )

        rewards_np = rewards.detach().cpu().numpy()
        terminated_np = terminated.detach().cpu().numpy()
        truncated_np = truncated.detach().cpu().numpy()

        for env_id, reward in enumerate(rewards_np):
            buffers[env_id].reward += float(reward)

        done_ids = np.flatnonzero(terminated_np | truncated_np).tolist()
        if not done_ids:
            continue

        for env_id in done_ids:
            attempts += 1

            is_success = bool(terminated_np[env_id])
            num_steps = len(buffers[env_id].actions)
            enough_steps = num_steps >= args.min_episode_steps

            if is_success and enough_steps and successes < args.successful_episodes_per_layout:
                start_goal_distance = episode_start_goal_distance(buffers[env_id])

                successes += 1

                path, render_seconds = save_success(
                    buffers[env_id],
                    env,
                    env_id,
                    args.output_dir,
                    args.instruction,
                    checkpoint,
                    args.image_size,
                    args.background,
                    static_background,
                    args.ppo_deterministic,
                    args.custom,
                    False,
                )

                render_seconds_total += render_seconds
                saved_distance_sum += start_goal_distance
                saved_distance_min = (
                    start_goal_distance
                    if saved_distance_min is None
                    else min(saved_distance_min, start_goal_distance)
                )
                saved_distance_max = (
                    start_goal_distance
                    if saved_distance_max is None
                    else max(saved_distance_max, start_goal_distance)
                )

                print(
                    f"[{layout}] success {successes}/{args.successful_episodes_per_layout} "
                    f"attempts={attempts} "
                    f"steps={num_steps} "
                    f"start_goal_dist={start_goal_distance:.4f} "
                    f"render={render_seconds:.2f}s "
                    f"saved={path.name}",
                    flush=True,
                )

            elif is_success:
                short_successes += 1
                start_goal_distance = episode_start_goal_distance(buffers[env_id])
                print(
                    f"[{layout}] discard_success reason=short "
                    f"attempts={attempts} "
                    f"steps={num_steps} "
                    f"start_goal_dist={start_goal_distance:.4f} "
                    f"min_episode_steps={args.min_episode_steps}",
                    flush=True,
                )

        if attempts >= args.max_attempts_per_layout and successes < args.successful_episodes_per_layout:
            raise RuntimeError(
                f"[{layout}] reached {attempts} attempts with only {successes} usable successes. "
                f"short_successes_discarded={short_successes}, "
                f"reset_resampled_pairs_total={reset_resampled_pairs_total}. "
                "If success count is too low, check whether PPO can solve longer start-goal tasks."
            )

        if successes >= args.successful_episodes_per_layout:
            break

        reset_ids = torch.as_tensor(done_ids, device=env.device, dtype=torch.long)

        obs, _, resampled_pairs = reset_until_valid_start_goal(
            env=env,
            reset_ids=reset_ids,
            min_x_distance=args.min_x_distance,
            min_y_distance=args.min_y_distance,
            max_x_distance=args.max_x_distance,
            max_y_distance=args.max_y_distance,
            max_resample_rounds=args.max_start_goal_resample_rounds,
            custom_endpoint_pool=custom_endpoint_pool,
        )
        reset_resampled_pairs_total += resampled_pairs

        for env_id in done_ids:
            buffers[env_id] = new_buffer(env, env_id, layout)

    saved_distance_mean = saved_distance_sum / max(successes, 1)

    return {
        "layout": layout,
        "successful_episodes": successes,
        "completed_attempts": attempts,
        "short_successes_discarded": short_successes,
        "success_rate": successes / max(attempts, 1),
        "render_seconds": render_seconds_total,
        "ppo_deterministic": args.ppo_deterministic,
        "endpoint_sampling": "custom_pool" if args.custom else "reset_rejection",
        "min_x_distance": args.min_x_distance,
        "min_y_distance": args.min_y_distance,
        "max_x_distance": args.max_x_distance,
        "max_y_distance": args.max_y_distance,
        "reset_resampled_pairs_total": reset_resampled_pairs_total,
        "saved_start_goal_distance_min": saved_distance_min,
        "saved_start_goal_distance_mean": saved_distance_mean,
        "saved_start_goal_distance_max": saved_distance_max,
    }


def collect_worker(
    device: str,
    indexed_layouts: list[tuple[int, str]],
    checkpoint: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run one persistent collection process on one GPU."""
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = str(resolve_torch_device(device))

    layout_names = [layout for _, layout in indexed_layouts]

    print(
        f"[worker pid={os.getpid()}] device={worker_args.device} policy_device=cpu layouts={layout_names} "
        f"num_envs_per_layout={worker_args.num_envs}",
        flush=True,
    )

    model = PPO.load(str(checkpoint), device="cpu")

    return [
        collect_layout(model, checkpoint, metadata, layout, worker_args, layout_index)
        for layout_index, layout in indexed_layouts
    ]


def main() -> None:
    args = build_arg_parser().parse_args()

    fixed_clot_values = (args.clot_fixed_branch_idx, args.clot_fixed_start_idx, args.clot_fixed_end_idx)
    if any(value is not None for value in fixed_clot_values) and not all(
        value is not None for value in fixed_clot_values
    ):
        raise ValueError(
            "Fixed clot position requires --clot-fixed-branch, "
            "--clot-fixed-start-idx, and --clot-fixed-end-idx together."
        )
    if args.successful_episodes_per_layout < 1:
        raise ValueError("--successful-episodes-per-layout must be positive.")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive.")
    if args.min_episode_steps < 1:
        raise ValueError("--min-episode-steps must be positive.")
    for axis in ("x", "y"):
        minimum = getattr(args, f"min_{axis}_distance")
        maximum = getattr(args, f"max_{axis}_distance")
        if minimum < 0:
            raise ValueError(f"--min-{axis}-distance must be non-negative.")
        if maximum is not None and maximum < 0:
            raise ValueError(f"--max-{axis}-distance must be non-negative.")
        if maximum is not None and maximum < minimum:
            raise ValueError(
                f"--max-{axis}-distance must be greater than or equal to --min-{axis}-distance."
            )
    if args.max_start_goal_resample_rounds < 1:
        raise ValueError("--max-start-goal-resample-rounds must be positive.")
    if args.no_goal_distance:
        print(
            "[warn] --no-goal-distance is ignored during collection; "
            "states.npy always records goal_delta. Use --no-goal-distance when converting.",
            flush=True,
        )

    checkpoint, metadata, layouts = load_run_config(args)
    validate_fixed_clot_span_for_layouts(args, layouts)
    devices = parse_devices(args)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint={checkpoint}")
    print(f"output_dir={args.output_dir}")
    print(f"devices={devices} workers={len(devices)} num_envs_per_layout={args.num_envs}")
    print(f"layouts={layouts}")
    print(f"successful_episodes_per_layout={args.successful_episodes_per_layout}")
    print(f"min_episode_steps={args.min_episode_steps}")
    print(f"ppo_deterministic={args.ppo_deterministic}")
    print(f"no_goal_distance={args.no_goal_distance}")
    print(f"randomize_start_override={args.randomize_start}")
    print(f"randomize_goal_override={args.randomize_goal}")
    print(f"min_x_distance={args.min_x_distance}")
    print(f"min_y_distance={args.min_y_distance}")
    print(f"max_x_distance={args.max_x_distance}")
    print(f"max_y_distance={args.max_y_distance}")
    print(f"custom_endpoint_sampling={args.custom}")
    print(f"max_start_goal_resample_rounds={args.max_start_goal_resample_rounds}")

    collection_started_at = datetime.now()

    shards: list[list[tuple[int, str]]] = [[] for _ in devices]
    for layout_index, layout in enumerate(layouts):
        shards[layout_index % len(devices)].append((layout_index, layout))

    assignments = [(device, shard) for device, shard in zip(devices, shards) if shard]

    summaries: list[dict[str, Any]] = []

    if len(assignments) == 1:
        device, shard = assignments[0]
        summaries.extend(collect_worker(device, shard, checkpoint, metadata, args))
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(assignments),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(collect_worker, device, shard, checkpoint, metadata, args)
                for device, shard in assignments
            ]
            for future in concurrent.futures.as_completed(futures):
                summaries.extend(future.result())

    layout_order = {layout: index for index, layout in enumerate(layouts)}
    summaries.sort(key=lambda item: layout_order[item["layout"]])

    summary = {
        "source": "ppo_expert",
        "checkpoint": str(checkpoint),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "instruction": args.instruction,
        "state_dim": 6,
        "state_fields": VLA_STATE_FIELDS,
        "proprio_mode": "position_velocity_goal_delta",
        "goal_delta_recorded": True,
        "randomize_start_override": args.randomize_start,
        "randomize_goal_override": args.randomize_goal,
        "only_successful_episodes_saved": True,
        "ppo_deterministic": args.ppo_deterministic,
        "endpoint_sampling": "custom_pool" if args.custom else "reset_rejection",
        "clot_config_source": (
            "current_VesselTrainConfig" if args.use_current_clot_config else "run_metadata"
        ),
        "clot_event_override": args.clot_event,
        "clot_fixed_span": (
            None
            if args.clot_fixed_branch_idx is None
            else {
                "branch_idx": args.clot_fixed_branch_idx,
                "start_idx": args.clot_fixed_start_idx,
                "end_idx": args.clot_fixed_end_idx,
            }
        ),
        "devices": devices,
        "worker_processes": len(assignments),
        "num_envs_per_layout": args.num_envs,
        "min_episode_steps": args.min_episode_steps,
        "min_x_distance": args.min_x_distance,
        "min_y_distance": args.min_y_distance,
        "max_x_distance": args.max_x_distance,
        "max_y_distance": args.max_y_distance,
        "max_start_goal_resample_rounds": args.max_start_goal_resample_rounds,
        "layouts": summaries,
    }

    summary_path = args.output_dir / "collection_summary.json"
    log_path = write_collection_log(args.output_dir, summary, summary_path, collection_started_at)

    print("collection complete")
    print(f"collection log: {log_path}")


if __name__ == "__main__":
    main()
