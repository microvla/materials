"""
Convert successful PPO expert demos into a manifest-based OpenVLA-OFT dataset.

The PPO collector stores episodes recursively under layout/clot_on or
layout/no_clot directories. This converter discovers all of them recursively.

Example:
  python3 MicroEnv/scripts/convert_ppo_demos_to_openvla_dataset.py \
    --input-dir MicroEnv/vla_demos_ppo_goal6_clot \
    --output-dir MicroEnv/openvla_datasets/microenv_ppo_goal6_clot \
    --dataset-name microenv_ppo_goal6_clot \
    --num-actions-chunk 8 --val-ratio 0.1 --seed 7 \
    --sample-episodes 1000 \
    --instruction "Move the blue agent to the yellow goal dot through the vessel."
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Support both direct script execution and ``python -m`` invocation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MicroEnv.scripts._bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(__file__)
MICROENV_ROOT = PROJECT_ROOT / "MicroEnv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PPO MicroEnv demos to an OpenVLA finetuning dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=MICROENV_ROOT / "vla_demos_ppo_goal6_clot",
        help="PPO demo root; episodes are discovered recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MICROENV_ROOT / "openvla_datasets" / "microenv_ppo_goal6_clot",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="microenv_ppo_goal6_clot", help="Dataset name stored in manifests."
    )
    parser.add_argument("--num-actions-chunk", type=int, default=8, help="Action chunk length for each training sample.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Use every Nth observation as a training-sample start while keeping each "
            "action chunk at the original control rate. The final episode frame is always retained."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of episodes held out for validation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for episode split.")
    parser.add_argument(
        "--sample-episodes",
        "--sample-n",
        type=int,
        default=None,
        help="Select the first N valid trajectories in sorted path order; default uses all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="CPU processes used to build per-episode manifest shards; default: min(8, CPU count).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override every trajectory instruction; default preserves each metadata.json instruction.",
    )
    parser.add_argument(
        "--no-goal-distance",
        "--no-goal-delta",
        action="store_true",
        help=(
            "Do not expose goal_delta to the training dataset. Input states.npy is left unchanged; "
            "converted proprio remains 6-D as [x, y, velocity_x, velocity_y, 0, 0]."
        ),
    )
    parser.add_argument(
        "--no-clot-only",
        action="store_true",
        help="Exclude clot episodes. By default both clot and no-clot episodes are included.",
    )
    parser.add_argument("--include-failures", action="store_true", help="Include non-success episodes in conversion.")
    return parser


def _compute_stats(array: np.ndarray) -> Dict[str, List[float]]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "mask": (np.ptp(arr, axis=0) > 1e-6).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


def _episode_has_clot(metadata: Dict, episode_dir: Path) -> bool:
    clot = metadata.get("clot")
    if isinstance(clot, dict) and "on" in clot:
        return bool(clot["on"])
    final_env = metadata.get("final_env")
    if isinstance(final_env, dict) and "clot_on" in final_env:
        return bool(final_env["clot_on"])
    if "clot_on" in episode_dir.parts:
        return True
    if "clot_off" in episode_dir.parts or "no_clot" in episode_dir.parts:
        return False
    return False


def _show_progress(label: str, current: int, total: int, started: float, detail: str = "") -> None:
    total = max(total, 1)
    if current != total and current != 1 and current % max(total // 100, 1) != 0:
        return
    ratio = min(current / total, 1.0)
    width = 24
    filled = int(width * ratio)
    elapsed = max(time.perf_counter() - started, 1e-6)
    rate = current / elapsed
    suffix = f" {detail}" if detail else ""
    print(
        f"\r{label:<20} [{'#' * filled}{'-' * (width - filled)}] "
        f"{current}/{total} ({ratio:6.1%}) {rate:7.1f}/s{suffix}",
        end="\n" if current == total else "",
        flush=True,
    )


def _discover_episode_dirs(root: Path, no_clot_only: bool) -> List[Path]:
    # Do not use rglob here: every episode contains an images/ directory with
    # many PNGs, making a recursive NFS walk unnecessarily expensive.
    meta_paths = set()
    for pattern in (
        "episode_*/metadata.json",
        "*/episode_*/metadata.json",
        "*/*/episode_*/metadata.json",
    ):
        meta_paths.update(root.glob(pattern))
    episodes = []
    for meta_path in sorted(meta_paths):
        episode_dir = meta_path.parent
        if any(part.startswith(".") for part in episode_dir.relative_to(root).parts):
            continue
        # PPO collector encodes clot status in the parent bucket. Filter these
        # paths before opening metadata files; unknown layouts are checked later.
        if no_clot_only and "clot_on" in episode_dir.parts:
            continue
        episodes.append(episode_dir)
    return episodes


def _build_samples_for_episode(
    episode_dir: Path,
    chunk_len: int,
    frame_skip: int,
    dataset_name: str,
    instruction_override: Optional[str],
    no_goal_distance: bool,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    states = np.load(episode_dir / "states.npy").astype(np.float32)
    actions = np.load(episode_dir / "actions.npy").astype(np.float32)
    images = sorted((episode_dir / "images").glob("*.png"))

    if not (len(states) == len(actions) == len(images)):
        raise ValueError(f"Mismatched lengths in {episode_dir}: states={len(states)} actions={len(actions)} images={len(images)}")
    if no_goal_distance:
        if states.ndim != 2 or states.shape[1] < 6:
            raise ValueError(f"--no-goal-distance expects states with at least 6 columns in {episode_dir}: {states.shape}")
        states = states.copy()
        states[:, 4:6] = 0.0

    try:
        episode_id = str(episode_dir.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        episode_id = str(episode_dir.resolve())

    instruction = instruction_override or metadata.get("instruction")
    if not instruction:
        raise ValueError(f"Missing instruction in {episode_dir / 'metadata.json'}; pass --instruction.")

    samples = []
    action_dim = int(actions.shape[1])
    step_indices = list(range(0, len(actions), frame_skip))
    if step_indices[-1] != len(actions) - 1:
        step_indices.append(len(actions) - 1)
    for step_idx in step_indices:
        image_path = episode_dir / "images" / images[step_idx].name
        rel_image_path = image_path.resolve().relative_to(PROJECT_ROOT) if str(image_path.resolve()).startswith(str(PROJECT_ROOT.resolve())) else image_path.resolve()
        available_actions = actions[step_idx : step_idx + chunk_len]
        action_chunk = np.zeros((chunk_len, action_dim), dtype=np.float32)
        action_chunk[: len(available_actions)] = available_actions
        samples.append(
            {
                "dataset_name": dataset_name,
                "instruction": instruction,
                "episode_id": episode_id,
                "step_idx": step_idx,
                "image_path": str(rel_image_path),
                "proprio": states[step_idx].tolist(),
                "actions": action_chunk.tolist(),
            }
        )
    return samples, states, actions


def _convert_episode_worker(job: Tuple) -> Dict:
    (
        episode_index,
        episode_dir_string,
        split,
        chunk_len,
        frame_skip,
        dataset_name,
        instruction_override,
        no_goal_distance,
        shard_root_string,
    ) = job
    episode_dir = Path(episode_dir_string)
    samples, states, actions = _build_samples_for_episode(
        episode_dir,
        chunk_len,
        frame_skip,
        dataset_name,
        instruction_override,
        no_goal_distance,
    )
    shard_path = Path(shard_root_string) / f"{episode_index:08d}_{split}.jsonl"
    with shard_path.open("w") as output:
        for sample in samples:
            output.write(json.dumps(sample) + "\n")
    return {
        "episode_index": episode_index,
        "split": split,
        "shard_path": str(shard_path),
        "num_samples": len(samples),
        "states": states,
        "actions": actions,
    }


def _merge_shards(output_path: Path, results: List[Dict], split: str) -> int:
    selected = sorted(
        (result for result in results if result["split"] == split),
        key=lambda result: result["episode_index"],
    )
    started = time.perf_counter()
    sample_count = 0
    with output_path.open("w") as output:
        for index, result in enumerate(selected, start=1):
            with Path(result["shard_path"]).open() as shard:
                shutil.copyfileobj(shard, output, length=1024 * 1024)
            sample_count += int(result["num_samples"])
            _show_progress(f"Merging {split}", index, len(selected), started, f"samples={sample_count}")
    if not selected:
        print(f"Merging {split:<10} [empty]", flush=True)
    return sample_count


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_actions_chunk < 1:
        raise SystemExit("--num-actions-chunk must be >= 1")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be >= 1")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.sample_episodes is not None and args.sample_episodes < 1:
        raise SystemExit("--sample-episodes must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    discover_started = time.perf_counter()
    discovered_episodes = _discover_episode_dirs(input_dir, no_clot_only=args.no_clot_only)
    print(
        f"Discovered {len(discovered_episodes)} candidate episodes in "
        f"{time.perf_counter() - discover_started:.2f}s",
        flush=True,
    )
    rng = random.Random(args.seed)
    # Deterministic first-N selection; randomness is used only for train/val split.
    candidates = discovered_episodes
    episodes = []
    skipped_empty = []
    select_started = time.perf_counter()
    selection_target = min(args.sample_episodes or len(candidates), len(candidates))
    for candidate_index, episode_dir in enumerate(candidates, start=1):
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        if not (args.include_failures or bool(metadata.get("success"))):
            _show_progress(
                "Selecting episodes",
                candidate_index,
                len(candidates),
                select_started,
                f"selected={len(episodes)}/{selection_target}",
            )
            continue
        if args.no_clot_only and _episode_has_clot(metadata, episode_dir):
            _show_progress(
                "Selecting episodes",
                candidate_index,
                len(candidates),
                select_started,
                f"selected={len(episodes)}/{selection_target}",
            )
            continue
        actions = np.load(episode_dir / "actions.npy", mmap_mode="r")
        if len(actions) == 0:
            skipped_empty.append(str(episode_dir))
        else:
            episodes.append(episode_dir)
        _show_progress(
            "Selecting episodes",
            candidate_index,
            len(candidates),
            select_started,
            f"selected={len(episodes)}/{selection_target}",
        )
        if args.sample_episodes is not None and len(episodes) >= args.sample_episodes:
            print(
                f"\nSelected {len(episodes)} episodes after scanning {candidate_index} candidates "
                f"in {time.perf_counter() - select_started:.2f}s",
                flush=True,
            )
            break
    if not episodes:
        raise SystemExit(f"No episodes found in {input_dir}")

    available_episode_count = len(discovered_episodes)

    shuffled = episodes[:]
    rng.shuffle(shuffled)
    if len(shuffled) == 1 or args.val_ratio == 0.0:
        train_eps, val_eps = shuffled, []
    else:
        val_count = max(1, int(round(len(shuffled) * args.val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
        val_eps, train_eps = shuffled[:val_count], shuffled[val_count:]

    train_eps_set = set(train_eps)
    worker_count = min(args.workers, len(episodes))
    print(f"Processing {len(episodes)} episodes with {worker_count} CPU workers", flush=True)
    with tempfile.TemporaryDirectory(prefix="ppo_openvla_shards_") as shard_root:
        jobs = [
            (
                episode_index,
                str(episode_dir),
                "train" if episode_dir in train_eps_set else "validation",
                args.num_actions_chunk,
                args.frame_skip,
                args.dataset_name,
                args.instruction,
                args.no_goal_distance,
                shard_root,
            )
            for episode_index, episode_dir in enumerate(episodes)
        ]
        process_started = time.perf_counter()
        results: List[Dict] = []
        if worker_count == 1:
            for completed, job in enumerate(jobs, start=1):
                results.append(_convert_episode_worker(job))
                _show_progress("Processing episodes", completed, len(jobs), process_started)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_convert_episode_worker, job) for job in jobs]
                for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    results.append(future.result())
                    _show_progress("Processing episodes", completed, len(futures), process_started)

        train_sample_count = _merge_shards(
            output_dir / "train_samples.jsonl", results, "train"
        )
        val_sample_count = _merge_shards(
            output_dir / "val_samples.jsonl", results, "validation"
        )

        if not train_sample_count:
            raise SystemExit(
                "No train samples generated. Collect longer episodes or reduce --num-actions-chunk."
            )
        all_actions = [result["actions"] for result in results]
        all_states = [result["states"] for result in results]

    action_array = np.concatenate(all_actions, axis=0)
    state_array = np.concatenate(all_states, axis=0)
    dataset_statistics = {
        args.dataset_name: {
            "action": _compute_stats(action_array),
            "proprio": _compute_stats(state_array),
            "num_trajectories": len(episodes),
            "num_transitions": int(action_array.shape[0]),
        }
    }
    with (output_dir / "dataset_statistics.json").open("w") as f:
        json.dump(dataset_statistics, f, indent=2)

    conversion_metadata = {
        "input_dir": str(input_dir),
        "dataset_name": args.dataset_name,
        "num_actions_chunk": args.num_actions_chunk,
        "frame_skip": args.frame_skip,
        "seed": args.seed,
        "sample_episodes": args.sample_episodes,
        "selection": "first_n_sorted_paths",
        "workers": worker_count,
        "available_episodes": available_episode_count,
        "selected_episodes": len(episodes),
        "instruction_override": args.instruction,
        "no_goal_distance": args.no_goal_distance,
        "proprio_mode": "position_velocity" if args.no_goal_distance else "position_velocity_goal_delta",
        "goal_delta_recorded": not args.no_goal_distance,
        "no_clot_only": args.no_clot_only,
        "include_failures": args.include_failures,
        "tail_action_padding": "zero",
        "all_episode_frames_preserved": args.frame_skip == 1,
        "final_episode_frame_preserved": True,
        "num_skipped_empty_episodes": len(skipped_empty),
        "num_train_episodes": len(train_eps),
        "num_val_episodes": len(val_eps),
        "num_train_samples": train_sample_count,
        "num_val_samples": val_sample_count,
    }
    with (output_dir / "conversion_metadata.json").open("w") as f:
        json.dump(conversion_metadata, f, indent=2)

    print(f"Wrote dataset to {output_dir}")
    print(
        f"episodes={len(episodes)}/{available_episode_count} train_episodes={len(train_eps)} val_episodes={len(val_eps)} "
        f"train_samples={train_sample_count} val_samples={val_sample_count}"
    )


if __name__ == "__main__":
    main()
