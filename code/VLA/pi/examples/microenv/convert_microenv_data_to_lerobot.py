"""
Convert MicroEnv PPO demos to a local LeRobot dataset for OpenPI fine-tuning.

Example:
  HF_LEROBOT_HOME=/path/to/lerobot-cache \
  uv run examples/microenv/convert_microenv_data_to_lerobot.py \
    --data-dir /path/to/vla_demos_ppo_no_distance_2000 \
    --repo-id anonymous/microenv_ppo_no_distance_2000 \
    --overwrite
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro


def _discover_episode_dirs(root: Path, include_failures: bool) -> list[Path]:
    meta_paths = set()
    for pattern in (
        "episode_*/metadata.json",
        "*/episode_*/metadata.json",
        "*/*/episode_*/metadata.json",
    ):
        meta_paths.update(root.glob(pattern))

    episodes: list[Path] = []
    for meta_path in sorted(meta_paths):
        episode_dir = meta_path.parent
        if any(part.startswith(".") for part in episode_dir.relative_to(root).parts):
            continue
        metadata = json.loads(meta_path.read_text())
        if include_failures or bool(metadata.get("success")):
            episodes.append(episode_dir)
    return episodes


def _prepare_states(states: np.ndarray, no_goal_distance: bool) -> np.ndarray:
    states = states.astype(np.float32, copy=False)
    if no_goal_distance:
        if states.ndim != 2 or states.shape[1] < 6:
            raise ValueError(f"--no-goal-distance expects states with at least 6 columns: {states.shape}")
        states = states.copy()
        states[:, 4:6] = 0.0
    return states


def main(
    data_dir: Path,
    *,
    repo_id: str = "anonymous/microenv_ppo_no_distance_2000",
    fps: int = 10,
    overwrite: bool = False,
    include_failures: bool = False,
    max_episodes: int | None = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    use_videos: bool = True,
    no_goal_distance: bool = False,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    episodes = _discover_episode_dirs(data_dir, include_failures=include_failures)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"No episodes found in {data_dir}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="microenv",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["actions"],
            },
        },
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        use_videos=use_videos,
    )

    kept_frames = 0
    for episode_index, episode_dir in enumerate(episodes, start=1):
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        states = _prepare_states(np.load(episode_dir / "states.npy"), no_goal_distance)
        actions = np.load(episode_dir / "actions.npy").astype(np.float32)
        image_paths = sorted((episode_dir / "images").glob("*.png"))

        if not (len(states) == len(actions) == len(image_paths)):
            raise ValueError(
                f"Mismatched lengths in {episode_dir}: "
                f"states={len(states)} actions={len(actions)} images={len(image_paths)}"
            )

        task = metadata.get("instruction")
        if not task:
            raise ValueError(f"Missing instruction in {episode_dir / 'metadata.json'}")

        for state, action, image_path in zip(states, actions, image_paths, strict=True):
            image = np.asarray(Image.open(image_path).convert("RGB"))
            dataset.add_frame(
                {
                    "image": image,
                    "state": state,
                    "actions": action,
                    "task": task,
                }
            )
            kept_frames += 1
        dataset.save_episode()

        if episode_index == 1 or episode_index == len(episodes) or episode_index % 25 == 0:
            print(
                f"converted {episode_index}/{len(episodes)} episodes "
                f"({kept_frames} frames) into {output_path}",
                flush=True,
            )

    print(f"Wrote LeRobot dataset to {output_path}")
    print(f"episodes={len(episodes)} frames={kept_frames} repo_id={repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
