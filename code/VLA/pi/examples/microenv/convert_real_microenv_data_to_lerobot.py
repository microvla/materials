"""
Convert the real MicroEnv demonstrations collected by
``collect_real_vla_dataset_v2.py`` into a local LeRobot dataset.

The converter intentionally keeps the collector contract unchanged:

* image: undistorted BGR frame -> direct 224x224 INTER_AREA resize -> RGB
* state: [x_mm, y_mm, dir_x, dir_y]
* action: current normalized controller command [x, y]
* time alignment: observation_t -> action_t (no one-step shift)

Example:
  HF_LEROBOT_HOME=/path/to/lerobot-cache \
  uv run examples/microenv/convert_real_microenv_data_to_lerobot.py \
    --data-dir datasets/microenv_real_20260721/source/vla_demo_real_2026721 \
    --repo-id anonymous/microenv_real_20260721 \
    --manifest-path datasets/microenv_real_20260721/conversion_manifest.json
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


STATE_FIELDS = ["x", "y", "dir_x", "dir_y"]
STATE_UNITS = ["mm", "mm", "unitless", "unitless"]
ACTION_FIELDS = ["x", "y"]
IMAGE_PREPROCESS = "undistorted_bgr_direct_square_resize_to_rgb"


def _discover_episode_dirs(root: Path, include_unconfirmed: bool) -> list[Path]:
    episodes: list[Path] = []
    for metadata_path in sorted(root.glob("episode_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("success") is True:
            episodes.append(metadata_path.parent)
        elif include_unconfirmed and metadata.get("success") is None:
            episodes.append(metadata_path.parent)
    return episodes


def _validate_episode(
    episode_dir: Path,
) -> tuple[dict, np.ndarray, np.ndarray, list[Path], np.ndarray]:
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    states = np.load(episode_dir / "states.npy", allow_pickle=False).astype(np.float32, copy=False)
    actions = np.load(episode_dir / "actions.npy", allow_pickle=False).astype(np.float32, copy=False)
    image_paths = sorted((episode_dir / "images").glob("*.png"))
    sample_times = np.asarray(metadata.get("sample_times_unix", []), dtype=np.float64)

    expected_steps = int(metadata.get("num_steps", -1))
    lengths = (expected_steps, len(states), len(actions), len(image_paths), len(sample_times))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Mismatched lengths in {episode_dir}: metadata/states/actions/images/times={lengths}"
        )
    if states.shape != (expected_steps, 4):
        raise ValueError(f"Expected states [{expected_steps}, 4], got {states.shape} in {episode_dir}")
    if actions.shape != (expected_steps, 2):
        raise ValueError(f"Expected actions [{expected_steps}, 2], got {actions.shape} in {episode_dir}")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"Non-finite state/action in {episode_dir}")
    if np.max(np.abs(actions), initial=0.0) > 1.0001:
        raise ValueError(f"Action outside [-1, 1] in {episode_dir}")
    if np.max(np.linalg.norm(actions, axis=1), initial=0.0) > 1.0002:
        raise ValueError(f"Action norm exceeds 1 in {episode_dir}")

    expected_names = [f"{index:06d}.png" for index in range(expected_steps)]
    if [path.name for path in image_paths] != expected_names:
        raise ValueError(f"Image sequence is not contiguous in {episode_dir}")

    metadata_contract = {
        "state_dim": 4,
        "state_fields": STATE_FIELDS,
        "state_units": STATE_UNITS,
        "action_dim": 2,
        "action_fields": ACTION_FIELDS,
        "image_size": [224, 224],
        "image_preprocess": IMAGE_PREPROCESS,
    }
    for key, expected in metadata_contract.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Metadata contract mismatch for {key} in {episode_dir}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )

    if not metadata.get("instruction"):
        raise ValueError(f"Missing instruction in {episode_dir / 'metadata.json'}")
    if len(sample_times) > 1 and np.any(np.diff(sample_times) <= 0):
        raise ValueError(f"Non-increasing sample times in {episode_dir}")

    return metadata, states, actions, image_paths, sample_times


def main(
    data_dir: Path,
    *,
    repo_id: str = "anonymous/microenv_real_20260721",
    fps: int = 6,
    overwrite: bool = False,
    include_unconfirmed: bool = False,
    max_episodes: int | None = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    use_videos: bool = True,
    manifest_path: Path | None = None,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    episodes = _discover_episode_dirs(data_dir, include_unconfirmed=include_unconfirmed)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"No eligible episodes found in {data_dir}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="microenv_real",
        fps=int(fps),
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (4,),
                "names": STATE_FIELDS,
            },
            "actions": {
                "dtype": "float32",
                "shape": (2,),
                "names": ACTION_FIELDS,
            },
        },
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        use_videos=use_videos,
    )

    kept_frames = 0
    all_sample_deltas: list[float] = []
    prompt_counts: dict[str, int] = {}
    converted_episodes: list[dict] = []
    for episode_index, episode_dir in enumerate(episodes, start=1):
        metadata, states, actions, image_paths, sample_times = _validate_episode(episode_dir)
        task = str(metadata["instruction"])
        prompt_counts[task] = prompt_counts.get(task, 0) + 1
        if len(sample_times) > 1:
            all_sample_deltas.extend(np.diff(sample_times).tolist())

        for state, action, image_path in zip(states, actions, image_paths, strict=True):
            with Image.open(image_path) as pil_image:
                if pil_image.mode != "RGB" or pil_image.size != (224, 224):
                    raise ValueError(
                        f"Expected 224x224 RGB image, got {pil_image.size} {pil_image.mode} at {image_path}"
                    )
                image = np.asarray(pil_image, dtype=np.uint8)
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
        converted_episodes.append(
            {
                "name": episode_dir.name,
                "steps": len(actions),
                "instruction": task,
                "status": metadata.get("status"),
                "success": metadata.get("success"),
            }
        )

        if episode_index == 1 or episode_index == len(episodes) or episode_index % 10 == 0:
            print(
                f"converted {episode_index}/{len(episodes)} episodes "
                f"({kept_frames} frames) into {output_path}",
                flush=True,
            )

    deltas = np.asarray(all_sample_deltas, dtype=np.float64)
    manifest = {
        "repo_id": repo_id,
        "output_path": str(output_path),
        "source_data_dir": str(data_dir),
        "selection": "success is True" if not include_unconfirmed else "success is True or None",
        "episodes": len(episodes),
        "frames": kept_frames,
        "prompt_episode_counts": prompt_counts,
        "lerobot_fps": int(fps),
        "source_timing_seconds": {
            "median": float(np.median(deltas)),
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        },
        "pipeline_contract": {
            "image": IMAGE_PREPROCESS,
            "image_shape": [224, 224, 3],
            "image_dtype": "uint8 RGB",
            "state_fields": STATE_FIELDS,
            "state_units": STATE_UNITS,
            "action_fields": ACTION_FIELDS,
            "action_semantics": "normalized continuous controller command; L2 norm <= 1",
            "observation_action_alignment": "observation_t -> action_t",
            "hardware_mapping": "collector quantize_wave_action(diagonal_ratio=0.45), FY8300 CH1/CH2",
        },
        "converted_episodes": converted_episodes,
    }
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote conversion manifest to {manifest_path}")

    print(f"Wrote LeRobot dataset to {output_path}")
    print(f"episodes={len(episodes)} frames={kept_frames} repo_id={repo_id} fps={fps}")


if __name__ == "__main__":
    tyro.cli(main)
