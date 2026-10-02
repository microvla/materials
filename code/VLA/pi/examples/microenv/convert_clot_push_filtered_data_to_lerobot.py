"""Convert the strictly jump-filtered real clot-pushing dataset to LeRobot.

This keeps the validated real-MicroEnv tensor/image conversion unchanged while
replacing the old single-controller provenance with the merged-dataset contract.
Only episodes explicitly marked ``success is True`` are selected by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import tyro

import convert_real_microenv_data_to_lerobot as _base


REPO_ID = "anonymous/microenv_clot_push_filtered_20260802"


def _validate_filtered_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_counts = {
        "kept_episodes": 21,
        "kept_frames": 3008,
        "kept_success_episodes": 20,
        "kept_success_frames": 2925,
        "quarantined_episodes": 10,
        "detected_jump_transitions": 46,
    }
    if manifest.get("instruction") != "push the clot":
        raise ValueError("The filtered dataset instruction must be exactly 'push the clot'.")
    for key, expected in expected_counts.items():
        actual = manifest.get("counts", {}).get(key)
        if actual != expected:
            raise ValueError(f"Filtered manifest count mismatch for {key}: {actual} != {expected}")
    if manifest.get("filter_policy", {}).get("position_jump_threshold_mm") != 10.0:
        raise ValueError("Expected the strict 10 mm position-jump filter.")
    return manifest


def main(
    data_dir: Path,
    *,
    repo_id: str = REPO_ID,
    fps: int = 6,
    overwrite: bool = False,
    include_unconfirmed: bool = False,
    max_episodes: int | None = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    use_videos: bool = True,
    manifest_path: Path,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    source_manifest = _validate_filtered_manifest(data_dir)

    _base.main(
        data_dir,
        repo_id=repo_id,
        fps=fps,
        overwrite=overwrite,
        include_unconfirmed=include_unconfirmed,
        max_episodes=max_episodes,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        use_videos=use_videos,
        manifest_path=manifest_path,
    )

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())

    manifest.update(
        {
            "source_dataset_schema": source_manifest["schema_version"],
            "source_roles": source_manifest["counts"]["by_source_role"],
            "position_jump_filter": source_manifest["filter_policy"],
            "position_jump_statistics_before_filter": source_manifest["source_jump_statistics"],
            "training_selection_expected": {
                "episodes": 21 if include_unconfirmed else 20,
                "frames": 3008 if include_unconfirmed else 2925,
                "instruction": "push the clot",
            },
        }
    )
    manifest["pipeline_contract"].update(
        {
            "action_alignment_between_sources": "identity",
            "canonical_action_preprocessing": "deadzone_then_y_inversion_then_l2_clip_to_1",
            "hardware_mapping": (
                "source-specific controller backends retained in each episode metadata; "
                "training consumes the shared normalized 2-D [x, y] action"
            ),
            "controller_sources": {
                "legacy_upper": "collect_real_vla_dataset_v2.py:eight_direction_additive",
                "clot_real": "collect_real_vla_dataset_v3.py:continuous_xy_fixed_z",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rewrote merged-controller provenance in {manifest_path}")


if __name__ == "__main__":
    tyro.cli(main)
