from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.shared import normalize
from openpi.training import config as cfg


def main() -> None:
    input_dir = Path("/path/to/vla_demos_ppo_no_distance_2000")
    config = cfg.get_config("pi05_microenv_lora")
    data_config = config.data.create(config.assets_dirs, config.model)
    output_dir = config.assets_dirs / data_config.repo_id

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    episode_count = 0
    frame_count = 0

    for meta_path in sorted(input_dir.glob("*/*/episode_*/metadata.json")):
        metadata = json.loads(meta_path.read_text())
        if not metadata.get("success"):
            continue
        episode_dir = meta_path.parent
        states = np.load(episode_dir / "states.npy").astype(np.float32)
        actions = np.load(episode_dir / "actions.npy").astype(np.float32)
        if len(states) != len(actions):
            raise ValueError(f"Length mismatch in {episode_dir}: states={len(states)} actions={len(actions)}")

        state_stats.update(states)
        action_stats.update(actions)
        episode_count += 1
        frame_count += len(actions)

    normalize.save(
        output_dir,
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
    )
    print(f"Wrote stats to: {output_dir}")
    print(f"episodes={episode_count} frames={frame_count}")


if __name__ == "__main__":
    main()
