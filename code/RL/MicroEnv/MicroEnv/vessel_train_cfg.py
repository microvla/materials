from pathlib import Path

from MicroEnv.MicroEnv.load_vessel_branches import (
    get_branch_layout,
    list_branch_layouts,
)

class VesselTrainConfig:
    class Train:
        total_timesteps = 100_000_000
        num_envs = 8192
        eval_num_envs = 256
        n_steps = 512
        batch_size = 65536
        n_epochs = 10
        gamma = 0.99
        gae_lambda = 0.95
        ent_coef = 0.01
        learning_rate = 3e-4
        # Torch state baseline with 7D observation:
        # [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, centerline_offset]
        # and progress-based reward shaping.
        save_dir = Path("./checkpoints/torch_state_progress")
        model_name = "ppo_vessel_torch_state_progress"
        device = "cuda:2"
        progress_bar = True
        # Evaluate once every N total environment steps.
        eval_freq = 2_000_00
        checkpoint_every_updates = 5000

    class Env:
        image_height = 64
        image_width = 64
        grayscale = False
        seed = 42
        sim_dt = 1.0 / 1000.0
        decimation = 10
        episode_seconds = 10.0
        eval_episode_seconds = 5.0 # eval steps = sec / (decimation * dt)
        step_scale = 0.1
        action_noise_std = 0.01
        drift_gain = 0.01
        inertia_decay = 0.005
        max_speed = 0.5
        wall_margin = 0.08
        agent_radius = 0.20
        # clot event
        clot_event = True
        clot_probability = 1.0
        clot_randomize_on_reset = True
        clot_region = (0.15, 0.85)
        clot_length_range = (2, 5)
        clot_max_length = 5.0
        clot_junction_margin = 1.0
        clot_min_y_slice_paths = 2
        clot_y_slice_samples = 5
        clot_y_merge_tolerance = 0.45
        clot_branch_randomize = True
        clot_radius_factor = 0.
        background = "black"
        branch_layout = "10_long_gentle_s_with_y"
        branches = get_branch_layout(branch_layout)
        start = (3.0, 10.0)
        goal = (17.0, 10.0)
        randomize_start = True
        randomize_goal = True
        min_clot_distance = 1.0
        goal_radius = 0.8
        fps_tty = 60

CONFIG = VesselTrainConfig
