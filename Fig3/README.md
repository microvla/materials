# Fig. 3 source data

This directory holds the source data we used for Fig. 3 (panels a–e).

## Contents

- `learning_curves/combined_eval_success.csv`
  - Fig. 3a: `condition=standard` and `condition=no_progress`, using only `timesteps <= 10,000,000`.
  - Fig. 3b: `condition=standard`, using only `timesteps <= 50,000,000`.
  - Each condition covers 10 environment layouts. We plot the arithmetic mean
    across layouts with the between-layout sample standard deviation (`ddof=1`)
    as shading, and the table keeps the unsmoothed per-layout curves.
- `learning_curves/mixed_layout_eval_success.csv`
  - Fig. 3b: `condition=mixed_standard`, using only the 10 layouts from the paper and `timesteps <= 50,000,000`.
  - Recorded on the same evaluation suite and step grid as the single-layout curves.
- `navigation_success/open_pi_eval_success_bins_20.csv`
  - MicroVLA in Fig. 3c and MicroVLA (flow matching) in Fig. 3d. Each row is a 20-episode success-rate bin.
- `navigation_success/ppo_fused_eval_records.csv`
  - Fused PPO in Fig. 3c. Each row is a 128-episode evaluation bin.
- `navigation_success/L1_regression_eval_rollouts.csv`
  - OpenVLA-L1 in Fig. 3d. Each row is one rollout.
- `navigation_success/openvla_diffusion_eval_rollouts.csv`
  - OpenVLA-diffusion in Fig. 3d. Each row is one rollout.
- `latency/inference_latency.csv`
  - Raw latency records for Fig. 3e. We plotted the groups `openvla-oft L1`,
    `openvla-oft L1 diffusion` and `pi05`, 100 repeats each; the PPO records in
    this CSV were not drawn in Fig. 3e. The `checkpoint` column gives the
    checkpoint file name.

## Statistics as plotted

- Fig. 3a: the last included point is 9,961,472 environment steps; the final
  layout-mean success rate is 0.809765625 with the progress reward and
  0.8609375 without it.
- Fig. 3b: the last included point is 49,807,360 environment steps; the final
  layout-mean success rate is 0.996875 for single-layout training and 0.89375
  for mixed-layout training.
- Fig. 3c–d: we sorted the rollout tables by layout and episode and aggregated
  every 20 episodes into one success-rate bin; the already-binned MicroVLA table
  uses its `success_rate` column directly, and the fused-PPO table uses its
  existing 128-episode bins.
- Fig. 3e: we plot the median and interquartile range. OpenVLA-L1:
  130.371376 ms [129.7983095, 131.3314015]; OpenVLA-diffusion:
  5722.7245145 ms [5712.06686475, 5724.718442]; MicroVLA (`pi05`):
  155.7525065 ms [155.36029575, 156.1974785].
