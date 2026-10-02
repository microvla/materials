# Fig. 5 source data

This directory contains the CSV data we used for Fig. 5 of the manuscript.

## Files

- `fig5a_continuous_run_trajectory_points.csv`: tracker points from the
  four-stage continuous run (pixel coordinates and video time).
- `fig5b_selected_frame_times.csv`: the 20 time-lapse timestamps displayed in
  panel b. We transcribed them from the displayed frames.
- `fig5c_historical_neighborhood_first_actions.csv`: historical actions within
  5 mm of the matched positions.
- `fig5c_fixed_observation_first_actions.csv`: 128 first-action samples for
  each of the three prompts.
- `fig5d_all_rollouts_speed_points.csv`: 14,029 retained states from 21 physical
  rollouts, including calibrated position and speed.
- `fig5e_fixed_observation_action_chunks_long.csv`: 384 sampled action chunks ×
  8 steps = 3,072 long-form rows, converted losslessly from the logged
  action-chunk records.
- `fig5_quantitative_summary.csv`: the counts and summary statistics quoted in
  the Fig. 5 text.

## Definitions

- `speed_mm_s`: statewise instantaneous speed in mm s^-1 from calibrated
  positions and recorded timestamps.
- `stop_like_norm_le_0p1`: the L2 norm of the two-dimensional predicted action
  is <= 0.1.
- Panel-c direction percentages apply the deployment stop gate first (L2 norm
  <= 0.1 becomes stop), then use the sign of the remaining x component. The
  summary also reports the un-gated raw-sign fraction separately.
- Panel-e right/left fractions use the raw sign of the predicted x component;
  stop-like fractions are reported separately.
- Speed percentiles use linear interpolation over the pooled statewise values.
