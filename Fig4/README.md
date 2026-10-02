# Fig. 4 source data

This directory collects the CSV tables behind the plotted panels of Fig. 4.

## Panel mapping

- `panel_b_flow_robustness/`: our matched high-flow comparison at flow gain 0.1.
  The plot uses ten adjacent 20-episode bins per layout. The overall success
  rates reproduce 94.7% for MicroVLA, 54.2% for direct action mapping and 57.4%
  for fused PPO.
- `panel_c_unseen_layouts/`: source tables for layouts 11–15. Reapplying our
  plotting aggregation reproduces mean success rates of 84.9% for MicroVLA,
  46.7% for fused PPO, 41.9% for OpenVLA diffusion and 71.9% for OpenVLA L1.
- `panel_f_training_loss/`: the raw W&B action-L1-loss export behind the three
  training-loss curves, plus its compact derived summary.

Panels a and d are conceptual schematics, panel e is a representative failure
schematic, and panel g is a coverage visualization; these panels have no
underlying tabular data.
