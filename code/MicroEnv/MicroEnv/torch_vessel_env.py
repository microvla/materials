from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from MicroEnv.MicroEnv.vessel_train_cfg import VesselTrainConfig


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def resolve_torch_device(device: str | torch.device | None, log=print) -> torch.device:
    if isinstance(device, torch.device):
        requested = str(device)
    elif device is None:
        requested = "auto"
    else:
        requested = str(device).strip() or "auto"

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if log is not None:
            log("[device] CUDA is not available; using CPU.")
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            if log is not None:
                log(f"[device] Requested '{requested}' but CUDA is not available; using CPU.")
            return torch.device("cpu")
        if ":" in normalized:
            try:
                index = int(normalized.split(":", 1)[1])
            except ValueError:
                index = None
            if index is not None and not (0 <= index < torch.cuda.device_count()):
                if log is not None:
                    log(
                        f"[device] Requested '{requested}' but only {torch.cuda.device_count()} CUDA device(s) "
                        "are visible; using CPU."
                    )
                return torch.device("cpu")
        return torch.device(requested)

    return torch.device(requested)


@dataclass
class TorchLumenInfo:
    closest: torch.Tensor
    distance: torch.Tensor
    radius: torch.Tensor
    navigable_radius: torch.Tensor
    clearance: torch.Tensor
    tangent: torch.Tensor
    branch_index: torch.Tensor
    segment_index: torch.Tensor
    segment_t: torch.Tensor
    flow: torch.Tensor


class TorchVesselBatchEnv:
    """
    Minimal batched Torch simulator for state-based vessel navigation.

    This class is intentionally not a full gym.Env replacement yet. It focuses on:
    - batched state tensors
    - batched reset/step
    - clot/start-goal randomization
    - state observations only
    """

    OBS_DIM = 8

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int = VesselTrainConfig.Env.seed,
        episode_seconds: float = VesselTrainConfig.Env.episode_seconds,
        sim_dt: float = VesselTrainConfig.Env.sim_dt,
        decimation: int = VesselTrainConfig.Env.decimation,
        step_scale: float = VesselTrainConfig.Env.step_scale,
        action_noise_std: float = VesselTrainConfig.Env.action_noise_std,
        drift_gain: float = VesselTrainConfig.Env.drift_gain,
        inertia_decay: float = VesselTrainConfig.Env.inertia_decay,
        max_speed: float = VesselTrainConfig.Env.max_speed,
        wall_margin: float = VesselTrainConfig.Env.wall_margin,
        agent_radius: float = VesselTrainConfig.Env.agent_radius,
        clot_event: bool = VesselTrainConfig.Env.clot_event,
        clot_probability: float = VesselTrainConfig.Env.clot_probability,
        clot_randomize_on_reset: bool = VesselTrainConfig.Env.clot_randomize_on_reset,
        clot_region: Tuple[float, float] = VesselTrainConfig.Env.clot_region,
        clot_length_range: Tuple[int, int] = VesselTrainConfig.Env.clot_length_range,
        clot_max_length: float = VesselTrainConfig.Env.clot_max_length,
        clot_junction_margin: float = VesselTrainConfig.Env.clot_junction_margin,
        clot_min_y_slice_paths: int = VesselTrainConfig.Env.clot_min_y_slice_paths,
        clot_y_slice_samples: int = VesselTrainConfig.Env.clot_y_slice_samples,
        clot_y_merge_tolerance: float = VesselTrainConfig.Env.clot_y_merge_tolerance,
        clot_branch_randomize: bool = VesselTrainConfig.Env.clot_branch_randomize,
        clot_radius_factor: float = VesselTrainConfig.Env.clot_radius_factor,
        clot_fixed_branch_idx: Optional[int] = None,
        clot_fixed_start_idx: Optional[int] = None,
        clot_fixed_end_idx: Optional[int] = None,
        branches: Optional[List[Dict]] = VesselTrainConfig.Env.branches,
        start: Tuple[float, float] = VesselTrainConfig.Env.start,
        goal: Tuple[float, float] = VesselTrainConfig.Env.goal,
        randomize_start: bool = VesselTrainConfig.Env.randomize_start,
        randomize_goal: bool = VesselTrainConfig.Env.randomize_goal,
        min_clot_distance: float = VesselTrainConfig.Env.min_clot_distance,
        goal_radius: float = VesselTrainConfig.Env.goal_radius,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = resolve_torch_device(device)
        self.seed = int(seed)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

        self.sim_dt = max(1e-6, float(sim_dt))
        self.decimation = max(1, int(decimation))
        self.control_dt = self.sim_dt * self.decimation
        self.episode_seconds = max(self.control_dt, float(episode_seconds))
        self.max_steps = max(1, int(round(self.episode_seconds / self.control_dt)))

        self.step_scale = float(step_scale)
        self.action_noise_std = max(0.0, float(action_noise_std))
        self.drift_gain = float(drift_gain)
        self.inertia_decay = float(inertia_decay)
        self.max_speed = float(max_speed)
        self.wall_margin = float(wall_margin)
        self.agent_radius = float(agent_radius)
        self.goal_radius = float(goal_radius)

        self.command_speed = self.step_scale / self.control_dt
        self.flow_speed = self.drift_gain / self.control_dt
        self.inertia_gain = self.inertia_decay / self.control_dt
        self.max_speed_per_second = self.max_speed / self.control_dt
        self.reward_rate_scale = 1.0 / self.control_dt

        self.clot_event = bool(clot_event)
        self.clot_probability = float(clot_probability)
        self.clot_randomize_on_reset = bool(clot_randomize_on_reset)
        self.clot_region = clot_region
        self.clot_length_range = clot_length_range
        self.clot_max_length = max(0.0, float(clot_max_length))
        self.clot_junction_margin = max(0.0, float(clot_junction_margin))
        self.clot_min_y_slice_paths = max(1, int(clot_min_y_slice_paths))
        self.clot_y_slice_samples = max(1, int(clot_y_slice_samples))
        self.clot_y_merge_tolerance = max(1e-6, float(clot_y_merge_tolerance))
        self.clot_branch_randomize = bool(clot_branch_randomize)
        self.clot_radius_factor = _clamp(float(clot_radius_factor), 0.0, 1.0)
        fixed_values = (clot_fixed_branch_idx, clot_fixed_start_idx, clot_fixed_end_idx)
        if all(value is None for value in fixed_values):
            self.clot_fixed_span: Optional[Tuple[int, int, int]] = None
        elif any(value is None for value in fixed_values):
            raise ValueError(
                "Fixed clot position requires clot_fixed_branch_idx, "
                "clot_fixed_start_idx, and clot_fixed_end_idx."
            )
        else:
            self.clot_fixed_span = (
                int(clot_fixed_branch_idx),
                int(clot_fixed_start_idx),
                int(clot_fixed_end_idx),
            )

        self.default_start = torch.tensor(start, dtype=torch.float32, device=self.device)
        self.default_goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        self.randomize_start = bool(randomize_start)
        self.randomize_goal = bool(randomize_goal)
        self.min_clot_distance = max(0.0, float(min_clot_distance))

        self.bounds = torch.tensor([0.0, 20.0, 0.0, 20.0], dtype=torch.float32, device=self.device)
        self._build_geometry(branches or VesselTrainConfig.Env.branches)
        if self.clot_fixed_span is not None:
            self._validate_fixed_clot_span()
        self._allocate_buffers()
        self.reset()

    def _build_geometry(self, branches: List[Dict]) -> None:
        branch_names: List[str] = []
        seg_a = []
        seg_b = []
        seg_r0 = []
        seg_r1 = []
        seg_branch_idx = []
        seg_idx_within_branch = []
        seg_flow = []

        self.branch_segment_ranges: List[Tuple[int, int]] = []
        branch_points: List[torch.Tensor] = []
        branch_radii: List[torch.Tensor] = []
        branch_seg_lengths: List[torch.Tensor] = []
        branch_cum_lengths: List[torch.Tensor] = []
        branch_total_lengths: List[float] = []
        branch_clot_allowed: List[bool] = []

        offset = 0
        for branch_idx, branch in enumerate(branches):
            branch_names.append(str(branch["name"]))
            branch_clot_allowed.append(bool(branch.get("clot_allowed", True)))
            pts = torch.as_tensor(branch["points"], dtype=torch.float32, device=self.device)
            radii = torch.as_tensor(branch["radii"], dtype=torch.float32, device=self.device)
            flow = torch.as_tensor(branch.get("flow", [1.0, 0.0]), dtype=torch.float32, device=self.device)
            branch_points.append(pts)
            branch_radii.append(radii)

            if pts.shape[0] < 2:
                branch_seg_lengths.append(torch.zeros(0, dtype=torch.float32, device=self.device))
                branch_cum_lengths.append(torch.zeros(1, dtype=torch.float32, device=self.device))
                branch_total_lengths.append(0.0)
                self.branch_segment_ranges.append((offset, offset))
                continue

            seg_lengths = torch.linalg.norm(pts[1:] - pts[:-1], dim=1)
            cum_lengths = torch.cat(
                [torch.zeros(1, dtype=torch.float32, device=self.device), torch.cumsum(seg_lengths, dim=0)],
                dim=0,
            )
            branch_seg_lengths.append(seg_lengths)
            branch_cum_lengths.append(cum_lengths)
            branch_total_lengths.append(float(cum_lengths[-1].item()))

            for i in range(pts.shape[0] - 1):
                seg_a.append(pts[i])
                seg_b.append(pts[i + 1])
                seg_r0.append(radii[i])
                seg_r1.append(radii[i + 1])
                seg_branch_idx.append(branch_idx)
                seg_idx_within_branch.append(i)
                seg_flow.append(flow)
            next_offset = offset + (pts.shape[0] - 1)
            self.branch_segment_ranges.append((offset, next_offset))
            offset = next_offset

        self.branch_names = branch_names
        self.branch_points = branch_points
        self.branch_radii = branch_radii
        self.branch_seg_lengths = branch_seg_lengths
        self.branch_cum_lengths = branch_cum_lengths
        self.branch_total_lengths = torch.tensor(branch_total_lengths, dtype=torch.float32, device=self.device)
        self.branch_clot_allowed = branch_clot_allowed
        self.clot_branch_choices = [idx for idx, allowed in enumerate(branch_clot_allowed) if allowed]
        if not self.clot_branch_choices:
            self.clot_branch_choices = list(range(len(branch_names)))
        self.branch_junction_distances = self._build_branch_junction_distances()

        self.seg_a = torch.stack(seg_a, dim=0)
        self.seg_b = torch.stack(seg_b, dim=0)
        self.seg_r0 = torch.stack(seg_r0, dim=0)
        self.seg_r1 = torch.stack(seg_r1, dim=0)
        self.seg_branch_idx = torch.tensor(seg_branch_idx, dtype=torch.long, device=self.device)
        self.seg_idx_within_branch = torch.tensor(seg_idx_within_branch, dtype=torch.long, device=self.device)
        self.seg_flow = torch.stack(seg_flow, dim=0)
        self.seg_ab = self.seg_b - self.seg_a
        self.seg_ab_len_sq = torch.sum(self.seg_ab * self.seg_ab, dim=1).clamp_min(1e-8)
        self.seg_tangent = self.seg_ab / torch.sqrt(self.seg_ab_len_sq).unsqueeze(1)
        self.seg_lengths = torch.sqrt(self.seg_ab_len_sq)
        self.total_vessel_length = float(self.seg_lengths.sum().item())

    def _build_branch_junction_distances(self) -> List[List[float]]:
        threshold = max(0.15, self.agent_radius * 1.25)
        junctions_by_branch: List[List[float]] = []
        for branch_idx, pts in enumerate(self.branch_points):
            cum_lengths = self.branch_cum_lengths[branch_idx]
            branch_junctions = [0.0, float(cum_lengths[-1].item())]
            for point_idx in range(pts.shape[0]):
                point = pts[point_idx]
                distance_along_branch = float(cum_lengths[point_idx].item())
                found = False
                for other_idx, other_pts in enumerate(self.branch_points):
                    if other_idx == branch_idx:
                        continue
                    for seg_idx in range(max(0, other_pts.shape[0] - 1)):
                        distance = self._point_to_segment_distance(point, other_pts[seg_idx], other_pts[seg_idx + 1])
                        if distance <= threshold:
                            branch_junctions.append(distance_along_branch)
                            found = True
                            break
                    if found:
                        break
            junctions_by_branch.append(sorted(set(round(value, 6) for value in branch_junctions)))
        return junctions_by_branch

    @staticmethod
    def _point_to_segment_distance(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
        ab = b - a
        # Avoid torch.dot here: on CUDA it initializes cuBLAS, which is brittle
        # during multi-process geometry setup and unnecessary for 2D vectors.
        ab_len_sq = float(torch.sum(ab * ab).item())
        if ab_len_sq <= 1e-8:
            return float(torch.linalg.norm(point - a).item())
        t = float(torch.sum((point - a) * ab).item()) / ab_len_sq
        t = _clamp(t, 0.0, 1.0)
        closest = a + t * ab
        return float(torch.linalg.norm(point - closest).item())

    def _clot_span_near_junction(self, branch_idx: int, span_start: float, span_end: float) -> bool:
        if self.clot_junction_margin <= 0.0:
            return False
        for junction in self.branch_junction_distances[branch_idx]:
            if span_start <= junction + self.clot_junction_margin and span_end >= junction - self.clot_junction_margin:
                return True
        return False

    def _clot_span_on_single_y_slice(self, branch_idx: int, start_idx: int, end_idx: int) -> bool:
        if self.clot_min_y_slice_paths <= 1:
            return False
        pts = self.branch_points[branch_idx].detach().cpu().numpy()
        span_pts = pts[start_idx : end_idx + 2]
        x_min = float(np.min(span_pts[:, 0]))
        x_max = float(np.max(span_pts[:, 0]))
        if abs(x_max - x_min) <= 1e-6:
            x_samples = [0.5 * (x_min + x_max)]
        else:
            x_samples = np.linspace(x_min, x_max, self.clot_y_slice_samples + 2)[1:-1]
        return any(self._path_count_at_x(float(x)) < self.clot_min_y_slice_paths for x in x_samples)

    def _path_count_at_x(self, x: float) -> int:
        ys = []
        eps = 1e-6
        for pts_tensor in self.branch_points:
            pts = pts_tensor.detach().cpu().numpy()
            for idx in range(len(pts) - 1):
                ax, ay = float(pts[idx][0]), float(pts[idx][1])
                bx, by = float(pts[idx + 1][0]), float(pts[idx + 1][1])
                if x < min(ax, bx) - eps or x > max(ax, bx) + eps:
                    continue
                dx = bx - ax
                if abs(dx) <= eps:
                    if abs(x - ax) <= eps:
                        ys.append(0.5 * (ay + by))
                    continue
                t = (x - ax) / dx
                if -eps <= t <= 1.0 + eps:
                    ys.append(ay + t * (by - ay))

        unique_ys = []
        for y in sorted(ys):
            if not unique_ys or abs(y - unique_ys[-1]) > self.clot_y_merge_tolerance:
                unique_ys.append(y)
        return len(unique_ys)

    def _validate_fixed_clot_span(self) -> None:
        if self.clot_fixed_span is None:
            return
        branch_idx, start_idx, end_idx = self.clot_fixed_span
        if not (0 <= branch_idx < len(self.branch_points)):
            raise ValueError(
                f"Invalid fixed clot branch {branch_idx}; expected 0..{len(self.branch_points) - 1}."
            )
        segment_count = max(0, self.branch_points[branch_idx].shape[0] - 1)
        if segment_count < 1:
            raise ValueError(f"Fixed clot branch {branch_idx} has no segments.")
        if not (0 <= start_idx <= end_idx < segment_count):
            raise ValueError(
                f"Invalid fixed clot span ({start_idx}, {end_idx}) for branch {branch_idx}; "
                f"expected 0 <= start_idx <= end_idx < {segment_count}."
            )

    def _allocate_buffers(self) -> None:
        n = self.num_envs
        self.pos = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.velocity = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.start = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.goal = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.distance_to_goal = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.clot_on = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.clot_branch_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clot_start_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clot_end_idx = torch.zeros(n, dtype=torch.long, device=self.device)

    def _rand(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self.rng, dtype=torch.float32, device=self.device)

    def _randint(self, low: int, high: int, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self.rng, dtype=torch.int64, device=self.device)

    def _sample_points_in_vessel(self, count: int) -> torch.Tensor:
        if self.total_vessel_length <= 1e-8:
            return self.default_start.unsqueeze(0).repeat(count, 1)

        seg_probs = self.seg_lengths / self.seg_lengths.sum()
        seg_ids = torch.multinomial(seg_probs, count, replacement=True, generator=self.rng).to(self.device)
        t = self._rand((count, 1))
        return self.seg_a[seg_ids] + t * self.seg_ab[seg_ids]

    @staticmethod
    def _distance_to_segments(points: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
        ab = seg_b - seg_a
        ab_len_sq = torch.sum(ab * ab, dim=1).clamp_min(1e-8)
        ap = points.unsqueeze(1) - seg_a.unsqueeze(0)
        t = torch.sum(ap * ab.unsqueeze(0), dim=2) / ab_len_sq.unsqueeze(0)
        t = torch.clamp(t, 0.0, 1.0)
        closest = seg_a.unsqueeze(0) + t.unsqueeze(2) * ab.unsqueeze(0)
        return torch.linalg.norm(points.unsqueeze(1) - closest, dim=2)

    def _distance_to_clot(self, points: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        distances = torch.full((env_ids.shape[0],), float("inf"), dtype=torch.float32, device=self.device)
        for branch_idx in range(len(self.branch_names)):
            mask = self.clot_on[env_ids] & (self.clot_branch_idx[env_ids] == branch_idx)
            if not torch.any(mask):
                continue
            local_ids = env_ids[mask]
            pts = self.branch_points[branch_idx]
            for row, env_id in enumerate(local_ids):
                start_idx = int(self.clot_start_idx[env_id].item())
                end_idx = int(self.clot_end_idx[env_id].item())
                seg_a = pts[start_idx : end_idx + 1]
                seg_b = pts[start_idx + 1 : end_idx + 2]
                dist = self._distance_to_segments(points[mask][row : row + 1], seg_a, seg_b).min()
                distances[mask.nonzero(as_tuple=False)[row, 0]] = dist
        return distances

    def _sample_points_away_from_clot(self, env_ids: torch.Tensor) -> torch.Tensor:
        points = self._sample_points_in_vessel(env_ids.shape[0])
        if self.min_clot_distance <= 0.0:
            return points
        for _ in range(128):
            bad = self._points_blocked_by_clot(points, env_ids)
            if not torch.any(bad):
                break
            points[bad] = self._sample_points_in_vessel(int(bad.sum().item()))
        return points

    def _points_blocked_by_clot(self, points: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        if self.min_clot_distance <= 0.0:
            return torch.zeros((env_ids.shape[0],), dtype=torch.bool, device=self.device)
        active = self.clot_on[env_ids]
        return active & (self._distance_to_clot(points, env_ids) < self.min_clot_distance)

    def _repair_points_away_from_clot(self, points: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        if self.min_clot_distance <= 0.0:
            return points
        repaired = points.clone()
        for _ in range(8):
            bad = self._points_blocked_by_clot(repaired, env_ids)
            if not torch.any(bad):
                return repaired
            bad_ids = env_ids[bad]
            repaired[bad] = self._sample_points_away_from_clot(bad_ids)

        bad = self._points_blocked_by_clot(repaired, env_ids)
        if torch.any(bad):
            # Endpoint validity has priority over keeping a clot in rare over-constrained resets.
            self.clot_on[env_ids[bad]] = False
        return repaired

    def _randomize_clot_for_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self.clot_fixed_span is not None:
            branch_idx, start_idx, end_idx = self.clot_fixed_span
            self.clot_branch_idx[env_ids] = branch_idx
            self.clot_start_idx[env_ids] = start_idx
            self.clot_end_idx[env_ids] = end_idx
            return

        def _branch_order() -> List[int]:
            choices = list(self.clot_branch_choices)
            if self.clot_branch_randomize:
                perm = torch.randperm(len(choices), generator=self.rng, device=self.device).tolist()
                return [choices[int(idx)] for idx in perm]
            preferred = 0 if 0 in choices else choices[0]
            return [preferred] + [idx for idx in choices if idx != preferred]

        def _candidate_spans(branch_idx: int, restrict_to_region: bool) -> List[Tuple[int, int]]:
            seg_lengths = self.branch_seg_lengths[branch_idx]
            if seg_lengths.numel() == 0:
                return []
            cum_lengths = self.branch_cum_lengths[branch_idx]
            total_len = float(cum_lengths[-1].item())
            if restrict_to_region:
                region_start = max(0.0, min(1.0, float(self.clot_region[0]))) * total_len
                region_end = max(0.0, min(1.0, float(self.clot_region[1]))) * total_len
                if region_end < region_start:
                    region_start, region_end = region_end, region_start
                candidate_starts = [
                    i for i in range(seg_lengths.shape[0])
                    if float(cum_lengths[i].item()) >= region_start and float(cum_lengths[i].item()) < region_end
                ]
            else:
                region_start = 0.0
                region_end = total_len
                candidate_starts = list(range(seg_lengths.shape[0]))
            if not candidate_starts:
                return []

            min_len = max(1, int(self.clot_length_range[0]))
            max_len_target = max(min_len, int(self.clot_length_range[1]))
            spans: List[Tuple[int, int]] = []
            for start_idx in candidate_starts:
                for length in range(min_len, max_len_target + 1):
                    end_idx = min(start_idx + length - 1, seg_lengths.shape[0] - 1)
                    span_start = float(cum_lengths[start_idx].item())
                    span_end = float(cum_lengths[end_idx + 1].item())
                    if span_end > region_end + 1e-8:
                        continue
                    span_length = span_end - span_start
                    if self.clot_max_length > 0.0 and span_length > self.clot_max_length + 1e-8:
                        continue
                    if self._clot_span_on_single_y_slice(branch_idx, start_idx, end_idx):
                        continue
                    if self._clot_span_near_junction(branch_idx, span_start, span_end):
                        continue
                    spans.append((start_idx, end_idx))
            return spans

        for row, env_id in enumerate(env_ids.tolist()):
            chosen = None
            order = _branch_order()
            for branch_idx in order:
                spans = _candidate_spans(branch_idx, restrict_to_region=True)
                if spans:
                    span_idx = int(self._randint(0, len(spans), (1,)).item())
                    chosen = (branch_idx, *spans[span_idx])
                    break
            if chosen is None:
                for branch_idx in order:
                    spans = _candidate_spans(branch_idx, restrict_to_region=False)
                    if spans:
                        span_idx = int(self._randint(0, len(spans), (1,)).item())
                        chosen = (branch_idx, *spans[span_idx])
                        break
            if chosen is None:
                self.clot_on[env_id] = False
                self.clot_branch_idx[env_id] = 0
                self.clot_start_idx[env_id] = 0
                self.clot_end_idx[env_id] = 0
                continue
            branch_idx, start_idx, end_idx = chosen
            self.clot_branch_idx[env_id] = branch_idx
            self.clot_start_idx[env_id] = start_idx
            self.clot_end_idx[env_id] = end_idx

    def _resolve_endpoints(self, env_ids: torch.Tensor) -> None:
        self.start[env_ids] = self.default_start
        self.goal[env_ids] = self.default_goal
        if self.randomize_start:
            self.start[env_ids] = self._sample_points_away_from_clot(env_ids)
        self.start[env_ids] = self._repair_points_away_from_clot(self.start[env_ids], env_ids)
        if self.randomize_goal:
            self.goal[env_ids] = self._sample_points_away_from_clot(env_ids)
            min_goal_dist = max(self.goal_radius * 2.0, 1.0)
            for _ in range(16):
                d = torch.linalg.norm(self.goal[env_ids] - self.start[env_ids], dim=1)
                bad = d <= min_goal_dist
                if not torch.any(bad):
                    break
                bad_ids = env_ids[bad]
                self.goal[bad_ids] = self._sample_points_away_from_clot(bad_ids)
        self.goal[env_ids] = self._repair_points_away_from_clot(self.goal[env_ids], env_ids)

    def _effective_radius(self, env_ids: torch.Tensor) -> torch.Tensor:
        radius_scale = torch.ones((env_ids.shape[0], self.seg_a.shape[0]), dtype=torch.float32, device=self.device)
        if self.clot_radius_factor < 1.0:
            clot_mask = (
                self.clot_on[env_ids].unsqueeze(1)
                & (self.seg_branch_idx.unsqueeze(0) == self.clot_branch_idx[env_ids].unsqueeze(1))
                & (self.seg_idx_within_branch.unsqueeze(0) >= self.clot_start_idx[env_ids].unsqueeze(1))
                & (self.seg_idx_within_branch.unsqueeze(0) <= self.clot_end_idx[env_ids].unsqueeze(1))
            )
            radius_scale = torch.where(clot_mask, torch.full_like(radius_scale, self.clot_radius_factor), radius_scale)
        return radius_scale

    def _nearest_lumen_info(self, points: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> TorchLumenInfo:
        if env_ids is None:
            if points.shape[0] != self.num_envs:
                raise ValueError("env_ids must be provided when querying a subset of environments.")
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        ap = points.unsqueeze(1) - self.seg_a.unsqueeze(0)
        t = torch.sum(ap * self.seg_ab.unsqueeze(0), dim=2) / self.seg_ab_len_sq.unsqueeze(0)
        t = torch.clamp(t, 0.0, 1.0)
        closest = self.seg_a.unsqueeze(0) + t.unsqueeze(2) * self.seg_ab.unsqueeze(0)
        base_radius = (1.0 - t) * self.seg_r0.unsqueeze(0) + t * self.seg_r1.unsqueeze(0)
        eff_scale = self._effective_radius(env_ids)
        radius = base_radius * eff_scale
        navigable_radius = torch.clamp(radius - self.agent_radius, min=0.0)
        offset = points.unsqueeze(1) - closest
        distance = torch.linalg.norm(offset, dim=2)
        clearance = navigable_radius - distance

        # Choose the geometrically nearest centerline first, then break ties by clearance.
        # This prevents points on a blocked branch from "borrowing" clearance from a nearby
        # but disconnected branch segment with a larger radius.
        min_distance = torch.min(distance, dim=1, keepdim=True).values
        distance_tolerance = 1e-4
        near_mask = distance <= (min_distance + distance_tolerance)
        masked_clearance = torch.where(near_mask, clearance, torch.full_like(clearance, float("-inf")))
        best_idx = torch.argmax(masked_clearance, dim=1)
        batch_idx = torch.arange(points.shape[0], device=self.device)
        return TorchLumenInfo(
            closest=closest[batch_idx, best_idx],
            distance=distance[batch_idx, best_idx],
            radius=radius[batch_idx, best_idx],
            navigable_radius=navigable_radius[batch_idx, best_idx],
            clearance=clearance[batch_idx, best_idx],
            tangent=self.seg_tangent[best_idx],
            branch_index=self.seg_branch_idx[best_idx],
            segment_index=best_idx,
            segment_t=t[batch_idx, best_idx],
            flow=self.seg_flow[best_idx],
        )

    def _project_to_safe_lumen(
        self,
        points: torch.Tensor,
        margin: float,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TorchLumenInfo]:
        info = self._nearest_lumen_info(points, env_ids)
        safe_radius = torch.clamp(info.navigable_radius - margin, min=0.0)
        offset = points - info.closest
        distance = torch.linalg.norm(offset, dim=1)
        projected = points.clone()

        near_zero = distance <= 1e-8
        if torch.any(near_zero):
            normal = torch.stack([-info.tangent[:, 1], info.tangent[:, 0]], dim=1)
            normal_norm = torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-8)
            normal = normal / normal_norm
            projected[near_zero] = info.closest[near_zero] + normal[near_zero] * torch.minimum(
                safe_radius[near_zero].unsqueeze(1),
                torch.full((int(near_zero.sum().item()), 1), 0.2, device=self.device),
            )

        needs_project = (~near_zero) & (distance > safe_radius)
        if torch.any(needs_project):
            projected[needs_project] = info.closest[needs_project] + (
                offset[needs_project] / distance[needs_project].unsqueeze(1).clamp_min(1e-8)
            ) * safe_radius[needs_project].unsqueeze(1)

        projected_info = self._nearest_lumen_info(projected, env_ids)
        return projected, projected_info

    def _path_is_safe(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        margin: float,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TorchLumenInfo]:
        if env_ids is None:
            if start.shape[0] != self.num_envs:
                raise ValueError("env_ids must be provided when checking a subset of environments.")
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        delta = end - start
        distance = torch.linalg.norm(delta, dim=1)
        sample_counts = torch.clamp(torch.ceil(distance / 0.05).to(torch.long), min=2)
        max_samples = int(sample_counts.max().item())
        last_info = self._nearest_lumen_info(start, env_ids)
        effective_margin = torch.minimum(
            torch.full_like(last_info.clearance, float(margin)),
            torch.clamp(last_info.clearance, min=0.0) + 1e-4,
        )
        safe = torch.ones(start.shape[0], dtype=torch.bool, device=self.device)

        for i in range(1, max_samples + 1):
            active = sample_counts >= i
            if not torch.any(active):
                break
            alpha = (torch.full_like(distance, float(i)) / sample_counts.to(torch.float32)).unsqueeze(1)
            probe = start + alpha * delta
            info = self._nearest_lumen_info(probe, env_ids)
            last_info = info
            active_bad = active & (info.clearance < effective_margin)
            safe = safe & (~active_bad)
        return safe, last_info

    def _compute_obs(self) -> torch.Tensor:
        info = self._nearest_lumen_info(self.pos)
        obs = torch.cat(
            [
                self.pos,
                self.velocity,
                self.goal - self.pos,
                info.distance.unsqueeze(1),
                self.clot_on.to(dtype=torch.float32).unsqueeze(1),
            ],
            dim=1,
        )
        return obs

    def compute_vla_state(self, eps: float = 1e-6) -> torch.Tensor:
        """Return compact VLA proprio state: x, y, unit motion direction x/y."""
        speed = torch.linalg.norm(self.velocity, dim=1, keepdim=True)
        direction = torch.where(speed > eps, self.velocity / speed.clamp_min(eps), torch.zeros_like(self.velocity))
        return torch.cat([self.pos, direction], dim=1)

    def _world_to_pixel_np(self, point: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        height, width = int(image_size[0]), int(image_size[1])
        bounds = self.bounds.detach().cpu().numpy()
        x_min, x_max, y_min, y_max = [float(v) for v in bounds]
        x = (float(point[0]) - x_min) / max(x_max - x_min, 1e-8)
        y = (float(point[1]) - y_min) / max(y_max - y_min, 1e-8)
        return np.array(
            [
                np.clip(x * (width - 1), 0, width - 1),
                np.clip((1.0 - y) * (height - 1), 0, height - 1),
            ],
            dtype=np.float32,
        )

    def _radius_to_pixels_np(self, radius: float, image_size: Tuple[int, int]) -> float:
        height, width = int(image_size[0]), int(image_size[1])
        bounds = self.bounds.detach().cpu().numpy()
        world_w = max(float(bounds[1] - bounds[0]), 1e-8)
        world_h = max(float(bounds[3] - bounds[2]), 1e-8)
        return float(radius) * min((width - 1) / world_w, (height - 1) / world_h)

    @staticmethod
    def _paint_disk_np(frame: np.ndarray, center: np.ndarray, radius: float, color: Tuple[int, int, int]) -> None:
        height, width = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        r = max(float(radius), 0.5)
        x0 = max(int(np.floor(cx - r)), 0)
        x1 = min(int(np.ceil(cx + r)) + 1, width)
        y0 = max(int(np.floor(cy - r)), 0)
        y1 = min(int(np.ceil(cy + r)) + 1, height)
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        frame[y0:y1, x0:x1][mask] = color

    @staticmethod
    def _paint_capsule_np(
        frame: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: Tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        r = max(float(radius), 0.5)
        x0 = max(int(np.floor(min(ax, bx) - r)), 0)
        x1 = min(int(np.ceil(max(ax, bx) + r)) + 1, width)
        y0 = max(int(np.floor(min(ay, by) - r)), 0)
        y1 = min(int(np.ceil(max(ay, by) + r)) + 1, height)
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        abx, aby = bx - ax, by - ay
        ab_len_sq = max(abx * abx + aby * aby, 1e-8)
        t = ((xx - ax) * abx + (yy - ay) * aby) / ab_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        mask = (xx - closest_x) ** 2 + (yy - closest_y) ** 2 <= r ** 2
        frame[y0:y1, x0:x1][mask] = color

    def render_rgb(
        self,
        env_id: int = 0,
        image_size: Tuple[int, int] = (224, 224),
        background: str = "white",
    ) -> np.ndarray:
        """Render one batched environment as an RGB uint8 array for VLA observations."""
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"env_id {env_id} is out of range for num_envs={self.num_envs}.")

        height, width = int(image_size[0]), int(image_size[1])
        bg = background.lower()
        if bg in {"black", "k"}:
            frame = np.full((height, width, 3), _hex_to_rgb("09101c"), dtype=np.uint8)
        else:
            frame = np.full((height, width, 3), 255, dtype=np.uint8)

        wall_color = _hex_to_rgb("d0d0d0")
        lumen_color = _hex_to_rgb("f7f7f7")
        clot_outer = _hex_to_rgb("cc3d3d")
        clot_inner = _hex_to_rgb("7a1f1f")
        start_color = _hex_to_rgb("2e8b57")
        goal_color = _hex_to_rgb("d4a017")
        agent_color = _hex_to_rgb("1261a0")
        agent_edge = _hex_to_rgb("082032")

        for pts_t, radii_t in zip(self.branch_points, self.branch_radii):
            pts = pts_t.detach().cpu().numpy()
            radii = radii_t.detach().cpu().numpy()
            for idx in range(len(pts) - 1):
                a_px = self._world_to_pixel_np(pts[idx], image_size)
                b_px = self._world_to_pixel_np(pts[idx + 1], image_size)
                mean_radius = 0.5 * (float(radii[idx]) + float(radii[idx + 1]))
                lumen_radius_px = max(self._radius_to_pixels_np(mean_radius - self.agent_radius, image_size), 1.0)
                wall_radius_px = max(self._radius_to_pixels_np(mean_radius, image_size), lumen_radius_px + 1.0)
                self._paint_capsule_np(frame, a_px, b_px, wall_radius_px, wall_color)
                self._paint_capsule_np(frame, a_px, b_px, lumen_radius_px, lumen_color)

        if bool(self.clot_on[env_id].item()):
            branch_idx = int(self.clot_branch_idx[env_id].item())
            start_idx = int(self.clot_start_idx[env_id].item())
            end_idx = int(self.clot_end_idx[env_id].item())
            pts = self.branch_points[branch_idx].detach().cpu().numpy()
            radii = self.branch_radii[branch_idx].detach().cpu().numpy()
            for idx in range(start_idx, min(end_idx + 1, len(pts) - 1)):
                a_px = self._world_to_pixel_np(pts[idx], image_size)
                b_px = self._world_to_pixel_np(pts[idx + 1], image_size)
                mean_radius = 0.5 * (float(radii[idx]) + float(radii[idx + 1]))
                # ``clot_radius_factor`` is the remaining lumen-radius ratio in
                # physics, so the rendered obstruction must use its complement.
                render_radius_factor = 0.65 * (1.0 - self.clot_radius_factor)
                clot_radius_px = max(
                    self._radius_to_pixels_np(mean_radius * render_radius_factor, image_size),
                    1.0,
                )
                self._paint_capsule_np(frame, a_px, b_px, clot_radius_px, clot_outer)
                self._paint_capsule_np(frame, a_px, b_px, max(clot_radius_px * 0.45, 1.0), clot_inner)

        start = self.start[env_id].detach().cpu().numpy()
        goal = self.goal[env_id].detach().cpu().numpy()
        pos = self.pos[env_id].detach().cpu().numpy()
        marker_radius = max(self._radius_to_pixels_np(self.agent_radius * 1.25, image_size), 2.0)
        agent_radius = max(self._radius_to_pixels_np(self.agent_radius, image_size), 2.0)

        self._paint_disk_np(frame, self._world_to_pixel_np(start, image_size), marker_radius, start_color)
        self._paint_disk_np(frame, self._world_to_pixel_np(goal, image_size), marker_radius, goal_color)
        self._paint_disk_np(frame, self._world_to_pixel_np(pos, image_size), agent_radius + 1.0, agent_edge)
        self._paint_disk_np(frame, self._world_to_pixel_np(pos, image_size), agent_radius, agent_color)

        return np.ascontiguousarray(frame, dtype=np.uint8)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)

        self.clot_on[env_ids] = False
        if self.clot_event:
            clot_draw = self._rand((env_ids.shape[0],)) < self.clot_probability
            self.clot_on[env_ids] = clot_draw
            active_ids = env_ids[clot_draw]
            if active_ids.numel() > 0:
                self._randomize_clot_for_envs(active_ids)

        self._resolve_endpoints(env_ids)
        self.pos[env_ids] = self.start[env_ids]
        self.velocity[env_ids] = 0.0
        self.steps[env_ids] = 0
        self.distance_to_goal[env_ids] = torch.linalg.norm(self.goal[env_ids] - self.pos[env_ids], dim=1)

        obs = self._compute_obs()
        info = {
            "start": self.start[env_ids].clone(),
            "goal": self.goal[env_ids].clone(),
            "clot_on": self.clot_on[env_ids].clone(),
        }
        return obs, info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        actions = torch.clamp(actions.to(self.device, dtype=torch.float32), -1.0, 1.0)
        self.steps += 1
        previous_distance = self.distance_to_goal.clone()

        current_info = self._nearest_lumen_info(self.pos)
        action_noise = torch.zeros_like(actions)
        if self.action_noise_std > 0.0:
            action_noise = torch.randn(actions.shape, generator=self.rng, dtype=torch.float32, device=self.device) * self.action_noise_std

        previous_velocity_rate = self.velocity / self.control_dt
        command_substep = actions * (self.command_speed * self.sim_dt)
        noise_substep = action_noise / self.decimation
        inertia_substep = previous_velocity_rate * self.inertia_gain * self.sim_dt
        max_substep_speed = self.max_speed_per_second * self.sim_dt

        start_pos = self.pos.clone()
        accepted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        penalty = torch.full(
            (self.num_envs,),
            -0.01 * self.reward_rate_scale * self.control_dt,
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(self.decimation):
            current_info = self._nearest_lumen_info(self.pos)
            flow_substep = current_info.flow * (self.flow_speed * self.sim_dt)
            total_substep = inertia_substep + command_substep + noise_substep + flow_substep
            speed = torch.linalg.norm(total_substep, dim=1, keepdim=True).clamp_min(1e-8)
            total_substep = torch.where(speed > max_substep_speed, total_substep * (max_substep_speed / speed), total_substep)

            proposal = self.pos + total_substep
            proposal_safe, _ = self._path_is_safe(self.pos, proposal, self.wall_margin)

            new_pos = self.pos.clone()
            if torch.any(proposal_safe):
                ok_ids = proposal_safe.nonzero(as_tuple=False).squeeze(1)
                projected, _ = self._project_to_safe_lumen(proposal[proposal_safe], self.wall_margin, ok_ids)
                new_pos[ok_ids] = projected
                accepted[ok_ids] = True

            blocked = ~proposal_safe
            if torch.any(blocked):
                blocked_ids = blocked.nonzero(as_tuple=False).squeeze(1)
                center_pull = current_info.closest[blocked] - self.pos[blocked]
                normal_norm = torch.linalg.norm(center_pull, dim=1, keepdim=True).clamp_min(1e-8)
                wall_normal = center_pull / normal_norm
                inertia_normal = torch.sum(inertia_substep[blocked] * wall_normal, dim=1, keepdim=True) * wall_normal
                inertia_substep[blocked] = inertia_substep[blocked] - inertia_normal
                tangent_push = current_info.tangent[blocked] * torch.sum(
                    total_substep[blocked] * current_info.tangent[blocked], dim=1, keepdim=True
                )
                # Favor tangential sliding over a strong centerline pull to avoid
                # a visible "bounce" when the proposal hits the wall.
                inward_fallback = self.pos[blocked] + tangent_push * 0.85 + center_pull * 0.25
                inward_safe, _ = self._path_is_safe(
                    self.pos[blocked],
                    inward_fallback,
                    max(0.0, self.wall_margin * 0.5),
                    blocked_ids,
                )
                if torch.any(inward_safe):
                    ok_ids = blocked_ids[inward_safe]
                    projected, _ = self._project_to_safe_lumen(inward_fallback[inward_safe], self.wall_margin * 0.5, ok_ids)
                    new_pos[ok_ids] = projected
                    accepted[ok_ids] = True
                    penalty[ok_ids] -= (0.02 * self.reward_rate_scale * self.control_dt) / self.decimation

                still_blocked = blocked.clone()
                if torch.any(inward_safe):
                    still_blocked[blocked_ids[inward_safe]] = False

                if torch.any(still_blocked):
                    still_ids = still_blocked.nonzero(as_tuple=False).squeeze(1)
                    tangent_push = current_info.tangent[still_blocked] * torch.sum(
                        total_substep[still_blocked] * current_info.tangent[still_blocked], dim=1, keepdim=True
                    )
                    fallback = self.pos[still_blocked] + tangent_push * 0.6
                    fallback_safe, _ = self._path_is_safe(
                        self.pos[still_blocked],
                        fallback,
                        max(0.0, self.wall_margin * 0.5),
                        still_ids,
                    )
                    if torch.any(fallback_safe):
                        ok_ids = still_ids[fallback_safe]
                        projected, _ = self._project_to_safe_lumen(fallback[fallback_safe], self.wall_margin * 0.5, ok_ids)
                        new_pos[ok_ids] = projected
                        accepted[ok_ids] = True
                        penalty[ok_ids] -= (0.03 * self.reward_rate_scale * self.control_dt) / self.decimation
                    if torch.any(~fallback_safe):
                        bad_ids = still_ids[~fallback_safe]
                        penalty[bad_ids] -= (0.08 * self.reward_rate_scale * self.control_dt) / self.decimation

            self.pos = new_pos

        self.velocity = self.pos - start_pos
        self.distance_to_goal = torch.linalg.norm(self.goal - self.pos, dim=1)
        final_info = self._nearest_lumen_info(self.pos)
        progress = previous_distance - self.distance_to_goal
        reward = penalty + 2.0 * progress - 0.02 * final_info.distance
        reward = torch.where(self.distance_to_goal <= self.goal_radius, reward + 5.0, reward)

        terminated = self.distance_to_goal <= self.goal_radius
        truncated = self.steps >= self.max_steps
        info = {
            "accepted_move": accepted,
            "distance_to_goal": self.distance_to_goal.clone(),
            "clearance": final_info.clearance.clone(),
            "clot_on": self.clot_on.clone(),
        }
        return self._compute_obs(), reward, terminated, truncated, info


def make_torch_vessel_batch_env(
    num_envs: int,
    device: str | torch.device = "cpu",
    **kwargs,
) -> TorchVesselBatchEnv:
    return TorchVesselBatchEnv(num_envs=num_envs, device=device, **kwargs)
