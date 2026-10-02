import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {color}")
    return np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)

from MicroEnv.MicroEnv.vessel_train_cfg import VesselTrainConfig

class VesselContinuousEnv(gym.Env):
    """
    Continuous vessel navigation environment.

    The agent moves in continuous 2D space inside a vessel tree composed of
    piecewise-linear centerlines with varying radii. Movement outside the lumen
    is rejected, and stenosis/clot events can dynamically shrink a local segment.

    Action space:
      Box([-1, -1], [1, 1]) where the two values represent desired local motion.

    Observation:
      Dict with:
        - position: current (x, y)
        - velocity: last accepted displacement
        - goal_vector: vector from agent to goal
        - centerline_offset: distance to nearest centerline
        - local_radius: effective lumen radius at current location
        - clearance: distance margin to vessel wall
        - clot_on: whether stenosis is active
    """

    metadata = {"render_modes": ["human", "rgb_array", "rgba_array", "ansi"], "render_fps": 120}
    FLAT_OBS_KEYS = (
        "position",
        "velocity",
        "goal_vector",
        "centerline_offset",
        "clot_on",
    )

    def __init__(
        self,
        max_steps: Optional[int] = None,
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
        branches: Optional[List[Dict[str, np.ndarray]]] = VesselTrainConfig.Env.branches,
        start: Tuple[float, float] = VesselTrainConfig.Env.start,
        goal: Tuple[float, float] = VesselTrainConfig.Env.goal,
        randomize_start: bool = VesselTrainConfig.Env.randomize_start,
        randomize_goal: bool = VesselTrainConfig.Env.randomize_goal,
        min_clot_distance: float = VesselTrainConfig.Env.min_clot_distance,
        render_mode: Optional[str] = None,
        background: str = VesselTrainConfig.Env.background,
        seed: Optional[int] = None,
        goal_radius = VesselTrainConfig.Env.goal_radius
    ):
        super().__init__()

        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.background = background
        self.sim_dt = max(1e-6, float(sim_dt))
        self.decimation = max(1, int(decimation))
        self.control_dt = self.sim_dt * self.decimation
        self.sim_hz = 1.0 / self.sim_dt
        self.policy_hz = 1.0 / self.control_dt
        self.episode_seconds = max(self.control_dt, float(episode_seconds))
        if max_steps is None:
            self.max_steps = max(1, int(round(self.episode_seconds * self.policy_hz)))
        else:
            self.max_steps = int(max_steps)
            self.episode_seconds = self.max_steps * self.control_dt
        self.step_scale = float(step_scale)
        self.action_noise_std = max(0.0, float(action_noise_std))
        self.drift_gain = float(drift_gain)
        self.inertia_decay = float(inertia_decay)
        self.max_speed = float(max_speed)
        self.command_speed = self.step_scale * self.policy_hz
        self.flow_speed = self.drift_gain * self.policy_hz
        self.inertia_gain = self.inertia_decay * self.policy_hz
        self.max_speed_per_second = self.max_speed * self.policy_hz
        self.reward_rate_scale = self.policy_hz
        self.wall_margin = float(wall_margin)
        self.agent_radius = float(agent_radius)
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

        self.bounds = (0.0, 20.0, 0.0, 20.0)
        self.default_start = np.asarray(start, dtype=np.float32)
        self.default_goal = np.asarray(goal, dtype=np.float32)
        self.start = self.default_start.copy()
        self.goal = self.default_goal.copy()
        self.goal_radius = goal_radius

        # Each branch is a polyline with radii defined at nodes.
        self.branches = self._normalize_branches(branches)
        self.randomize_start = bool(randomize_start)
        self.randomize_goal = bool(randomize_goal)
        self.min_clot_distance = max(0.0, float(min_clot_distance))
        self.clot_branch_name = "main"
        self.clot_segment_index = 1
        self.clot_segments = {self.clot_segment_index}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), high=np.array([20.0, 20.0], dtype=np.float32), dtype=np.float32),
                "velocity": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
                "goal_vector": spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32),
                "centerline_offset": spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32),
                "local_radius": spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32),
                "effective_radius": spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32),
                "clearance": spaces.Box(low=-1.0, high=3.0, shape=(1,), dtype=np.float32),
                "clot_on": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self.np_random = None
        self._set_seed(seed)

        self.pos = self.start.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.clot_step = None
        self._auto_clot_active = False
        self._manual_clot_override = None
        self.clot_on = False
        self.last_info = {}
        self.last_done_reason = None

        self.fig = None
        self.ax = None
        self.status_text = None
        self._layout_applied = False
        self._interactive_keys = {"up": False, "down": False, "left": False, "right": False}
        self._control_latches = {"c": False, "r": False}
        self._interactive_running = False

        self.reset(seed=seed)

    @classmethod
    def flat_observation_keys(cls) -> List[str]:
        return list(cls.FLAT_OBS_KEYS)

    @staticmethod
    def _normalize_branches(branches: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        normalized = []
        for branch in branches:
            if not isinstance(branch, dict):
                raise ValueError("Each branch must be a dict with name/points/radii/flow.")
            name = str(branch.get("name", "branch"))
            points = np.asarray(branch.get("points"), dtype=np.float32)
            radii = np.asarray(branch.get("radii"), dtype=np.float32)
            flow = np.asarray(branch.get("flow", [1.0, 0.0]), dtype=np.float32)
            clot_allowed = bool(branch.get("clot_allowed", True))
            normalized.append({"name": name, "points": points, "radii": radii, "flow": flow, "clot_allowed": clot_allowed})
        if not normalized:
            raise ValueError("branches must contain at least one branch.")
        return normalized

    def _set_seed(self, seed: Optional[int]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _sample_point_in_vessel(self) -> np.ndarray:
        segments = []
        total_len = 0.0
        for branch in self.branches:
            pts = branch["points"]
            for idx in range(len(pts) - 1):
                seg_len = float(np.linalg.norm(pts[idx + 1] - pts[idx]))
                if seg_len <= 1e-8:
                    continue
                segments.append((pts[idx], pts[idx + 1], seg_len))
                total_len += seg_len

        if not segments or total_len <= 1e-8:
            return self.default_start.copy()

        target_len = float(self.np_random.uniform(0.0, total_len))
        accum = 0.0
        for a, b, seg_len in segments:
            next_accum = accum + seg_len
            if target_len <= next_accum:
                t = (target_len - accum) / seg_len
                return (a + (b - a) * t).astype(np.float32)
            accum = next_accum

        a, b, _ = segments[-1]
        return np.asarray(b, dtype=np.float32).copy()

    @staticmethod
    def _distance_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq <= 1e-8:
            return float(np.linalg.norm(point - a))
        t = float(np.dot(point - a, ab) / ab_len_sq)
        t = _clamp(t, 0.0, 1.0)
        closest = a + t * ab
        return float(np.linalg.norm(point - closest))

    def _distance_to_clot(self, point: np.ndarray) -> float:
        branch = next((b for b in self.branches if b["name"] == self.clot_branch_name), None)
        if branch is None:
            return float("inf")

        best = float("inf")
        pts = branch["points"]
        for idx in self.clot_segments:
            if idx < 0 or idx + 1 >= len(pts):
                continue
            best = min(best, self._distance_to_segment(point, pts[idx], pts[idx + 1]))
        return best

    def _point_blocked_by_clot(self, point: np.ndarray) -> bool:
        if (not self.clot_on) or self.min_clot_distance <= 0.0:
            return False
        return self._distance_to_clot(point) < self.min_clot_distance

    def _sample_point_away_from_clot(self) -> np.ndarray:
        if (not self.clot_on) or self.min_clot_distance <= 0.0:
            return self._sample_point_in_vessel()

        point = self._sample_point_in_vessel()
        for _ in range(128):
            if not self._point_blocked_by_clot(point):
                break
            point = self._sample_point_in_vessel()
        return point

    def _repair_point_away_from_clot(self, point: np.ndarray) -> np.ndarray:
        if (not self.clot_on) or self.min_clot_distance <= 0.0:
            return point
        repaired = np.asarray(point, dtype=np.float32).copy()
        for _ in range(8):
            if not self._point_blocked_by_clot(repaired):
                return repaired
            repaired = self._sample_point_away_from_clot()
        if self._point_blocked_by_clot(repaired):
            self.clot_on = False
            self._auto_clot_active = False
        return repaired

    def _resolve_episode_endpoints(self, options: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        options = options or {}
        start = np.asarray(options.get("start", self.default_start), dtype=np.float32).copy()
        goal = np.asarray(options.get("goal", self.default_goal), dtype=np.float32).copy()

        if bool(options.get("randomize_start", self.randomize_start)):
            start = self._sample_point_away_from_clot()
        start = self._repair_point_away_from_clot(start)
        if bool(options.get("randomize_goal", self.randomize_goal)):
            goal = self._sample_point_away_from_clot()
            for _ in range(16):
                if float(np.linalg.norm(goal - start)) > max(self.goal_radius * 2.0, 1.0):
                    break
                goal = self._sample_point_away_from_clot()
        goal = self._repair_point_away_from_clot(goal)
        return start.astype(np.float32), goal.astype(np.float32)

    def _refresh_clot_state(self):
        if self._manual_clot_override is None:
            self.clot_on = bool(self._auto_clot_active)
        else:
            self.clot_on = bool(self._manual_clot_override)

    def _effective_radius(self, branch_name: str, segment_index: int, radius: float) -> float:
        if self.clot_on and branch_name == self.clot_branch_name and segment_index in self.clot_segments:
            return radius * self.clot_radius_factor
        return radius

    def _randomize_clot_segments(self):
        allowed_branches = [b for b in self.branches if b.get("clot_allowed", True)]
        if not allowed_branches:
            allowed_branches = self.branches

        def _branch_order():
            if self.clot_branch_randomize:
                order = list(allowed_branches)
                self.np_random.shuffle(order)
                return order
            preferred = next((b for b in allowed_branches if b["name"] == self.clot_branch_name), allowed_branches[0])
            return [preferred] + [b for b in allowed_branches if b is not preferred]

        def _candidate_spans(branch, restrict_to_region):
            pts = branch["points"]
            if len(pts) < 2:
                return []
            seg_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_len = float(cum_lengths[-1])
            if total_len <= 1e-8:
                return []

            if restrict_to_region:
                region_start = max(0.0, min(1.0, float(self.clot_region[0]))) * total_len
                region_end = max(0.0, min(1.0, float(self.clot_region[1]))) * total_len
                if region_end < region_start:
                    region_start, region_end = region_end, region_start
                candidate_starts = [
                    i for i in range(len(seg_lengths))
                    if cum_lengths[i] >= region_start and cum_lengths[i] < region_end
                ]
            else:
                region_end = total_len
                candidate_starts = list(range(len(seg_lengths)))
            if not candidate_starts:
                return []

            min_len = max(1, int(self.clot_length_range[0]))
            max_len_target = max(min_len, int(self.clot_length_range[1]))
            junctions = self._branch_junction_distances(branch["name"], pts, cum_lengths)
            spans = []
            for start_idx in candidate_starts:
                for length in range(min_len, max_len_target + 1):
                    end_idx = min(start_idx + length - 1, len(seg_lengths) - 1)
                    span_start = float(cum_lengths[start_idx])
                    span_end = float(cum_lengths[end_idx + 1])
                    if span_end > region_end + 1e-8:
                        continue
                    span_length = span_end - span_start
                    if self.clot_max_length > 0.0 and span_length > self.clot_max_length + 1e-8:
                        continue
                    if self._clot_span_on_single_y_slice(branch, start_idx, end_idx):
                        continue
                    if self._clot_span_near_junction(span_start, span_end, junctions):
                        continue
                    spans.append((start_idx, end_idx))
            return spans

        chosen = None
        order = _branch_order()
        for branch in order:
            spans = _candidate_spans(branch, restrict_to_region=True)
            if spans:
                start_idx, end_idx = spans[int(self.np_random.integers(0, len(spans)))]
                chosen = (branch, start_idx, end_idx)
                break
        if chosen is None:
            for branch in order:
                spans = _candidate_spans(branch, restrict_to_region=False)
                if spans:
                    start_idx, end_idx = spans[int(self.np_random.integers(0, len(spans)))]
                    chosen = (branch, start_idx, end_idx)
                    break
        if chosen is None:
            self.clot_on = False
            self._auto_clot_active = False
            self.clot_segments = {self.clot_segment_index}
            return

        branch, start_idx, end_idx = chosen
        self.clot_branch_name = str(branch["name"])
        self.clot_segments = set(range(start_idx, end_idx + 1))
        self.clot_segment_index = start_idx

    def _branch_junction_distances(self, branch_name: str, pts: np.ndarray, cum_lengths: np.ndarray) -> List[float]:
        junctions = [0.0, float(cum_lengths[-1])]
        threshold = max(0.15, self.agent_radius * 1.25)
        for point_idx, point in enumerate(pts):
            distance_along_branch = float(cum_lengths[point_idx])
            for other in self.branches:
                if other["name"] == branch_name:
                    continue
                other_pts = other["points"]
                for idx in range(len(other_pts) - 1):
                    if self._distance_to_segment(point, other_pts[idx], other_pts[idx + 1]) <= threshold:
                        junctions.append(distance_along_branch)
                        break
                else:
                    continue
                break
        return sorted(set(round(value, 6) for value in junctions))

    def _clot_span_near_junction(self, span_start: float, span_end: float, junctions: List[float]) -> bool:
        if self.clot_junction_margin <= 0.0:
            return False
        for junction in junctions:
            if span_start <= junction + self.clot_junction_margin and span_end >= junction - self.clot_junction_margin:
                return True
        return False

    def _clot_span_on_single_y_slice(self, branch: Dict[str, np.ndarray], start_idx: int, end_idx: int) -> bool:
        if self.clot_min_y_slice_paths <= 1:
            return False
        pts = branch["points"]
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
        for branch in self.branches:
            pts = branch["points"]
            for idx in range(len(pts) - 1):
                a = pts[idx]
                b = pts[idx + 1]
                ax, ay = float(a[0]), float(a[1])
                bx, by = float(b[0]), float(b[1])
                x_low = min(ax, bx) - eps
                x_high = max(ax, bx) + eps
                if x < x_low or x > x_high:
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

    def _segment_projection(
        self,
        p: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        r0: float,
        r1: float,
        branch_name: str,
        segment_index: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq <= 1e-8:
            return None

        t = float(np.dot(p - a, ab) / ab_len_sq)
        t = _clamp(t, 0.0, 1.0)
        closest = a + t * ab
        base_radius = (1.0 - t) * r0 + t * r1
        radius = self._effective_radius(branch_name, segment_index, float(base_radius))
        navigable_radius = max(0.0, radius - self.agent_radius)

        offset = p - closest
        distance = float(np.linalg.norm(offset))
        tangent = ab / np.sqrt(ab_len_sq)
        return {
            "closest": closest,
            "distance": distance,
            "radius": radius,
            "navigable_radius": navigable_radius,
            "clearance": navigable_radius - distance,
            "tangent": tangent,
            "branch": branch_name,
            "segment_index": segment_index,
            "segment_t": t,
        }

    def _nearest_lumen_info(self, point: np.ndarray) -> Dict[str, np.ndarray]:
        best = None
        best_distance = None
        distance_tolerance = 1e-6
        for branch in self.branches:
            pts = branch["points"]
            radii = branch["radii"]
            for idx in range(len(pts) - 1):
                candidate = self._segment_projection(
                    point,
                    pts[idx],
                    pts[idx + 1],
                    float(radii[idx]),
                    float(radii[idx + 1]),
                    branch["name"],
                    idx,
                )
                if candidate is None:
                    continue
                candidate_distance = float(candidate["distance"])
                if (
                    best is None
                    or candidate_distance < best_distance - distance_tolerance
                    or (
                        abs(candidate_distance - best_distance) <= distance_tolerance
                        and candidate["clearance"] > best["clearance"]
                    )
                ):
                    best = candidate
                    best_distance = candidate_distance

        if best is None:
            raise RuntimeError("No vessel geometry defined.")
        return best

    def _inside_lumen(self, point: np.ndarray) -> Tuple[bool, Dict[str, np.ndarray]]:
        info = self._nearest_lumen_info(point)
        inside = bool(info["clearance"] >= -1e-6)
        return inside, info

    def _project_to_safe_lumen(self, point: np.ndarray, margin: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        info = self._nearest_lumen_info(point)
        closest = np.asarray(info["closest"], dtype=np.float32)
        offset = point - closest
        distance = float(np.linalg.norm(offset))
        safe_radius = max(0.0, float(info["navigable_radius"]) - margin)

        if distance <= 1e-8:
            normal = np.array([-info["tangent"][1], info["tangent"][0]], dtype=np.float32)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                normal = np.array([0.0, 1.0], dtype=np.float32)
            else:
                normal /= norm
            projected = closest + normal * min(safe_radius, 0.2)
        elif distance <= safe_radius:
            projected = np.asarray(point, dtype=np.float32)
        else:
            projected = closest + (offset / distance) * safe_radius

        projected_info = self._nearest_lumen_info(projected)
        return projected.astype(np.float32), projected_info

    def _path_is_safe(self, start: np.ndarray, end: np.ndarray, margin: float) -> Tuple[bool, Dict[str, np.ndarray]]:
        delta = end - start
        distance = float(np.linalg.norm(delta))
        samples = max(2, int(np.ceil(distance / 0.05)))
        last_info = self._nearest_lumen_info(start)
        effective_margin = min(float(margin), max(0.0, float(last_info["clearance"])) + 1e-6)

        for i in range(1, samples + 1):
            alpha = i / samples
            probe = start + alpha * delta
            inside, info = self._inside_lumen(probe)
            last_info = info
            if (not inside) or info["clearance"] < effective_margin:
                return False, info
        return True, last_info

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        info = self._nearest_lumen_info(self.pos)
        return {
            "position": self.pos.astype(np.float32).copy(),
            "velocity": self.velocity.astype(np.float32).copy(),
            "goal_vector": (self.goal - self.pos).astype(np.float32),
            "centerline_offset": np.array([info["distance"]], dtype=np.float32),
            "local_radius": np.array([info["radius"]], dtype=np.float32),
            "effective_radius": np.array([info["navigable_radius"]], dtype=np.float32),
            "clearance": np.array([info["clearance"]], dtype=np.float32),
            "clot_on": np.array([1.0 if self.clot_on else 0.0], dtype=np.float32),
        }

    def flatten_observation(self, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        if obs is None:
            obs = self._compute_observation()

        chunks = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in self.FLAT_OBS_KEYS]
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._set_seed(seed)

        self._auto_clot_active = False
        self._manual_clot_override = None
        if self.clot_event:
            # Decide clot presence once per reset (no mid-episode activation).
            self._auto_clot_active = bool(self.np_random.random() < self.clot_probability)
            if self._auto_clot_active:
                self._randomize_clot_segments()
            else:
                self.clot_segments = {self.clot_segment_index}
            self.clot_step = None
        else:
            self.clot_segments = {self.clot_segment_index}
            self.clot_step = None
        self._refresh_clot_state()

        self.start, self.goal = self._resolve_episode_endpoints(options)
        self.pos = self.start.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.last_info = {}
        self.last_done_reason = None

        obs = self._compute_observation()
        info = {
            "steps": self.steps,
            "clot_on": self.clot_on,
            "clot_step": self.clot_step,
            "start": self.start.copy(),
            "goal": self.goal.copy(),
        }
        return obs, info

    def _compute_flow(self, lumen_info: Dict[str, np.ndarray]) -> np.ndarray:
        for branch in self.branches:
            if branch["name"] == lumen_info["branch"]:
                return branch["flow"]
        return np.array([1.0, 0.0], dtype=np.float32)

    def _compute_reward(self, penalty: float, progress: float, centerline_offset: float, distance_to_goal: float,) -> float:
        reward = float(penalty)
        reward += 2.0 * float(progress) - 0.02 * float(centerline_offset)
        if distance_to_goal <= self.goal_radius:
            reward += 5.0
        return reward

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.steps += 1
        previous_distance = float(np.linalg.norm(self.goal - self.pos))

        current_info = self._nearest_lumen_info(self.pos)
        action_noise = np.zeros(2, dtype=np.float32)
        if self.action_noise_std > 0.0:
            action_noise = self.np_random.normal(0.0, self.action_noise_std, size=2).astype(np.float32)
        previous_velocity_rate = self.velocity / self.control_dt
        command_substep = action * (self.command_speed * self.sim_dt)
        noise_substep = action_noise / self.decimation if self.action_noise_std > 0.0 else np.zeros(2, dtype=np.float32)
        inertia_substep = previous_velocity_rate * self.inertia_gain * self.sim_dt
        max_substep_speed = self.max_speed_per_second * self.sim_dt

        start_pos = self.pos.copy()
        accepted = False
        penalty = -0.01 * self.reward_rate_scale * self.control_dt
        proposal_info = current_info

        for _ in range(self.decimation):
            current_info = self._nearest_lumen_info(self.pos)
            flow = self._compute_flow(current_info)
            flow_substep = flow * (self.flow_speed * self.sim_dt)
            total_substep = inertia_substep + command_substep + noise_substep + flow_substep

            speed = float(np.linalg.norm(total_substep))
            if speed > max_substep_speed:
                total_substep *= max_substep_speed / speed

            proposal = self.pos + total_substep
            proposal_safe, proposal_info = self._path_is_safe(self.pos, proposal, self.wall_margin)

            if proposal_safe:
                new_pos, proposal_info = self._project_to_safe_lumen(proposal, self.wall_margin)
                accepted = True
            else:
                center_pull = current_info["closest"] - self.pos
                normal_norm = max(float(np.linalg.norm(center_pull)), 1e-8)
                wall_normal = center_pull / normal_norm
                inertia_normal = wall_normal * float(np.dot(inertia_substep, wall_normal))
                inertia_substep = inertia_substep - inertia_normal
                tangent_push = current_info["tangent"] * float(np.dot(total_substep, current_info["tangent"]))
                # Favor tangential sliding over a strong centerline pull to avoid
                # a visible "bounce" when the proposal hits the wall.
                inward_fallback = self.pos + tangent_push * 0.85 + center_pull * 0.25
                inward_safe, inward_info = self._path_is_safe(self.pos, inward_fallback, max(0.0, self.wall_margin * 0.5))
                if inward_safe:
                    new_pos, proposal_info = self._project_to_safe_lumen(inward_fallback, self.wall_margin * 0.5)
                    accepted = True
                    penalty -= (0.02 * self.reward_rate_scale * self.control_dt) / self.decimation
                else:
                    fallback = self.pos + tangent_push * 0.6
                    fallback_safe, fallback_info = self._path_is_safe(self.pos, fallback, max(0.0, self.wall_margin * 0.5))
                    if fallback_safe:
                        new_pos, proposal_info = self._project_to_safe_lumen(fallback, self.wall_margin * 0.5)
                        accepted = True
                        penalty -= (0.03 * self.reward_rate_scale * self.control_dt) / self.decimation
                    else:
                        new_pos = self.pos.copy()
                        proposal_info = fallback_info
                        penalty -= (0.08 * self.reward_rate_scale * self.control_dt) / self.decimation

            self.pos = new_pos.astype(np.float32)

        self.velocity = (self.pos - start_pos).astype(np.float32)
        self.distance_to_goal = float(np.linalg.norm(self.goal - self.pos))
        progress = previous_distance - self.distance_to_goal
        
        reward = self._compute_reward(
            penalty=penalty,
            progress=progress,
            centerline_offset=float(proposal_info["distance"]),
            distance_to_goal=self.distance_to_goal,
        )

        terminated = False
        truncated = False
        if self.distance_to_goal <= self.goal_radius:
            terminated = True
            self.last_done_reason = "goal"
        elif self.steps >= self.max_steps:
            truncated = True
            self.last_done_reason = "max_steps"

        obs = self._compute_observation()
        info = {
            "steps": self.steps,
            "clot_on": self.clot_on,
            "clot_step": self.clot_step,
            "accepted_move": accepted,
            "distance_to_goal": self.distance_to_goal,
            "branch": proposal_info["branch"],
            "clearance": proposal_info["clearance"],
            "sim_dt": self.sim_dt,
            "control_dt": self.control_dt,
            "decimation": self.decimation,
            "episode_seconds": self.episode_seconds,
        }
        self.last_info = info
        return obs, float(reward), terminated, truncated, info

    def _render_ansi(self) -> str:
        obs = self._compute_observation()
        return (
            f"pos=({obs['position'][0]:.2f}, {obs['position'][1]:.2f}) "
            f"vel=({obs['velocity'][0]:.2f}, {obs['velocity'][1]:.2f}) "
            f"radius={obs['local_radius'][0]:.2f} "
            f"nav_radius={obs['effective_radius'][0]:.2f} "
            f"clearance={obs['clearance'][0]:.2f} "
            f"clot_on={self.clot_on}"
        )

    def _ensure_canvas(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))
            self._apply_background()
            self._layout_applied = False

    def _plot_branch(self, branch: Dict[str, np.ndarray], wall_color: str, lumen_color: str):
        pts = branch["points"]
        xs = pts[:, 0]
        ys = pts[:, 1]
        mean_radius = float(np.mean(branch["radii"]))

        self.ax.plot(
            xs,
            ys,
            color=wall_color,
            linewidth=36 * mean_radius,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        self.ax.plot(
            xs,
            ys,
            color=lumen_color,
            linewidth=22 * mean_radius,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )

    def _draw_clot_overlay(self):
        branch = next(branch for branch in self.branches if branch["name"] == self.clot_branch_name)
        for idx in sorted(self.clot_segments):
            if idx < 0 or idx + 1 >= len(branch["points"]):
                continue
            a = branch["points"][idx]
            b = branch["points"][idx + 1]
            self.ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#df7c2d",
                linewidth=14,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )
            self.ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#7a3414",
                linewidth=6,
                alpha=0.95,
                solid_capstyle="round",
                zorder=4,
            )

    def _draw_marker(self, point: np.ndarray, color: str, size: float, edgecolor: str = "white", zorder: int = 6):
        self.ax.scatter(
            [float(point[0])],
            [float(point[1])],
            s=size,
            c=[color],
            edgecolors=edgecolor,
            linewidths=1.4,
            zorder=zorder,
        )

    def _draw_scene(self):
        self._ensure_canvas()
        self.ax.clear()
        self._apply_background()

        wall_color = "#6d1731"
        lumen_color = "#f8c9d2"

        for branch in self.branches:
            self._plot_branch(branch, wall_color, lumen_color)

        if self.clot_on:
            self._draw_clot_overlay()

        self._draw_marker(self.start, "#39d98a", 110)
        self._draw_marker(self.goal, "#ff5e76", 110)
        self._draw_marker(self.pos, "#4db8ff", max(120.0, 700.0 * self.agent_radius), edgecolor="#082032", zorder=7)

        self.ax.text(self.start[0], self.start[1] - 1.0, "Start", color="#b7ffd7", fontsize=10, ha="center", zorder=8)
        self.ax.text(self.goal[0], self.goal[1] - 1.0, "Goal", color="#ffd3da", fontsize=10, ha="center", zorder=8)

        obs = self._compute_observation()
        if self.status_text is None:
            self.status_text = self.ax.text(
                0.02,
                -0.08,
                "",
                transform=self.ax.transAxes,
                fontsize=10,
                color="#e6edf7",
            )
        self.status_text.set_text(
            f"steps={self.steps}  pos=({obs['position'][0]:.2f},{obs['position'][1]:.2f})  "
            f"clearance={obs['clearance'][0]:.2f}  radius={obs['local_radius'][0]:.2f}  "
            f"nav_radius={obs['effective_radius'][0]:.2f}  "
            f"clot_on={self.clot_on}"
        )
        if self.last_done_reason == "goal":
            self.ax.text(
                0.5,
                0.06,
                "Reached goal",
                transform=self.ax.transAxes,
                fontsize=13,
                color="#7CFFB2",
                ha="center",
                zorder=9,
            )
        elif self.last_done_reason == "max_steps":
            self.ax.text(
                0.5,
                0.06,
                "Max steps reached",
                transform=self.ax.transAxes,
                fontsize=13,
                color="#FFD580",
                ha="center",
                zorder=9,
            )

        self.ax.set_title("VesselContinuousEnv", color="#f4f7fb", pad=12)
        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[3], self.bounds[2])
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        if not self._layout_applied:
            self.fig.tight_layout()
            self._layout_applied = True

    def _render_rgb_array(self) -> np.ndarray:
        self._draw_scene()
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[..., :3])

    def _render_rgba_array(self) -> np.ndarray:
        self._draw_scene()
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba)

    def _world_to_pixel(self, point: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        height, width = int(image_size[0]), int(image_size[1])
        xmin, xmax, ymin, ymax = self.bounds
        x = (float(point[0]) - xmin) / max(xmax - xmin, 1e-6)
        y = (float(point[1]) - ymin) / max(ymax - ymin, 1e-6)
        return np.array([x * (width - 1), y * (height - 1)], dtype=np.float32)

    def _radius_to_pixels(self, radius: float, image_size: Tuple[int, int]) -> float:
        height, width = int(image_size[0]), int(image_size[1])
        xmin, xmax, ymin, ymax = self.bounds
        scale_x = (width - 1) / max(xmax - xmin, 1e-6)
        scale_y = (height - 1) / max(ymax - ymin, 1e-6)
        return float(radius) * min(scale_x, scale_y)

    @staticmethod
    def _paint_mask(frame: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
        frame[mask] = color

    def _paint_disk(self, frame: np.ndarray, center_px: np.ndarray, radius_px: float, color: np.ndarray) -> None:
        radius_px = max(float(radius_px), 1.0)
        height, width = frame.shape[:2]
        cx, cy = float(center_px[0]), float(center_px[1])
        xmin = max(int(np.floor(cx - radius_px)), 0)
        xmax = min(int(np.ceil(cx + radius_px)) + 1, width)
        ymin = max(int(np.floor(cy - radius_px)), 0)
        ymax = min(int(np.ceil(cy + radius_px)) + 1, height)
        if xmin >= xmax or ymin >= ymax:
            return
        yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2
        patch = frame[ymin:ymax, xmin:xmax]
        patch[mask] = color

    def _paint_capsule(
        self,
        frame: np.ndarray,
        a_px: np.ndarray,
        b_px: np.ndarray,
        radius_px: float,
        color: np.ndarray,
    ) -> None:
        radius_px = max(float(radius_px), 1.0)
        height, width = frame.shape[:2]
        ax, ay = float(a_px[0]), float(a_px[1])
        bx, by = float(b_px[0]), float(b_px[1])
        xmin = max(int(np.floor(min(ax, bx) - radius_px)), 0)
        xmax = min(int(np.ceil(max(ax, bx) + radius_px)) + 1, width)
        ymin = max(int(np.floor(min(ay, by) - radius_px)), 0)
        ymax = min(int(np.ceil(max(ay, by) + radius_px)) + 1, height)
        if xmin >= xmax or ymin >= ymax:
            return

        yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 1e-8:
            mask = (xx - ax) ** 2 + (yy - ay) ** 2 <= radius_px ** 2
        else:
            t = ((xx - ax) * abx + (yy - ay) * aby) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest_x = ax + t * abx
            closest_y = ay + t * aby
            mask = (xx - closest_x) ** 2 + (yy - closest_y) ** 2 <= radius_px ** 2
        patch = frame[ymin:ymax, xmin:xmax]
        patch[mask] = color

    def _render_fast_rgb(self, image_size: Tuple[int, int]) -> np.ndarray:
        height, width = int(image_size[0]), int(image_size[1])
        bg = (self.background or "transparent").lower()
        if bg in {"white", "w"}:
            frame = np.full((height, width, 3), 255, dtype=np.uint8)
        elif bg in {"black", "k"}:
            frame = np.full((height, width, 3), _hex_to_rgb("09101c"), dtype=np.uint8)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        wall_color = _hex_to_rgb("6d1731")
        lumen_color = _hex_to_rgb("f8c9d2")
        clot_outer = _hex_to_rgb("df7c2d")
        clot_inner = _hex_to_rgb("7a3414")
        start_color = _hex_to_rgb("39d98a")
        goal_color = _hex_to_rgb("ff5e76")
        agent_color = _hex_to_rgb("4db8ff")
        agent_edge = _hex_to_rgb("082032")

        for branch in self.branches:
            pts = branch["points"]
            radii = branch["radii"]
            for idx in range(len(pts) - 1):
                a_px = self._world_to_pixel(pts[idx], image_size)
                b_px = self._world_to_pixel(pts[idx + 1], image_size)
                mean_radius = 0.5 * (float(radii[idx]) + float(radii[idx + 1]))
                lumen_radius_px = max(self._radius_to_pixels(mean_radius, image_size), 1.0)
                wall_radius_px = max(lumen_radius_px * 1.45, lumen_radius_px + 1.0)
                self._paint_capsule(frame, a_px, b_px, wall_radius_px, wall_color)
                self._paint_capsule(frame, a_px, b_px, lumen_radius_px, lumen_color)

        if self.clot_on:
            branch = next(branch for branch in self.branches if branch["name"] == self.clot_branch_name)
            pts = branch["points"]
            radii = branch["radii"]
            for idx in sorted(self.clot_segments):
                if idx < 0 or idx + 1 >= len(pts):
                    continue
                a_px = self._world_to_pixel(pts[idx], image_size)
                b_px = self._world_to_pixel(pts[idx + 1], image_size)
                mean_radius = 0.5 * (float(radii[idx]) + float(radii[idx + 1]))
                clot_radius_px = max(self._radius_to_pixels(mean_radius * 0.75, image_size), 1.0)
                self._paint_capsule(frame, a_px, b_px, clot_radius_px, clot_outer)
                self._paint_capsule(frame, a_px, b_px, max(clot_radius_px * 0.45, 1.0), clot_inner)

        start_px = self._world_to_pixel(self.start, image_size)
        goal_px = self._world_to_pixel(self.goal, image_size)
        pos_px = self._world_to_pixel(self.pos, image_size)
        marker_radius = max(self._radius_to_pixels(self.agent_radius * 1.15, image_size), 2.0)
        agent_radius = max(self._radius_to_pixels(self.agent_radius, image_size), 2.0)

        self._paint_disk(frame, start_px, marker_radius, start_color)
        self._paint_disk(frame, goal_px, marker_radius, goal_color)
        self._paint_disk(frame, pos_px, agent_radius + 1.0, agent_edge)
        self._paint_disk(frame, pos_px, agent_radius, agent_color)

        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _apply_background(self):
        bg = (self.background or "transparent").lower()
        if bg in {"white", "w"}:
            self.fig.patch.set_facecolor("white")
            self.fig.patch.set_alpha(1.0)
            self.ax.set_facecolor("white")
        elif bg in {"black", "k"}:
            self.fig.patch.set_facecolor("#09101c")
            self.fig.patch.set_alpha(1.0)
            self.ax.set_facecolor("#09101c")
        else:
            self.fig.patch.set_facecolor("none")
            self.fig.patch.set_alpha(0.0)
            self.ax.set_facecolor("none")

    def render(self):
        mode = self.render_mode or "ansi"
        if mode == "ansi":
            return self._render_ansi()
        if mode == "rgb_array":
            return self._render_rgb_array()
        if mode == "rgba_array":
            return self._render_rgba_array()
        if mode == "human":
            frame = self._render_rgb_array()
            plt.show(block=False)
            plt.pause(1.0 / self.metadata["render_fps"])
            return frame
        raise ValueError(f"Unsupported render_mode: {mode}")

    def close(self):
        self._interactive_running = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.status_text = None
            self._layout_applied = False

    def _key_to_action(self) -> np.ndarray:
        dx = 0.0
        dy = 0.0
        if self._interactive_keys["left"]:
            dx -= 1.0
        if self._interactive_keys["right"]:
            dx += 1.0
        if self._interactive_keys["up"]:
            dy -= 1.0
        if self._interactive_keys["down"]:
            dy += 1.0

        action = np.array([dx, dy], dtype=np.float32)
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action /= norm
        return action

    def _on_key_press(self, event):
        key = (event.key or "").lower()
        if key in self._interactive_keys:
            self._interactive_keys[key] = True
        elif key == "c":
            if not self._control_latches["c"]:
                self._manual_clot_override = not self.clot_on
                self._refresh_clot_state()
                self._control_latches["c"] = True
        elif key == "r":
            if not self._control_latches["r"]:
                self.reset()
                self._control_latches["r"] = True
        elif key == "escape":
            self._interactive_running = False
            plt.close(self.fig)

    def _on_key_release(self, event):
        key = (event.key or "").lower()
        if key in self._interactive_keys:
            self._interactive_keys[key] = False
        elif key in self._control_latches:
            self._control_latches[key] = False

    def run_interactive(self):
        """
        Open a live window and let the user drive with arrow keys.

        Controls:
          Arrow keys: move
          c: toggle clot/stenosis
          r: reset
          Esc: quit
        """
        if self.render_mode != "human":
            self.render_mode = "human"

        self._ensure_canvas()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

        self._interactive_running = True
        self.render()
        print("Interactive controls: arrows move, c toggles clot, r resets, Esc exits.")

        last_time = time.time()
        while self._interactive_running and plt.fignum_exists(self.fig.number):
            now = time.time()
            dt = now - last_time
            last_time = now

            action = self._key_to_action()
            _, _, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                self.render()
                print(f"Episode ended: {self.last_done_reason}. Press r to reset or Esc to exit.")
                while self._interactive_running and plt.fignum_exists(self.fig.number):
                    plt.pause(0.001)
                    if self.last_done_reason is None:
                        break
                continue

            self.render()

            if dt < 1.0 / self.metadata["render_fps"]:
                plt.pause((1.0 / self.metadata["render_fps"]) - dt)
            else:
                plt.pause(0.001)


class VesselContinuousRLWrapper(gym.Wrapper):
    """
    RL-friendly wrapper that converts the Dict observation into a flat Box vector.

    Flattened observation order:
      [position(2), velocity(2), goal_vector(2), local_radius(1),
       effective_radius(1), clearance(1), clot_on(1)]
    """

    def __init__(self, env: VesselContinuousEnv):
        super().__init__(env)
        self.obs_keys = tuple(env.flat_observation_keys())
        lows = []
        highs = []
        for key in self.obs_keys:
            space = env.observation_space.spaces[key]
            lows.append(np.asarray(space.low, dtype=np.float32).reshape(-1))
            highs.append(np.asarray(space.high, dtype=np.float32).reshape(-1))
        self.observation_space = spaces.Box(
            low=np.concatenate(lows, axis=0).astype(np.float32),
            high=np.concatenate(highs, axis=0).astype(np.float32),
            dtype=np.float32,
        )

    def _flatten(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        flat_obs = self.env.flatten_observation(obs)
        if flat_obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Unexpected flattened observation shape {flat_obs.shape}, "
                f"expected {self.observation_space.shape}."
            )
        return flat_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["obs_dict"] = obs
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["obs_dict"] = obs
        return self._flatten(obs), reward, terminated, truncated, info

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        return self.env._compute_observation()


class VesselImageObservationWrapper(gym.Wrapper):
    """
    Visual observation wrapper for pixel-based RL.

    By default it returns channel-first RGB images with shape (3, H, W),
    which is a better default for PyTorch-based PPO/CNN policies.
    """

    def __init__(
        self,
        env: VesselContinuousEnv,
        image_size: Tuple[int, int] = (84, 84),
        grayscale: bool = False,
        channel_first: bool = True,
    ):
        super().__init__(env)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.grayscale = bool(grayscale)
        self.channel_first = bool(channel_first)

        channels = 1 if self.grayscale else 3
        if self.channel_first:
            obs_shape = (channels, self.image_size[0], self.image_size[1])
        else:
            obs_shape = (self.image_size[0], self.image_size[1], channels)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        dst_h, dst_w = self.image_size
        row_idx = np.linspace(0, src_h - 1, dst_h).astype(np.int32)
        col_idx = np.linspace(0, src_w - 1, dst_w).astype(np.int32)
        return frame[row_idx][:, col_idx]

    def _encode_frame(self) -> np.ndarray:
        frame = self.env._render_fast_rgb(self.image_size)

        if self.grayscale:
            frame = np.mean(frame, axis=2, keepdims=True).astype(np.uint8)

        if self.channel_first:
            frame = np.transpose(frame, (2, 0, 1))

        return np.ascontiguousarray(frame, dtype=np.uint8)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        info = dict(info)
        info["obs_dict"] = self.env._compute_observation()
        return self._encode_frame(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["obs_dict"] = self.env._compute_observation()
        return self._encode_frame(), reward, terminated, truncated, info


def make_vessel_env(flatten_observation: bool = True, **kwargs):
    """
    Create the vessel environment.

    Args:
      flatten_observation: True returns a flat Box observation for RL libraries.
      **kwargs: forwarded to VesselContinuousEnv.
    """
    env = VesselContinuousEnv(**kwargs)
    if flatten_observation:
        return VesselContinuousRLWrapper(env)
    return env


def make_vessel_image_env(
    image_size: Tuple[int, int] = (84, 84),
    grayscale: bool = False,
    channel_first: bool = True,
    **kwargs,
):
    """
    Create a pixel-observation vessel environment for visual RL.

    Args:
      image_size: output image size as (height, width).
      grayscale: True returns a single-channel image.
      channel_first: True returns (C, H, W), otherwise (H, W, C).
      **kwargs: forwarded to VesselContinuousEnv.
    """
    env = VesselContinuousEnv(**kwargs)
    return VesselImageObservationWrapper(
        env,
        image_size=image_size,
        grayscale=grayscale,
        channel_first=channel_first,
    )

if __name__ == "__main__":
    env = VesselContinuousEnv(render_mode="human", seed=0, clot_event=True)
    obs, info = env.reset()
    print("Initial observation keys:", obs.keys())
    print("Initial state:", env.render())
    env.run_interactive()
    env.close()
