"""Shared VLA renderer for PPO dataset collection and OpenVLA evaluation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# A 1:1 physical radius visually merges tightly spaced parallel branches in
# layout 02. This scale preserves their separation while keeping sampled
# endpoint centres visibly inside the rendered lumen.
VESSEL_RENDER_RADIUS_SCALE = 0.65


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _build_static_layout_background_numpy(env, image_size: int, background: str) -> np.ndarray:
    bg_color = _hex_rgb("#09101c") if background.lower() in {"black", "k"} else (255, 255, 255)
    frame = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)
    image_shape = frame.shape[:2]

    for points_tensor, radii_tensor in zip(env.branch_points, env.branch_radii):
        points = points_tensor.detach().cpu().numpy()
        radii = radii_tensor.detach().cpu().numpy()
        for color, radius_offset in (
            (_hex_rgb("#6d1731"), 0.0),
            (_hex_rgb("#f8c9d2"), -float(env.agent_radius)),
        ):
            for index in range(len(points) - 1):
                mean_radius = 0.5 * (float(radii[index]) + float(radii[index + 1]))
                radius = max((mean_radius + radius_offset) * VESSEL_RENDER_RADIUS_SCALE, 0.0)
                radius_px = max(env._radius_to_pixels_np(radius, image_shape), 1.0)
                start_px = env._world_to_pixel_np(points[index], image_shape)
                end_px = env._world_to_pixel_np(points[index + 1], image_shape)
                env._paint_capsule_np(frame, start_px, end_px, radius_px, color)

    return frame


def _build_static_layout_background_legacy_vla(env, image_size: int, background: str) -> np.ndarray:
    image_shape = (image_size, image_size)
    bg_color = _hex_rgb("#09101c") if background.lower() in {"black", "k"} else (255, 255, 255)
    frame = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)
    wall_color = _hex_rgb("#d0d0d0")
    lumen_color = _hex_rgb("#f7f7f7")
    for points_tensor, radii_tensor in zip(env.branch_points, env.branch_radii):
        points = points_tensor.detach().cpu().numpy()
        radii = radii_tensor.detach().cpu().numpy()
        for index in range(len(points) - 1):
            start_px = env._world_to_pixel_np(points[index], image_shape)
            end_px = env._world_to_pixel_np(points[index + 1], image_shape)
            mean_radius = 0.5 * (float(radii[index]) + float(radii[index + 1]))
            lumen_radius = max(
                env._radius_to_pixels_np(mean_radius - env.agent_radius, image_shape), 1.0
            )
            wall_radius = max(
                env._radius_to_pixels_np(mean_radius, image_shape), lumen_radius + 1.0
            )
            env._paint_capsule_np(frame, start_px, end_px, wall_radius, wall_color)
            env._paint_capsule_np(frame, start_px, end_px, lumen_radius, lumen_color)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _build_static_layout_background_matplotlib(env, image_size: int, background: str) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100.0
    bg_color = "#09101c" if background.lower() in {"black", "k"} else "white"
    figure = Figure(
        figsize=(image_size / dpi, image_size / dpi),
        dpi=dpi,
        facecolor=bg_color,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=bg_color)

    for points_tensor, radii_tensor in zip(env.branch_points, env.branch_radii):
        points = points_tensor.detach().cpu().numpy()
        mean_radius = float(radii_tensor.detach().cpu().numpy().mean())
        # Matplotlib line widths are measured in points. Convert the actual
        # simulator radii to pixels first, then pixels to points. This makes
        # the visible lumen agree with the collision geometry and prevents
        # valid randomized endpoints from appearing outside a too-thin tube.
        wall_radius_px = env._radius_to_pixels_np(
            mean_radius * VESSEL_RENDER_RADIUS_SCALE, (image_size, image_size)
        )
        lumen_radius_px = env._radius_to_pixels_np(
            max(mean_radius - float(env.agent_radius), 0.0)
            * VESSEL_RENDER_RADIUS_SCALE,
            (image_size, image_size),
        )
        wall_width_pt = 2.0 * wall_radius_px * 72.0 / dpi
        lumen_width_pt = max(2.0 * lumen_radius_px * 72.0 / dpi, 1.0)
        for color, width, zorder in (
            ("#6d1731", wall_width_pt, 1),
            ("#f8c9d2", lumen_width_pt, 2),
        ):
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=width,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=zorder,
            )

    bounds = env.bounds.detach().cpu().numpy()
    axis.set_xlim(float(bounds[0]), float(bounds[1]))
    # Match TorchVesselBatchEnv._world_to_pixel_np: +world-y is visually up.
    axis.set_ylim(float(bounds[2]), float(bounds[3]))
    axis.set_aspect("equal")
    axis.set_axis_off()
    canvas.draw()
    frame = np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[..., :3], dtype=np.uint8)
    figure.clear()
    return frame


def build_static_layout_background(env, image_size: int, background: str) -> np.ndarray:
    """Render one seamless, snapshot-style vessel background for a layout."""
    backend = os.environ.get("MICROENV_VLA_RENDER_BACKEND", "matplotlib").strip().lower()
    if backend in {"legacy_vla", "legacy-vla", "legacy"}:
        return _build_static_layout_background_legacy_vla(env, image_size, background)
    if backend in {"numpy", "np", "fast"}:
        return _build_static_layout_background_numpy(env, image_size, background)
    return _build_static_layout_background_matplotlib(env, image_size, background)


def add_episode_clot(env, static_background: np.ndarray, clot: dict[str, Any]) -> np.ndarray:
    """Paint an episode clot onto a cached layout background."""
    frame = static_background.copy()
    if not bool(clot.get("on", False)):
        return frame

    image_shape = frame.shape[:2]
    branch_idx = int(clot["branch_idx"])
    start_idx = int(clot["start_idx"])
    end_idx = int(clot["end_idx"])
    radius_factor = float(clot["radius_factor"])
    render_radius_factor = 0.65 * (1.0 - radius_factor)
    points = env.branch_points[branch_idx].detach().cpu().numpy()
    radii = env.branch_radii[branch_idx].detach().cpu().numpy()
    capsules = []
    for index in range(start_idx, min(end_idx + 1, len(points) - 1)):
        start_px = env._world_to_pixel_np(points[index], image_shape)
        end_px = env._world_to_pixel_np(points[index + 1], image_shape)
        mean_radius = 0.5 * (float(radii[index]) + float(radii[index + 1]))
        clot_radius = max(
            env._radius_to_pixels_np(mean_radius * render_radius_factor, image_shape), 1.0
        )
        capsules.append((start_px, end_px, clot_radius))
    for start_px, end_px, radius in capsules:
        env._paint_capsule_np(frame, start_px, end_px, radius, _hex_rgb("df7c2d"))
    for start_px, end_px, radius in capsules:
        env._paint_capsule_np(
            frame, start_px, end_px, max(radius * 0.45, 1.0), _hex_rgb("7a3414")
        )
    return frame


def render_vla_frame(
    env,
    episode_background: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Paint the dynamic start, goal and agent markers."""
    frame = episode_background.copy()
    image_shape = frame.shape[:2]
    marker_radius = max(env._radius_to_pixels_np(env.agent_radius * 1.15, image_shape), 2.0)
    agent_radius = max(env._radius_to_pixels_np(env.agent_radius, image_shape), 2.0)
    backend = os.environ.get("MICROENV_VLA_RENDER_BACKEND", "matplotlib").strip().lower()
    if backend in {"legacy_vla", "legacy-vla", "legacy"}:
        marker_radius = max(env._radius_to_pixels_np(env.agent_radius * 1.25, image_shape), 2.0)
        start_color = _hex_rgb("#2e8b57")
        goal_color = _hex_rgb("#d4a017")
        agent_color = _hex_rgb("#1261a0")
    else:
        start_color = _hex_rgb("#39d98a")
        goal_color = _hex_rgb("#ff5e76")
        agent_color = _hex_rgb("#4db8ff")
    env._paint_disk_np(frame, env._world_to_pixel_np(start, image_shape), marker_radius, start_color)
    env._paint_disk_np(frame, env._world_to_pixel_np(goal, image_shape), marker_radius, goal_color)
    agent_center = env._world_to_pixel_np(position, image_shape)
    env._paint_disk_np(frame, agent_center, agent_radius + 1.0, _hex_rgb("082032"))
    env._paint_disk_np(frame, agent_center, agent_radius, agent_color)
    return frame


def _scalar(value, env_id: int):
    item = value[env_id]
    return item.detach().cpu().item() if hasattr(item, "detach") else item


class VLAFrameRenderer:
    """Stateful renderer that caches static layout and episode clot layers."""

    def __init__(self, env, image_size: int, background: str):
        self.env = env
        self.image_size = int(image_size)
        self.background = background
        self.static_background = build_static_layout_background(env, self.image_size, background)
        self._clot_signature = None
        self._episode_background = self.static_background

    def _clot_state(self, env_id: int) -> dict[str, Any]:
        on = bool(_scalar(self.env.clot_on, env_id)) if hasattr(self.env, "clot_on") else False
        return {
            "on": on,
            "branch_idx": int(_scalar(self.env.clot_branch_idx, env_id)) if on else -1,
            "start_idx": int(_scalar(self.env.clot_start_idx, env_id)) if on else -1,
            "end_idx": int(_scalar(self.env.clot_end_idx, env_id)) if on else -1,
            "radius_factor": float(self.env.clot_radius_factor),
        }

    def render(self, env_id: int = 0) -> np.ndarray:
        clot = self._clot_state(env_id)
        signature = tuple(clot.values())
        if signature != self._clot_signature:
            self._episode_background = add_episode_clot(self.env, self.static_background, clot)
            self._clot_signature = signature
        start = self.env.start[env_id].detach().cpu().numpy()
        goal = self.env.goal[env_id].detach().cpu().numpy()
        position = self.env.pos[env_id].detach().cpu().numpy()
        return render_vla_frame(self.env, self._episode_background, start, goal, position)


def render_vla_env_frame(env, env_id: int, image_size: int, background: str) -> np.ndarray:
    """Render an env frame using a renderer cached by size/background on the env."""
    cache = getattr(env, "_vla_frame_renderer_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_vla_frame_renderer_cache", cache)
    key = (int(image_size), background.lower())
    if key not in cache:
        cache[key] = VLAFrameRenderer(env, image_size, background)
    return cache[key].render(env_id)
