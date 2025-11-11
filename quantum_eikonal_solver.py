#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optical Neuromorphic Eikonal Solver
===================================

This module implements a fully functional optical/quantum-inspired solver for the
Eikonal equation using the existing photonic processor architecture. The GPU is
tasked with rendering successive frames of a texture grid; each frame encodes the
state of a neuromorphic medium whose evolution solves a fast-marching / fast-sweeping
problem. The solution emerges from the diffusion of travel-times across the grid,
while direction states (four spin-like orientations) guide photonic propagation.

Key features
------------
- Real-time propagation of geodesic distance fields via ModernGL fragment shaders.
- Four-state directional memory per cell, updated neuromorphically with decay.
- Interactive manipulation: place sources, targets, obstacles, and variable media.
- On-the-fly optimal path extraction through gradient descent on the GPU field.
- Optional CPU validation using a Dijkstra-based reference solver for accuracy.

Usage
-----
    python quantum_eikonal_solver.py

Controls
--------
Left click                : Set target
Shift + Left click        : Set source
Right click               : Paint obstacle (opaque medium)
Shift + Right click       : Erase obstacle
Mouse wheel               : Adjust brush radius
Space                     : Toggle simulation run/pause
R                         : Reset travel-time field (preserve directional memory)
V                         : Toggle CPU validation (prints mean absolute error)
1                         : Paint slow medium (speed = 0.4)
2                         : Paint fast medium (speed = 1.8)
C                         : Clear media modifiers (restore unit speed)
M                         : Generate random maze (auto source/target)
N                         : Generate synthetic city grid (auto source/target)
On-screen panel           : Clickable controls for all the above actions
ESC                       : Exit

The window title displays the current frame rate, propagation iteration,
path length (if available), and the last CPU validation error when enabled.
"""

from __future__ import annotations

import math
import time
import heapq
from typing import Any, Dict, List, Optional, Tuple, cast

import glfw
import moderngl
import numpy as np


DEFAULT_GRID_SIZE = 256
WINDOW_SCALE = 3
LARGE_TIME_VALUE = 1.0e6
UI_PANEL_WIDTH = 280
UI_PADDING = 16
UI_BUTTON_HEIGHT = 40
TEXT_CELL_SIZE = 6

GLYPH_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "A": (
        "010",
        "101",
        "111",
        "101",
        "101",
    ),
    "B": (
        "110",
        "101",
        "110",
        "101",
        "110",
    ),
    "C": (
        "011",
        "100",
        "100",
        "100",
        "011",
    ),
    "D": (
        "110",
        "101",
        "101",
        "101",
        "110",
    ),
    "E": (
        "111",
        "100",
        "111",
        "100",
        "111",
    ),
    "F": (
        "111",
        "100",
        "111",
        "100",
        "100",
    ),
    "G": (
        "011",
        "100",
        "101",
        "101",
        "011",
    ),
    "H": (
        "101",
        "101",
        "111",
        "101",
        "101",
    ),
    "I": (
        "111",
        "010",
        "010",
        "010",
        "111",
    ),
    "L": (
        "100",
        "100",
        "100",
        "100",
        "111",
    ),
    "M": (
        "10101",
        "11111",
        "10101",
        "10101",
        "10101",
    ),
    "N": (
        "101",
        "111",
        "111",
        "111",
        "101",
    ),
    "O": (
        "010",
        "101",
        "101",
        "101",
        "010",
    ),
    "P": (
        "110",
        "101",
        "110",
        "100",
        "100",
    ),
    "R": (
        "110",
        "101",
        "110",
        "101",
        "101",
    ),
    "S": (
        "011",
        "100",
        "010",
        "001",
        "110",
    ),
    "T": (
        "111",
        "010",
        "010",
        "010",
        "010",
    ),
    "U": (
        "101",
        "101",
        "101",
        "101",
        "111",
    ),
    "V": (
        "101",
        "101",
        "101",
        "101",
        "010",
    ),
    "Y": (
        "101",
        "101",
        "010",
        "010",
        "010",
    ),
    "Z": (
        "111",
        "001",
        "010",
        "100",
        "111",
    ),
    "0": (
        "111",
        "101",
        "101",
        "101",
        "111",
    ),
    "1": (
        "010",
        "110",
        "010",
        "010",
        "111",
    ),
    "2": (
        "111",
        "001",
        "111",
        "100",
        "111",
    ),
    "3": (
        "111",
        "001",
        "111",
        "001",
        "111",
    ),
    "4": (
        "101",
        "101",
        "111",
        "001",
        "001",
    ),
    "5": (
        "111",
        "100",
        "111",
        "001",
        "111",
    ),
    "6": (
        "111",
        "100",
        "111",
        "101",
        "111",
    ),
    "7": (
        "111",
        "001",
        "001",
        "001",
        "001",
    ),
    "8": (
        "111",
        "101",
        "111",
        "101",
        "111",
    ),
    "9": (
        "111",
        "101",
        "111",
        "001",
        "111",
    ),
    ":": (
        "0",
        "1",
        "0",
        "1",
        "0",
    ),
    " ": (
        "0",
        "0",
        "0",
        "0",
        "0",
    ),
    ".": (
        "0",
        "0",
        "0",
        "0",
        "1",
    ),
    "-": (
        "000",
        "000",
        "111",
        "000",
        "000",
    ),
}


def grid_to_clip(x: int, y: int, size: int) -> Tuple[float, float]:
    """Convert grid coordinates to clip-space coordinates (-1, 1)."""
    fx = (x / (size - 1)) * 2.0 - 1.0
    fy = (y / (size - 1)) * 2.0 - 1.0
    return fx, fy


class OpticalEikonalSolver:
    """Interactive optical neuromorphic solver for the Eikonal equation."""

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        window_scale: int = WINDOW_SCALE,
        iterations_per_frame: int = 4,
    ):
        self.grid_size = grid_size
        self.window_width = grid_size * window_scale
        self.window_height = grid_size * window_scale
        self.iterations_per_frame = iterations_per_frame

        self.huge_time = LARGE_TIME_VALUE
        self.relaxation = 0.95
        self.memory_mix = 0.08
        self.validate_cpu = True
        self.validation_interval = 2.5
        self.last_validation_error: Optional[float] = None
        self.last_validation_time = 0.0

        self.is_running = True
        self.iteration_counter = 0
        self.rng = np.random.default_rng()

        self.source_pos = (grid_size // 8, grid_size // 2)
        self.target_pos = (grid_size - grid_size // 8, grid_size // 2)
        self.brush_radius = 5

        self.speed_field = np.ones((grid_size, grid_size), dtype=np.float32)
        self.obstacle_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.source_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.target_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.source_field[self.source_pos[1], self.source_pos[0]] = 1.0
        self.target_field[self.target_pos[1], self.target_pos[0]] = 1.0

        self.path_points: List[Tuple[int, int]] = []
        self.path_cost: Optional[float] = None
        self.path_length: Optional[float] = None
        self.path_vbo: Optional[moderngl.Buffer] = None
        self.path_vao: Optional[moderngl.VertexArray] = None

        self.ui_buttons: List[Dict[str, Any]] = []
        self.panel_rect: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

        self._init_glfw()
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._create_quad()
        self._create_textures()
        self._create_programs()
        self._create_path_pipeline()

        self.current_buffer_index = 0
        self.time_range = (0.0, 1.0)

        self._upload_static_textures()
        self._reset_time_field(preserve_state=False)
        self._update_path(force_cpu_validation=True)

    def _screen_to_clip(self, x: float, y: float) -> Tuple[float, float]:
        cx = (x / self.window_width) * 2.0 - 1.0
        cy = 1.0 - (y / self.window_height) * 2.0
        return cx, cy

    def _add_ui_quad(
        self,
        vertices: List[float],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Tuple[float, float, float, float],
    ) -> None:
        cx0, cy0 = self._screen_to_clip(x0, y0)
        cx1, cy1 = self._screen_to_clip(x1, y1)
        r, g, b, a = color
        vertices.extend(
            [
                cx0,
                cy0,
                r,
                g,
                b,
                a,
                cx1,
                cy0,
                r,
                g,
                b,
                a,
                cx1,
                cy1,
                r,
                g,
                b,
                a,
                cx0,
                cy0,
                r,
                g,
                b,
                a,
                cx1,
                cy1,
                r,
                g,
                b,
                a,
                cx0,
                cy1,
                r,
                g,
                b,
                a,
            ]
        )

    def _build_text_vertices(
        self,
        text: str,
        x: float,
        y: float,
        scale: float,
        color: Tuple[float, float, float, float],
    ) -> List[float]:
        verts: List[float] = []
        cursor_x = x
        cell = TEXT_CELL_SIZE * scale

        for ch in text.upper():
            pattern = GLYPH_PATTERNS.get(ch)
            if not pattern:
                cursor_x += cell * 4
                continue

            cols = len(pattern[0])
            for row_idx, row in enumerate(pattern):
                for col_idx, value in enumerate(row):
                    if value == "1":
                        x0 = cursor_x + col_idx * cell
                        y0 = y + row_idx * cell
                        x1 = x0 + cell
                        y1 = y0 + cell
                        self._add_ui_quad(verts, x0, y0, x1, y1, color)

            cursor_x += (cols + 1) * cell

        return verts

    def _point_in_rect(self, x: float, y: float, rect: Tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    @staticmethod
    def _set_uniform(program: moderngl.Program, name: str, value: Any) -> None:
        try:
            uniform = program[name]
        except KeyError:
            return
        cast(Any, uniform).value = value

    # ------------------------------------------------------------------ GLFW --
    def _init_glfw(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(
            self.window_width,
            self.window_height,
            "Optical Neuromorphic Eikonal Solver",
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        self.cursor_pos = (0.0, 0.0)

    # ------------------------------------------------------------- GPU Setup --
    def _create_quad(self) -> None:
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
            ],
            dtype="f4",
        )
        self.quad_buffer = self.ctx.buffer(quad_vertices.tobytes())

    def _create_textures(self) -> None:
        size = (self.grid_size, self.grid_size)
        self.time_textures = []
        self.state_textures = []
        self.framebuffers = []

        for _ in range(2):
            time_tex = self.ctx.texture(size, 1, dtype="f4")
            time_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            time_tex.repeat_x = False
            time_tex.repeat_y = False

            state_tex = self.ctx.texture(size, 4, dtype="f4")
            state_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            state_tex.repeat_x = False
            state_tex.repeat_y = False

            fbo = self.ctx.framebuffer(color_attachments=[time_tex, state_tex])

            self.time_textures.append(time_tex)
            self.state_textures.append(state_tex)
            self.framebuffers.append(fbo)

        self.speed_texture = self.ctx.texture(size, 1, dtype="f4")
        self.speed_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.source_texture = self.ctx.texture(size, 1, dtype="f4")
        self.source_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.target_texture = self.ctx.texture(size, 1, dtype="f4")
        self.target_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.obstacle_texture = self.ctx.texture(size, 1, dtype="f4")
        self.obstacle_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _create_programs(self) -> None:
        quad_vs = """
        #version 430 core
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            v_uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """

        propagation_fs = """
        #version 430 core

        in vec2 v_uv;
        layout(location = 0) out float out_time;
        layout(location = 1) out vec4 out_state;

        uniform sampler2D u_time_tex;
        uniform sampler2D u_state_tex;
        uniform sampler2D u_speed_tex;
        uniform sampler2D u_source_tex;
        uniform sampler2D u_obstacle_tex;

        uniform vec2 u_resolution;
        uniform float u_relaxation;
        uniform float u_memory_mix;
        uniform float u_huge_time;

        float sample_time(vec2 offset) {
            vec2 texel = 1.0 / u_resolution;
            vec2 clamped = clamp(v_uv + offset * texel, texel * 0.5, 1.0 - texel * 0.5);
            return texture(u_time_tex, clamped).r;
        }

        void main() {
            float speed = texture(u_speed_tex, v_uv).r;
            float source = texture(u_source_tex, v_uv).r;
            float obstacle = texture(u_obstacle_tex, v_uv).r;
            float current = texture(u_time_tex, v_uv).r;
            vec4 prev_state = texture(u_state_tex, v_uv);

            if (source > 0.5) {
                out_time = 0.0;
                out_state = vec4(0.25, 0.25, 0.25, 0.25);
                return;
            }

            if (obstacle > 0.5 || speed <= 1e-6) {
                out_time = u_huge_time;
                out_state = mix(prev_state, vec4(0.0), u_memory_mix);
                return;
            }

            float t_left = sample_time(vec2(-1.0, 0.0));
            float t_right = sample_time(vec2(1.0, 0.0));
            float t_down = sample_time(vec2(0.0, -1.0));
            float t_up = sample_time(vec2(0.0, 1.0));

            float a = min(t_left, t_right);
            float b = min(t_down, t_up);
            float inv_speed = 1.0 / max(speed, 1e-4);

            float candidate;
            float diff = abs(a - b);
            if (diff >= inv_speed) {
                candidate = min(a, b) + inv_speed;
            } else {
                float rad = max(0.0, 2.0 * inv_speed * inv_speed - diff * diff);
                candidate = 0.5 * (a + b + sqrt(rad));
            }

            candidate = min(candidate, min(min(t_left, t_right), min(t_down, t_up)) + inv_speed);
            candidate = min(candidate, current);

            float relaxed = mix(current, candidate, u_relaxation);
            out_time = relaxed;

            float north = max(0.0, t_up - relaxed);
            float east  = max(0.0, t_right - relaxed);
            float south = max(0.0, t_down - relaxed);
            float west  = max(0.0, t_left - relaxed);
            vec4 flow = vec4(north, east, south, west);
            float total = max(dot(flow, vec4(1.0)), 1e-6);
            flow /= total;

            vec4 blended = mix(prev_state, flow, u_memory_mix);
            blended = max(blended, vec4(0.0));
            float blend_sum = max(dot(blended, vec4(1.0)), 1e-6);
            out_state = blended / blend_sum;
        }
        """

        display_fs = """
        #version 430 core

        in vec2 v_uv;
        out vec4 fragColor;

        uniform sampler2D u_time_tex;
        uniform sampler2D u_state_tex;
        uniform sampler2D u_speed_tex;
        uniform sampler2D u_source_tex;
        uniform sampler2D u_target_tex;
        uniform sampler2D u_obstacle_tex;
        uniform vec2 u_time_range;

        vec3 colormap(float t) {
            t = clamp(t, 0.0, 1.0);
            vec3 c1 = vec3(0.06, 0.09, 0.18);
            vec3 c2 = vec3(0.18, 0.34, 0.58);
            vec3 c3 = vec3(0.32, 0.72, 0.78);
            vec3 c4 = vec3(0.94, 0.95, 0.92);
            if (t < 0.33) {
                float k = smoothstep(0.0, 0.33, t);
                return mix(c1, c2, k);
            } else if (t < 0.66) {
                float k = smoothstep(0.33, 0.66, t);
                return mix(c2, c3, k);
            } else {
                float k = smoothstep(0.66, 1.0, t);
                return mix(c3, c4, k);
            }
        }

        void main() {
            float time_val = texture(u_time_tex, v_uv).r;
            vec4 state = texture(u_state_tex, v_uv);
            float speed = texture(u_speed_tex, v_uv).r;
            float source = texture(u_source_tex, v_uv).r;
            float target = texture(u_target_tex, v_uv).r;
            float obstacle = texture(u_obstacle_tex, v_uv).r;

            float min_t = u_time_range.x;
            float max_t = u_time_range.y;
            float norm_t = (time_val - min_t) / max(max_t - min_t, 1e-6);
            vec3 base = colormap(norm_t);

            float anisotropy = max(max(state.x, state.y), max(state.z, state.w)) -
                               min(min(state.x, state.y), min(state.z, state.w));
            base += anisotropy * vec3(0.10, 0.15, 0.10);

            if (speed > 1.05) {
                base = mix(base, vec3(0.28, 0.85, 0.60), 0.25);
            } else if (speed < 0.8) {
                base = mix(base, vec3(0.75, 0.35, 0.25), 0.25);
            }

            if (obstacle > 0.5) {
                base *= 0.1;
            }

            if (source > 0.5) {
                base = vec3(1.0, 0.2, 0.2);
            }

            if (target > 0.5) {
                base = mix(base, vec3(0.2, 0.95, 0.2), 0.7);
            }

            fragColor = vec4(base, 1.0);
        }
        """

        path_vs = """
        #version 430 core
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """

        path_fs = """
        #version 430 core
        out vec4 fragColor;
        uniform vec3 u_path_color;
        void main() {
            fragColor = vec4(u_path_color, 1.0);
        }
        """

        self.propagation_program = self.ctx.program(
            vertex_shader=quad_vs, fragment_shader=propagation_fs
        )
        self.display_program = self.ctx.program(
            vertex_shader=quad_vs, fragment_shader=display_fs
        )
        self.path_program = self.ctx.program(
            vertex_shader=path_vs, fragment_shader=path_fs
        )
        ui_vs = """
        #version 430 core
        layout(location = 0) in vec2 in_pos;
        layout(location = 1) in vec4 in_color;
        out vec4 v_color;
        void main() {
            v_color = in_color;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """

        ui_fs = """
        #version 430 core
        in vec4 v_color;
        out vec4 fragColor;
        void main() {
            fragColor = v_color;
        }
        """

        self.ui_program = self.ctx.program(vertex_shader=ui_vs, fragment_shader=ui_fs)

        self.propagation_vao = self.ctx.simple_vertex_array(
            self.propagation_program, self.quad_buffer, "in_pos"
        )
        self.display_vao = self.ctx.simple_vertex_array(
            self.display_program, self.quad_buffer, "in_pos"
        )

    def _create_path_pipeline(self) -> None:
        self._set_uniform(self.path_program, "u_path_color", (0.2, 1.0, 0.2))
        self.path_vao = None

    # ------------------------------------------------------------- Utilities --
    def _upload_static_textures(self) -> None:
        self.speed_texture.write(self.speed_field.tobytes())
        self.source_texture.write(self.source_field.tobytes())
        self.target_texture.write(self.target_field.tobytes())
        self.obstacle_texture.write(self.obstacle_field.tobytes())

        initial_state = np.full(
            (self.grid_size, self.grid_size, 4), 0.25, dtype=np.float32
        )
        state_bytes = initial_state.tobytes()
        for tex in self.state_textures:
            tex.write(state_bytes)

    def _reset_time_field(self, preserve_state: bool = True) -> None:
        base = np.full((self.grid_size, self.grid_size), self.huge_time, dtype=np.float32)
        base[self.source_pos[1], self.source_pos[0]] = 0.0
        data = base.tobytes()
        for tex in self.time_textures:
            tex.write(data)

        if not preserve_state:
            initial_state = np.full(
                (self.grid_size, self.grid_size, 4), 0.25, dtype=np.float32
            )
            state_bytes = initial_state.tobytes()
            for tex in self.state_textures:
                tex.write(state_bytes)

        self.iteration_counter = 0

    def _toggle_running(self) -> None:
        self.is_running = not self.is_running

    def _screen_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.clip(x / self.window_width * self.grid_size, 0, self.grid_size - 1))
        gy = int(
            np.clip(
                (self.window_height - y) / self.window_height * self.grid_size,
                0,
                self.grid_size - 1,
            )
        )
        return gx, gy

    # -------------------------------------------------------------- Callbacks --
    def _cursor_callback(self, window, xpos: float, ypos: float) -> None:
        self.cursor_pos = (xpos, ypos)

    def _mouse_button_callback(self, window, button, action, mods) -> None:
        if action != glfw.PRESS:
            return

        if self._handle_ui_click(button, mods):
            return

        grid_pos = self._screen_to_grid(*self.cursor_pos)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if mods & glfw.MOD_SHIFT:
                self._set_source(grid_pos)
            else:
                self._set_target(grid_pos)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            erase = bool(mods & glfw.MOD_SHIFT)
            if mods & glfw.MOD_CONTROL:
                self._paint_medium(grid_pos, slow=True, erase=erase)
            elif mods & glfw.MOD_ALT:
                self._paint_medium(grid_pos, slow=False, erase=erase)
            else:
                self._paint_obstacle(grid_pos, erase=erase)

    def _key_callback(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_SPACE:
            self._toggle_running()
        elif key == glfw.KEY_R:
            self._reset_time_field(preserve_state=True)
        elif key == glfw.KEY_V:
            self.validate_cpu = not self.validate_cpu
        elif key == glfw.KEY_C:
            self._clear_media()
        elif key == glfw.KEY_1:
            self._paint_medium(self._screen_to_grid(*self.cursor_pos), slow=True, erase=False)
        elif key == glfw.KEY_2:
            self._paint_medium(self._screen_to_grid(*self.cursor_pos), slow=False, erase=False)
        elif key == glfw.KEY_M:
            self._generate_random_maze()
        elif key == glfw.KEY_N:
            self._generate_city_network()

    def _scroll_callback(self, window, xoffset, yoffset) -> None:
        self.brush_radius = int(np.clip(self.brush_radius + yoffset, 1, 25))

    # ------------------------------------------------------------- Modifiers --
    def _set_source(self, pos: Tuple[int, int]) -> None:
        self.source_field.fill(0.0)
        sx, sy = pos
        self.source_field[sy, sx] = 1.0
        self.source_pos = pos
        self.source_texture.write(self.source_field.tobytes())
        self._reset_time_field(preserve_state=True)
        self._update_path(force_cpu_validation=True)

    def _set_target(self, pos: Tuple[int, int]) -> None:
        self.target_field.fill(0.0)
        tx, ty = pos
        self.target_field[ty, tx] = 1.0
        self.target_pos = pos
        self.target_texture.write(self.target_field.tobytes())
        self._update_path(force_cpu_validation=True)

    def _paint_obstacle(self, center: Tuple[int, int], erase: bool) -> None:
        cx, cy = center
        rr = self.brush_radius
        y_indices = slice(max(0, cy - rr), min(self.grid_size, cy + rr + 1))
        x_indices = slice(max(0, cx - rr), min(self.grid_size, cx + rr + 1))
        sub = self.obstacle_field[y_indices, x_indices]

        yy = np.arange(y_indices.start, y_indices.stop)[:, None]
        xx = np.arange(x_indices.start, x_indices.stop)[None, :]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr

        if erase:
            sub[mask] = 0.0
        else:
            sub[mask] = 1.0

        self.obstacle_texture.write(self.obstacle_field.tobytes())
        self._update_path(force_cpu_validation=False)

    def _paint_medium(self, center: Tuple[int, int], slow: bool, erase: bool) -> None:
        cx, cy = center
        rr = self.brush_radius
        y_indices = slice(max(0, cy - rr), min(self.grid_size, cy + rr + 1))
        x_indices = slice(max(0, cx - rr), min(self.grid_size, cx + rr + 1))
        sub = self.speed_field[y_indices, x_indices]
        yy = np.arange(y_indices.start, y_indices.stop)[:, None]
        xx = np.arange(x_indices.start, x_indices.stop)[None, :]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr

        if erase:
            sub[mask] = 1.0
        else:
            sub[mask] = 0.4 if slow else 1.8

        self.speed_texture.write(self.speed_field.tobytes())
        self._update_path(force_cpu_validation=True)

    def _clear_media(self) -> None:
        self.obstacle_field.fill(0.0)
        self.speed_field.fill(1.0)
        self.obstacle_texture.write(self.obstacle_field.tobytes())
        self.speed_texture.write(self.speed_field.tobytes())
        self._reset_time_field(preserve_state=True)
        self._update_path(force_cpu_validation=True)

    def _assign_endpoints(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int],
        preserve_state: bool = False,
    ) -> None:
        sx, sy = source
        tx, ty = target
        sx = int(np.clip(sx, 0, self.grid_size - 1))
        sy = int(np.clip(sy, 0, self.grid_size - 1))
        tx = int(np.clip(tx, 0, self.grid_size - 1))
        ty = int(np.clip(ty, 0, self.grid_size - 1))

        self.source_field.fill(0.0)
        self.target_field.fill(0.0)
        self.source_field[sy, sx] = 1.0
        self.target_field[ty, tx] = 1.0

        self.source_pos = (sx, sy)
        self.target_pos = (tx, ty)

        self.source_texture.write(self.source_field.tobytes())
        self.target_texture.write(self.target_field.tobytes())

        self._reset_time_field(preserve_state=preserve_state)
        self._update_path(force_cpu_validation=True)

    def _find_walkable_near(self, start: Tuple[int, int]) -> Tuple[int, int]:
        x0 = int(np.clip(start[0], 0, self.grid_size - 1))
        y0 = int(np.clip(start[1], 0, self.grid_size - 1))

        if self.obstacle_field[y0, x0] < 0.5:
            return (x0, y0)

        for radius in range(1, self.grid_size):
            xmin = max(0, x0 - radius)
            xmax = min(self.grid_size - 1, x0 + radius)
            ymin = max(0, y0 - radius)
            ymax = min(self.grid_size - 1, y0 + radius)
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    if self.obstacle_field[y, x] < 0.5:
                        return (x, y)

        return (x0, y0)

    def _generate_random_maze(self) -> None:
        cell_scale = max(2, self.grid_size // 64)
        cells = self.grid_size // cell_scale
        if cells < 5:
            cells = 5
        if cells % 2 == 0:
            cells -= 1

        maze = np.ones((cells, cells), dtype=np.uint8)
        visited = np.zeros_like(maze, dtype=bool)
        stack: List[Tuple[int, int]] = []
        start = (1, 1)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        maze[start[1], start[0]] = 0
        visited[start[1], start[0]] = True
        stack.append(start)

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in dirs:
                nx = x + 2 * dx
                ny = y + 2 * dy
                if 0 < nx < cells - 1 and 0 < ny < cells - 1 and not visited[ny, nx]:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                idx = int(self.rng.integers(len(neighbors)))
                nx, ny, dx, dy = neighbors[idx]
                maze[y + dy, x + dx] = 0
                maze[ny, nx] = 0
                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        maze[cells - 2, cells - 2] = 0

        self.obstacle_field.fill(1.0)
        self.speed_field.fill(0.3)

        for y in range(cells):
            y0 = y * cell_scale
            y1 = min(self.grid_size, y0 + cell_scale)
            for x in range(cells):
                x0 = x * cell_scale
                x1 = min(self.grid_size, x0 + cell_scale)
                if maze[y, x] == 0:
                    self.obstacle_field[y0:y1, x0:x1] = 0.0
                    self.speed_field[y0:y1, x0:x1] = 1.2

        self.obstacle_texture.write(self.obstacle_field.tobytes())
        self.speed_texture.write(self.speed_field.tobytes())

        def coarse_to_grid(cx: int, cy: int) -> Tuple[int, int]:
            x0 = cx * cell_scale + cell_scale // 2
            y0 = cy * cell_scale + cell_scale // 2
            return (
                int(np.clip(x0, 0, self.grid_size - 1)),
                int(np.clip(y0, 0, self.grid_size - 1)),
            )

        start_px = self._find_walkable_near(coarse_to_grid(1, 1))
        goal_px = self._find_walkable_near(coarse_to_grid(cells - 2, cells - 2))

        print(
            f"[Maze] Generated random maze with {cells} coarse cells (scale={cell_scale})."
        )
        self._assign_endpoints(start_px, goal_px, preserve_state=False)

    def _generate_city_network(self) -> None:
        size = self.grid_size
        self.obstacle_field.fill(0.0)
        self.speed_field.fill(0.6)

        block = max(12, size // 12)
        road_width = max(2, block // 4)

        for x in range(0, size, block):
            x1 = min(size, x + road_width)
            self.speed_field[:, x:x1] = 1.6

        for y in range(0, size, block):
            y1 = min(size, y + road_width)
            self.speed_field[y:y1, :] = 1.6

        for _ in range(3):
            x0 = int(self.rng.integers(0, size))
            y0 = int(self.rng.integers(0, size))
            x1 = int(self.rng.integers(0, size))
            y1 = int(self.rng.integers(0, size))
            num = max(2, int(np.hypot(x1 - x0, y1 - y0)))
            xs = np.linspace(x0, x1, num=num, dtype=int)
            ys = np.linspace(y0, y1, num=num, dtype=int)
            half = max(1, road_width // 2)
            for x, y in zip(xs, ys):
                xmin = max(0, x - half)
                xmax = min(size, x + half)
                ymin = max(0, y - half)
                ymax = min(size, y + half)
                self.speed_field[ymin:ymax, xmin:xmax] = 1.9

        building_attempts = size // 3
        for _ in range(building_attempts):
            w = int(self.rng.integers(max(road_width * 2, 4), max(block, road_width * 3)))
            h = int(self.rng.integers(max(road_width * 2, 4), max(block, road_width * 3)))
            if w >= size or h >= size:
                continue
            x0 = int(self.rng.integers(0, size - w))
            y0 = int(self.rng.integers(0, size - h))
            region = self.speed_field[y0:y0 + h, x0:x0 + w]
            if np.mean(region) > 1.2:
                continue
            self.obstacle_field[y0:y0 + h, x0:x0 + w] = 1.0
            self.speed_field[y0:y0 + h, x0:x0 + w] = 0.0

        self.obstacle_texture.write(self.obstacle_field.tobytes())
        self.speed_texture.write(self.speed_field.tobytes())

        source_guess = (road_width, road_width)
        target_guess = (size - road_width - 1, size - road_width - 1)
        source_px = self._find_walkable_near(source_guess)
        target_px = self._find_walkable_near(target_guess)

        print(
            f"[City] Generated synthetic city layout (block={block}, road_width={road_width})."
        )
        self._assign_endpoints(source_px, target_px, preserve_state=False)

    # ---------------------------------------------------------- Propagation --
    def _propagation_step(self) -> None:
        read_idx = self.current_buffer_index
        write_idx = 1 - read_idx

        self.time_textures[read_idx].use(location=0)
        self.state_textures[read_idx].use(location=1)
        self.speed_texture.use(location=2)
        self.source_texture.use(location=3)
        self.obstacle_texture.use(location=4)

        self._set_uniform(self.propagation_program, "u_time_tex", 0)
        self._set_uniform(self.propagation_program, "u_state_tex", 1)
        self._set_uniform(self.propagation_program, "u_speed_tex", 2)
        self._set_uniform(self.propagation_program, "u_source_tex", 3)
        self._set_uniform(self.propagation_program, "u_obstacle_tex", 4)
        self._set_uniform(
            self.propagation_program,
            "u_resolution",
            (float(self.grid_size), float(self.grid_size)),
        )
        self._set_uniform(self.propagation_program, "u_relaxation", float(self.relaxation))
        self._set_uniform(self.propagation_program, "u_memory_mix", float(self.memory_mix))
        self._set_uniform(self.propagation_program, "u_huge_time", float(self.huge_time))

        self.framebuffers[write_idx].use()
        self.propagation_vao.render(moderngl.TRIANGLES)

        self.current_buffer_index = write_idx
        self.iteration_counter += 1

    # ----------------------------------------------------------- Rendering --
    def _render_scene(self) -> None:
        self.ctx.screen.use()
        self.ctx.clear(0.03, 0.04, 0.06, 1.0)

        time_tex = self.time_textures[self.current_buffer_index]
        state_tex = self.state_textures[self.current_buffer_index]

        time_tex.use(location=0)
        state_tex.use(location=1)
        self.speed_texture.use(location=2)
        self.source_texture.use(location=3)
        self.target_texture.use(location=4)
        self.obstacle_texture.use(location=5)

        self._set_uniform(self.display_program, "u_time_tex", 0)
        self._set_uniform(self.display_program, "u_state_tex", 1)
        self._set_uniform(self.display_program, "u_speed_tex", 2)
        self._set_uniform(self.display_program, "u_source_tex", 3)
        self._set_uniform(self.display_program, "u_target_tex", 4)
        self._set_uniform(self.display_program, "u_obstacle_tex", 5)
        self._set_uniform(self.display_program, "u_time_range", self.time_range)

        self.display_vao.render(moderngl.TRIANGLES)

        if self.path_vao is not None:
            self._set_uniform(self.path_program, "u_path_color", (0.18, 0.95, 0.45))
            self.path_vao.render(mode=moderngl.TRIANGLES)

        self._render_ui()

    def _render_ui(self) -> None:
        margin = UI_PADDING
        panel_y0 = float(margin)
        panel_y1 = float(self.window_height - margin)
        available_height = max(40.0, panel_y1 - panel_y0)

        button_specs = [
            ("toggle", "PAUSE" if self.is_running else "RUN"),
            ("reset", "RESET"),
            ("maze", "MAZE"),
            ("city", "CITY"),
            ("clear", "CLEAR"),
        ]

        metrics: List[Tuple[str, str]] = []
        if self.path_length is not None:
            metrics.append(("LENGTH", f"{self.path_length:6.1f}"))
        if self.path_cost is not None:
            metrics.append(("COST", f"{self.path_cost:6.1f}"))
        if self.last_validation_error is not None:
            metrics.append(("ERROR", f"{self.last_validation_error:7.4f}"))
        metrics.append(("SOURCE", f"{self.source_pos[0]:03d} {self.source_pos[1]:03d}"))
        metrics.append(("TARGET", f"{self.target_pos[0]:03d} {self.target_pos[1]:03d}"))

        info_lines = [
            "LEFT CLICK    TARGET",
            "SHIFT + CLICK SOURCE",
            "RIGHT CLICK   OBSTACLE",
            "SCROLL        BRUSH",
            "V KEY         VALIDATE",
        ]

        n_buttons = len(button_specs)
        metrics_count = len(metrics)
        info_count = len(info_lines)

        base_total = (
            140.0
            + n_buttons * (UI_BUTTON_HEIGHT + 18.0)
            + metrics_count * 28.0
            + info_count * 26.0
            + 120.0
        )
        scale = float(np.clip(available_height / max(base_total, 1.0), 0.55, 1.2))

        for _ in range(3):
            button_height = max(32.0, UI_BUTTON_HEIGHT * scale)
            button_spacing = 16.0 * scale
            header_height = 90.0 * scale
            metrics_height = max(40.0 * scale, metrics_count * 24.0 * scale + 30.0 * scale)
            info_height = max(60.0 * scale, info_count * 20.0 * scale + 40.0 * scale)
            content_height = (
                header_height
                + n_buttons * button_height
                + max(n_buttons - 1, 0) * button_spacing
                + metrics_height
                + info_height
            )
            if content_height <= available_height:
                break
            adjust = available_height / max(content_height, 1.0)
            scale = max(0.45, scale * adjust)

        button_height = max(32.0, UI_BUTTON_HEIGHT * scale)
        button_spacing = 16.0 * scale

        def text_height(factor: float) -> float:
            return TEXT_CELL_SIZE * factor * 5.0

        panel_width = max(UI_PANEL_WIDTH, 220.0 * scale)
        panel_x1 = float(self.window_width - margin)
        panel_x0 = float(max(margin, panel_x1 - panel_width))

        self.panel_rect = (panel_x0, panel_y0, panel_x1, panel_y1)
        ui_vertices: List[float] = []

        background_color = (0.07, 0.09, 0.15, 0.94)
        border_color = (0.24, 0.30, 0.42, 0.96)
        self._add_ui_quad(ui_vertices, panel_x0, panel_y0, panel_x1, panel_y1, background_color)
        border = 2.0
        self._add_ui_quad(ui_vertices, panel_x0 - border, panel_y0 - border, panel_x1 + border, panel_y0, border_color)
        self._add_ui_quad(ui_vertices, panel_x0 - border, panel_y1, panel_x1 + border, panel_y1 + border, border_color)
        self._add_ui_quad(ui_vertices, panel_x0 - border, panel_y0, panel_x0, panel_y1, border_color)
        self._add_ui_quad(ui_vertices, panel_x1, panel_y0, panel_x1 + border, panel_y1, border_color)

        title_color = (0.85, 0.90, 0.96, 1.0)
        subtitle_color = (0.58, 0.72, 0.88, 1.0)
        text_color = (0.85, 0.90, 0.96, 1.0)
        accent_color = (0.32, 0.82, 0.92, 1.0)
        info_color = (0.58, 0.72, 0.88, 0.85)

        title_scale = max(0.7, 1.25 * scale)
        subtitle_scale = max(0.6, 0.9 * scale)
        title_y = panel_y0 + 14.0 * scale
        ui_vertices.extend(self._build_text_vertices("OPTICAL ROUTES", panel_x0 + 18.0 * scale, title_y, title_scale, title_color))

        subtitle_y = title_y + text_height(title_scale) + 8.0 * scale
        ui_vertices.extend(
            self._build_text_vertices(
                "NEUROMORPHIC SOLVER",
                panel_x0 + 18.0 * scale,
                subtitle_y,
                subtitle_scale,
                subtitle_color,
            )
        )

        button_top = subtitle_y + text_height(subtitle_scale) + 22.0 * scale
        button_left = panel_x0 + 16.0 * scale
        button_right = panel_x1 - 16.0 * scale
        cursor_x, cursor_y = self.cursor_pos
        btn_text_scale = max(0.55, 0.9 * scale)
        btn_text_height = text_height(btn_text_scale)

        self.ui_buttons = []
        for idx, (action, label) in enumerate(button_specs):
            y0 = button_top + idx * (button_height + button_spacing)
            y1 = y0 + button_height
            rect = (button_left, y0, button_right, y1)
            hover = self._point_in_rect(cursor_x, cursor_y, rect)

            base_color = (0.16, 0.22, 0.30, 0.92)
            hover_color = (0.24, 0.34, 0.46, 0.96)
            outline_color = (0.28, 0.42, 0.60, 1.0)
            fill_color = hover_color if hover else base_color

            self._add_ui_quad(ui_vertices, *rect, fill_color)
            self._add_ui_quad(ui_vertices, rect[0] - 1.2, rect[1] - 1.2, rect[2] + 1.2, rect[1], outline_color)
            self._add_ui_quad(ui_vertices, rect[0] - 1.2, rect[3], rect[2] + 1.2, rect[3] + 1.2, outline_color)
            self._add_ui_quad(ui_vertices, rect[0] - 1.2, rect[1], rect[0], rect[3], outline_color)
            self._add_ui_quad(ui_vertices, rect[2], rect[1], rect[2] + 1.2, rect[3], outline_color)

            text_x = rect[0] + 12.0 * scale
            text_y = y0 + 0.5 * (button_height - btn_text_height)
            ui_vertices.extend(self._build_text_vertices(label, text_x, text_y, btn_text_scale, text_color))

            self.ui_buttons.append({"id": action, "rect": rect})

        metrics_top = button_top + n_buttons * (button_height + button_spacing) - button_spacing + 28.0 * scale
        metric_label_scale = max(0.55, 0.85 * scale)
        metric_value_scale = max(0.55, 0.9 * scale)
        metric_line_height = text_height(metric_label_scale) + 6.0 * scale

        metric_y = metrics_top
        for label, value in metrics:
            ui_vertices.extend(
                self._build_text_vertices(label + ":", panel_x0 + 18.0 * scale, metric_y, metric_label_scale, subtitle_color)
            )
            ui_vertices.extend(
                self._build_text_vertices(value, panel_x0 + 18.0 * scale + 110.0 * scale, metric_y, metric_value_scale, accent_color)
            )
            metric_y += metric_line_height

        info_title_scale = max(0.55, 0.85 * scale)
        info_line_scale = max(0.5, 0.75 * scale)
        info_line_height = text_height(info_line_scale) + 4.0 * scale

        info_title_y = metric_y + 14.0 * scale
        ui_vertices.extend(
            self._build_text_vertices("INTERACTION", panel_x0 + 18.0 * scale, info_title_y, info_title_scale, subtitle_color)
        )

        info_y = info_title_y + text_height(info_title_scale) + 10.0 * scale
        for line in info_lines:
            ui_vertices.extend(self._build_text_vertices(line, panel_x0 + 18.0 * scale, info_y, info_line_scale, info_color))
            info_y += info_line_height
            if info_y > panel_y1 - 18.0 * scale:
                break

        if ui_vertices:
            data = np.array(ui_vertices, dtype="f4").tobytes()
            vbo = self.ctx.buffer(data)
            vao = self.ctx.vertex_array(self.ui_program, [(vbo, "2f 4f", "in_pos", "in_color")])
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()

    def _handle_ui_click(self, button: int, mods: int) -> bool:
        x, y = self.cursor_pos
        if not self._point_in_rect(x, y, self.panel_rect):
            return False

        if button != glfw.MOUSE_BUTTON_LEFT:
            return True

        for btn in self.ui_buttons:
            if self._point_in_rect(x, y, btn["rect"]):
                self._activate_button(btn["id"])
                return True
        return True

    def _activate_button(self, action: str) -> None:
        if action == "toggle":
            self._toggle_running()
        elif action == "reset":
            self._reset_time_field(preserve_state=True)
            self._update_path(force_cpu_validation=True)
        elif action == "maze":
            self._generate_random_maze()
        elif action == "city":
            self._generate_city_network()
        elif action == "clear":
            self._clear_media()

    # ----------------------------------------------------------- Analytics --
    def _read_time_field(self) -> np.ndarray:
        tex = self.time_textures[self.current_buffer_index]
        buffer = tex.read()
        field = np.frombuffer(buffer, dtype=np.float32).reshape(
            self.grid_size, self.grid_size
        )
        return field

    def _update_time_range(self, field: np.ndarray) -> None:
        mask = field < self.huge_time * 0.9
        if np.any(mask):
            min_t = float(np.min(field[mask]))
            max_t = float(np.max(field[mask]))
            if abs(max_t - min_t) < 1e-5:
                max_t = min_t + 1.0
            self.time_range = (min_t, max_t)
        else:
            self.time_range = (0.0, 1.0)

    def _extract_path_from_field(self, field: np.ndarray) -> List[Tuple[int, int]]:
        if self.target_pos is None or self.source_pos is None:
            return []

        tx, ty = self.target_pos
        sx, sy = self.source_pos
        tx = int(np.clip(tx, 0, self.grid_size - 1))
        ty = int(np.clip(ty, 0, self.grid_size - 1))
        sx = int(np.clip(sx, 0, self.grid_size - 1))
        sy = int(np.clip(sy, 0, self.grid_size - 1))

        path: List[Tuple[int, int]] = []
        current = (tx, ty)
        visited = set()
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in range(self.grid_size * self.grid_size):
            path.append(current)
            if current == (sx, sy):
                break

            visited.add(current)
            cx, cy = current
            best_neighbor = current
            best_value = field[cy, cx]

            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.obstacle_field[ny, nx] > 0.5:
                        continue
                    candidate = field[ny, nx]
                    if candidate < best_value - 1e-6:
                        best_value = candidate
                        best_neighbor = (nx, ny)

            if best_neighbor == current or best_neighbor in visited:
                break

            current = best_neighbor

        if not path or path[-1] != (sx, sy):
            return []
        return path

    def _compute_path_metrics(self, path: List[Tuple[int, int]]) -> Tuple[Optional[float], Optional[float]]:
        if len(path) < 2:
            return None, None

        total_cost = 0.0
        total_length = 0.0

        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            s0 = self.speed_field[y0, x0]
            s1 = self.speed_field[y1, x1]
            step_speed = max(0.5 * (s0 + s1), 1e-4)
            total_cost += 1.0 / step_speed
            total_length += math.hypot(x1 - x0, y1 - y0)

        return total_cost, total_length

    def _update_path(self, force_cpu_validation: bool) -> None:
        if self.target_pos is None:
            return

        field = self._read_time_field()
        self._update_time_range(field)

        path = self._extract_path_from_field(field)
        cpu_field: Optional[np.ndarray] = None

        if len(path) < 2:
            cpu_field = self._cpu_reference_solution()
            if cpu_field is not None:
                cpu_path = self._extract_path_from_field(cpu_field)
                if len(cpu_path) >= 2:
                    path = cpu_path

        self.path_points = path
        cost, length = self._compute_path_metrics(path)
        self.path_cost = cost
        self.path_length = length
        self._upload_path_buffer()

        now = time.time()
        if self.validate_cpu and (
            force_cpu_validation or now - self.last_validation_time > self.validation_interval
        ):
            if cpu_field is None:
                cpu_field = self._cpu_reference_solution()
            if cpu_field is not None:
                mask = (field < self.huge_time * 0.9) & (cpu_field < self.huge_time * 0.9)
                if np.any(mask):
                    error = float(np.mean(np.abs(cpu_field[mask] - field[mask])))
                    self.last_validation_error = error
                    self.last_validation_time = now

    def _upload_path_buffer(self) -> None:
        if not self.path_points:
            if self.path_vbo is not None:
                self.path_vbo.release()
                self.path_vbo = None
                self.path_vao = None
            return

        mesh = self._build_path_mesh(self.path_points)
        if mesh.size == 0:
            if self.path_vbo is not None:
                self.path_vbo.release()
                self.path_vbo = None
                self.path_vao = None
            return

        data = mesh.astype("f4").tobytes()

        if self.path_vbo is None:
            self.path_vbo = self.ctx.buffer(data)
        else:
            if self.path_vbo.size != len(data):
                self.path_vbo.release()
                self.path_vbo = self.ctx.buffer(data)
            else:
                self.path_vbo.write(data)

        self.path_vao = self.ctx.simple_vertex_array(self.path_program, self.path_vbo, "in_pos")

    def _build_path_mesh(self, path: List[Tuple[int, int]]) -> np.ndarray:
        if len(path) < 2:
            return np.array([], dtype=np.float32)

        thickness = max(2.5 / self.grid_size, 1.5e-2)
        vertices: List[float] = []

        for idx in range(len(path) - 1):
            x0, y0 = path[idx]
            x1, y1 = path[idx + 1]
            cx0, cy0 = grid_to_clip(x0, y0, self.grid_size)
            cx1, cy1 = grid_to_clip(x1, y1, self.grid_size)

            dx = cx1 - cx0
            dy = cy1 - cy0
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0.0:
                continue

            nx = -dy / length
            ny = dx / length

            ox = nx * thickness
            oy = ny * thickness

            v0 = (cx0 + ox, cy0 + oy)
            v1 = (cx0 - ox, cy0 - oy)
            v2 = (cx1 + ox, cy1 + oy)
            v3 = (cx1 - ox, cy1 - oy)

            vertices.extend([*v0, *v1, *v2])
            vertices.extend([*v1, *v3, *v2])

        return np.array(vertices, dtype=np.float32)

    def _cpu_reference_solution(self) -> Optional[np.ndarray]:
        sx, sy = self.source_pos
        speed = self.speed_field
        obstacles = self.obstacle_field > 0.5

        dist = np.full((self.grid_size, self.grid_size), self.huge_time, dtype=np.float32)
        visited = np.zeros_like(dist, dtype=bool)

        pq: List[Tuple[float, int, int]] = []
        dist[sy, sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while pq:
            time_val, x, y = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True

            for dx, dy in neighbors:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if obstacles[ny, nx]:
                        continue
                    step_speed = 0.5 * (speed[y, x] + speed[ny, nx])
                    step_speed = max(step_speed, 1e-4)
                    cost = 1.0 / step_speed
                    new_time = time_val + cost
                    if new_time < dist[ny, nx]:
                        dist[ny, nx] = new_time
                        heapq.heappush(pq, (new_time, nx, ny))

        return dist

    # ----------------------------------------------------------- Main Loop --
    def run(self) -> None:
        print("\nOptical Neuromorphic Eikonal Solver running...")
        print("Press ESC to exit, Space to toggle propagation.")
        last_time = time.time()
        frame_counter = 0

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.is_running:
                for _ in range(self.iterations_per_frame):
                    self._propagation_step()

            self._render_scene()
            glfw.swap_buffers(self.window)

            self._update_path(force_cpu_validation=False)

            frame_counter += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_counter / (current_time - last_time)
                title_parts = [
                    f"FPS {fps:5.1f}",
                    f"Iters {self.iteration_counter}",
                    f"Steps {len(self.path_points)}",
                ]
                if self.path_length is not None:
                    title_parts.append(f"Len {self.path_length:.1f}")
                if self.path_cost is not None:
                    title_parts.append(f"Cost {self.path_cost:.1f}")
                if self.last_validation_error is not None:
                    title_parts.append(f"MAE {self.last_validation_error:.4f}")
                glfw.set_window_title(self.window, " | ".join(title_parts))
                last_time = current_time
                frame_counter = 0

        glfw.terminate()


def main() -> None:
    solver = OpticalEikonalSolver()
    try:
        solver.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()


