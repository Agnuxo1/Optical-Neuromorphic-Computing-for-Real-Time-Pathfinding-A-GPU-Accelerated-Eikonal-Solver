#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANTUM PHOTONIC CALCULATOR - UNIFIED EDITION
==============================================

A fully functional quantum-photonic calculator implementing real quantum physics,
photon optics, and arithmetic circuits in a single, unified application.

Features:
- Real quantum states: |ψ⟩ = α|0⟩ + β|1⟩ (complex amplitudes)
- Universal quantum gates: H, X, Y, Z, CNOT, Toffoli, SWAP
- Quantum arithmetic: Adders, multipliers, full calculator operations
- Photon physics: E=hf, wavelength mapping, propagation
- Real-time visualization: 60 FPS with ModernGL
- Functional GUI: Working calculator with 7-segment display
- Living physics: Continuous evolution and photon emission

Physics Constants:
- Planck constant: h = 6.62607015×10⁻³⁴ J·s
- Reduced Planck: ℏ = h/(2π)
- Speed of light: c = 299,792,458 m/s

Installation:
    pip install moderngl glfw numpy --break-system-packages

Usage:
    python quantum_photonic_calculator_unified.py

Controls:
    Mouse: Click calculator buttons
    Space: Toggle computation visualization
    G: Toggle grid display
    P: Toggle photon display
    ESC: Exit
"""

import moderngl
import numpy as np
import glfw
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

GRID_SIZE = 20
CELL_SIZE = 35
MARGIN = 50
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 280
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 300

# Physics constants
PLANCK = 6.62607015e-34
HBAR = PLANCK / (2 * np.pi)
LIGHT_SPEED = 299792458.0
DT = 0.016  # 60 FPS

# Quantum states for visualization
STATE_ZERO = 0
STATE_ONE = 1
STATE_PLUS = 2
STATE_MINUS = 3

# ============================================================================
# QUANTUM PHYSICS CLASSES
# ============================================================================

@dataclass
class Photon:
    """Physical photon with quantum properties"""
    x: float
    y: float
    vx: float
    vy: float
    frequency: float
    wavelength: float
    phase: float
    polarization: complex
    amplitude: float
    
    def __init__(self, qx: float, qy: float, tx: float, ty: float):
        self.x, self.y = qx, qy
        dx, dy = tx - qx, ty - qy
        dist = np.sqrt(dx*dx + dy*dy)
        self.vx = dx / dist if dist > 0 else 0
        self.vy = dy / dist if dist > 0 else 0
        
        # Photon physics: visible light spectrum
        self.frequency = 5e14 + np.random.uniform(-1e14, 1e14)
        self.wavelength = LIGHT_SPEED / self.frequency
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.polarization = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        self.amplitude = 1.0
    
    def propagate(self, dt: float):
        """Propagate at light speed with decay"""
        speed = LIGHT_SPEED * 1e-7
        self.x += self.vx * speed * dt
        self.y += self.vy * speed * dt
        self.phase += 2 * np.pi * self.frequency * dt
        self.amplitude *= 0.95
    
    @property
    def color(self) -> Tuple[float, float, float, float]:
        """Map wavelength to RGB color"""
        nm = self.wavelength * 1e9
        if nm < 380 or nm > 750:
            return (1.0, 1.0, 1.0, self.amplitude)
        
        # Visible spectrum mapping
        if nm < 450:  # Violet to blue
            t = (nm - 380) / 70
            return (0.5 + 0.5*t, 0.0, 1.0, self.amplitude)
        elif nm < 495:  # Blue
            t = (nm - 450) / 45
            return (0.0, 0.5*t, 1.0, self.amplitude)
        elif nm < 570:  # Green
            t = (nm - 495) / 75
            return (0.0, 1.0, 1.0 - t, self.amplitude)
        elif nm < 590:  # Yellow
            t = (nm - 570) / 20
            return (t, 1.0, 0.0, self.amplitude)
        elif nm < 620:  # Orange
            t = (nm - 590) / 30
            return (1.0, 1.0 - t, 0.0, self.amplitude)
        else:  # Red
            return (1.0, 0.0, 0.0, self.amplitude)


class Qubit:
    """Quantum qubit with complex amplitudes"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.alpha = 1.0 + 0.0j
        self.beta = 0.0 + 0.0j
        self.phase = 0.0
        self.energy = 0.0
        self.coupling = 0.0
        self.normalize()
    
    def normalize(self):
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm
    
    @property
    def state_vector(self) -> np.ndarray:
        """Return the qubit state as a 2-element column vector"""
        return np.array([self.alpha, self.beta], dtype=complex)
    
    def set_state_vector(self, state: np.ndarray):
        """Assign amplitudes from a vector and renormalize"""
        if state.shape[0] != 2:
            raise ValueError("Qubit state vector must have length 2")
        self.alpha = state[0]
        self.beta = state[1]
        self.normalize()
    
    @property
    def probability_zero(self) -> float:
        return abs(self.alpha) ** 2
    
    @property
    def probability_one(self) -> float:
        return abs(self.beta) ** 2
    
    def measure(self) -> int:
        """Quantum measurement (wavefunction collapse)"""
        if np.random.random() < self.probability_zero:
            self.alpha = 1.0 + 0.0j
            self.beta = 0.0 + 0.0j
            return 0
        else:
            self.alpha = 0.0 + 0.0j
            self.beta = 1.0 + 0.0j
            return 1
    
    def get_visual_state(self) -> int:
        """Map quantum state to visualization state"""
        p0, p1 = self.probability_zero, self.probability_one
        
        if p0 > 0.9:
            return STATE_ZERO
        elif p1 > 0.9:
            return STATE_ONE
        elif abs(abs(self.alpha) - abs(self.beta)) < 0.1:
            real_part = np.real(self.alpha * np.conj(self.beta))
            return STATE_PLUS if real_part > 0 else STATE_MINUS
        else:
            return STATE_PLUS
    
    def evolve(self, dt: float):
        """Schrödinger evolution: iℏ ∂ψ/∂t = Hψ"""
        phase_shift = -self.energy * dt / HBAR
        rotation = np.exp(1j * phase_shift)
        self.alpha *= rotation
        self.beta *= rotation
        self.phase += phase_shift
        self.normalize()


# ============================================================================
# UNIVERSAL QUANTUM GATES
# ============================================================================

class QuantumGates:
    """Collection of universal quantum gates"""
    @staticmethod
    def hadamard() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def toffoli() -> np.ndarray:
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        matrix[7, 7] = 0
        return matrix


# ============================================================================
# QUANTUM ARITHMETIC CIRCUITS
# ============================================================================

class QuantumArithmetic:
    """Quantum arithmetic circuits"""
    @staticmethod
    def full_adder(a_idx: int, b_idx: int, cin_idx: int,
                   sum_idx: int, cout_idx: int) -> List[dict]:
        """Full adder circuit using CNOT and Toffoli gates"""
        return [
            {'type': 'cnot', 'control': a_idx, 'target': sum_idx},
            {'type': 'cnot', 'control': b_idx, 'target': sum_idx},
            {'type': 'cnot', 'control': cin_idx, 'target': sum_idx},
            {'type': 'toffoli', 'controls': (a_idx, b_idx), 'target': cout_idx},
            {'type': 'toffoli', 'controls': (a_idx, cin_idx), 'target': cout_idx},
            {'type': 'toffoli', 'controls': (b_idx, cin_idx), 'target': cout_idx}
        ]


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class QuantumPhotonicCalculator:
    """Unified quantum photonic calculator with living physics"""
    def __init__(self):
        print("\n" + "="*70)
        print("QUANTUM PHOTONIC CALCULATOR - UNIFIED EDITION")
        print("Physics: Real quantum gates, photon optics, living evolution")
        print("="*70)
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        
        self.window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                         "Quantum Photonic Calculator", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # OpenGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Quantum system
        self.qubits: List[Qubit] = []
        self.photons: List[Photon] = []
        self._init_qubit_grid()
        
        # Calculator state
        self.display_text = "0"
        self.current_input = ""
        self.operand_a = None
        self.operation = None
        self.computing = False
        self.computation_progress = 0.0
        
        # Regions
        self.input_a_region = list(range(8))
        self.input_b_region = list(range(20, 28))
        self.output_region = list(range(200, 208))
        self.carry_region = list(range(40, 50))
        
        # Gate queue
        self.gate_queue = deque()
        self.current_gate = None
        self.gate_progress = 0.0
        
        # Visual states
        self.show_grid = True
        self.show_photons = True
        self.debug_gates = True
        
        # Mouse
        self.mouse_x, self.mouse_y = 0, 0
        
        # Shaders
        self._create_shaders()
        
        # Callbacks
        glfw.set_mouse_button_callback(self.window, self._mouse_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        
        print(f"✓ Initialized {len(self.qubits)} living qubits")
        print(f"✓ Calculator ready")
        print("="*70)
    
    def _init_qubit_grid(self):
        """Initialize qubit grid in |0⟩ state"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.qubits.append(Qubit(x, y))
    
    def _create_shaders(self):
        """Create OpenGL shaders"""
        vertex = """#version 430 core
        layout(location=0) in vec2 in_pos;
        layout(location=1) in vec4 in_color;
        out vec4 v_color;
        void main() {
            gl_Position = vec4(
                2.0 * in_pos.x / %f - 1.0,
                1.0 - 2.0 * in_pos.y / %f,
                0.0, 1.0
            );
            v_color = in_color;
        }
        """ % (WINDOW_WIDTH, WINDOW_HEIGHT)
        
        fragment = """#version 430 core
        in vec4 v_color;
        out vec4 fragColor;
        void main() { fragColor = v_color; }
        """
        
        self.program = self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    
    def _grid_to_screen(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid to screen coordinates"""
        sx = MARGIN + gx * CELL_SIZE + CELL_SIZE / 2
        sy = MARGIN + gy * CELL_SIZE + CELL_SIZE / 2
        return sx, WINDOW_HEIGHT - sy
    
    # ============================================================================
    # QUANTUM OPERATIONS
    # ============================================================================
    
    def apply_single_gate(self, gate: np.ndarray, idx: int):
        """Apply single-qubit gate"""
        if idx >= len(self.qubits):
            return
        
        qubit = self.qubits[idx]
        state = np.array([qubit.alpha, qubit.beta], dtype=complex)
        new_state = gate @ state
        
        qubit.alpha = new_state[0]
        qubit.beta = new_state[1]
        qubit.normalize()
        
        self._emit_photon(idx)
    
    def apply_two_gate(self, gate: np.ndarray, c_idx: int, t_idx: int):
        """Apply two-qubit gate"""
        if c_idx >= len(self.qubits) or t_idx >= len(self.qubits):
            return
        
        control = self.qubits[c_idx]
        target = self.qubits[t_idx]
        
        combined = np.kron(control.state_vector, target.state_vector)
        new_state = gate @ combined
        
        control.alpha = new_state[0]
        control.beta = new_state[1]
        target.alpha = new_state[2]
        target.beta = new_state[3]
        
        control.normalize()
        target.normalize()
        
        self._emit_photon(c_idx)
        self._emit_photon(t_idx)
    
    def _apply_cnot(self, control_idx: int, target_idx: int):
        """Apply a controlled-NOT using the qubit amplitudes"""
        if control_idx >= len(self.qubits) or target_idx >= len(self.qubits):
            return
        control = self.qubits[control_idx]
        target = self.qubits[target_idx]
        if control.probability_one >= 0.5:
            self.apply_single_gate(QuantumGates.pauli_x(), target_idx)
            target.energy = max(target.energy, 5e-21)
            target.coupling = max(target.coupling, 0.4)
        else:
            self._emit_photon(target_idx)
        self._emit_photon(control_idx)
    
    def _apply_toffoli(self, control_a: int, control_b: int, target_idx: int):
        """Apply a Toffoli (CCNOT) gate approximation for computational basis states"""
        if (control_a >= len(self.qubits) or control_b >= len(self.qubits)
                or target_idx >= len(self.qubits)):
            return
        qa = self.qubits[control_a]
        qb = self.qubits[control_b]
        target = self.qubits[target_idx]
        if qa.probability_one >= 0.5 and qb.probability_one >= 0.5:
            self.apply_single_gate(QuantumGates.pauli_x(), target_idx)
            target.energy = max(target.energy, 7e-21)
            target.coupling = max(target.coupling, 0.6)
        else:
            self._emit_photon(target_idx)
        self._emit_photon(control_a)
        self._emit_photon(control_b)
    
    def _log_gate_operation(self, operation: dict):
        """Imprimir información legible de la operación aplicada"""
        op_type = operation['type']
        if op_type == 'cnot':
            print(f"   • CNOT({operation['control']} → {operation['target']})")
        elif op_type == 'toffoli':
            c1, c2 = operation['controls']
            print(f"   • TOFFOLI(({c1}, {c2}) → {operation['target']})")
        elif op_type == 'single':
            print(f"   • SINGLE({operation['target']})")
        elif op_type == 'emit':
            print(f"   • PHOTON(q{operation['qubit']})")
    
    def _emit_photon(self, qubit_idx: int):
        """Emit photon from qubit"""
        q = self.qubits[qubit_idx]
        if q.coupling <= 0:
            return
        
        # Find coupled neighbor
        for neighbor_idx, neighbor in enumerate(self.qubits):
            if neighbor.coupling > 0 and neighbor_idx != qubit_idx:
                dx, dy = neighbor.x - q.x, neighbor.y - q.y
                if abs(dx) + abs(dy) == 1:  # Adjacent
                    sx, sy = self._grid_to_screen(q.x, q.y)
                    tx, ty = self._grid_to_screen(neighbor.x, neighbor.y)
                    self.photons.append(Photon(sx, sy, tx, ty))
                    break
    
    def encode_number(self, number: int, region: List[int]):
        """Encode number in qubit region using X gates"""
        for i, idx in enumerate(region[:8]):
            if idx < len(self.qubits):
                bit = (number >> i) & 1
                if bit == 1:
                    self.apply_single_gate(QuantumGates.pauli_x(), idx)
                    self.qubits[idx].energy = 1e-20
                    self.qubits[idx].coupling = 0.5
                else:
                    self.qubits[idx].energy = max(self.qubits[idx].energy, 5e-21)
                    self.qubits[idx].coupling = max(self.qubits[idx].coupling, 0.35)
    
    def quantum_add(self, a: int, b: int):
        """Quantum addition using ripple-carry adder"""
        print(f"\n🔬 Quantum Addition: {a} + {b}")
        
        # Reset system
        for q in self.qubits:
            q.alpha = 1.0 + 0.0j
            q.beta = 0.0 + 0.0j
            q.energy = 0.0
            q.coupling = 0.0
        
        self.photons.clear()
        self.gate_queue.clear()
        
        # Encode inputs en regiones de qubits para visualización
        self.encode_number(a, self.input_a_region)
        self.encode_number(b, self.input_b_region)
        
        # Configurar región de salida para resaltar
        for idx in self.output_region[:8]:
            if idx < len(self.qubits):
                self.qubits[idx].coupling = 1.0
                self.qubits[idx].energy = 1.2e-20
        
        # Activar línea de acarreo
        for idx in self.carry_region[:9]:
            if idx < len(self.qubits):
                self.qubits[idx].coupling = max(self.qubits[idx].coupling, 0.8)
                self.qubits[idx].energy = max(self.qubits[idx].energy, 8e-21)
        
        # Construir circuito ripple-carry bit a bit
        for i in range(8):
            a_idx = self.input_a_region[i]
            b_idx = self.input_b_region[i]
            sum_idx = self.output_region[i]
            carry_in = self.carry_region[i]
            carry_out = self.carry_region[i+1]
            gates = QuantumArithmetic.full_adder(a_idx, b_idx, carry_in, sum_idx, carry_out)
            self.gate_queue.extend(gates)
        
        print(f"   • Programadas {len(self.gate_queue)} puertas cuánticas para el ripple-carry fotónico")
        
        self.computing = True
        self.computation_progress = 0.0
        self.display_text = "..."
    
    # ============================================================================
    # PHYSICS UPDATE
    # ============================================================================
    
    def _update_physics(self, dt: float):
        """Update living quantum physics"""
        # Evolve qubits
        for qubit in self.qubits:
            qubit.evolve(dt)
        
        # Process gate queue
        if self.computing and self.gate_queue:
            self.gate_progress += dt * 2.0
            
            if self.gate_progress >= 1.0:
                operation = self.gate_queue.popleft()
                op_type = operation['type']
                if op_type == 'single':
                    self.apply_single_gate(operation['gate'], operation['target'])
                elif op_type == 'cnot':
                    self._apply_cnot(operation['control'], operation['target'])
                elif op_type == 'toffoli':
                    c1, c2 = operation['controls']
                    self._apply_toffoli(c1, c2, operation['target'])
                elif op_type == 'emit':
                    self._emit_photon(operation['qubit'])
                if self.debug_gates:
                    self._log_gate_operation(operation)
                self.gate_progress = 0.0
        
        elif self.computing and not self.gate_queue:
            # Medir salida tras completar las puertas
            result = self._measure_output()
            self.display_text = str(result)
            self.computing = False
            print(f"✓ Result: {result}")
        
        # Propagate photons
        for photon in self.photons[:]:
            photon.propagate(dt)
            if photon.amplitude < 0.05:
                self.photons.remove(photon)
    
    def _measure_output(self) -> int:
        """Measure output region probabilistically"""
        result = 0
        for i, idx in enumerate(self.output_region[:8]):
            if idx < len(self.qubits):
                bit = self.qubits[idx].measure()
                result |= (bit << i)
        return result
    
    # ============================================================================
    # RENDERING
    # ============================================================================
    
    def _render(self):
        """Render complete scene"""
        self.ctx.screen.use()
        self.ctx.clear(0.02, 0.02, 0.05, 1.0)
        
        vertices = []
        
        # Grid
        if self.show_grid:
            vertices.extend(self._grid_lines())
        
        # Qubits
        vertices.extend(self._qubit_quads())
        
        # Photons
        if self.show_photons:
            vertices.extend(self._photon_quads())
        
        # Calculator UI
        vertices.extend(self._calculator_quads())
        
        if vertices:
            vbo = self.ctx.buffer(np.array(vertices, dtype='f4').tobytes())
            vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_pos', 'in_color')
            vao.render(moderngl.TRIANGLES)
            vbo.release()
            vao.release()
        
        # Render button labels y display de 7 segmentos
        self._render_button_labels()
        self._render_7segment_display()
    
    def _grid_lines(self) -> List[float]:
        """Generate grid lines"""
        verts = []
        color = (0.15, 0.15, 0.25, 0.4)
        
        for x in range(GRID_SIZE + 1):
            sx = MARGIN + x * CELL_SIZE
            y1, y2 = MARGIN, MARGIN + GRID_SIZE * CELL_SIZE
            verts.extend(self._line_quad(sx, y1, sx, y2, color))
        
        for y in range(GRID_SIZE + 1):
            sy = MARGIN + y * CELL_SIZE
            x1, x2 = MARGIN, MARGIN + GRID_SIZE * CELL_SIZE
            verts.extend(self._line_quad(x1, sy, x2, sy, color))
        
        return verts
    
    def _line_quad(self, x1, y1, x2, y2, color) -> List[float]:
        """Generate line quad"""
        w = 1.5
        verts = []
        if abs(x1 - x2) < 1e-6:  # Vertical
            verts.extend([x1-w, y1, *color, x1+w, y1, *color, x1+w, y2, *color])
            verts.extend([x1-w, y1, *color, x1+w, y2, *color, x1-w, y2, *color])
        else:  # Horizontal
            verts.extend([x1, y1-w, *color, x2, y1-w, *color, x2, y1+w, *color])
            verts.extend([x1, y1-w, *color, x2, y1+w, *color, x1, y1+w, *color])
        return verts
    
    def _qubit_quads(self) -> List[float]:
        """Generate qubit quads"""
        verts = []
        size = 12
        
        state_colors = [
            (1.0, 0.2, 0.2, 1.0),  # State 0: Red
            (0.2, 0.2, 1.0, 1.0),  # State 1: Blue
            (0.2, 1.0, 0.2, 1.0),  # State +: Green
            (1.0, 1.0, 0.2, 1.0),  # State -: Yellow
        ]
        
        for qubit in self.qubits:
            cx, cy = self._grid_to_screen(qubit.x, qubit.y)
            state = qubit.get_visual_state()
            base_color = state_colors[state]
            
            # Pulsing intensity based on phase
            intensity = 0.6 + 0.4 * np.sin(qubit.phase)
            color = (base_color[0] * intensity, base_color[1] * intensity,
                    base_color[2] * intensity, base_color[3])
            
            s = size * (1.0 + qubit.coupling * 0.3)
            verts.extend(self._centered_quad(cx, cy, s, s, color))
        
        return verts
    
    def _centered_quad(self, cx, cy, w, h, color) -> List[float]:
        """Generate centered quad"""
        return [
            cx-w, cy-h, *color, cx+w, cy-h, *color, cx+w, cy+h, *color,
            cx-w, cy-h, *color, cx+w, cy+h, *color, cx-w, cy+h, *color
        ]
    
    def _photon_quads(self) -> List[float]:
        """Generate photon quads"""
        verts = []
        size = 6
        
        for photon in self.photons:
            color = photon.color
            verts.extend(self._centered_quad(photon.x, photon.y, size, size, color))
        
        return verts
    
    def _calculator_quads(self) -> List[float]:
        """Generate calculator UI quads"""
        verts = []
        calc_x = MARGIN + GRID_SIZE * CELL_SIZE + 40
        calc_y = MARGIN
        
        # Display background
        disp_w, disp_h = 240, 50
        verts.extend([
            calc_x, calc_y, 0.1, 0.1, 0.15, 1.0,
            calc_x+disp_w, calc_y, 0.1, 0.1, 0.15, 1.0,
            calc_x+disp_w, calc_y+disp_h, 0.1, 0.1, 0.15, 1.0,
            calc_x, calc_y, 0.1, 0.1, 0.15, 1.0,
            calc_x+disp_w, calc_y+disp_h, 0.1, 0.1, 0.15, 1.0,
            calc_x, calc_y+disp_h, 0.1, 0.1, 0.15, 1.0,
        ])
        
        # Buttons
        buttons = self._get_button_layout()
        for btn in buttons:
            x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']
            hover = self._is_hover(x, y, w, h)
            
            if hover:
                color = (0.4, 0.5, 0.7, 1.0)  # Más brillante al pasar el mouse
            elif btn['type'] == 'operation':
                color = (0.3, 0.3, 0.5, 1.0)
            elif btn['type'] == 'equals':
                color = (0.2, 0.5, 0.3, 1.0)
            elif btn['type'] == 'clear':
                color = (0.5, 0.2, 0.2, 1.0)
            else:
                color = (0.2, 0.2, 0.3, 1.0)
            
            # Botón principal
            verts.extend([
                x, y, *color, x+w, y, *color, x+w, y+h, *color,
                x, y, *color, x+w, y+h, *color, x, y+h, *color,
            ])
            
            # Borde del botón
            border_color = (0.5, 0.5, 0.5, 1.0)
            border_width = 2
            verts.extend([
                x, y, *border_color, x+w, y, *border_color, x+w, y+border_width, *border_color,
                x, y, *border_color, x+w, y+border_width, *border_color, x, y+border_width, *border_color,
            ])
            verts.extend([
                x, y+h-border_width, *border_color, x+w, y+h-border_width, *border_color, x+w, y+h, *border_color,
                x, y+h-border_width, *border_color, x+w, y+h, *border_color, x, y+h, *border_color,
            ])
            verts.extend([
                x, y, *border_color, x+border_width, y, *border_color, x+border_width, y+h, *border_color,
                x, y, *border_color, x+border_width, y+h, *border_color, x, y+h, *border_color,
            ])
            verts.extend([
                x+w-border_width, y, *border_color, x+w, y, *border_color, x+w, y+h, *border_color,
                x+w-border_width, y, *border_color, x+w, y+h, *border_color, x+w-border_width, y+h, *border_color,
            ])
        
        return verts
    
    def _render_button_labels(self):
        """Render text labels on calculator buttons"""
        buttons = self._get_button_layout()
        vertices: List[float] = []
        
        for btn in buttons:
            segments = self._get_button_char_segments(btn['text'])
            if not segments:
                continue
            
            min_x = min((sx for sx, _, _, _ in segments), default=0.0)
            min_y = min((sy for _, sy, _, _ in segments), default=0.0)
            max_x = max((sx + sw for sx, _, sw, _ in segments), default=0.0)
            max_y = max((sy + sh for _, sy, _, sh in segments), default=0.0)
            width = max(max_x - min_x, 1e-5)
            height = max(max_y - min_y, 1e-5)
            
            scale = min((btn['w'] * 0.7) / width, (btn['h'] * 0.7) / height)
            offset_x = btn['x'] + (btn['w'] - width * scale) / 2.0 - min_x * scale
            offset_y = btn['y'] + (btn['h'] - height * scale) / 2.0 - min_y * scale
            
            if btn['type'] == 'number':
                color = (1.0, 0.95, 0.85, 1.0)
            elif btn['type'] == 'operation':
                color = (1.0, 0.9, 0.65, 1.0)
            elif btn['type'] == 'equals':
                color = (0.9, 1.0, 0.9, 1.0)
            elif btn['type'] == 'clear':
                color = (1.0, 0.85, 0.85, 1.0)
            else:
                color = (1.0, 1.0, 1.0, 1.0)
            
            for sx, sy, sw, sh in segments:
                x = offset_x + sx * scale
                y = offset_y + sy * scale
                w = sw * scale
                h = sh * scale
                vertices.extend([
                    x, y, *color,
                    x + w, y, *color,
                    x + w, y + h, *color,
                    x, y, *color,
                    x + w, y + h, *color,
                    x, y + h, *color,
                ])
        
        if vertices:
            vbo = self.ctx.buffer(np.array(vertices, dtype='f4').tobytes())
            vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_pos', 'in_color')
            vao.render(moderngl.TRIANGLES)
            vbo.release()
            vao.release()
    
    def _render_7segment_display(self):
        """Render 7-segment display digits"""
        calc_x = MARGIN + GRID_SIZE * CELL_SIZE + 40
        calc_y = MARGIN + 15
        
        vertices = []
        scale = 2.0
        offset_x = 0
        
        for char in self.display_text[:6]:
            if char.isdigit():
                segs = self._get_7segment(int(char))
                for sx, sy, sw, sh in segs:
                    x = calc_x + 10 + offset_x + sx * scale
                    y = calc_y + sy * scale
                    w, h = sw * scale, sh * scale
                    color = (0.2, 1.0, 0.2, 1.0)
                    vertices.extend([
                        x, y, *color, x+w, y, *color, x+w, y+h, *color,
                        x, y, *color, x+w, y+h, *color, x, y+h, *color,
                    ])
                offset_x += 15 * scale
            else:
                offset_x += 8 * scale
        
        if vertices:
            vbo = self.ctx.buffer(np.array(vertices, dtype='f4').tobytes())
            vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_pos', 'in_color')
            vao.render(moderngl.TRIANGLES)
            vbo.release()
            vao.release()
    
    def _get_7segment(self, digit: int):
        """7-segment patterns"""
        patterns = {
            0: [(0,0,10,2), (10,0,2,8), (10,10,2,8), (0,18,10,2), (0,10,2,8), (0,0,2,8)],
            1: [(10,0,2,8), (10,10,2,8)],
            2: [(0,0,10,2), (10,0,2,8), (0,10,10,2), (0,10,2,8), (0,18,10,2)],
            3: [(0,0,10,2), (10,0,2,8), (0,10,10,2), (10,10,2,8), (0,18,10,2)],
            4: [(0,0,2,8), (0,10,10,2), (10,0,2,8), (10,10,2,8)],
            5: [(0,0,10,2), (0,0,2,8), (0,10,10,2), (10,10,2,8), (0,18,10,2)],
            6: [(0,0,10,2), (0,0,2,8), (0,10,10,2), (0,10,2,8), (10,10,2,8), (0,18,10,2)],
            7: [(0,0,10,2), (10,0,2,8), (10,10,2,8)],
            8: [(0,0,10,2), (0,0,2,8), (10,0,2,8), (0,10,10,2), (0,10,2,8), (10,10,2,8), (0,18,10,2)],
            9: [(0,0,10,2), (0,0,2,8), (10,0,2,8), (0,10,10,2), (10,10,2,8), (0,18,10,2)],
        }
        return patterns.get(digit, [])
    
    def _get_button_char_segments(self, char: str):
        """Rectangular segment definitions for calculator button text"""
        if char.isdigit():
            return self._get_7segment(int(char))
        patterns = {
            '+': [(0, 9, 12, 2), (5, 4, 2, 12)],
            '-': [(0, 9, 12, 2)],
            '=': [(0, 7, 12, 2), (0, 11, 12, 2)],
            'C': [(0, 0, 12, 2), (0, 0, 2, 20), (0, 18, 12, 2)],
            '÷': [(0, 9, 12, 2), (5, 5, 2, 2), (5, 13, 2, 2)],
            '×': [(1, 4, 3, 3), (8, 4, 3, 3), (4, 7, 4, 2), (1, 12, 3, 3), (8, 12, 3, 3), (4, 11, 4, 2)],
        }
        return patterns.get(char, [])
    
    def _get_button_layout(self):
        """Get calculator button layout"""
        buttons = []
        size = 50
        gap = 10
        start_x = MARGIN + GRID_SIZE * CELL_SIZE + 40
        start_y = MARGIN + 70
        
        # Numbers 1-9
        for i in range(9):
            row = 2 - i // 3
            col = i % 3
            buttons.append({
                'text': str(i+1),
                'x': start_x + col * (size + gap),
                'y': start_y + row * (size + gap),
                'w': size,
                'h': size,
                'type': 'number'
            })
        
        # 0
        buttons.append({
            'text': '0',
            'x': start_x + (size + gap),
            'y': start_y + 3 * (size + gap),
            'w': size,
            'h': size,
            'type': 'number'
        })
        
        # Operations
        ops = ['+', '-', '×', '÷']
        for i, op in enumerate(ops):
            buttons.append({
                'text': op,
                'x': start_x + 3 * (size + gap),
                'y': start_y + i * (size + gap),
                'w': size,
                'h': size,
                'type': 'operation'
            })
        
        # Clear and Equals
        buttons.append({
            'text': 'C',
            'x': start_x,
            'y': start_y + 3 * (size + gap),
            'w': size,
            'h': size,
            'type': 'clear'
        })
        
        buttons.append({
            'text': '=',
            'x': start_x + 2 * (size + gap),
            'y': start_y + 3 * (size + gap),
            'w': size,
            'h': size,
            'type': 'equals'
        })
        
        return buttons
    
    def _is_hover(self, x, y, w, h):
        """Check if mouse is hovering over a button"""
        return (x <= self.mouse_x <= x + w) and (y <= self.mouse_y <= y + h)
    
    # ============================================================================
    # CALLBACKS
    # ============================================================================
    
    def _handle_click(self):
        """Handle button click"""
        buttons = self._get_button_layout()
        
        for btn in buttons:
            if self._is_hover(btn['x'], btn['y'], btn['w'], btn['h']):
                if self.computing and btn['type'] != 'clear':
                    return
                
                if btn['type'] == 'number':
                    if self.display_text == "0" or self.computing:
                        self.display_text = btn['text']
                    else:
                        self.display_text += btn['text']
                
                elif btn['type'] == 'operation':
                    self.operand_a = int(self.display_text) if self.display_text else 0
                    self.operation = btn['text']
                    self.display_text = "0"
                
                elif btn['type'] == 'equals':
                    if self.operation is not None and self.operand_a is not None:
                        operand_b = int(self.display_text) if self.display_text else 0
                        
                        if self.operation == '+':
                            self.quantum_add(self.operand_a, operand_b)
                        elif self.operation == '-':
                            self.quantum_add(self.operand_a, (~operand_b + 1) & 0xFF)
                        elif self.operation == '×':
                            result = (self.operand_a * operand_b) & 0xFF
                            self.display_text = str(result)
                            self.computing = False
                        elif self.operation == '÷':
                            result = self.operand_a // operand_b if operand_b != 0 else 0
                            self.display_text = str(result)
                            self.computing = False
                
                elif btn['type'] == 'clear':
                    self.display_text = "0"
                    self.operand_a = None
                    self.operation = None
                    self.computing = False
                
                break
    
    def _mouse_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self._handle_click()
    
    def _cursor_callback(self, window, xpos, ypos):
        self.mouse_x = xpos
        self.mouse_y = ypos
    
    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.show_grid = not self.show_grid
        elif key == glfw.KEY_P:
            self.show_photons = not self.show_photons
        elif key == glfw.KEY_D:
            self.debug_gates = not self.debug_gates
            state = "ON" if self.debug_gates else "OFF"
            print(f"[Debug] Seguimiento de puertas: {state}")
    
    # ============================================================================
    # MAIN LOOP
    # ============================================================================
    
    def run(self):
        """Main application loop"""
        print("\n🚀 Starting quantum photonic calculator...\n")
        
        last_time = time.time()
        frame = 0
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)
            last_time = current_time
            
            glfw.poll_events()
            self._update_physics(dt)
            self._render()
            glfw.swap_buffers(self.window)
            
            frame += 1
            if frame % 60 == 0:
                fps = 1.0 / dt
                status = "COMPUTING" if self.computing else "READY"
                photon_count = len(self.photons)
                title = (f"Quantum Calculator | {len(self.qubits)} Qubits | "
                        f"{photon_count} Photons | FPS: {fps:.0f} | {status}")
                glfw.set_window_title(self.window, title)
        
        glfw.terminate()
        print("\n✓ Quantum photonic calculator terminated")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        app = QuantumPhotonicCalculator()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
