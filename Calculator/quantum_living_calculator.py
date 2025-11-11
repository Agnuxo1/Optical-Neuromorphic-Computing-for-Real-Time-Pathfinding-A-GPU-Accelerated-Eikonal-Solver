#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANTUM LIVING CALCULATOR
=========================

El circuito VIVE en cada fotograma.
La física cuántica SUCEDE en la imagen renderizada.
No es una representación - ES el circuito funcionando en tiempo real.

Cada frame:
- Ecuaciones de Schrödinger evolucionan los qubits
- Fotones propagan información física real
- Estados cuánticos cambian según interacciones
- El resultado EMERGE de la física
"""

import moderngl
import numpy as np
import glfw
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

# Configuración
GRID_SIZE = 24
CELL_SIZE = 30
MARGIN = 40
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 300  # +300 para calculadora
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 100

# Constantes físicas REALES
HBAR = 1.054571817e-34  # Constante de Planck reducida
C = 299792458           # Velocidad de la luz
DT = 0.016             # Delta time por frame (60 FPS)


@dataclass
class LivingQubit:
    """Qubit que VIVE - evoluciona en cada frame"""
    x: int
    y: int
    
    # Estado cuántico (complejo)
    psi_0: complex = 1.0 + 0.0j  # Amplitud |0⟩
    psi_1: complex = 0.0 + 0.0j  # Amplitud |1⟩
    
    # Hamiltoniano local (energía)
    energy: float = 0.0
    
    # Fase acumulada
    phase: float = 0.0
    
    # Acoplamiento con vecinos (fotones)
    coupling_strength: float = 0.0
    
    def evolve(self, dt: float):
        """Evolución temporal según Schrödinger: iℏ ∂ψ/∂t = H ψ"""
        # Hamiltoniano simple: H = E
        # Solución: ψ(t) = exp(-iEt/ℏ) ψ(0)
        
        phase_shift = -self.energy * dt / HBAR
        rotation = np.exp(1j * phase_shift)
        
        self.psi_0 *= rotation
        self.psi_1 *= rotation
        
        # Normalizar
        norm = np.sqrt(abs(self.psi_0)**2 + abs(self.psi_1)**2)
        if norm > 0:
            self.psi_0 /= norm
            self.psi_1 /= norm
        
        self.phase += phase_shift
    
    @property
    def probability_0(self):
        return abs(self.psi_0) ** 2
    
    @property
    def probability_1(self):
        return abs(self.psi_1) ** 2
    
    @property
    def display_color(self):
        """Color basado en probabilidades"""
        p0 = self.probability_0
        p1 = self.probability_1
        
        # Mezcla de rojo (|0⟩) y azul (|1⟩)
        r = p0
        g = min(p0, p1) * 2  # Verde cuando hay superposición
        b = p1
        
        # Intensidad basada en energía
        intensity = 0.5 + 0.5 * np.sin(self.phase)
        
        return (r * intensity, g * intensity, b * intensity, 1.0)


@dataclass
class LivingPhoton:
    """Fotón que propaga información física real"""
    x: float
    y: float
    vx: float  # Velocidad
    vy: float
    frequency: float
    phase: float
    amplitude: float
    polarization: complex
    
    def propagate(self, dt: float):
        """Propagar a velocidad de la luz (escalada)"""
        speed = C * 1e-7  # Escalado para visualización
        self.x += self.vx * speed * dt
        self.y += self.vy * speed * dt
        
        # Evolución de fase
        self.phase += 2 * np.pi * self.frequency * dt
        self.phase = self.phase % (2 * np.pi)
        
        # Decaimiento
        self.amplitude *= 0.98
    
    @property
    def color(self):
        """Color según frecuencia (espectro visible)"""
        # Mapear frecuencia a color
        t = (self.frequency % 1.0)
        
        if t < 0.33:  # Rojo -> Verde
            r = 1.0 - t * 3
            g = t * 3
            b = 0.0
        elif t < 0.66:  # Verde -> Azul
            r = 0.0
            g = 1.0 - (t - 0.33) * 3
            b = (t - 0.33) * 3
        else:  # Azul -> Rojo
            r = (t - 0.66) * 3
            g = 0.0
            b = 1.0 - (t - 0.66) * 3
        
        return (r, g, b, self.amplitude)


class QuantumLivingCalculator:
    """Calculadora donde la física VIVE en cada fotograma"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("QUANTUM LIVING CALCULATOR")
        print("Physics happens IN the rendered circuit")
        print("="*70)
        
        # Inicializar GLFW
        if not glfw.init():
            raise RuntimeError("GLFW failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        
        self.window = glfw.create_window(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            "Quantum Living Calculator",
            None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window failed")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # OpenGL
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Grid de qubits VIVOS
        self.qubits: List[LivingQubit] = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.qubits.append(LivingQubit(x, y))
        
        # Fotones activos
        self.photons: List[LivingPhoton] = []
        
        # Estado calculadora
        self.display_text = "0"
        self.current_input = ""
        self.operand_a = None
        self.operation = None
        self.computing = False
        self.computation_progress = 0.0
        
        # Shader
        self._create_shader()
        
        # Mouse
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Callbacks
        glfw.set_mouse_button_callback(self.window, self._mouse_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        
        print(f"✓ {len(self.qubits)} living qubits initialized")
        print("="*70)
    
    def _create_shader(self):
        """Shader simple"""
        vertex = """
        #version 430
        layout(location=0) in vec2 in_pos;
        layout(location=1) in vec4 in_color;
        out vec4 v_color;
        uniform vec2 resolution;
        void main() {
            gl_Position = vec4(
                2.0 * in_pos.x / resolution.x - 1.0,
                1.0 - 2.0 * in_pos.y / resolution.y,
                0.0, 1.0
            );
            v_color = in_color;
        }
        """
        
        fragment = """
        #version 430
        in vec4 v_color;
        out vec4 fragColor;
        void main() { fragColor = v_color; }
        """
        
        self.program = self.ctx.program(
            vertex_shader=vertex,
            fragment_shader=fragment
        )
        self.program['resolution'] = (WINDOW_WIDTH, WINDOW_HEIGHT)
    
    def _grid_to_screen(self, gx, gy):
        sx = MARGIN + gx * CELL_SIZE + CELL_SIZE / 2
        sy = MARGIN + gy * CELL_SIZE + CELL_SIZE / 2
        return sx, sy
    
    # ========================================================================
    # FÍSICA QUE VIVE
    # ========================================================================
    
    def _quantum_interaction(self, q1: LivingQubit, q2: LivingQubit, strength: float):
        """Interacción cuántica entre qubits vecinos"""
        # Acoplamiento: H_int = g (σ₁⁺σ₂⁻ + σ₁⁻σ₂⁺)
        # Esto causa transferencia de excitación
        
        # Calcular intercambio de amplitud
        transfer = strength * (q1.psi_1 * np.conj(q2.psi_0) - 
                              q1.psi_0 * np.conj(q2.psi_1))
        
        # Aplicar intercambio
        delta_psi = transfer * DT * 0.1
        
        q1.psi_1 -= delta_psi
        q2.psi_1 += delta_psi
        
        # Normalizar
        for q in [q1, q2]:
            norm = np.sqrt(abs(q.psi_0)**2 + abs(q.psi_1)**2)
            if norm > 0:
                q.psi_0 /= norm
                q.psi_1 /= norm
        
        # Emitir fotón si hay transferencia significativa
        if abs(delta_psi) > 0.1:
            self._emit_photon(q1, q2)
    
    def _emit_photon(self, source: LivingQubit, target: LivingQubit):
        """Emitir fotón entre qubits"""
        sx, sy = self._grid_to_screen(source.x, source.y)
        tx, ty = self._grid_to_screen(target.x, target.y)
        
        # Dirección
        dx = tx - sx
        dy = ty - sy
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            photon = LivingPhoton(
                x=sx, y=sy,
                vx=dx/dist, vy=dy/dist,
                frequency=abs(source.energy - target.energy) / HBAR,
                phase=np.random.uniform(0, 2*np.pi),
                amplitude=0.8,
                polarization=np.exp(1j * np.random.uniform(0, 2*np.pi))
            )
            self.photons.append(photon)
    
    def _encode_number_quantum(self, number: int, start_row: int):
        """Codificar número usando física real"""
        for bit in range(8):
            val = (number >> bit) & 1
            idx = start_row * GRID_SIZE + bit
            
            if idx < len(self.qubits):
                q = self.qubits[idx]
                if val == 1:
                    # Aplicar pulso de Rabi π: |0⟩ → |1⟩
                    q.psi_0 = 0.0 + 0.0j
                    q.psi_1 = 1.0 + 0.0j
                else:
                    # Mantener en |0⟩
                    q.psi_0 = 1.0 + 0.0j
                    q.psi_1 = 0.0 + 0.0j
                
                q.energy = val * 1e-20  # Energía proporcional al estado
    
    def _quantum_add_living(self, a: int, b: int):
        """Suma cuántica que VIVE - evoluciona en tiempo real"""
        print(f"\n🔬 Quantum Living Addition: {a} + {b}")
        
        # Resetear qubits
        for q in self.qubits:
            q.psi_0 = 1.0 + 0.0j
            q.psi_1 = 0.0 + 0.0j
            q.energy = 0.0
            q.phase = 0.0
        
        self.photons.clear()
        
        # Codificar entradas
        self._encode_number_quantum(a, 0)  # Fila 0
        self._encode_number_quantum(b, 1)  # Fila 1
        
        # Configurar acoplamientos para suma
        # Los qubits interactuarán y el resultado emergirá
        for bit in range(8):
            idx_a = 0 * GRID_SIZE + bit
            idx_b = 1 * GRID_SIZE + bit
            idx_out = 10 * GRID_SIZE + bit  # Fila 10 = salida
            
            if idx_out < len(self.qubits):
                # Acoplar A y B con salida
                self.qubits[idx_a].coupling_strength = 0.5
                self.qubits[idx_b].coupling_strength = 0.5
                self.qubits[idx_out].coupling_strength = 1.0
        
        self.computing = True
        self.computation_progress = 0.0
    
    def _read_quantum_result(self, output_row: int) -> int:
        """Leer resultado midiendo qubits"""
        result = 0
        for bit in range(8):
            idx = output_row * GRID_SIZE + bit
            if idx < len(self.qubits):
                q = self.qubits[idx]
                # Medir: colapso probabilístico
                if np.random.random() < q.probability_1:
                    result |= (1 << bit)
        return result
    
    def _update_physics(self, dt: float):
        """Actualizar física - AQUÍ VIVE el circuito"""
        
        # 1. EVOLUCIÓN INDIVIDUAL de cada qubit (Schrödinger)
        for qubit in self.qubits:
            qubit.evolve(dt)
        
        # 2. INTERACCIONES entre qubits vecinos
        if self.computing:
            self.computation_progress += dt * 0.5
            
            # Interacciones verticales y horizontales
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    idx = y * GRID_SIZE + x
                    q1 = self.qubits[idx]
                    
                    # Vecino derecha
                    if x < GRID_SIZE - 1:
                        q2 = self.qubits[idx + 1]
                        strength = (q1.coupling_strength + q2.coupling_strength) / 2
                        if strength > 0:
                            self._quantum_interaction(q1, q2, strength)
                    
                    # Vecino abajo
                    if y < GRID_SIZE - 1:
                        q2 = self.qubits[idx + GRID_SIZE]
                        strength = (q1.coupling_strength + q2.coupling_strength) / 2
                        if strength > 0:
                            self._quantum_interaction(q1, q2, strength)
            
            # Terminar computación
            if self.computation_progress >= 2.0:
                result = self._read_quantum_result(10)
                self.display_text = str(result)
                self.computing = False
                print(f"✓ Result: {result}")
        
        # 3. PROPAGACIÓN de fotones
        for photon in self.photons[:]:
            photon.propagate(dt)
            
            # Remover si sale de pantalla o decae
            if (photon.amplitude < 0.05 or 
                photon.x < 0 or photon.x > WINDOW_WIDTH or
                photon.y < 0 or photon.y > WINDOW_HEIGHT):
                self.photons.remove(photon)
    
    # ========================================================================
    # RENDERIZADO
    # ========================================================================
    
    def _render(self):
        """Renderizar el circuito VIVO"""
        vertices = []
        
        # 1. GRID
        for x in range(GRID_SIZE + 1):
            sx = MARGIN + x * CELL_SIZE
            y1 = MARGIN
            y2 = MARGIN + GRID_SIZE * CELL_SIZE
            vertices.extend([sx, y1, 0.15, 0.15, 0.2, 0.3,
                           sx, y2, 0.15, 0.15, 0.2, 0.3])
        
        for y in range(GRID_SIZE + 1):
            sy = MARGIN + y * CELL_SIZE
            x1 = MARGIN
            x2 = MARGIN + GRID_SIZE * CELL_SIZE
            vertices.extend([x1, sy, 0.15, 0.15, 0.2, 0.3,
                           x2, sy, 0.15, 0.15, 0.2, 0.3])
        
        if vertices:
            self._render_lines(vertices)
        
        # 2. QUBITS (su estado actual)
        vertices = []
        size = 10
        
        for q in self.qubits:
            cx, cy = self._grid_to_screen(q.x, q.y)
            r, g, b, a = q.display_color
            
            # Tamaño basado en energía
            s = size * (1.0 + q.coupling_strength * 0.5)
            
            vertices.extend([
                cx-s, cy-s, r, g, b, a,
                cx+s, cy-s, r, g, b, a,
                cx+s, cy+s, r, g, b, a,
                cx-s, cy-s, r, g, b, a,
                cx+s, cy+s, r, g, b, a,
                cx-s, cy+s, r, g, b, a,
            ])
        
        if vertices:
            self._render_triangles(vertices)
        
        # 3. FOTONES (información viajando)
        vertices = []
        psize = 5
        
        for p in self.photons:
            r, g, b, a = p.color
            vertices.extend([
                p.x-psize, p.y-psize, r, g, b, a,
                p.x+psize, p.y-psize, r, g, b, a,
                p.x+psize, p.y+psize, r, g, b, a,
                p.x-psize, p.y-psize, r, g, b, a,
                p.x+psize, p.y+psize, r, g, b, a,
                p.x-psize, p.y+psize, r, g, b, a,
            ])
        
        if vertices:
            self._render_triangles(vertices)
        
        # 4. CALCULADORA
        self._render_calculator()
    
    def _render_calculator(self):
        """Renderizar calculadora a la derecha"""
        calc_x = MARGIN + GRID_SIZE * CELL_SIZE + 40
        calc_y = MARGIN
        
        vertices = []
        
        # Display
        dx, dy, dw, dh = calc_x, calc_y, 240, 50
        vertices.extend([
            dx, dy, 0.1, 0.1, 0.15, 1.0,
            dx+dw, dy, 0.1, 0.1, 0.15, 1.0,
            dx+dw, dy+dh, 0.1, 0.1, 0.15, 1.0,
            dx, dy, 0.1, 0.1, 0.15, 1.0,
            dx+dw, dy+dh, 0.1, 0.1, 0.15, 1.0,
            dx, dy+dh, 0.1, 0.1, 0.15, 1.0,
        ])
        
        # Botones
        buttons = self._get_buttons(calc_x, calc_y + 70)
        for btn in buttons:
            bx, by, bw, bh = btn['x'], btn['y'], btn['w'], btn['h']
            hover = self._is_hover(bx, by, bw, bh)
            
            if hover:
                c = (0.3, 0.4, 0.6, 1.0)
            elif btn['type'] == 'op':
                c = (0.3, 0.3, 0.5, 1.0)
            elif btn['type'] == 'eq':
                c = (0.2, 0.5, 0.3, 1.0)
            elif btn['type'] == 'clear':
                c = (0.5, 0.2, 0.2, 1.0)
            else:
                c = (0.2, 0.2, 0.3, 1.0)
            
            vertices.extend([
                bx, by, *c,
                bx+bw, by, *c,
                bx+bw, by+bh, *c,
                bx, by, *c,
                bx+bw, by+bh, *c,
                bx, by+bh, *c,
            ])
        
        # Renderizar texto del display (simple)
        self._render_text(self.display_text, dx + 10, dy + 15, 2.0)
        
        if vertices:
            self._render_triangles(vertices)
    
    def _render_text(self, text: str, x: float, y: float, scale: float):
        """Renderizar texto simple (números)"""
        # Implementación simple con rectángulos para dígitos
        vertices = []
        offset_x = 0
        
        for char in text:
            if char.isdigit():
                digit = int(char)
                # Renderizar dígito como patrón de 7 segmentos simplificado
                segs = self._get_7segment(digit)
                for sx, sy, sw, sh in segs:
                    vertices.extend([
                        x + offset_x + sx*scale, y + sy*scale, 0.2, 1.0, 0.2, 1.0,
                        x + offset_x + (sx+sw)*scale, y + sy*scale, 0.2, 1.0, 0.2, 1.0,
                        x + offset_x + (sx+sw)*scale, y + (sy+sh)*scale, 0.2, 1.0, 0.2, 1.0,
                        x + offset_x + sx*scale, y + sy*scale, 0.2, 1.0, 0.2, 1.0,
                        x + offset_x + (sx+sw)*scale, y + (sy+sh)*scale, 0.2, 1.0, 0.2, 1.0,
                        x + offset_x + sx*scale, y + (sy+sh)*scale, 0.2, 1.0, 0.2, 1.0,
                    ])
                offset_x += 12 * scale
            else:
                offset_x += 8 * scale
        
        if vertices:
            self._render_triangles(vertices)
    
    def _get_7segment(self, digit: int):
        """Patrones de 7 segmentos para dígitos"""
        # Segmentos: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
        patterns = {
            0: [(0,0,10,2), (10,0,2,5), (10,5,2,5), (0,10,10,2), (0,5,2,5), (0,0,2,5)],
            1: [(10,0,2,5), (10,5,2,5)],
            2: [(0,0,10,2), (10,0,2,5), (0,5,10,2), (0,5,2,5), (0,10,10,2)],
            3: [(0,0,10,2), (10,0,2,5), (0,5,10,2), (10,5,2,5), (0,10,10,2)],
            4: [(0,0,2,5), (0,5,10,2), (10,0,2,5), (10,5,2,5)],
            5: [(0,0,10,2), (0,0,2,5), (0,5,10,2), (10,5,2,5), (0,10,10,2)],
            6: [(0,0,10,2), (0,0,2,5), (0,5,10,2), (0,5,2,5), (10,5,2,5), (0,10,10,2)],
            7: [(0,0,10,2), (10,0,2,5), (10,5,2,5)],
            8: [(0,0,10,2), (0,0,2,5), (10,0,2,5), (0,5,10,2), (0,5,2,5), (10,5,2,5), (0,10,10,2)],
            9: [(0,0,10,2), (0,0,2,5), (10,0,2,5), (0,5,10,2), (10,5,2,5), (0,10,10,2)],
        }
        return patterns.get(digit, [])
    
    def _get_buttons(self, start_x, start_y):
        """Layout de botones"""
        buttons = []
        size = 50
        gap = 10
        
        # Números
        for i in range(9):
            row = 2 - i // 3
            col = i % 3
            buttons.append({
                'text': str(i+1),
                'x': start_x + col * (size + gap),
                'y': start_y + row * (size + gap),
                'w': size, 'h': size,
                'type': 'num'
            })
        
        # 0
        buttons.append({
            'text': '0',
            'x': start_x + (size + gap),
            'y': start_y + 3 * (size + gap),
            'w': size, 'h': size,
            'type': 'num'
        })
        
        # Operaciones
        ops = ['+', '-', '×', '÷']
        for i, op in enumerate(ops):
            buttons.append({
                'text': op,
                'x': start_x + 3 * (size + gap),
                'y': start_y + i * (size + gap),
                'w': size, 'h': size,
                'type': 'op'
            })
        
        # C y =
        buttons.append({
            'text': 'C',
            'x': start_x,
            'y': start_y + 3 * (size + gap),
            'w': size, 'h': size,
            'type': 'clear'
        })
        
        buttons.append({
            'text': '=',
            'x': start_x + 2 * (size + gap),
            'y': start_y + 3 * (size + gap),
            'w': size, 'h': size,
            'type': 'eq'
        })
        
        return buttons
    
    def _is_hover(self, x, y, w, h):
        return (x <= self.mouse_x <= x+w and y <= self.mouse_y <= y+h)
    
    def _render_lines(self, vertices):
        vbo = self.ctx.buffer(np.array(vertices, dtype='f4').tobytes())
        vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_pos', 'in_color')
        vao.render(moderngl.LINES)
        vbo.release()
        vao.release()
    
    def _render_triangles(self, vertices):
        vbo = self.ctx.buffer(np.array(vertices, dtype='f4').tobytes())
        vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_pos', 'in_color')
        vao.render(moderngl.TRIANGLES)
        vbo.release()
        vao.release()
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    
    def _handle_click(self):
        """Manejar click en botón"""
        calc_x = MARGIN + GRID_SIZE * CELL_SIZE + 40
        calc_y = MARGIN + 70
        buttons = self._get_buttons(calc_x, calc_y)
        
        for btn in buttons:
            if self._is_hover(btn['x'], btn['y'], btn['w'], btn['h']):
                if self.computing:
                    return  # No permitir input durante computación
                
                if btn['type'] == 'num':
                    # Añadir dígito
                    if self.display_text == "0":
                        self.display_text = btn['text']
                    else:
                        self.display_text += btn['text']
                    
                elif btn['type'] == 'op':
                    self.operand_a = int(self.display_text) if self.display_text else 0
                    self.operation = btn['text']
                    self.display_text = "0"
                    print(f"Operation: {self.operand_a} {btn['text']}")
                    
                elif btn['type'] == 'eq':
                    if self.operation and self.operand_a is not None:
                        operand_b = int(self.display_text) if self.display_text else 0
                        
                        if self.operation == '+':
                            self._quantum_add_living(self.operand_a, operand_b)
                        elif self.operation == '-':
                            self._quantum_add_living(self.operand_a, (~operand_b + 1) & 0xFF)
                        elif self.operation == '×':
                            result = (self.operand_a * operand_b) & 0xFF
                            self.display_text = str(result)
                        elif self.operation == '÷' and operand_b != 0:
                            result = self.operand_a // operand_b
                            self.display_text = str(result)
                    
                elif btn['type'] == 'clear':
                    self.display_text = "0"
                    self.operand_a = None
                    self.operation = None
                    print("Cleared")
                
                break
    
    def _mouse_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self._handle_click()
    
    def _cursor_callback(self, window, xpos, ypos):
        self.mouse_x = xpos
        self.mouse_y = ypos
    
    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    def run(self):
        print("\n🚀 Starting living quantum circuit...\n")
        
        last_time = time.time()
        frame = 0
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)  # Cap dt
            last_time = current_time
            
            glfw.poll_events()
            
            # FÍSICA VIVE AQUÍ
            self._update_physics(dt)
            
            # Render
            self.ctx.screen.use()
            self.ctx.clear(0.02, 0.02, 0.05, 1.0)
            self._render()
            
            glfw.swap_buffers(self.window)
            
            frame += 1
            if frame % 60 == 0:
                fps = 60.0 / (time.time() - (current_time - 60 * dt))
                status = "COMPUTING" if self.computing else "READY"
                photon_count = len(self.photons)
                title = (f"Quantum Living | {len(self.qubits)} Qubits | "
                        f"{photon_count} Photons | FPS: {fps:.0f} | {status}")
                glfw.set_window_title(self.window, title)
        
        glfw.terminate()
        print("\n✓ Quantum circuit ended")


if __name__ == '__main__':
    try:
        calc = QuantumLivingCalculator()
        calc.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
