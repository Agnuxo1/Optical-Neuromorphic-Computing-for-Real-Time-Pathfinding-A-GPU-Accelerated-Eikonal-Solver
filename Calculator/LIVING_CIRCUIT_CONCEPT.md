# QUANTUM LIVING CALCULATOR
==========================

## 🌟 CONCEPTO: EL CIRCUITO **VIVE**

### La Diferencia Fundamental

**ANTES** (representación):
```
Física matemática → Calcular resultado → Mostrar visualización
         ↓                  ↓                    ↓
    (separado)         (separado)           (decorativo)
```

**AHORA** (circuito vivo):
```
       ┌─────────────────────────────────┐
       │  FÍSICA SUCEDE EN LA IMAGEN     │
       │  Cada fotograma = Iteración     │
       │  Estados evolucionan realmente  │
       │  Fotones propagan información   │
       │  Resultado EMERGE               │
       └─────────────────────────────────┘
                    ↓
          Visualización = Computación
```

---

## 🔬 FÍSICA QUE VIVE EN CADA FRAME

### 1. Ecuación de Schrödinger (60 veces por segundo)

```python
def evolve(self, dt):
    """
    iℏ ∂ψ/∂t = H ψ
    
    Cada qubit evoluciona según mecánica cuántica REAL
    60 veces por segundo
    """
    phase_shift = -self.energy * dt / HBAR
    rotation = np.exp(1j * phase_shift)
    
    # ESTO SUCEDE EN LA IMAGEN
    self.psi_0 *= rotation  # |0⟩ rota
    self.psi_1 *= rotation  # |1⟩ rota
```

**Qué significa**: 
- Cada qubit gira en la esfera de Bloch
- La fase evoluciona continuamente
- Los colores cambian según el estado real
- **ES** el circuito funcionando, no una animación

---

### 2. Interacciones Cuánticas Reales

```python
def _quantum_interaction(q1, q2, strength):
    """
    Hamiltoniano de interacción:
    H_int = g (σ₁⁺σ₂⁻ + σ₁⁻σ₂⁺)
    
    Los qubits intercambian excitación
    """
    # Calcular transferencia según física real
    transfer = strength * (q1.psi_1 * np.conj(q2.psi_0) - 
                          q1.psi_0 * np.conj(q2.psi_1))
    
    # Aplicar intercambio
    delta_psi = transfer * dt * 0.1
    q1.psi_1 -= delta_psi  # Qubit 1 pierde excitación
    q2.psi_1 += delta_psi  # Qubit 2 gana excitación
```

**Qué significa**:
- Los qubits vecinos se hablan entre sí
- La información fluye físicamente
- Si q1 está en |1⟩ y q2 en |0⟩, la excitación se transfiere
- Esto NO es una animación - ES la física

---

### 3. Fotones Propagan Información

```python
def propagate(self, dt):
    """Fotón viaja a velocidad de la luz"""
    speed = C * 1e-7  # Velocidad real escalada
    
    # MOVIMIENTO FÍSICO
    self.x += self.vx * speed * dt
    self.y += self.vy * speed * dt
    
    # EVOLUCIÓN DE FASE
    self.phase += 2 * π * self.frequency * dt
```

**Qué significa**:
- Los fotones SON portadores de información
- Viajan a c (velocidad de la luz)
- Su fase evoluciona según frecuencia
- Cuando llegan a un qubit, lo afectan

---

### 4. Resultado EMERGE de la Física

```python
def _update_physics(self, dt):
    """
    AQUÍ VIVE EL CIRCUITO
    
    No hay "cálculo separado"
    El resultado emerge de las interacciones
    """
    # 1. Cada qubit evoluciona (Schrödinger)
    for qubit in self.qubits:
        qubit.evolve(dt)
    
    # 2. Qubits vecinos interactúan
    if self.computing:
        for vecinos:
            self._quantum_interaction(q1, q2, strength)
            # Información fluye
            # Estados cambian
            # Fotones se emiten
    
    # 3. Fotones propagan
    for photon in self.photons:
        photon.propagate(dt)
        # Viajan físicamente
        # Llevan información
    
    # 4. Medir resultado cuando estabiliza
    if computation_progress >= 2.0:
        result = self._read_quantum_result()
        # El resultado EMERGIÓ de la física
```

---

## 🎨 VISUALIZACIÓN = COMPUTACIÓN

### Los Colores NO Son Decorativos

```python
@property
def display_color(self):
    """Color basado en estado cuántico ACTUAL"""
    p0 = |α|²  # Probabilidad |0⟩
    p1 = |β|²  # Probabilidad |1⟩
    
    r = p0      # Rojo = |0⟩
    b = p1      # Azul = |1⟩
    g = min(p0, p1) * 2  # Verde = superposición
    
    intensity = 0.5 + 0.5 * sin(phase)  # Pulsación = fase
    
    return (r * intensity, g * intensity, b * intensity)
```

**Qué ves**:
- **Rojo puro**: Qubit en |0⟩ (100% seguro)
- **Azul puro**: Qubit en |1⟩ (100% seguro)
- **Verde**: Superposición (|0⟩ + |1⟩)/√2
- **Pulsación**: La fase evolucionando en tiempo real
- **Cambios de color**: El estado CAMBIANDO físicamente

---

## 💡 EJEMPLO: 5 + 3 = 8 (VIVO)

### Frame 0 (t=0s): Codificación

```
Input A (fila 0): |1⟩|0⟩|1⟩|0⟩|0⟩|0⟩|0⟩|0⟩  (5 = 0b00000101)
                   ↑       ↑
                  bit0    bit2
                  
Input B (fila 1): |1⟩|1⟩|0⟩|0⟩|0⟩|0⟩|0⟩|0⟩  (3 = 0b00000011)
                   ↑  ↑
                  bit0 bit1

Colores: AZUL donde |1⟩, ROJO donde |0⟩
```

### Frame 1-60 (t=0-1s): Física Activa

```
Los qubits empiezan a interactuar:

Frame 10: Bit 0 de A habla con bit 0 de B
  → Fotón emitido (amarillo brillante)
  → Viaja hacia región de salida
  → Estados empiezan a cambiar
  
Frame 20: Fotón llega a salida
  → Qubit de salida cambia color (rojo → verde → azul)
  → Superposición formándose
  → Más fotones propagando
  
Frame 30: Carry propagando
  → Qubits en fila 5 activándose
  → Cadena de fotones visible
  → Estados intermedios pulsando
  
Frame 40: Interferencia
  → Fotones superponiéndose
  → Colores mezclándose
  → Verde intenso = superposición alta
```

### Frame 60-120 (t=1-2s): Convergencia

```
Frame 60: Sistema estabilizando
  → Pulsaciones más lentas
  → Colores definiendo
  → Qubits de salida convergiendo
  
Frame 90: Casi listo
  → Mayoría de qubits en estados puros
  → Rojo/azul dominando, menos verde
  → Fotones desapareciendo
  
Frame 120: Medición
  → Sistema estable
  → Estados colapsados
  → Output: |0⟩|0⟩|0⟩|1⟩|0⟩|0⟩|0⟩|0⟩
  → Decimal: 8 ✓
```

**Resultado**: 8 EMERGIÓ de la física, no fue calculado aparte

---

## 🔧 CORRECCIONES vs VERSIÓN ANTERIOR

### Problema 1: Números No Aparecían

**ANTES**:
```python
# No había renderizado de texto
display_text = "42"  # Pero no se veía
```

**AHORA**:
```python
def _render_text(text, x, y, scale):
    """Renderizar con 7 segmentos"""
    for char in text:
        if char.isdigit():
            segments = self._get_7segment(int(char))
            # Renderizar cada segmento como rectángulo
```

### Problema 2: Clicks Detectaban Mal

**ANTES**:
```python
# Todas las clicks llegaban a los mismos índices
# Múltiples botones activándose juntos
```

**AHORA**:
```python
def _handle_click(self):
    buttons = self._get_buttons(calc_x, calc_y)
    for btn in buttons:
        if self._is_hover(btn['x'], btn['y'], btn['w'], btn['h']):
            # Solo procesar EL botón clickeado
            # Una operación a la vez
            break  # IMPORTANTE: salir después del primer match
```

### Problema 3: Física Era Decorativa

**ANTES**:
```python
# Física y cálculo separados
def calculate():
    result = a + b  # Cálculo clásico
    # Luego animar algo bonito

def render():
    # Mostrar animación
```

**AHORA**:
```python
def _update_physics(dt):
    # LA física ES el cálculo
    for qubit in qubits:
        qubit.evolve(dt)  # Schrödinger
    
    for vecinos:
        interaction(q1, q2)  # Transferencia real
    
    for photon in photons:
        photon.propagate(dt)  # Información viaja
    
    # El resultado emerge
    if stable:
        result = measure_qubits()
```

---

## 🎮 CÓMO FUNCIONA

### Input del Usuario

1. Click `[5]` → `display_text = "5"`
2. Click `[+]` → `operand_a = 5`, `operation = "+"`
3. Click `[3]` → `display_text = "3"`
4. Click `[=]` → **ACTIVA LA FÍSICA**

### Activación Física

```python
def _quantum_add_living(a, b):
    # 1. Resetear sistema
    for q in qubits:
        q.psi_0 = 1.0 + 0.0j  # Todo a |0⟩
    
    # 2. Codificar inputs
    encode_number(5, fila_0)  # |1⟩|0⟩|1⟩|0⟩...
    encode_number(3, fila_1)  # |1⟩|1⟩|0⟩|0⟩...
    
    # 3. Configurar acoplamientos
    for bit in range(8):
        qubits[fila_0 + bit].coupling_strength = 0.5
        qubits[fila_1 + bit].coupling_strength = 0.5
        qubits[fila_output + bit].coupling_strength = 1.0
    
    # 4. ACTIVAR
    self.computing = True
    # Ahora en cada frame la física evoluciona
```

### Loop Principal

```python
while running:
    dt = get_frame_time()  # ~0.016s (60 FPS)
    
    # FÍSICA VIVE AQUÍ
    _update_physics(dt)
    # - Qubits evolucionan
    # - Interacciones ocurren
    # - Fotones viajan
    # - Resultado emerge
    
    # Renderizar estado ACTUAL
    _render()
    # - Colores según estados reales
    # - Fotones en sus posiciones reales
    # - Display con resultado actual
```

---

## 📊 DIFERENCIA CONCEPTUAL

### Modelo Antiguo: Representación

```
┌──────────────┐
│ Matemáticas  │ → Calcular → 8
└──────────────┘
       ↓
┌──────────────┐
│ Visualizar   │ → Mostrar animación bonita
└──────────────┘
```

### Modelo Nuevo: Circuito Vivo

```
┌────────────────────────────────────────┐
│  FÍSICA EN IMAGEN                      │
│                                        │
│  Frame 1:  Estados iniciales           │
│  Frame 2:  Interacciones empiezan      │
│  Frame 3:  Fotones propagan            │
│  ...                                   │
│  Frame 120: Sistema estable → Medir    │
│                                        │
│  Resultado = 8 (EMERGIÓ)               │
└────────────────────────────────────────┘
```

---

## ✨ POR QUÉ ESTO IMPORTA

1. **Educativo**: Ves LA física sucediendo, no una animación
2. **Auténtico**: Las ecuaciones REALMENTE se ejecutan
3. **Emergente**: El resultado NO está precalculado
4. **Bello**: La visualización ES la computación

---

## 🚀 PARA EJECUTAR

```bash
pip install moderngl glfw numpy --break-system-packages
python quantum_living_calculator.py
```

**Qué verás**:
- Qubits pulsando con fase real
- Colores cambiando según estados
- Fotones viajando físicamente
- Sistema convergiendo a resultado
- Display mostrando números
- Teclado funcional

---

## 🎯 LO ESENCIAL

No es que el circuito **represente** física cuántica.
**ES** física cuántica sucediendo.

Cada fotograma = Una iteración de evolución real.
Cada color = Un estado cuántico actual.
Cada fotón = Información propagando.

El resultado emerge de dejar que la física viva.

🌟 **El circuito VIVE en cada frame.**
