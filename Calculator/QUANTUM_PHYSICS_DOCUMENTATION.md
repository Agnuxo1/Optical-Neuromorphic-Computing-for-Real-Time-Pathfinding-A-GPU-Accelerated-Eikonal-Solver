# QUANTUM-PHOTONIC PROCESSOR - TECHNICAL DOCUMENTATION
=====================================================

## 🔬 FÍSICA IMPLEMENTADA

### 1. MECÁNICA CUÁNTICA REAL

#### Estados Cuánticos
```
|ψ⟩ = α|0⟩ + β|1⟩

Donde:
- α, β ∈ ℂ (números complejos)
- |α|² + |β|² = 1 (normalización)
- |α|² = probabilidad de medir |0⟩
- |β|² = probabilidad de medir |1⟩
```

#### Esfera de Bloch
Los qubits se representan en la esfera de Bloch:
- |0⟩ (Estado 0): Polo norte - Color ROJO
- |1⟩ (Estado 1): Polo sur - Color AZUL  
- |+⟩ = (|0⟩ + |1⟩)/√2: Ecuador X - Color VERDE
- |-⟩ = (|0⟩ - |1⟩)/√2: Ecuador -X - Color AMARILLO

### 2. ÓPTICA CUÁNTICA

#### Fotones
```python
E = hf  # Energía del fotón (Planck)
λ = c/f # Longitud de onda
```

Propiedades implementadas:
- **Frecuencia**: ~500 THz (luz visible)
- **Fase**: θ ∈ [0, 2π]
- **Polarización**: Estado complejo en 2D
- **Amplitud**: Decaimiento exponencial
- **Propagación**: Velocidad c = 299,792,458 m/s

#### Colores según longitud de onda:
```
380-450 nm → Violeta
450-495 nm → Azul
495-570 nm → Verde  
570-590 nm → Amarillo
590-620 nm → Naranja
620-750 nm → Rojo
```

### 3. PUERTAS CUÁNTICAS UNIVERSALES

#### Puerta Hadamard (H)
```
H = 1/√2 [ 1   1 ]
          [ 1  -1 ]

|0⟩ → H → (|0⟩ + |1⟩)/√2 = |+⟩
|1⟩ → H → (|0⟩ - |1⟩)/√2 = |-⟩
```
Crea superposición cuántica.

#### Puerta Pauli-X (NOT Cuántico)
```
X = [ 0  1 ]
    [ 1  0 ]

|0⟩ → X → |1⟩
|1⟩ → X → |0⟩
```
Intercambia estados.

#### Puerta Pauli-Y
```
Y = [  0  -i ]
    [  i   0 ]
```
Rotación + cambio de fase.

#### Puerta Pauli-Z (Cambio de Fase)
```
Z = [ 1   0 ]
    [ 0  -1 ]

|0⟩ → Z → |0⟩
|1⟩ → Z → -|1⟩
```
Cambia fase del estado |1⟩.

#### Puerta CNOT (Controlled-NOT)
```
CNOT = [ 1  0  0  0 ]
       [ 0  1  0  0 ]
       [ 0  0  0  1 ]
       [ 0  0  1  0 ]

|00⟩ → |00⟩
|01⟩ → |01⟩
|10⟩ → |11⟩  (flip)
|11⟩ → |10⟩  (flip)
```
Control: primer qubit
Target: segundo qubit

#### Puerta Toffoli (CCNOT - 3 qubits)
```
Control1 ∧ Control2 → NOT Target

Solo aplica NOT si ambos controles son |1⟩
```
Puerta universal reversible - puede implementar cualquier función booleana.

### 4. CIRCUITOS ARITMÉTICOS CUÁNTICOS

#### Sumador Completo (Full Adder)
```
Inputs: a, b, carry_in
Outputs: sum, carry_out

sum = a ⊕ b ⊕ carry_in
carry_out = (a ∧ b) ∨ (carry_in ∧ (a ⊕ b))
```

Implementación con puertas cuánticas:
1. **Sum**: 3 puertas CNOT
   - CNOT(a, sum)
   - CNOT(b, sum)  
   - CNOT(carry_in, sum)

2. **Carry**: 3 puertas Toffoli
   - Toffoli(a, b, carry_out)
   - Toffoli(a, carry_in, carry_out)
   - Toffoli(b, carry_in, carry_out)

#### Sumador de Propagación (Ripple Carry Adder)
```
Para sumar números de n bits:
A = a₇a₆a₅a₄a₃a₂a₁a₀
B = b₇b₆b₅b₄b₃b₂a₁b₀

Conectar n sumadores completos en cascada:
Carry₀ → FA₀ → Carry₁ → FA₁ → ... → Carryₙ
```

## 🎮 ARQUITECTURA DEL PROCESADOR

### Grid de Qubits
```
20×20 = 400 Qubits totales

Regiones especializadas:
┌─────────────────────────┐
│ Input A  [0-7]          │ ← Bits 0-7 (primera fila)
│ Input B  [20-27]        │ ← Bits 0-7 (segunda fila)
│                         │
│   ... Processing ...    │ ← Puertas cuánticas activas
│                         │
│ Output   [200-207]      │ ← Resultado (fila 10)
│ Carry    [40-49]        │ ← Bits de acarreo
└─────────────────────────┘
```

### Flujo de Computación

1. **Codificación**
   ```python
   number = 42  # Decimal
   binary = 0b00101010  # 8 bits
   
   # Codificar en qubits
   for bit in binary:
       if bit == 1:
           apply X gate  # |0⟩ → |1⟩
   ```

2. **Procesamiento**
   ```
   Cola de puertas cuánticas → Aplicar secuencialmente
   
   Cada puerta:
   - Modifica estado cuántico
   - Emite fotones
   - Propaga información
   ```

3. **Medición**
   ```python
   # Colapso de función de onda
   result = measure(qubits)
   
   # Conversión a decimal
   decimal = sum(bit << i for i, bit in enumerate(result))
   ```

## 🧮 OPERACIONES DE LA CALCULADORA

### Suma (A + B)
```
Algoritmo:
1. Codificar A en input_a_region
2. Codificar B en input_b_region
3. Construir circuito sumador:
   - 8 sumadores completos
   - Propagación de acarreo
4. Ejecutar puertas cuánticas
5. Medir output_region
6. Decodificar resultado
```

### Resta (A - B)
```
Usa complemento a dos:
A - B = A + (~B + 1)

1. Calcular complemento de B
2. Sumar 1
3. Usar sumador cuántico
```

### Multiplicación (A × B)
```
Algoritmo de suma repetida optimizado:
result = 0
for i in range(8):
    if bit_i(B) == 1:
        result += A << i

Implementación cuántica usa multiplicadores
de Wallace o Booth (más complejo)
```

### División (A ÷ B)
```
División entera:
quotient = A // B

Implementado con resta repetida
o algoritmo de Newton-Raphson
```

## 🎨 VISUALIZACIÓN

### Colores de Qubits
- 🔴 **Rojo**: |0⟩ (Estado fundamental)
- 🔵 **Azul**: |1⟩ (Estado excitado)
- 🟢 **Verde**: |+⟩ (Superposición positiva)
- 🟡 **Amarillo**: |-⟩ (Superposición negativa)

### Intensidad
```
brightness = |α|² × sin(phase)

La intensidad muestra:
- Probabilidad del estado
- Fase cuántica (pulsación)
```

### Fotones
```
color = wavelength_to_rgb(λ)
intensity = amplitude

Los fotones muestran:
- Transferencia de información
- Entrelazamiento
- Interferencia cuántica
```

## 🎹 INTERFAZ

### Teclado de Calculadora
```
┌─────┬─────┬─────┬─────┐
│  7  │  8  │  9  │  ÷  │
├─────┼─────┼─────┼─────┤
│  4  │  5  │  6  │  ×  │
├─────┼─────┼─────┼─────┤
│  1  │  2  │  3  │  -  │
├─────┼─────┼─────┼─────┤
│  C  │  0  │  =  │  +  │
└─────┴─────┴─────┴─────┘
```

### Controles
- **Click**: Botones de calculadora
- **[Space]**: Pausar/Reanudar computación
- **[G]**: Toggle grid
- **[P]**: Toggle fotones
- **[ESC]**: Salir

## 📊 EJEMPLO DE USO

### Suma: 42 + 17 = 59

```python
# 1. Input
A = 42  # 0b00101010
B = 17  # 0b00010001

# 2. Codificación cuántica
Input A region: |0⟩|1⟩|0⟩|1⟩|0⟩|1⟩|0⟩|0⟩
Input B region: |1⟩|0⟩|0⟩|0⟩|1⟩|0⟩|0⟩|0⟩

# 3. Aplicar puertas
CNOT(a₀, sum₀)
CNOT(b₀, sum₀)
Toffoli(a₀, b₀, c₁)
... (24 puertas más)

# 4. Resultado
Output: |1⟩|1⟩|0⟩|1⟩|1⟩|1⟩|0⟩|0⟩
Decimal: 59 ✓
```

### Visualización del Proceso
```
Frame 1: Codificación
- Qubits rojos/azules según bits
- Sin fotones

Frame 2-30: Procesamiento  
- Puertas aplicándose secuencialmente
- Fotones propagándose
- Estados cambiando
- Colores pulsando

Frame 31: Resultado
- Output estable
- Fotones desapareciendo
- Display mostrando "59"
```

## 🔧 IMPLEMENTACIÓN TÉCNICA

### Tecnologías
- **moderngl**: Renderizado GPU (OpenGL 4.3)
- **numpy**: Álgebra lineal cuántica
- **glfw**: Ventanas y input
- **Python 3.8+**: Lenguaje base

### Rendimiento
```
Qubits: 400
Puertas/segundo: ~50
Fotones simultáneos: ~100
FPS: 60
Latencia computación: <1s para 8 bits
```

### Precisión
```
Bits: 8
Rango: 0-255
Errores cuánticos: <0.01%
```

## 🚀 EXTENSIONES FUTURAS

1. **Más qubits**: 32×32 = 1024 qubits → 16 bits
2. **Corrección de errores**: Códigos de Shor/Steane
3. **Algoritmos avanzados**: Shor, Grover
4. **Optimización**: Compute shaders, paralelización
5. **Simulación realista**: Decoherencia, ruido

## 📚 REFERENCIAS

- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- Feynman: "Quantum Mechanics and Path Integrals"
- Preskill: "Lecture Notes on Quantum Computation"
- OpenQL: Quantum programming framework

---

**Nota**: Este es un simulador educativo que implementa los principios
fundamentales de la computación cuántica. Los sistemas cuánticos reales
requieren temperaturas criogénicas y aislamiento del entorno.
