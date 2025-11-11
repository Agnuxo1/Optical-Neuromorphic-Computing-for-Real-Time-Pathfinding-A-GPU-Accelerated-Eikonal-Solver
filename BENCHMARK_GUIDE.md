# Optical Neuromorphic Eikonal Solver – Benchmark Guide

This guide describes how to obtain datasets, convert them to the solver format, and run reproducible GPU vs CPU benchmarks.

## 1. Preparar datasets

Todos los conversores se encuentran en `benchmarks/prepare_datasets.py`.

### CMAP Maze Benchmark
1. Descarga el generador oficial (`maze_generator.py`) desde http://mazebenchmark.github.io.
2. Genera los laberintos deseados:
   ```bash
   python maze_generator.py --size 256 --connectivity 0.3 --output raw/cmap/maze_256_c03.npy
   ```
3. Convierte al formato `.npz` del solver:
   ```bash
   python -m benchmarks.prepare_datasets cmap \
       --input raw/cmap/maze_256_c03.npy \
       --output cases/cmap/maze_256_c03.npz \
       --connectivity 0.3
   ```

### MovingAI Benchmarks
1. Descarga mapas y escenarios desde http://movingai.com/benchmarks/.
2. Convierte un mapa y sus escenarios asociados:
   ```bash
   python -m benchmarks.prepare_datasets movingai \
       --map raw/movingai/mazes/maze512-32-0.map \
       --scen raw/movingai/mazes/maze512-32-0.map.scen \
       --output cases/movingai/maze512-32-0 \
       --limit 50
   ```

### Suite sintética de referencia
```bash
python -m benchmarks.prepare_datasets synthetic --output cases/synthetic
```
El flag `--scale` permite ajustar los tamaños de los grids de forma uniforme (p. ej. `--scale 0.5`).

## 2. Ejecutar benchmarks

Usa `benchmarks/run_suite.py` para lanzar comparativas GPU vs CPU sobre un directorio de `.npz`.

```bash
python -m benchmarks.run_suite \
    --cases cases/cmap \
    --output results/cmap.csv \
    --iterations 300
```

- `--iterations` controla cuántos pasos de propagación se aplican a cada caso.
- `--grid` permite fijar un tamaño de grid del solver (si los casos ya fueron remuestreados).
- El script produce un CSV con métricas por caso e imprime un resumen en consola.

## 3. Resumen de resultados

Convierte uno o varios CSV a un informe Markdown:

```bash
python -m benchmarks.report \
    --results results/cmap.csv results/movingai.csv \
    --output reports/summary.md
```

El informe incluye:
- Número de casos evaluados.
- Speedup medio (CPU vs GPU).
- MAE medio entre los campos de tiempos.
- Ratio medio de longitud de camino (≈1.0 indica trayectorias óptimas).

## 4. Buenas prácticas

- **Headless GPU**: el harness crea una ventana GLFW oculta; ejecuta en un entorno con drivers OpenGL disponibles.
- **Semillas**: la suite sintética usa semillas fijas para garantizar reproducibilidad.
- **Escalabilidad**: para medir escalado empírico (`O(n²)`), genera suites sintéticas de 128, 256, 512 y 1024.
- **Validación**: revisa el MAE; valores < 0.02 se consideran equivalentes a Dijkstra.
- **Documentación**: adjunta `BENCHMARK_GUIDE.md`, los CSV generados y el informe Markdown cuando prepares material académico.

Con estos pasos, el Optical Neuromorphic Eikonal Solver puede reproducirse contra benchmarks estándar y reportar métricas comparables a la literatura en pathfinding.


