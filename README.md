# Proyecto final: Programación Genética (A, B y C)

Tres implementaciones del mismo sistema de programación genética (regresión simbólica con
árboles), con las mismas reglas y verificadas entre sí:

| Carpeta | Implementación | Categoría del torneo |
|---|---|---|
| [`GP_JL_A/`](GP_JL_A/) | A: convencional de alto nivel (Julia) | no compite (referencia) |
| [`GP_CPP20_B/`](GP_CPP20_B/) | B: convencional de bajo nivel (C++20) | 1: más rápida convencional |
| [`GP_JL_C/`](GP_JL_C/) | C: no convencional, toda la población en un solo grafo (Julia) | 2: más rápida no convencional |

**Para entender el proyecto empieza por [`docs/README.md`](docs/README.md):** explica, con
diagramas, qué es un individuo, el MSE, cada función del manual y **todas las técnicas de
eficiencia** de B y C, con sus mediciones.

- `benchmarks\benchmark_A.ps1`, `benchmark_B.ps1` y `benchmark_C.ps1`: el benchmark de cada
  implementación por separado, como lo pide el manual (crossover con 10,000 árboles en
  profundidades 7, 9, 10, 12 y 15; GP completo con población 5,000 en profundidades 7, 9 y 12),
  con 100 generaciones, semilla 5 por defecto, un proceso por profundidad y el mejor MSE de
  cada generación (en pantalla mientras corre y en un CSV). Comandos, parámetros que se pueden cambiar y cómo leer los resultados:
  [`benchmarks/LEEME_benchmarks.txt`](benchmarks/LEEME_benchmarks.txt).
- `correr_benchmarks.ps1`: las tres implementaciones juntas, con la verificación cruzada
  entre A, B y C antes de medir (Windows).
- `verificar_cambios.ps1`: verifica el último cambio de C (lista postfija y cache al
  reusarse): pruebas y GP completo con y sin el cambio; el MSE de cada generación debe ser igual.

**Optimizaciones que se pueden apagar** (B y C; todas dan exactamente el mismo resultado,
solo cambian el tiempo): `--no-fold` (plegado de constantes), `--no-cut` (corte temprano
exacto), `--no-order` (evaluar primero los hijos de mejores padres) y `--no-dup` (hijos
idénticos a su padre reciben su MSE sin evaluarse). Los scripts de benchmark las apagan
con `-SinPendientes`. Solo en C: `--no-tape` (evaluar con la recursión en vez de la lista
postfija) y `--reuse-min-size=0` (sin cache en los subárboles chicos que se reusan); el
profesor ya aceptó ambas ideas, así que no entran en `-SinPendientes`. C también recorre
la lista por bloques de 64 filas (evaluación por lotes, que el profesor dijo que no es
válida; se dejó por decisión del equipo): `--no-batch` la apaga con el mismo resultado. El detalle y las mediciones están en
[`docs/README.md`](docs/README.md#5-catálogo-de-técnicas-de-eficiencia).
