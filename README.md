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

- `correr_benchmarks.ps1`: pruebas, verificación cruzada y benchmarks del manual (Windows).
- `verificar_cambios.ps1`: compara C contra una copia de respaldo anterior.
