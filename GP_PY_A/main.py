"""Interfaz de terminal de la implementacion A (mismas opciones que B y C).

  python main.py --test
  python main.py --demo [--data=data/ejemplo.csv]
  python main.py --bench-crossover [--trees=10000] [--depths=7,9,10,12,15]
                                   [--variables=2] [--seed=42] [--batch-pairs=4]
  python main.py --bench-gp [--data=dataset.csv] [--population=5000]
                            [--depths=7,9,12] [--generations=10] [--seed=42]
                            [--engine=numpy|scalar]
  python main.py --eval-file --file=arboles.txt [--data=dataset.csv] [--engine=...]
       Lee una expresion por linea (formato de B/C) e imprime
       MSE<TAB>profundidad<TAB>nodos<TAB>expresion, con MSE en 17 digitos
       (misma salida que gp_c --eval-file).
  python main.py --export-trees [--count=100] [--depth=5] [--variables=2] [--seed=42]
                                [--full]
       Escribe arboles grow_random (o grow_full con --full) sin perdida.

El CSV tiene encabezado opcional y la ultima columna es y. El manual NO fija
el numero de generaciones (10 por defecto, igual que B y C).
"""

from __future__ import annotations

import sys
import time

import gp

DEFAULTS = {
    "data": "data/ejemplo.csv", "depths": "", "file": "", "trees": 10000,
    "population": 5000, "generations": 10, "variables": 2, "seed": 42,
    "batch_pairs": 4, "count": 100, "depth": 5, "engine": "auto", "full": False,
}


def parse_args(argv: list[str]) -> tuple[str, dict]:
    mode = argv[0] if argv else "--help"
    opts = dict(DEFAULTS)
    for arg in argv[1:]:
        if arg == "--full":
            opts["full"] = True
            continue
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"Se esperaba --opcion=valor: {arg}")
        key, value = arg[2:].split("=", 1)
        key = key.replace("-", "_")
        if key not in DEFAULTS:
            raise ValueError(f"Opcion desconocida: --{key}")
        opts[key] = int(value) if isinstance(DEFAULTS[key], int) else value
    if opts["engine"] not in ("auto", "numpy", "scalar"):
        raise ValueError("--engine debe ser auto, numpy o scalar")
    return mode, opts


def parse_depths(text: str, default: str) -> list[int]:
    depths = [int(item) for item in (text or default).split(",") if item.strip()]
    if not depths:
        raise ValueError("Lista de profundidades vacia")
    return depths


def main(argv: list[str]) -> int:
    mode, o = parse_args(argv)
    if mode in ("--help", "-h"):
        print(__doc__)
    elif mode == "--test":
        verbose = "-v" in argv
        count = gp.run_tests(verbose=verbose)
        engine = "numpy y escalar" if gp.np is not None else "solo escalar (sin numpy)"
        print(f"OK: {count} comprobaciones correctas (motores: {engine}).")
    elif mode == "--demo":
        data = gp.Dataset.from_csv(o["data"])
        result = gp.full_gp(data, gp.GPOptions(population_size=40, max_depth=4,
                                               generations=5, seed=o["seed"],
                                               engine=o["engine"]))
        print(f"MSE final: {result.best_fitness:.6g}")
        print(f"Expresion: {result.best.expression()}")
        print("Mejor MSE por generacion:")
        for i, value in enumerate(result.best_fitness_per_generation):
            print(f"  Generacion {i}: {value:.6g}")
    elif mode == "--bench-crossover":
        print("depth,trees,crossovers,elapsed_seconds,mean_seconds_per_crossover,mean_nodes",
              flush=True)
        for depth in parse_depths(o["depths"], "7,9,10,12,15"):
            t = gp.benchmark_crossover(o["trees"], depth, o["variables"], o["seed"],
                                       o["batch_pairs"])
            print(f"{t.depth},{t.trees},{t.crossovers},{t.seconds:.9g},"
                  f"{t.mean_seconds_per_crossover:.9g},{t.mean_nodes_per_tree:.6g}",
                  flush=True)
    elif mode == "--bench-gp":
        data = gp.Dataset.from_csv(o["data"])
        print("depth,population,generations,elapsed_seconds,best_mse,best_expression",
              flush=True)
        for depth in parse_depths(o["depths"], "7,9,12"):
            options = gp.GPOptions(population_size=o["population"], max_depth=depth,
                                   generations=o["generations"], seed=o["seed"],
                                   engine=o["engine"])
            start = time.perf_counter()
            result = gp.full_gp(data, options)
            elapsed = time.perf_counter() - start
            print(f'{depth},{options.population_size},{options.generations},{elapsed:.9g},'
                  f'{result.best_fitness:.9g},"{result.best.expression()}"', flush=True)
    elif mode == "--eval-file":
        if not o["file"]:
            raise ValueError("Falta --file=arboles.txt")
        data = gp.Dataset.from_csv(o["data"])
        with open(o["file"], encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                tree = gp.parse_expression(line)
                mse = gp.fitness(tree, data, o["engine"])
                print(f"{gp.format_number(mse, 17)}\t{tree.depth()}\t{tree.node_count()}\t{line}")
    elif mode == "--export-trees":
        import random
        rng = random.Random(o["seed"])
        grow = gp.grow_full if o["full"] else gp.grow_random
        for _ in range(o["count"]):
            print(grow(o["depth"], o["variables"], rng).expression(17))
    else:
        raise ValueError(f"Modo no reconocido: {mode}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, OSError, AssertionError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
