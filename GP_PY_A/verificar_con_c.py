"""Verificacion cruzada A <-> C (y por transitividad A <-> B).

C ya se verifico bit a bit contra B con cross_check. Este script comprueba que
A coincide con C usando el formato de texto comun (expression con 17 digitos):

  1. A genera arboles (grow_random y grow_full) -> C los evalua con --eval-file.
  2. C genera arboles (--export-trees)            -> A los evalua.
  3. Para cada arbol y cada dataset se compara: MSE bit a bit, profundidad y
     numero de nodos. Se usan tres datasets: el de ejemplo (u otro con --data),
     uno con ceros, negativos y magnitudes extremas (activa las protecciones)
     y uno que provoca desbordamientos (MSE = +inf).

Uso (Windows):
  python verificar_con_c.py --gp-c=..\\GP_CPP20_C\\gp_c.exe [--data=data\\ejemplo.csv]
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile

import gp


def write_csv(path: str, rows: list[list[float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        width = len(rows[0]) - 1
        handle.write(",".join([f"x{i + 1}" for i in range(width)] + ["y"]) + "\n")
        for row in rows:
            handle.write(",".join(repr(v) for v in row) + "\n")


def synthetic_datasets(folder: str) -> dict[str, str]:
    rng = random.Random(2026)
    extreme = []
    for i in range(97):
        x = [0.0 if (i + k) % 7 == 0 else rng.uniform(-2, 2) * 10 ** rng.choice([-13, -6, 0, 0, 4])
             for k in range(4)]
        extreme.append(x + [rng.uniform(-5, 5)])
    overflow = [[rng.uniform(-1, 1) * 10 ** rng.choice([0, 80, 160]) for _ in range(4)]
                + [rng.uniform(-1, 1)] for _ in range(40)]
    paths = {"extremos": os.path.join(folder, "extremos.csv"),
             "desborde": os.path.join(folder, "desborde.csv")}
    write_csv(paths["extremos"], extreme)
    write_csv(paths["desborde"], overflow)
    return paths


def run_c(gp_c: str, *args: str) -> list[str]:
    out = subprocess.run([gp_c, *args], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gp-c", required=True, help="ruta al ejecutable gp_c")
    parser.add_argument("--data", default="data/ejemplo.csv")
    parser.add_argument("--trees-per-case", type=int, default=25)
    args = parser.parse_args()

    engines = ["scalar"] + (["numpy"] if gp.np is not None else [])
    checks = failures = 0
    with tempfile.TemporaryDirectory() as folder:
        datasets = {"principal": args.data, **synthetic_datasets(folder)}
        for name, path in datasets.items():
            data = gp.Dataset.from_csv(path)
            nvars = data.variables

            # 1) Arboles de A.
            rng = random.Random(1234)
            trees_a = []
            for depth in range(0, 11):
                for v in range(1, nvars + 1):
                    for _ in range(args.trees_per_case // 5 + 1):
                        trees_a.append(gp.grow_random(depth, v, rng).expression(17))
                        trees_a.append(gp.grow_full(min(depth, 8), v, rng).expression(17))
            # 2) Arboles de C.
            trees_c = []
            for seed in (1, 2, 3):
                for depth in (0, 3, 6, 9):
                    for v in range(1, nvars + 1):
                        trees_c += run_c(args.gp_c, "--export-trees",
                                         f"--count={args.trees_per_case}", f"--depth={depth}",
                                         f"--variables={v}", f"--seed={seed}")

            file_trees = os.path.join(folder, f"arboles_{name}.txt")
            with open(file_trees, "w", encoding="utf-8") as handle:
                handle.write("\n".join(trees_a + trees_c) + "\n")
            c_lines = run_c(args.gp_c, "--eval-file", f"--file={file_trees}", f"--data={path}")
            if len(c_lines) != len(trees_a) + len(trees_c):
                print(f"[{name}] C devolvio {len(c_lines)} lineas, se esperaban "
                      f"{len(trees_a) + len(trees_c)}")
                return 1

            infinite = 0
            for line in c_lines:
                c_mse, c_depth, c_size, text = line.split("\t", 3)
                c_mse = float(c_mse)
                tree = gp.parse_expression(text)
                # El arbol leido por A debe reescribirse igual (sin perdida).
                ok = tree.expression(17) == text
                ok &= tree.depth() == int(c_depth) and tree.node_count() == int(c_size)
                for engine in engines:
                    ok &= gp.fitness(tree, data, engine).hex() == c_mse.hex()
                infinite += c_mse == float("inf")
                checks += 1
                if not ok:
                    failures += 1
                    if failures <= 10:
                        print(f"[{name}] DIFERENCIA: C={line}")
                        print(f"          A: mse={gp.fitness(tree, data):.17g} "
                              f"prof={tree.depth()} nodos={tree.node_count()}")
            print(f"[{name}] {len(c_lines)} arboles ({len(trees_a)} de A, {len(trees_c)} de C), "
                  f"{infinite} con MSE=+inf, motores A: {', '.join(engines)}")

    print(f"\nTotal: {checks} arboles comparados, {failures} diferencias.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
