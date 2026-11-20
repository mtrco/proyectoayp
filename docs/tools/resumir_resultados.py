#!/usr/bin/env python3
"""Resume los CSV de docs/resultados_contenedor/ en el JSON que usan las graficas.

    python docs/tools/resumir_resultados.py > docs/tools/resultados.json
    python docs/tools/generar_figuras.py docs/tools/resultados.json

Columnas `impl`: b_old_* / c_old = commit 4561dac (antes de estos cambios); b_new_* / c_new =
version actual. gp.csv: mediana de las semillas por implementacion y profundidad.
crossover.csv: microsegundos por cruce.
"""
import csv
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "resultados_contenedor")

gp = defaultdict(lambda: defaultdict(list))
with open(os.path.join(DATA, "gp.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gp[row["impl"]][row["depth"]].append(float(row["elapsed_seconds"]))
crossover = defaultdict(dict)
with open(os.path.join(DATA, "crossover.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        crossover[row["impl"]][row["depth"]] = float(row["mean_seconds_per_crossover"]) * 1e6

result = {
    "gp": {impl: {d: statistics.median(v) for d, v in depths.items()} for impl, depths in gp.items()},
    "crossover": crossover,
    # Medido con B instrumentado (g++ -O3, semilla 1, profundidad 12, version anterior):
    # porcentaje del tiempo total del GP completo (32.1 s).
    "breakdown_b": [["evaluar hijos", 69.4], ["evaluar población inicial", 11.8],
                    ["destruir árboles descartados", 5.2], ["clonar (mutación)", 4.1],
                    ["clonar (crossover)", 3.7], ["generar población", 1.6], ["resto", 4.2]],
    # Medido con @timed en Julia 1.13.4 (profundidad 12, 5,000 arboles iniciales), segundos.
    # Antes: 1.37 s de los 1.96 s de generar y 1.24 s de los 4.17 s de evaluar eran
    # recoleccion de basura.
    "breakdown_c": [[["generar 5,000 árboles", 1.96], ["evaluar su MSE", 4.17]],
                    [["generar 5,000 árboles", 0.37], ["evaluar su MSE", 1.70]]],
}
print(json.dumps(result, indent=1, ensure_ascii=False))
