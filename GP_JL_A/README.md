# Programacion Genetica (GP) - Implementacion A (alto nivel), Julia 1.9+

Version **CONVENCIONAL de alto nivel**. Implementa todas las partes del manual con la
jerarquia de tipos que propone: arboles, `expression`, `eval`, MSE, `grow_random`,
`grow_full`, clonacion, crossover, mutacion, versiones protegidas por profundidad,
GP completo, benchmarks y pruebas. No usa paquetes externos (solo `Random` y `Printf`,
que vienen con Julia).

> **Estado:** este codigo se escribio sin poder ejecutar Julia; en el entorno donde se
> preparo no habia Julia ni red para instalarlo. Antes de usarlo corre `--test` y
> `GP_JL_C/tools/cross_check.jl`, y reporta cualquier error.

## Diseno

```
Node (abstracto)          expression(n), evaluate(n, x)  (el eval del manual), clone(n),
 |                        node_count(n), tree_depth(n)
 |- NodeInternal          op, left, right      (sqrt usa solo left; right = nothing)
 |- NodeVariable          index                (1 -> x1)
 |- NodeConstant          value
```

- **Un metodo por tipo:** cada tipo define su `expression`, `evaluate` y `clone`
  (despacho multiple de Julia). El metodo del manual se llama `evaluate` porque `eval`
  esta reservado en Julia.
- **Hijos tipados con una union concreta**
  (`Union{NodeInternal, NodeVariable, NodeConstant}`) en vez del tipo abstracto. Asi
  Julia sabe en compilacion que metodo llamar y la recursion no paga despacho dinamico.
- **Crossover por intercambio de referencias**, como en el manual. Cada punto de corte
  es un `Slot` (dueno, lado). La raiz cuelga de un contenedor `Holder`, asi que
  intercambiar dos subarboles, incluso dos raices, es intercambiar dos referencias.
- **Protecciones por profundidad en O(1) por intento.** Cada slot guarda la profundidad
  del nodo y la altura de su subarbol. El intercambio `i <-> j` es valido si
  `prof_A(i) + altura_B(j) <= max` y `prof_B(j) + altura_A(i) <= max`. Es la misma regla
  que B: 128 intentos y, si ninguno sirve, intercambio de raices.

### Reglas del manual respetadas (las mismas que B y C)

- Raiz a profundidad 0; `grow_full(d)` pone todas las hojas exactamente a profundidad d.
- Terminales: variables repetibles y constantes uniformes en [-1, 1).
- Operadores: +, -, *, division protegida `a / (|b| < 1e-12 ? 1e-12 : b)` y raiz
  protegida `sqrt(max(x, 0))`.
- Mismas probabilidades de generacion que B: operador 1/5 cada uno, hoja constante o
  variable 1/2, y `grow_random` corta con probabilidad 1/4.
- GP: 100% `grow_full`, torneo binario repetido, 50% crossover / 50% mutacion,
  reemplazo con el mejor 50% actual + el mejor 50% de los descendientes.
- MSE: menor es mejor; un resultado no finito vale +Inf.
- Sin paralelismo: un solo hilo (regla confirmada por el profesor).

### Generador aleatorio y verificacion

A usa `Random.Xoshiro`. **C (en Julia) consume el generador exactamente en el mismo
orden**, asi que con la misma semilla genera los mismos arboles. Con C en modo
`--preorder`, ademas, obtiene los mismos hijos y la misma corrida completa del GP. Eso
permite comparar A y C sin depender de la entrada ni de la salida.

B (C++) usa `std::mt19937_64` y no puede reproducir la misma secuencia. Con B se compara
el **mismo arbol** evaluado por las tres implementaciones: `expression(arbol, 17)` es
sin perdida y el MSE debe coincidir bit a bit.

## Instalar y ejecutar (Windows)

1. Instala Julia 1.10 o mas reciente desde julialang.org (en Windows se recomienda
   el instalador `juliaup`). Comprueba con `julia --version`.
2. Desde esta carpeta:

```powershell
julia main.jl --test            # pruebas internas (-v para verlas una por una)
julia main.jl --demo
```

## Uso

```powershell
julia -O3 main.jl --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
julia -O3 main.jl --bench-gp --data=data\DATASET_OFICIAL.csv --population=5000 --depths=7,9,12 --generations=10
julia main.jl --eval-file --file=arboles.txt --data=data\ejemplo.csv
julia main.jl --export-trees --count=100 --depth=6 --variables=2 --seed=1 [--full]
```

| Opcion | Significado |
|---|---|
| `--batch-pairs=K` | Parejas de padres en memoria a la vez en el benchmark de crossover. `0` (por defecto) = los 10,000; `2500` = 5,000 a la vez |
| `--eval-file` | `MSE<TAB>profundidad<TAB>nodos<TAB>expresion`, MSE con 17 digitos (misma salida que `gp_b --eval-file`) |
| `--export-trees` | Arboles sin perdida (17 digitos), uno por linea |

Los benchmarks hacen primero una corrida pequena de calentamiento, para que **la
compilacion JIT de Julia no se cuente en el tiempo**. La salida CSV tiene las mismas
columnas que B y C.

### Protocolo del benchmark de crossover (acordado con el profesor)

- Los 10,000 padres (`grow_full`) estan en memoria al mismo tiempo. Si no alcanza la RAM,
  se permiten 5,000 (`--batch-pairs=2500`). La columna `resident_trees` lo dice.
- Se cronometra solo el crossover (clonacion de ambos padres + intercambio).
- Los hijos se sueltan cada 16 cruces y el recolector de basura de Julia corre fuera del
  cronometro (`GC.gc(false)`). Equivale a destruir los hijos fuera del cronometro en B.
- RAM estimada en profundidad 15 con 10,000 arboles: unos 4 GB (unos 25 bytes por nodo).
  No esta medida; revisala con el Administrador de tareas.

## Pruebas (`julia main.jl --test`)

- **Parte 1:** `expression`, `evaluate`, el MSE de la tabla del manual y la clonacion
  profunda. Tambien la division y la raiz protegidas, MSE = +Inf al desbordar, y que un
  operador binario sin hijo derecho se rechace.
- **Generacion:** `grow_full` pone todas las hojas a la profundidad exacta y `grow_random`
  respeta el maximo. Las constantes estan en [-1, 1] y se usan todas las variables. El
  tamano promedio es el teorico (`sum 1.8^k` y `E(d) = 1 + 1.35 E(d-1)`), lo que confirma
  las mismas probabilidades que B.
- **Formato de texto:** ida y vuelta `expression(17)` -> `parse_expression` sin perdida.
- **Crossover y mutacion:** el crossover conserva el total de nodos. Los padres quedan
  intactos y los hijos no comparten nodos con ellos. Las versiones protegidas respetan
  `max_depth`, incluso al limite.
- **GP:** elitismo (el mejor MSE nunca sube), y la misma semilla da la misma corrida.

Las verificaciones cruzadas estan en `GP_JL_C/tools/`: `cross_check.jl` (A contra C) y
`verificar_con_b.jl` (A y C contra B).

## Archivos

- `src/GPA.jl`: tipos, operadores, generacion, fitness, operadores geneticos, GP,
  benchmark y pruebas.
- `main.jl`: interfaz de terminal (mismas opciones que B y C).
- `data/ejemplo.csv`: datos ilustrativos (no oficiales).
