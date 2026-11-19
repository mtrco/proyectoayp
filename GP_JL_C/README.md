# Programacion Genetica (GP) - Implementacion C (alternativa), Julia 1.9+

Version **NO CONVENCIONAL**. Implementa todas las partes del manual (representacion,
`expression`, `eval`, MSE, `grow_random`, `grow_full`, clonacion, crossover, mutacion,
versiones protegidas, GP completo, benchmarks y pruebas) con una representacion distinta
a la de A y B. Es la misma idea que la version anterior de C en C++, llevada a Julia.
No usa paquetes externos.

> **Estado:** este codigo se escribio sin poder ejecutar Julia; en el entorno donde se
> preparo no habia Julia ni red para instalarlo. Antes de medir corre `--test`,
> `tools/cross_check.jl` y `tools/verificar_con_b.jl`, y reporta cualquier error.

## Idea central

| | A y B (convencionales) | C (esta implementacion) |
|---|---|---|
| Arbol | Nodos enlazados por referencias | **Arreglo plano en notacion postfija**, sin punteros |
| Memoria por nodo | Un objeto por nodo en el heap | **9 bytes** (1 `Float64` + 1 byte de operacion) en dos vectores |
| Clonar | Recorrido recursivo, un objeto por nodo | **Dos copias contiguas** (`copy`) |
| Crossover / mutacion | Intercambiar referencias entre clones | **Pegar segmentos**: un subarbol es un tramo contiguo |
| Proteccion de profundidad | Recorrido para calcular profundidades | Tamano, altura y profundidad en una pasada O(n); cada intento es O(1) |
| Evaluacion | Recursiva, fila por fila | **Por lotes**: cada instruccion se aplica a un bloque de filas |
| Dataset | Por filas | **Por columnas** |

Ejemplo: `x1 + (x2 * 0.5)` se guarda como

```
ops : VAR  VAR  CONST  MUL  ADD
vals: 1    2    0.5    0    0
```

El subarbol `(x2 * 0.5)` es el tramo `[2..4]`: termina en su raiz (indice 4) y empieza en
`subtree_start(4) = 2`, que se calcula contando operandos hacia atras.

### Decisiones de diseno

- **Metadatos bajo demanda.** Tamano, altura, profundidad y preorden no se guardan en
  el arbol. Solo los calculan, en una pasada O(n) y con buffers reutilizados, los
  operadores que los necesitan. El crossover simple del benchmark no paga ese costo.
- **Evaluacion por lotes.** El programa se interpreta una vez por bloque de `--block`
  filas (64 por defecto). La pila es una matriz `bloque x profundidad`, cada operacion es
  un ciclo `@simd` sobre una columna, y la raiz usa la instruccion IEEE sin comprobar el
  dominio. La suma del error es **secuencial, fila por fila y en el mismo orden que A y
  B**, por eso el MSE es identico bit a bit.
- **Mismo consumo del generador aleatorio que A.** Con la misma semilla, `grow_full` y
  `grow_random` producen **exactamente los mismos arboles** que A.
- **Seleccion de nodos (`--preorder`):**
  - Por defecto (`FAST`), el punto de corte se elige uniforme sobre el indice postfijo.
  - Con `--preorder`, se elige uniforme en preorden, como A. La distribucion es la misma,
    pero C reproduce **exactamente** las corridas de A con la misma semilla. Es el modo
    de verificacion.
- **Sin paralelismo:** un solo hilo (regla confirmada por el profesor). Lo unico que hay
  es la vectorizacion SIMD que hace el compilador dentro de un nucleo; esta pendiente
  confirmar con el profesor que eso no cuenta como paralelismo.

## Instalar y ejecutar (Windows)

1. Instala Julia 1.10 o mas reciente desde julialang.org (en Windows se recomienda
   el instalador `juliaup`). Comprueba con `julia --version`.
2. Coloca `GP_JL_A` y `GP_JL_C` en la misma carpeta, junto a `GP_CPP20_B`.

```powershell
cd GP_JL_C
julia main.jl --test                                   # pruebas internas
julia tools\cross_check.jl                             # A <-> C (busca ..\GP_JL_A)
julia tools\verificar_con_b.jl ..\GP_CPP20_B\out\build\x64-Release\gp_b.exe
julia main.jl --demo
```

## Uso

```powershell
julia -O3 main.jl --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
julia -O3 main.jl --bench-gp --data=data\DATASET_OFICIAL.csv --population=5000 --depths=7,9,12 --generations=10
```

| Opcion | Significado |
|---|---|
| `--batch-pairs=K` | Parejas de padres en memoria a la vez en el benchmark de crossover. `0` (por defecto) = los 10,000 arboles a la vez, protocolo acordado; `2500` = 5,000 a la vez |
| `--block=K` | Filas por lote en la evaluacion (probar con el dataset oficial) |
| `--preorder` | Seleccion de nodos como A (reproduce sus corridas) |
| `--eval-file --file=arboles.txt --data=...` | Lee expresiones (una por linea) e imprime MSE con 17 digitos, profundidad y nodos |
| `--export-trees --count=N --depth=D --variables=V --seed=S [--full]` | Escribe arboles sin perdida (17 digitos) |

Los benchmarks hacen primero una corrida pequena de calentamiento, para que **la
compilacion JIT de Julia no se cuente en el tiempo**.

### Protocolo del benchmark de crossover (acordado con el profesor)

- Los 10,000 padres (`grow_full`) estan en memoria al mismo tiempo. Si no alcanza la RAM,
  se permiten 5,000 (`--batch-pairs=2500`). La columna `resident_trees` lo dice.
- Se cronometra solo el crossover, incluida la creacion de los hijos.
- Los hijos se sueltan cada 16 cruces y el recolector de basura corre fuera del cronometro.
- RAM en profundidad 15 con 10,000 arboles: unos 1.4 GB (9 bytes por nodo, como en C++).

## Verificacion

`julia main.jl --test` comprueba:

- `expression`, `postfix`, `eval_row`, MSE, clonacion profunda y operadores protegidos.
- MSE = +Inf al desbordar, y que una variable fuera de rango se rechace.
- Profundidad exacta de `grow_full` y maxima de `grow_random`; arreglos bien formados.
- Ida y vuelta texto <-> arreglo sin perdida.
- **Evaluacion por lotes igual bit a bit a la escalar**, con bloques de 1, 7, 64 y 256 filas.
- `subtree_start`, tamanos, preorden y altura de la raiz.
- Crossover y mutacion en ambos modos: hijos validos, conservacion de nodos, padres
  intactos y limite de profundidad.
- Elitismo del GP, y que el tamano de bloque no altere el resultado.

`tools/cross_check.jl` carga A y C en un solo proceso y comprueba:

1. Mismos arboles con la misma semilla, para 40 semillas, profundidades 0-9 y 1-4
   variables, y el mismo estado final del generador.
2. Conversion A -> texto -> C -> texto sin perdida, con los mismos nodos y la misma profundidad.
3. `evaluate`/`eval_row` y MSE identicos bit a bit en tres datasets (`ejemplo.csv`, uno con
   ceros, negativos y magnitudes extremas, y uno que desborda), con cuatro tamanos de bloque.
4. Crossover, mutacion y sus versiones protegidas producen los mismos hijos (C en `PREORDER`).
5. El GP completo da el mismo historial del mejor MSE por generacion y el mismo mejor arbol.

`tools/verificar_con_b.jl` hace que B genere arboles y que B, A y C evaluen los mismos
arboles en los tres datasets. Compara el MSE bit a bit, la profundidad y el numero de nodos.

## Archivos

- `src/GPC.jl`: representacion, generacion, evaluacion por lotes, operadores, GP,
  benchmark y pruebas.
- `main.jl`: interfaz de terminal.
- `tools/cross_check.jl`: verificacion cruzada A <-> C.
- `tools/verificar_con_b.jl`: verificacion cruzada A, C <-> B.
- `data/ejemplo.csv`: datos ilustrativos (no oficiales).

## Mejoras pendientes (medir cada una por separado)

1. Reutilizar los buffers de los individuos descartados en vez de reservar memoria
   nueva para cada descendiente.
2. Evitar copiar las columnas de variables a la pila durante la evaluacion.
3. Barrer `--block` con el dataset oficial.
4. Probar `julia -O3 --check-bounds=no` (si el profesor lo permite).
5. Repetir los benchmarks con varias semillas y reportar media y desviacion.
