# Programacion Genetica (GP) - Implementacion C (alternativa), Julia 1.9+

Version **NO CONVENCIONAL**: arboles **inmutables** que **comparten subarboles**, con
**cache de resultados** en la evaluacion. Implementa todas las partes del manual
(representacion, `expression`, `eval`, MSE, `grow_random`, `grow_full`, clonacion,
crossover, mutacion, versiones protegidas, GP completo, benchmarks y pruebas). No usa
paquetes externos.

> **Estado:** este codigo se escribio sin poder ejecutar Julia; en el entorno donde se
> preparo no habia Julia ni red para instalarlo. Antes de medir corre `--test`,
> `tools/cross_check.jl` y `tools/verificar_con_b.jl`, y reporta cualquier error.

### Reglas confirmadas por el profesor que definen este diseno

| Idea | Respuesta del profesor | En esta version |
|---|---|---|
| Compartir subarboles en vez de copiarlos | Valido ("suena a una buena idea") | Si: copia de camino |
| Toda la poblacion en un solo grafo con cache de resultados | Valido | Si: DAG + cache por nodo |
| Evaluacion por lotes | **No valido** | No: se evalua **fila por fila** |
| Arbol como arreglo postfijo | Valido, pero sin beneficio claro | Ya no se usa |
| Paralelismo | No permitido | Un solo hilo |

> La version anterior de C (arreglo postfijo + evaluacion por lotes) quedo descartada
> porque usaba evaluacion por lotes. Sus tiempos no deben usarse en el reporte.

## Idea central

| | A y B (convencionales) | C (esta implementacion) |
|---|---|---|
| Nodos | Mutables; cada arbol es dueno de sus nodos | **Inmutables**; los subarboles se comparten entre individuos |
| Metadatos | Se recorren para calcularlos | **Tamano y altura guardados en cada nodo** (O(1)) |
| Clonar | Copia nodo por nodo | **Innecesario**: un nodo nunca cambia, compartirlo es seguro |
| Crossover / mutacion | Clonar padres completos e intercambiar | **Copia de camino**: solo los nodos de la raiz al punto de corte |
| Punto de corte | Lista de slots en preorden | Bajar por los tamanos hasta el k-esimo nodo en preorden, O(profundidad) |
| Evaluacion | Recursiva, fila por fila | Recursiva, fila por fila, **con cache** en los nodos grandes |
| Poblacion | Bosque de arboles independientes | **Grafo aciclico** (DAG) de subarboles compartidos |

### 1. Nodos inmutables con metadatos

- `OpNode`: operador, hijos, tamano (`size`) y altura (`height`), calculados al crearlo en
  O(1) a partir de los hijos. Ademas lleva el cache de resultados.
- `VarNode`: hay **una sola instancia por variable** (x1, x2, ...), compartida por todos.
- `ConstNode`: constante.

Como ningun operador modifica un nodo, los padres nunca cambian. Por eso `clone(arbol)`
devuelve el mismo arbol; para las pruebas hay `deep_copy_tree`.

### 2. Copia de camino (path copying)

- El k-esimo nodo en preorden se encuentra bajando por los tamanos de los hijos, en
  O(profundidad).
- Un hijo de crossover o de mutacion crea **solo los nodos del camino** entre la raiz y
  el punto de corte (a lo mas uno por nivel). Todo lo demas se comparte con el padre.
  En profundidad 15 son unos 16 nodos nuevos por hijo, contra unos 15,000 copiados en A o B.
- Crossover protegido: el intercambio `i <-> j` es valido si
  `prof_A(i) + altura_B(j) <= max` y `prof_B(j) + altura_A(i) <= max`. Es la misma regla
  que A y B: 128 intentos y, si ninguno sirve, intercambio de raices.

### 3. Evaluacion fila por fila con cache de resultados

- Para cada fila i se recorre el arbol recursivamente, como en el manual. No hay
  evaluacion por lotes.
- Los nodos con tamano >= S guardan su resultado para cada fila. Si el cache de un nodo
  esta completo para el dataset en uso, el nodo devuelve el valor guardado sin recorrer
  su subarbol.
- Un hijo comparte casi todo con sus padres, que ya estan evaluados. Solo se calculan los
  nodos nuevos del camino y los subarboles pequenos (tamano < S) que cuelgan de el.
- El cache vive en el nodo: cuando ningun individuo usa ya ese nodo, el recolector de
  basura de Julia lo libera junto con su cache.
- **S se elige solo** para que el cache de la poblacion quepa aproximadamente en
  `--cache-mb` megabytes (1024 por defecto). Con un dataset de muchas filas, S crece.
  `--cache-min-size=K` lo fija a mano; `--cache-mb=0` desactiva el cache.
- **El resultado no cambia**: mismas operaciones, en el mismo orden y sumando fila por
  fila como A y B. El MSE es identico bit a bit con y sin cache.

- **El dataset no debe modificarse despues de crearse.** El cache de cada nodo se asocia
  al identificador del `Dataset`; si se cambiaran sus valores, un nodo con cache devolveria
  el resultado anterior. El programa lee el CSV una vez y nunca lo modifica, y el
  constructor copia los datos. `--cache-mb=0` (o `cache_min_size = NO_CACHE`) no lee ni
  escribe ningun cache, ni siquiera uno que ya existiera.

### 4. Subarboles constantes precalculados (plegado de constantes)

- Cada nodo interno sabe, al crearse y en O(1), si su subarbol solo tiene constantes
  (ninguna variable). En ese caso calcula su valor **una vez**, con los mismos operadores
  protegidos, y lo guarda. Ejemplo: en `((sqrt(0.25) + 0.1) * x1)`, el nodo
  `(sqrt(0.25) + 0.1)` vale 0.6 para todas las filas.
- Al evaluar, ese nodo devuelve el valor guardado sin recorrer su subarbol en cada fila.
  Sigue siendo evaluacion fila por fila; no hay lotes.
- Los nodos originales se conservan: `expression`, profundidad, tamano y puntos de corte
  son los mismos, y el MSE es identico bit a bit (el valor se calcula con las mismas
  operaciones que la evaluacion normal). Los nodos constantes no ocupan cache.
- Como los nodos son inmutables, el valor nunca queda desactualizado: un hijo de crossover
  o de mutacion crea nodos nuevos en el camino, y cada uno calcula su propio valor.

### Mismo comportamiento que A

C consume el generador aleatorio en el mismo orden que A y elige el punto de corte
uniforme en preorden, como A. Con la misma semilla produce **exactamente** los mismos
arboles, los mismos hijos y la misma corrida del GP. `tools/cross_check.jl` lo comprueba.

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
julia --startup-file=no -O3 --threads=1 --gcthreads=1 main.jl --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
julia --startup-file=no -O3 --threads=1 --gcthreads=1 main.jl --bench-gp --data=data\DATASET_OFICIAL.csv --population=5000 --depths=7,9,12 --generations=10
```

| Opcion | Significado |
|---|---|
| `--batch-pairs=K` | Parejas de padres en memoria a la vez en el benchmark de crossover. `0` (por defecto) = los 10,000 arboles a la vez, protocolo acordado; `2500` = 5,000 a la vez |
| `--cache-mb=M` | Presupuesto aproximado del cache del GP, en MB (1024 por defecto; 0 = sin cache) |
| `--cache-min-size=K` | Tamano minimo de subarbol que guarda cache (0 = automatico segun `--cache-mb`) |
| `--eval-file --file=arboles.txt --data=...` | Lee expresiones (una por linea) e imprime MSE con 17 digitos, profundidad y nodos |
| `--export-trees --count=N --depth=D --variables=V --seed=S [--full]` | Escribe arboles sin perdida (17 digitos) |

Los benchmarks hacen primero una corrida pequena de calentamiento, para que **la
compilacion JIT de Julia no se cuente en el tiempo**.

### Protocolo del benchmark de crossover (acordado con el profesor)

- Los 10,000 padres (`grow_full`) estan en memoria al mismo tiempo. Si no alcanza la RAM,
  se permiten 5,000 (`--batch-pairs=2500`). La columna `resident_trees` lo dice.
- Se cronometra solo el crossover, incluida la creacion de los hijos (copia de camino).
- Los hijos se sueltan cada 16 cruces y el recolector de basura corre fuera del cronometro.
- **RAM:** los padres del benchmark se generan por separado y no comparten nada salvo las
  variables. En profundidad 15 con 10,000 arboles calculo unos 4 a 5 GB (unos 48 bytes
  por nodo interno y 16 por constante). No esta medido; revisalo con el Administrador de
  tareas y usa `--batch-pairs=2500` si no alcanza.

## Validacion de entradas

- Los constructores rechazan codigos de operacion desconocidos, operadores con el numero
  de hijos equivocado e indices de variable menores que 1.
- `eval_row` y la evaluacion comprueban `indice <= numero de variables`. `Dataset` exige
  filas de la misma longitud.
- `full_gp` rechaza `generations < 0`, y la linea de comandos rechaza enteros negativos.
- CSV: un campo vacio (por ejemplo "1,2,") es un error. La primera linea solo se toma como
  encabezado si ningun campo es numerico, igual que en A y B.

## Verificacion

`julia main.jl --test` comprueba:

- `expression`, `eval_row`, MSE (incluida la tabla del manual), operadores protegidos,
  MSE = +Inf al desbordar y variables fuera de rango.
- `grow_full` y `grow_random`: profundidades, todas las hojas a la profundidad exacta,
  y que el tamano y la altura guardados en cada nodo sean los reales.
- Ida y vuelta texto <-> arbol sin perdida.
- **Cache:** el MSE con cache (tamanos minimos 1, 4 y 16), sin cache y la referencia
  escalar son identicos bit a bit, en frio, en caliente y alternando entre dos datasets.
  Los hijos evaluados con los subarboles de sus padres ya en cache dan el mismo MSE.
- **Copia de camino:** el crossover conserva el total de nodos, los metadatos de los
  hijos son correctos, solo se crean los nodos del camino y los padres no cambian.
- Versiones protegidas: respetan `max_depth`, incluso al limite.
- GP: elitismo, y el cache no altera el resultado.
- **Plegado de constantes:** valores y marcas en un ejemplo conocido; que no cambie
  `expression` ni el tamano; que en 300 arboles aleatorios (incluidos arboles solo de
  constantes) cada subarbol constante valga, bit a bit, lo mismo que recorrerlo; y que un
  subarbol constante no finito (Inf - Inf) siga dando MSE = +Inf.
- `NO_CACHE` da el MSE correcto aunque los nodos tengan cache de una evaluacion anterior.

`tools/cross_check.jl` carga A y C en un solo proceso y comprueba:

1. Mismos arboles con la misma semilla, para 40 semillas, profundidades 0-9 y 1-4
   variables, y el mismo estado final del generador.
2. Conversion A -> texto -> C -> texto sin perdida, con los mismos nodos y profundidad.
3. `evaluate`/`eval_row` y MSE identicos bit a bit en tres datasets (`ejemplo.csv`, uno con
   ceros, negativos y magnitudes extremas, y uno que desborda), sin cache y con tres
   tamanos minimos, en frio y en caliente.
4. Crossover, mutacion y sus versiones protegidas producen los mismos hijos, y su MSE
   (con los subarboles compartidos ya en cache en C) es identico al de A.
5. El GP completo da el mismo historial del mejor MSE y el mismo mejor arbol que A, y el
   mismo resultado con y sin cache.

`tools/verificar_con_b.jl` hace que B genere arboles y que B, A y C evaluen los mismos
arboles en los tres datasets. Compara el MSE bit a bit, la profundidad y el numero de nodos.

## Archivos

- `src/GPC.jl`: nodos, copia de camino, evaluacion con cache, operadores, GP, benchmark y pruebas.
- `main.jl`: interfaz de terminal.
- `tools/cross_check.jl`: verificacion cruzada A <-> C.
- `tools/verificar_con_b.jl`: verificacion cruzada A, C <-> B.
- `data/ejemplo.csv`: datos ilustrativos (no oficiales).

## Mejoras pendientes (medir cada una por separado)

1. Barrer `--cache-min-size` (por ejemplo 4, 8, 16, 32) con el dataset oficial.
2. Unir subarboles identicos creados por separado (hash-consing), no solo los heredados.
   Es la otra mitad de la idea de Handley; ahorra memoria, pero cuesta calcular hashes al
   crear cada nodo.
3. Medir la ganancia del plegado de constantes con el dataset oficial (comparar contra la
   version anterior, guardada en `respaldo_20261119_2010/GP_JL_C`).
4. Repetir los benchmarks con varias semillas y reportar media y desviacion.
