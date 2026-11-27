# Programacion Genetica (GP) - Implementacion C (alternativa), Julia 1.9+

Version **NO CONVENCIONAL**: **toda la poblacion vive en un solo grafo guardado en
arreglos**, los arboles **comparten subarboles** y la evaluacion usa un **cache de
resultados**. Implementa todas las partes del manual (representacion, `expression`,
`eval`, MSE, `grow_random`, `grow_full`, clonacion, crossover, mutacion, versiones
protegidas, GP completo, benchmarks y pruebas). No usa paquetes externos.

Probado con Julia 1.13.4: `--test` (1,582 comprobaciones), `tools/cross_check.jl`
(56,116 comprobaciones contra A, 0 fallos) y `tools/verificar_con_b.jl` (3,280 arboles
contra B, 0 diferencias). La lista postfija, el cache al reusarse y la evaluacion por bloques
(secciones 9 a 11) se agregaron despues y se midieron en un modelo de C en C++; sus pruebas
suben `--test` a 1,598 comprobaciones y falta correrlas en Julia.

La explicacion detallada, con diagramas, esta en [`../docs/README.md`](../docs/README.md).

### Reglas confirmadas por el profesor que definen este diseno

| Idea | Respuesta del profesor | En esta version |
|---|---|---|
| Compartir subarboles en vez de copiarlos | Valido ("suena a una buena idea") | Si: copia de camino |
| Toda la poblacion en un solo grafo con cache de resultados | Valido | Si: un grafo en arreglos + cache |
| Evaluacion por lotes | **No valido** | **Si, por decision del equipo** (seccion 11): la lista postfija se recorre por bloques de 64 filas. `--no-batch` regresa a fila por fila |
| Arbol como arreglo postfijo | Valido, pero sin beneficio claro | Si, en los arboles de hasta 512 instrucciones (seccion 9; `--no-tape` la apaga) |
| Paralelismo | No permitido | Un solo hilo (el recolector de basura tambien: 1 hilo) |

## Idea central

| | A y B (convencionales) | C (esta implementacion) |
|---|---|---|
| Nodos | Un objeto por nodo; cada arbol es dueno de sus nodos | **Registros de 24 bytes en un arreglo**; los hijos son indices de 32 bits |
| Poblacion | Bosque de arboles independientes | **Un solo grafo** (DAG) de subarboles compartidos |
| Metadatos | Se recorren para calcularlos | **Tamano, altura y mayor variable guardados en cada nodo** (O(1)) |
| Clonar | Copia nodo por nodo | **Innecesario**: un nodo nunca cambia, compartirlo es seguro |
| Crossover / mutacion | Clonar padres completos e intercambiar | **Copia de camino**: solo los nodos de la raiz al punto de corte |
| Evaluacion | Recursiva, fila por fila | **Lista postfija por bloques de 64 filas** y **cache** en los nodos grandes y en los chicos que se reusan (fila por fila con `--no-batch`) |
| Memoria | Recolector de basura (A) o asignador propio (B) | **Sin objetos por nodo**: el recolector de basura casi no trabaja |

### 1. Un grafo en arreglos (`Graph`, `NodeRec`, `Tree`)

- `NodeRec` (24 bytes, sin punteros): hijos izquierdo y derecho (indices `Int32`; en una
  variable, `left` es su indice), tamano, altura, tipo, una bandera de plegado y `value`
  (la constante, el valor plegado o, en los demas nodos internos, el mayor indice de
  variable del subarbol).
- `Graph`: el `Vector{NodeRec}` con todos los nodos, una instancia compartida de cada
  variable y, en arreglos paralelos, el cache de resultados.
- `Tree`: (grafo, indice de la raiz). Copiar un `Tree` no copia nodos.
- Antes cada nodo era un objeto de Julia: en profundidad 12 la poblacion inicial tenia
  ~13 millones de objetos y **la mitad del tiempo era recoleccion de basura**. Ahora son
  registros dentro de un arreglo y la recoleccion es practicamente cero.
- `full_gp` y `benchmark_crossover` usan un grafo propio; el resto usa `DEFAULT_GRAPH`.
- Corridas largas: cuando el cache ocupa el doble de `--cache-mb`, el grafo se compacta
  (se copian solo los arboles vivos, conservando lo que comparten). No cambia el resultado.

### 2. Arboles en preorden

Al generar un arbol, el registro del padre se reserva antes que los de sus hijos: el
arbol nuevo queda en memoria en el orden en que se recorre, y el hijo izquierdo esta
justo despues del padre. El generador se consume igual que en A.

### 3. Copia de camino (path copying)

- El k-esimo nodo en preorden se encuentra bajando por los tamanos de los hijos, en
  O(profundidad).
- Un hijo de crossover o de mutacion crea **solo los nodos del camino** entre la raiz y
  el punto de corte (a lo mas uno por nivel). Todo lo demas se comparte con el padre.
  En profundidad 15 son unos 16 nodos nuevos por hijo, contra unos 15,000 copiados en A o B.
- Crossover protegido: el intercambio `i <-> j` es valido si
  `prof_A(i) + altura_B(j) <= max` y `prof_B(j) + altura_A(i) <= max`. Es la misma regla
  que A y B: 128 intentos y, si ninguno sirve, intercambio de raices.

### 4. Evaluacion fila por fila con cache de resultados

- Para cada fila i se recorre el arbol recursivamente, como en el manual (o su lista
  postfija, seccion 9). Por defecto la lista se recorre por bloques de filas (seccion 11);
  con `--no-batch`, fila por fila.
- Los nodos con tamano >= S guardan su resultado en cada fila (en bloques de 8 MB, no en
  un objeto por nodo). Si el cache de un nodo esta completo para el dataset en uso, el
  nodo devuelve el valor guardado sin recorrer su subarbol.
- Un hijo comparte casi todo con sus padres, ya evaluados: solo se calculan los nodos
  nuevos del camino y los subarboles pequenos (tamano < S) que cuelgan de el.
- **Subarboles pequenos sin revisar cache y sin saltos**: debajo de S ningun nodo tiene
  cache, asi que se usa un evaluador aparte que no lo revisa; elige el operador con
  `ifelse` (se calculan + - * / y se toma uno) y lee las hojas sin saltos. Se le pasa el
  indice del nodo, no su registro (pasar el registro obliga a copiarlo en cada llamada).
- **S automatico**: 16, o 32 en profundidad >= 11 (lo mas rapido medido: llenar el cache
  de la poblacion inicial cuesta), y mas grande si la estimacion del cache pasa de 4 veces
  `--cache-mb`. La estimacion es muy conservadora y la memoria real la acota la
  compactacion del grafo. `--cache-min-size=K` lo fija a mano; `--cache-mb=0` desactiva
  el cache.
- **Medido con dataset3a** (961 filas, 5000 individuos, 10 generaciones, semilla 1, Ryzen 7
  170): mismo `best_mse` con todos los S.

  | S | Prof. 7 | Prof. 9 | Prof. 12 |
  |---|---|---|---|
  | 16 | 6.14 s | 12.40 s | 72.4 s |
  | 32 | 6.80 s | 11.59 s | 39.9 s |
  | 64 | 9.16 s | 14.16 s | 39.1 s |
  | automatico anterior (16, 32, 256) | 6.18 s | 12.25 s | 53.6 s |

  El automatico actual elige 16, 16 y 64. En profundidad 12 el pico de RAM con S = 64 fue
  de unos 4 GB. Con `--cache-mb=4096` tardaba lo mismo pero el pico subia a 8.7 GB, porque
  la compactacion esperaba a que el cache llegara a 8 GB.
- **El resultado no cambia**: mismas operaciones, en el mismo orden y sumando fila por
  fila como A y B. El MSE es identico bit a bit con y sin cache.
- El indice de variable se valida una vez por arbol (el mayor esta guardado en la raiz).

### 5. Plegado de constantes (`--no-fold` lo apaga)

Un nodo cuyo subarbol no tiene variables (por ejemplo `(sqrt(0.25) + 0.1)`) calcula su
valor al crearse, con las mismas operaciones protegidas, y lo guarda en `value`. La
evaluacion lo lee sin recorrer el subarbol. `expression`, tamano, profundidad y puntos de
corte no cambian, y el MSE es identico bit a bit.

### 6. Corte temprano exacto (`--no-cut` lo apaga)

Solo sobrevive el mejor 50% de los hijos. Un monticulo guarda los N/2 mejores MSE de los
hijos ya evaluados; si el MSE parcial de un hijo (suma/n) ya supera al (N/2)-esimo mejor,
ese hijo terminaria peor que N/2 hijos y nunca sobreviviria: se deja de evaluar y recibe
+infinito. Los sobrevivientes conservan su MSE exacto; con el orden estable del
reemplazo, el GP es identico al de A (que no usa corte).

### 7. Orden de evaluacion de los hijos (`--no-order` lo apaga)

Igual que en B: primero se **crean** todos los hijos (mismo orden y mismo consumo del
generador) y despues se **evaluan empezando por los de mejores padres** (el padre del que el
hijo hereda la raiz; orden estable). El umbral del corte temprano baja antes y se cortan mas
hijos. No cambia ningun MSE de los sobrevivientes ni el orden de creacion, que decide los
empates: el GP es identico (lo comprueba `--test` y `tools/cross_check.jl` contra A).
En B, con `dataset3a.csv`, ahorro entre 5% y 6.6% del tiempo del GP completo; en C falta
medirlo (`julia main.jl --bench-gp ... --no-order` da la version anterior).

### 8. Hijos identicos a su padre (`--no-dup` lo apaga)

Igual que en B: un crossover que intercambia dos subarboles iguales, una mutacion que pone
un subarbol igual al que quita o el intercambio de raices producen un hijo **identico a su
padre**. En C se detecta al crear el hijo (se comparan los dos subarboles; casi siempre son
hojas) y el hijo es el padre mismo: no se copia el camino ni se evaluan las filas, y recibe
el MSE del padre. Con el corte temprano se registra igual que si se hubiera evaluado, asi
que el GP es identico (lo comprueba `--test`). En B, entre 6% y 15% de los hijos son
copias y el GP tarda de 7% a 12% menos; en C falta medirlo.

### 9. Lista postfija (`--no-tape` la apaga)

Medido en un modelo de C en C++, el 93% del tiempo del GP era evaluar los hijos, y casi
todo eso era **recorrer** el arbol, no calcular: en cada una de las 961 filas, la recursion
vuelve a leer el registro de cada nodo del camino, revisa su cache y llama a la funcion del
hijo (en el modelo, unos 13 ns por cada nodo del camino y por fila).

- Antes de evaluar un arbol, se traduce **una vez** a una lista postfija (`_compile!`):
  hijo izquierdo, hijo derecho y luego el operador. Cada instruccion (`TapeOp`, 24 bytes)
  apila una variable, una constante o el valor guardado en el cache de un nodo, aplica un
  operador, o guarda el tope de la pila en el cache de un nodo.
- Despues, **para cada fila**, se recorre la lista con una pila chica (`_run_tape`): cada
  fila pasa por todo el arbol antes de empezar la siguiente. Asi es con `--no-batch`; por
  defecto la lista se recorre por bloques de 64 filas (seccion 11).
- Un nodo con cache completo es **una sola instruccion**: su subarbol ni siquiera se recorre
  al traducir. Como los hijos comparten casi todo con sus padres, sus listas son cortas.
- **Hibrido**: si la lista pasa de 512 instrucciones (`TAPE_MAX`), se usa la recursion. En
  la poblacion inicial de profundidad 12 hay arboles de miles de nodos sin nada en cache, y
  con listas tan largas el procesador ya no adivina la siguiente instruccion: la lista fue
  2.5 veces mas lenta que la recursion (91 s contra 37 s en la poblacion inicial del modelo).
- Mismas operaciones, en el mismo orden: el MSE es identico bit a bit (`--test` alterna
  lista y recursion sobre el mismo grafo y compara contra el MSE de referencia).

### 10. Cache al reusarse (`--reuse-min-size=K`, 4 por defecto; 0 lo apaga)

- Con S = 16, un subarbol de 4 a 15 nodos nunca tiene cache: cada hijo que lo contiene lo
  vuelve a calcular en las 961 filas. Bajar S para todos no sirve: el cache de la poblacion
  inicial no cabe (con S = 4 en profundidad 9 el modelo se quedo sin memoria).
- En cambio, los nodos con K <= tamano < S guardan su cache **la segunda vez** que se
  evaluan; la primera solo se marcan (`SEEN` en `cache_id`). Asi gastan memoria solo los
  subarboles chicos que de verdad reaparecen (en el camino de un hijo o en otra generacion).
- **Solo guarda el subarbol chico mas externo**: si un nodo chico va a guardar su cache,
  los nodos de adentro no guardan ni se marcan (la proxima vez se lee el de afuera y no se
  baja hasta ellos). Sin esta regla, en profundidad 15 (S = 512) cada hijo guardaba cientos
  de subarboles anidados y la memoria se disparaba: en el modelo, 2,000 arboles de
  profundidad 15 llegaban a 5.9 GB contra 2.0 GB con la regla (y 1.8 GB sin cache al
  reusarse). En profundidades 7 a 12 el tiempo es el mismo y en 12 el pico baja de 5.2 a
  4.4 GB.
- Funciona con la lista postfija; si un arbol se evalua con la recursion, sus nodos chicos
  quedan marcados pero no guardan cache.
- El cache guarda exactamente el valor que da la operacion: el MSE no cambia.

**Medido en el modelo de C en C++** (mismas estructuras y misma logica de evaluacion que
C; otro generador aleatorio, asi que su MSE no es el de Julia; calibrado contra Julia: 37.5 s
contra 36.6 s en profundidad 7). `dataset3a.csv`, poblacion 5,000, 100 generaciones, un
hilo. El historial del mejor MSE es identico con y sin los cambios en las tres profundidades:

| Profundidad | Antes | Lista postfija + cache al reusarse | Pico de RAM antes / despues |
|---|---|---|---|
| 7 | 35.1 s | 18.3 s (1.9x) | 2.3 / 2.3 GB |
| 9 | 53.5 s | 29.8 s (1.8x) | 2.5 / 2.6 GB |
| 12 | 148.4 s | 66.0 s (2.2x) | 4.2 / 5.1 GB |

Falta medirlo en Julia: `--no-tape --reuse-min-size=0` da la version anterior para comparar.

### 11. Evaluacion por bloques de filas (`--no-batch` la apaga)

> **Ojo:** el profesor dijo que la evaluacion por lotes **no es valida** para C. Se agrego y
> se dejo encendida por decision del equipo; `--no-batch` da la version fila por fila si el
> profesor la objeta (mismo resultado, solo cambia el tiempo).

- La lista postfija (seccion 9) se recorre **una vez por bloque de 64 filas** en vez de una
  vez por fila (`_run_tape_batch!`, `_fitness_tape_batch`). Cada lugar de la pila es una
  columna de 64 valores: una instruccion apila la columna de una variable (el dataset
  guarda sus columnas seguidas, `Dataset.cols`), una constante, o el tramo del cache de un
  nodo; un operador combina las dos columnas de arriba.
- El costo de decidir cada instruccion (el que dominaba con la lista fila por fila) se
  reparte entre 64 filas, y los ciclos por columna son sencillos. Por eso aqui tambien
  conviene la lista en los arboles largos: no se usa `TAPE_MAX`.
- La suma del error se hace **fila por fila, en el mismo orden**, con el mismo corte
  temprano, y las operaciones son las mismas (division y raiz protegidas incluidas): el
  MSE es identico bit a bit (`--test` lo compara con el MSE de referencia, tambien con
  arboles que llegan a Inf o NaN).
- Con `--no-tape` no hay lista y la evaluacion es la recursion, fila por fila.

Medido en el modelo de C en C++ (mismo protocolo que la seccion 10; historial del mejor
MSE identico en las tres profundidades):

| Profundidad | Lista fila por fila | Lista por bloques | Pico de RAM |
|---|---|---|---|
| 7 | 17.4 s | 4.7 s (3.7x) | 2.3 GB |
| 9 | 29.8 s | 8.4 s (3.5x) | 2.6 GB |
| 12 | 66.0 s | 15.0 s (4.4x) | 5.2 GB |

Falta medirlo en Julia.

### Resultado medido (mismo resultado, menos tiempo)

Contenedor Linux, Julia 1.13.4 `-O3`, un hilo, `data/ejemplo.csv`, mediana de 3 semillas;
"antes" = version anterior de C (objetos inmutables + cache por nodo, commit 4561dac):

| Profundidad | GP antes (s) | GP ahora (s) | Crossover antes (us) | Crossover ahora (us) |
|---|---|---|---|---|
| 7 | 0.63 | 0.40 | 1.88 | 1.05 |
| 9 | 1.29 | 0.84 | 2.88 | 2.00 |
| 10 | - | - | 4.42 | 2.27 |
| 12 | 6.01 | 2.83 | 5.00 | 3.01 |
| 15 | - | - | 7.88 | 7.09 |

En profundidad 12, antes la poblacion inicial tardaba ~6.1 s, de ellos ~2.6 s de
recoleccion de basura; ahora ~2.1 s y practicamente sin recoleccion.

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
julia --startup-file=no -O3 main.jl --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
# GP completo: un comando por profundidad, 100 generaciones, mejor MSE de cada generacion
julia --startup-file=no -O3 main.jl --bench-gp --data=..\datos\dataset3a.csv --population=5000 --depths=7 --generations=100 --history=historial_7.csv
```

| Opcion | Significado |
|---|---|
| `--batch-pairs=K` | Parejas de padres en memoria a la vez en el benchmark de crossover. `0` (por defecto) = los 10,000 arboles a la vez, protocolo acordado; `2500` = 5,000 a la vez |
| `--generations=G` | Generaciones del GP completo (100 por defecto, lo acordado) |
| `--seed=S` | Semilla del generador. Sin ella, C elige una al azar (de 1 a 999,999) y la imprime por stderr para poder repetir la corrida; los scripts de benchmark siempre pasan una (5 por defecto) |
| `--history=archivo.csv` | Escribe `depth,generation,best_mse` (17 digitos) fuera del cronometro; la generacion 0 es la poblacion inicial |
| `--quiet` | No mostrar en pantalla el mejor MSE de cada generacion mientras corre el GP (ver abajo) |
| `--cache-mb=M` | Presupuesto aproximado del cache del GP, en MB (1024 por defecto; 0 = sin cache) |
| `--cache-min-size=K` | Tamano minimo de subarbol que guarda cache (0 = automatico) |
| `--reuse-min-size=K` | Los subarboles de K nodos o mas (y menores que el anterior) guardan cache la segunda vez que se evaluan (4 por defecto; 0 = no; mismo resultado) |
| `--no-tape` | Evaluar con la recursion en vez de la lista postfija (mismo resultado) |
| `--no-batch` | Recorrer la lista postfija fila por fila en vez de por bloques de 64 filas (mismo resultado; no tiene que ver con `--batch-pairs`) |
| `--no-fold` | Sin plegado de constantes (mismo resultado) |
| `--no-cut` | Sin corte temprano exacto (mismo resultado) |
| `--no-order` | Evaluar los hijos en orden de creacion (mismo resultado) |
| `--no-dup` | Evaluar tambien los hijos identicos a su padre (mismo resultado) |
| `--eval-file --file=arboles.txt --data=...` | Lee expresiones (una por linea) e imprime MSE con 17 digitos, profundidad y nodos |
| `--export-trees --count=N --depth=D --variables=V --seed=S [--full]` | Escribe arboles sin perdida (17 digitos) |

`--bench-gp` muestra **en pantalla, mientras corre**, una linea por generacion con el mejor
MSE y el tiempo que lleva (`  prof 7  gen  12/100  mejor MSE 0.1102234512  (5.3 s)`). Esas
lineas salen por stderr, asi que stdout sigue siendo solo el CSV que leen los scripts, y el
tiempo de imprimirlas se descuenta del cronometro (al final de cada profundidad dice cuanto
fue: unos 0.04 ms por linea en una terminal de Linux). `--quiet` las apaga.

`--no-fold`, `--no-cut` y `--no-dup` existen porque esas optimizaciones estaban pendientes
de confirmar con el profesor. Si no las acepta, se apagan sin cambiar el resultado.
`--no-tape` y `--reuse-min-size=0` no son por eso (el profesor ya acepto el arreglo
postfijo y el cache): sirven para medir cuanto ahorra cada una. `--no-batch` si: el
profesor dijo que la evaluacion por lotes no es valida (seccion 11).

Los benchmarks hacen primero una corrida pequena de calentamiento, para que **la
compilacion JIT de Julia no se cuente en el tiempo**.

### Protocolo del benchmark de crossover (acordado con el profesor)

- Los 10,000 padres (`grow_full`) estan en memoria al mismo tiempo. Si no alcanza la RAM,
  se permiten 5,000 (`--batch-pairs=2500`). La columna `resident_trees` lo dice.
- Se cronometra solo el crossover, incluida la creacion de los hijos (copia de camino).
- Los hijos se sueltan cada 16 cruces, fuera del cronometro (se recorta el arreglo de
  nodos a su tamano anterior).
- **RAM:** 24 bytes por nodo. En profundidad 15 con 10,000 arboles (~15,200 nodos cada
  uno) son unos 3.7 GB.

## Validacion de entradas

- Los constructores rechazan codigos de operacion desconocidos, operadores con el numero
  de hijos equivocado, hijos inexistentes e indices de variable menores que 1.
- `fitness` comprueba una vez por arbol que sus variables existan en el dataset;
  `eval_row` lo comprueba en cada acceso. `Dataset` exige filas de la misma longitud.
- No se pueden mezclar en un crossover arboles de grafos distintos.
- `full_gp` rechaza `generations < 0`, y la linea de comandos rechaza enteros negativos.
- CSV: un campo vacio (por ejemplo "1,2,") es un error. La primera linea solo se toma como
  encabezado si ningun campo es numerico, igual que en A y B.

## Verificacion

`julia main.jl --test` comprueba:

- `expression`, `eval_row`, MSE (incluida la tabla del manual), operadores protegidos,
  MSE = +Inf al desbordar y variables fuera de rango.
- El registro de nodo mide 24 bytes y no tiene punteros.
- El operador sin saltos da lo mismo, bit a bit, que el operador con saltos (incluidos
  Inf, NaN, ceros y valores diminutos).
- `grow_full` y `grow_random`: profundidades, todas las hojas a la profundidad exacta,
  metadatos guardados = reales, y que los arboles nuevos esten en preorden.
- Ida y vuelta texto <-> arbol sin perdida.
- **Plegado:** valores y marcas en un ejemplo conocido, que no cambie `expression` ni el
  tamano, que en 300 arboles cada subarbol plegado valga bit a bit lo mismo que recorrerlo
  (y este plegado si y solo si no tiene variables), y que Inf - Inf siga dando +Inf.
- **Cache:** el MSE con cache (tamanos minimos 1, 4 y 16), sin cache y la referencia
  escalar son identicos bit a bit, en frio, en caliente, alternando dos datasets y con y
  sin plegado. Los hijos evaluados con los subarboles de sus padres en cache dan el mismo MSE.
- **Corte temprano:** si devuelve un numero, es el MSE exacto; +Inf solo si el MSE real
  supera el corte.
- **Copia de camino:** el crossover conserva el total de nodos, los metadatos de los
  hijos son correctos, solo se crean los nodos del camino y los padres no cambian.
- Versiones protegidas: respetan `max_depth`, incluso al limite.
- GP: elitismo; el cache, el plegado, el corte temprano y la compactacion del grafo no
  alteran la corrida.

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

- `src/GPC.jl`: grafo, copia de camino, evaluacion con cache, operadores, GP, benchmark y pruebas.
- `main.jl`: interfaz de terminal.
- `tools/cross_check.jl`: verificacion cruzada A <-> C.
- `tools/verificar_con_b.jl`: verificacion cruzada A, C <-> B.
- `data/ejemplo.csv`: datos ilustrativos (no oficiales).

## Mejoras pendientes (medir cada una por separado)

1. ~~Barrer `--cache-min-size` con el dataset oficial~~: hecho (ver "Medido con dataset3a").
2. Unir subarboles identicos creados por separado (hash-consing). Medido en un modelo:
   casi no hay repetidos, porque las constantes son aleatorias; ganancia esperada baja.
3. Repetir los benchmarks con varias semillas y reportar media y desviacion.
