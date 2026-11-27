# Programacion Genetica (GP) - Implementacion B, C++20

Version **CONVENCIONAL** con nodos enlazados mediante `std::unique_ptr`.
Incluye todas las partes tecnicas del manual: arboles, expresion, eval, MSE,
`grow_random`, `grow_full`, clonacion, crossover, mutacion, operadores
protegidos, GP completo, benchmarks y pruebas.

> Esta carpeta corresponde a la implementacion B. La implementacion C del manual
> debe ser una implementacion alternativa completa e independiente, NO una copia
> del mismo arbol con distintos parametros.

## Abrir en Visual Studio (Windows)

1. Visual Studio -> Archivo -> Abrir -> Carpeta.
2. Selecciona esta carpeta (`GP_CPP20_B`). Visual Studio reconocera `CMakeLists.txt`.
3. Selecciona la configuracion **x64-Release** para rendimiento.
4. Compila `gp_b` (Compilar -> Compilar todo).
5. En el menu de depuracion puedes pasar argumentos al ejecutable, o usar
   **Developer PowerShell for VS** desde la carpeta del proyecto.

## Compilar desde Developer PowerShell de Visual Studio

```powershell
cl /std:c++20 /O2 /GL /EHsc /W4 /I include src\main.cpp src\gp.cpp /link /LTCG /OPT:REF /OPT:ICF /OUT:gp_b.exe
```

Durante desarrollo y depuracion tambien puedes usar:

```powershell
cl /std:c++20 /Od /Zi /EHsc /W4 /I include src\main.cpp src\gp.cpp /Fe:gp_b_debug.exe
```

Alternativa (CMake en terminal con compiladores configurados):

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

La ruta final de `gp_b.exe` depende del generador CMake elegido.

## Pruebas antes de hacer benchmarks

```powershell
.\gp_b.exe --test
.\gp_b.exe --demo
```

El dataset `data/ejemplo.csv` **es sintetico e ilustrativo**; no es el dataset oficial
que el docente debe proporcionar. Formato: `x1,x2,...,y` (ultima columna la salida).

## Benchmarks rapidos (comenzar por estos)

```powershell
.\gp_b.exe --bench-crossover --trees=40 --depths=3,4 --variables=2
.\gp_b.exe --bench-gp --data=data/ejemplo.csv --population=40 --depths=3,4 --generations=3
```

## Benchmarks solicitados por el manual

```powershell
.\gp_b.exe --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
# GP completo: un comando por profundidad (no las tres juntas), 100 generaciones,
# y el mejor MSE de cada generacion en un archivo aparte.
.\gp_b.exe --bench-gp --data=..\datos\dataset3a.csv --population=5000 --depths=7 --generations=100 --history=historial_7.csv
.\gp_b.exe --bench-gp --data=..\datos\dataset3a.csv --population=5000 --depths=9 --generations=100 --history=historial_9.csv
.\gp_b.exe --bench-gp --data=..\datos\dataset3a.csv --population=5000 --depths=12 --generations=100 --history=historial_12.csv
```

`--history=archivo.csv` escribe `depth,generation,best_mse` (MSE con 17 digitos) despues
de detener el cronometro; la generacion 0 es la poblacion inicial.

Para medir sin las optimizaciones pendientes de confirmar (mismo resultado):
`--no-fold --no-cut --no-dup`.

El manual **no establece el numero de generaciones**; se acordo usar **100** (valor por
defecto de `--generations` en A, B y C). El dataset oficial es `datos/dataset3a.csv`.

## Intercambio de arboles con A y C (verificacion cruzada)

```powershell
.\gp_b.exe --export-trees --count=100 --depth=6 --variables=2 --seed=1 [--full]
.\gp_b.exe --eval-file --file=arboles.txt --data=data\ejemplo.csv
```

`--export-trees` escribe arboles sin perdida (constantes con 17 digitos), uno por linea.
`--eval-file` lee una expresion por linea e imprime
`MSE<TAB>profundidad<TAB>nodos<TAB>expresion`, con el MSE en 17 digitos. A y C (Julia)
tienen los mismos modos; `GP_JL_C/tools/verificar_con_b.jl` los usa para comparar el MSE
de los mismos arboles bit a bit entre las tres implementaciones.

### Protocolo del benchmark de crossover (acordado con el profesor)

- **Los 10,000 padres estan en memoria al mismo tiempo** (opcion por defecto,
  `--batch-pairs=0`). Si no alcanza la RAM, el profesor acepta 5,000 a la vez:
  `--batch-pairs=2500` (2,500 parejas). La columna `resident_trees` del CSV dice cuantos
  padres hubo en memoria a la vez; hay que reportarla.
- Se cronometra SOLO `crossover`: clonacion de ambos padres + intercambio. No se cuenta
  la generacion de padres ni la destruccion de los hijos.
- Los hijos se liberan cada 16 cruces, fuera del cronometro, para que no dupliquen la RAM.
- **RAM necesaria en profundidad 15** (unos 15,200 nodos por arbol, ~45 bytes por nodo):
  ~7 GB con 10,000 arboles y ~3.5 GB con 5,000. En profundidad 12 o menos basta con ~1.3 GB.
  Revisa la RAM del equipo del torneo antes de elegir.
- Sin paralelismo: todo corre en un solo hilo (regla confirmada).

### Reglas del manual respetadas

- Raiz en profundidad 0; `grow_full(d)` tiene hojas exactamente a profundidad d.
- Terminales: variables repetibles y constantes uniformes en [-1,1].
- Operadores: +, -, *, division protegida, raiz protegida.
- Division protegida: a / (abs(b)<1e-12 ? 1e-12 : b).
- Raiz protegida: sqrt(max(x,0)).
- Crossover clona ambos padres, intercambia subarboles y produce dos hijos.
- Mutacion clona padre y sustituye subarbol generado aleatoriamente.
- Proteccion por profundidad en crossover y mutacion.
- Seleccion: torneo binario repetido.
- Variacion: 50% descendientes por crossover, 50% por mutacion.
- Reemplazo: mejores 50% actuales + mejores 50% descendientes (orden estable).
- GP inicializa 100% de la poblacion usando `grow_full`.
- MSE: menor es mejor; resultados no finitos reciben +infinito.

## Archivos

- `include/gp.hpp`: API y representacion convencional del arbol.
- `src/gp.cpp`: operaciones, fitness, GP completo, pruebas y benchmark crossover.
- `src/main.cpp`: interfaz de terminal y benchmark del GP completo.
- `data/ejemplo.csv`: datos ilustrativos, no oficiales.
- `CMakeLists.txt`: proyecto CMake para Visual Studio/GCC/Clang.

## Optimizacion de crossover y mutacion (resultados identicos)

- **Clonar y recolectar en una pasada.** Antes, cada padre se recorria tres veces:
  `clone`, `node_count` para reservar el vector de slots, y `collect_slots`. Ahora se
  clona y se registran los slots en preorden en un solo recorrido, y los vectores de
  slots se reutilizan entre llamadas.
- **Crossover protegido en O(1) por intento.** Antes, cada uno de los hasta 128
  intentos intercambiaba los subarboles y volvia a medir `tree_depth` de los dos hijos.
  Ahora cada slot guarda la profundidad del nodo y la altura de su subarbol, y el
  intento `i <-> j` es valido si `prof_A(i) + altura_B(j) <= max` y
  `prof_B(j) + altura_A(i) <= max`. Esa condicion es exactamente equivalente a la
  anterior: mismos intentos, mismo consumo del generador y mismos hijos.
- **Verificado:** `--test` pasa; `cross_check` (B contra C++ C, que reproduce a B con
  `--preorder`) da 43,930 comprobaciones y 0 fallos; y `--bench-gp` da el mismo mejor MSE
  y la misma expresion que antes en las profundidades 7, 9 y 12.
- **Medido** en el mismo contenedor Linux (g++ 13.3 `-O3`, un hilo), con los 10,000 padres
  en memoria; en profundidad 15, con 5,000:

| Profundidad | Crossover antes (us) | Crossover ahora (us) | GP completo antes (s) | GP completo ahora (s) |
|---|---|---|---|---|
| 7 | 13.4-14.2 | 9.0 | 0.99 | 0.86 |
| 9 | 46.1-47.4 | 30.4-30.8 | 3.27 | 3.03 |
| 10 | 78.7-82.3 | 60.7-62.7 | - | - |
| 12 | 247-258 | 230-238 | 36.9 | 35.4 |
| 15 | 1,620 | 1,017 | - | - |

## Asignador propio, nodo compacto y evaluacion ligera (resultados identicos)

B sigue siendo **convencional**: cada nodo es un objeto independiente enlazado con
`std::unique_ptr`, y clonar copia nodo por nodo. Solo cambia la gestion de memoria y
una comprobacion repetida:

- **Asignador propio** (`detail::NodePool`, en `operator new`/`operator delete` de
  `Node`). Los nodos salen de bloques de 2 MB y los liberados se reutilizan con una
  lista libre, en vez de pedir y liberar cada nodo al heap del sistema. `make_unique` y
  `unique_ptr` lo usan sin cambiar el resto del codigo. La memoria reservada no se
  devuelve al sistema hasta que termina el programa.
- **Nodo de 32 bytes** (antes 40). `value` y `variable_index` comparten una union,
  porque un nodo es constante o variable, nunca ambas cosas.
- **Evaluacion ligera.** `fitness` valida los indices de variable una sola vez por
  arbol y despues evalua todas las filas con una version sin comprobaciones y con un
  puntero simple en lugar de `std::span`. Las hojas (constantes y variables) se leen
  directamente desde el nodo padre (`operand`, forzada en linea), sin llamada
  recursiva: cerca de la mitad de los nodos son hojas, asi que se evita casi la mitad
  de las llamadas. Sigue siendo una evaluacion recursiva, fila por fila, con las mismas
  operaciones en el mismo orden, por lo que el MSE es identico bit a bit. Con MSVC es la
  optimizacion que mas pesa en el GP completo (unas 1.6 veces en el equipo de pruebas).
- **Verificado:** `--test` pasa; `cross_check` contra C++ C da 0 fallos; y `--bench-gp`
  da el mismo mejor MSE y la misma expresion que antes en las profundidades 7, 9 y 12.
- **Medido** en el contenedor Linux (g++ 13.3 `-O3`, un hilo). En Windows el heap del
  sistema es mas lento que el de Linux, asi que la ganancia del asignador deberia ser
  mayor ahi; hay que medirla en el equipo del torneo.

| Profundidad | Crossover antes (us) | Crossover ahora (us) | GP antes (s) | GP ahora (s) |
|---|---|---|---|---|
| 7 | 8.66 | 3.54 | 0.94 | 0.84 |
| 9 | 29.5 | 13.0 | 3.65 | 2.79-3.19 |
| 12 | 244 | 92 | 39.8 | 34.8-37.2 |
| 15 | 1,061 * | 560 * | - | - |

\* Con 5,000 padres a la vez.

- **RAM con 1,000 arboles de profundidad 15:** 716 MB antes y 479 MB ahora. Con
  10,000 arboles serian unos 4.8 GB en lugar de unos 7 GB.

## Evaluacion sin saltos, plegado, corte temprano y arboles en preorden (resultados identicos)

Medido en el contenedor Linux (g++ 13.3 y clang 18 `-O3`, un hilo, poblacion 5000, 10
generaciones, `data/ejemplo.csv`), la evaluacion era ~80% del tiempo del GP en
profundidad 12, y casi todo ese costo eran **saltos mal adivinados** por el procesador.

- **Evaluacion sin saltos** (`apply_binary`, `direct_value`). El `switch` por tipo de nodo
  es un salto condicional; en un arbol de miles de nodos el procesador no puede memorizar la
  secuencia y falla muy seguido (cada fallo cuesta ~15-20 ciclos). Ahora se calculan las
  cuatro operaciones binarias y se elige el resultado con el tipo de nodo como indice, y la
  hoja se lee eligiendo la direccion (`variable ? x + i : &value`). Mismas operaciones IEEE:
  MSE identico bit a bit. Evaluacion en profundidad 12: 5.6 -> 2.9 ns por nodo y fila (g++),
  5.25 -> 2.0 (clang).
- **Plegado de constantes** (`prepare_tree`, `--no-fold` lo apaga). La pasada que ya validaba
  los indices de variable ahora tambien calcula, una sola vez, el valor de cada subarbol sin
  variables (por ejemplo `(0.3 * -0.5)`); la evaluacion lo lee sin recorrerlo. Se pliegan
  17-19% de los nodos internos. El valor queda en la union del nodo y la marca `direct` en
  un byte del relleno: el nodo sigue midiendo 32 bytes.
- **Corte temprano exacto** (`--no-cut` lo apaga). Solo sobrevive el mejor 50% de los hijos.
  Un monticulo guarda los N/2 mejores MSE vistos; si el MSE parcial de un hijo ya supera
  al (N/2)-esimo mejor, el hijo nunca sobreviviria: se deja de evaluar y recibe +infinito.
- **Orden estable en el reemplazo** (`stable_sort` en lugar de `nth_element`). El torneo
  elige individuos por posicion; con `nth_element` esa posicion dependia de la biblioteca
  estandar y de todos los puntajes. Con orden estable (igual que A y C) el GP da el mismo
  resultado con y sin corte temprano. **Esto cambia las corridas de B respecto a versiones
  anteriores** (mismo algoritmo, otro orden de los sobrevivientes); las anteriores tampoco
  eran comparables entre compiladores.
- **Arboles generados en preorden**. `grow_impl` reserva el nodo padre antes que sus hijos,
  asi que el arbol queda en memoria en el orden en que clone y la evaluacion lo recorren.
  El generador se consume igual (mismos arboles; `--test` lo comprueba contra la version
  anterior). Benchmark de crossover: 1.4 a 1.9 veces mas rapido.
- **Verificado:** `--test` (incluye MSE identico a `Node::eval` en 400 arboles con datos
  extremos, plegado, preorden y el GP igual con y sin `--no-fold --no-cut`),
  `verificar_con_b.jl` (3,280 arboles contra A y C, 0 diferencias).

Mediana de 3 semillas en el mismo contenedor ("antes" = commit 4561dac):

| Profundidad | GP antes g++ (s) | GP ahora g++ (s) | GP antes clang (s) | GP ahora clang (s) | Crossover antes (us) | Crossover ahora (us) |
|---|---|---|---|---|---|---|
| 7 | 0.87 | 0.92 * | 0.88 | 0.66 | 5.30 | 2.61 |
| 9 | 3.11 | 2.72 | 3.09 | 2.09 | 14.99 | 8.92 |
| 10 | - | - | - | - | 25.55 | 16.29 |
| 12 | 31.20 | 18.20 | 31.40 | 14.94 | 101.4 | 79.2 |
| 15 | - | - | - | - | 523.8 | 329.5 |

\* Dentro del ruido de medicion (otras corridas de la misma version: 0.74-0.80 s).
La explicacion completa, con diagramas, esta en [`../docs/README.md`](../docs/README.md).

### Compilador

clang aprovecha mucho mejor la evaluacion sin saltos (2.0 contra 2.9 ns por nodo y fila).
En Windows conviene medir **clang-cl** (Visual Studio Installer -> "C++ Clang tools for
Windows"; en CMake, toolset `ClangCL`). clang-cl usa la biblioteca estandar de MSVC, asi que
las distribuciones aleatorias, y por lo tanto las corridas, son las mismas que con MSVC.
Confirmar con el profesor que se permite.

## Orden de evaluacion de los hijos (resultados identicos; `--no-order` lo apaga)

El corte temprano necesita un umbral: el (N/2)-esimo mejor MSE entre los hijos ya
evaluados. Antes, los hijos se evaluaban en el orden en que se creaban, asi que el umbral
tardaba en bajar. Ahora primero se **crean** todos los hijos (mismo orden y mismo consumo
del generador aleatorio) y despues se **evaluan empezando por los de mejores padres** (el
padre del que el hijo hereda la raiz; orden estable). Los buenos hijos aparecen antes, el
umbral baja antes y se dejan de evaluar mas hijos que no pueden sobrevivir.

El orden de evaluacion no cambia el MSE de ningun sobreviviente ni el orden de creacion,
que es el que decide los empates: el GP da exactamente el mismo resultado.

- **Verificado:** `--test` (el GP con y sin el orden nuevo es identico) y un programa que
  enlaza la version anterior y esta en un mismo ejecutable: 33,219 comprobaciones bit a bit,
  0 fallos, incluidas corridas completas del GP con `dataset3a.csv`.
- **Medido** con el dataset oficial `dataset3a.csv` (961 filas), poblacion 5000,
  10 generaciones, g++ 13 `-O3`, un hilo, Linux. Mismo MSE en todas las corridas.

| Profundidad | Antes (s) | Ahora (s) | Mejora |
|---|---|---|---|
| 7 (3 semillas) | 10.68 / 10.88 / 10.92 | 10.21 / 10.97 / 9.72 | -5% |
| 9 (3 semillas) | 35.18 / 34.41 / 31.58 | 32.28 / 33.23 / 30.08 | -5.5% |
| 12 (2 semillas) | 250.9 / 210.8 | 239.0 / 197.0 | -4.7% / -6.6% |

Filas evaluadas en profundidad 7 (semilla 2): 49.8 millones antes, 48.0 millones ahora,
de 52.9 millones posibles. Con estos datos el corte temprano ahorra poco porque la mayoria de
los hijos tiene un MSE parecido al umbral y lo cruza tarde.

Otras variantes medidas en las mismas condiciones y descartadas (mismo MSE, mas lentas):
sin plegado (`--no-fold`) +25%; sin corte (`--no-cut`) +5% a +8%; division solo en nodos de
division +4% a +7% (con g++; con MSVC puede ser distinto, ver la revision); `switch` en
lugar de la evaluacion sin saltos +4% a +12%; PGO de g++ mas lento.

## Hijos identicos a su padre (resultados identicos; `--no-dup` lo apaga)

Entre 6% y 15% de los hijos son **identicos a su padre**: un crossover que intercambia dos
subarboles iguales (por ejemplo, la hoja `x1` por la hoja `x1`), una mutacion que pone un
subarbol igual al que quita, o el intercambio de raices. Mismo arbol => mismo MSE, asi que
el hijo recibe el MSE del padre sin evaluar las 961 filas. Detectarlo cuesta casi nada: al
crear el hijo se compara el subarbol que sale con el que entra (casi siempre una hoja). Las
constantes solo cuentan como iguales si tienen los mismos bits. Idea tomada de la deteccion
de duplicados de Operon (Burlacu, hash de Zobrist), en version minima: solo copias del padre.

Con el corte temprano el hijo se registra igual que si se hubiera evaluado (+infinito si
su MSE pasa del umbral), asi que el GP es exactamente el mismo con y sin `--no-dup`.

- **Verificado:** `--test` (3,000 cruces y mutaciones: mismos hijos, mismo consumo del
  generador y cada copia reportada es identica a su padre; el GP igual con y sin `--no-dup`).
- **Medido** con `dataset3a.csv`, poblacion 5000, profundidad 7, 40 generaciones, semilla 5,
  g++ 13 `-O2`, Linux: 24.3 / 24.1 s contra 26.1 / 27.3 s con `--no-dup` (7% a 12% menos),
  mismo MSE (0.296032601).

## Validacion de entradas (no afecta el tiempo medido)

- Los argumentos numericos de la linea de comandos deben ser enteros no negativos
  escritos solo con digitos. Antes, `std::stoull("-1")` daba 18446744073709551615.
- CSV: una coma al final de la linea ("1,2,") es un error. La primera linea solo se toma
  como encabezado si **ningun** campo es numerico (por ejemplo `x1,x2,y`); una primera
  fila numerica mal formada ya no se descarta en silencio. A y C (Julia) siguen la misma regla.

## Limitaciones / mejoras propuestas

1. Esta es una version correcta de referencia funcional, **no la mas rapida posible**.
2. ~~Pool/arena para los nodos~~: hecho (`detail::NodePool`).
3. El crossover protegido prueba hasta 128 cortes y puede terminar en intercambio
   de raices; esta estrategia es valida en profundidad, pero puede reducir diversidad.
4. Generaciones son parametrizables; el manual no define una cifra.
5. Validar con el docente lo que falta del protocolo (semillas, dataset, repeticiones).
6. Probar `/O2 /GL /LTCG` y medir antes de introducir mas optimizaciones.
7. ~~Nodo mas compacto y arena de nodos~~: hecho; en profundidad 15 con 10,000 arboles
   la RAM baja de unos 7 GB a unos 4.8 GB.
8. Pendientes de confirmar con el profesor: PGO de MSVC y compilar con clang-cl.
   El plegado de subarboles constantes y el corte temprano del MSE ya estan hechos y
   activos por defecto, igual que reutilizar el MSE de los hijos identicos a su padre; si el
   profesor no los acepta, `--no-fold --no-cut --no-dup` los apagan (el resultado es el
   mismo, solo cambia el tiempo).
