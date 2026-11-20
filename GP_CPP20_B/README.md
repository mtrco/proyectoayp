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
.\gp_b.exe --bench-gp --data=data/DATASET_OFICIAL.csv --population=5000 --depths=7,9,12 --generations=10
```

El manual **no establece el numero de generaciones**, ni la cantidad concreta de
variables para el primer benchmark ni el formato del dataset oficial. Sustituye
`--generations=10`, el dataset y `--variables=2` cuando acuerden el protocolo.

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
- Reemplazo: mejores 50% actuales + mejores 50% descendientes.
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

## Validacion de entradas (no afecta el tiempo medido)

- Los argumentos numericos de la linea de comandos deben ser enteros no negativos
  escritos solo con digitos. Antes, `std::stoull("-1")` daba 18446744073709551615.
- CSV: una coma al final de la linea ("1,2,") es un error. La primera linea solo se toma
  como encabezado si **ningun** campo es numerico (por ejemplo `x1,x2,y`); una primera
  fila numerica mal formada ya no se descarta en silencio. A y C (Julia) siguen la misma regla.

## Limitaciones / mejoras propuestas

1. Esta es una version correcta de referencia funcional, **no la mas rapida posible**.
2. La clonacion usa `std::make_unique` por nodo; un pool/arena podria ser mas rapido
   (en profundidad 15 es ahora el costo dominante del crossover).
3. El crossover protegido prueba hasta 128 cortes y puede terminar en intercambio
   de raices; esta estrategia es valida en profundidad, pero puede reducir diversidad.
4. Generaciones son parametrizables; el manual no define una cifra.
5. Validar con el docente lo que falta del protocolo (semillas, dataset, repeticiones).
6. Probar `/O2 /GL /LTCG` y medir antes de introducir mas optimizaciones.
7. Con los 10,000 arboles en memoria, cada nodo de ~45 bytes en el heap limita la
   profundidad 15 a equipos con mas de 8 GB. Un nodo mas compacto o un arena de nodos
   reduciria la RAM y aceleraria la clonacion.
