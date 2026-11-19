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

### Nota muy importante sobre memoria del benchmark de crossover

`grow_full(15)` puede crear arboles enormes (maximo teorico de 65,535 nodos
con solo operadores binarios). Almacenar 10,000 simultaneamente podria agotar la RAM.
Esta implementacion **genera 10,000 arboles en lotes** de 8 (4 parejas), realiza
5,000 cruces y libera la memoria antes de crear mas. El tiempo cronometrado
incluye `crossover`, su clonacion y el intercambio, pero NO la generacion de padres
ni la destruccion de los hijos. Reportar explicita y claramente este protocolo:
si el docente exige los 10,000 arboles simultaneos, hay que modificarlo.

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

## Limitaciones / mejoras propuestas

1. Esta es una version correcta de referencia funcional, **no la mas rapida posible**.
2. La clonacion usa `std::make_unique` por nodo; un pool/arena podria ser mas rapido.
3. El crossover protegido prueba hasta 128 cortes y puede terminar en intercambio
   de raices; esta estrategia es valida en profundidad, pero puede reducir diversidad.
4. Generaciones son parametrizables; el manual no define una cifra.
5. Validar con el docente el protocolo final (semillas, dataset, repeticiones, hilos).
6. Probar `/O2 /GL /LTCG` y medir antes de introducir mas optimizaciones.
