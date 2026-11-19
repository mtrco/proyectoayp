# Programacion Genetica (GP) - Implementacion A (alto nivel), Python 3.10+

Version **CONVENCIONAL de alto nivel**. Implementa las partes tecnicas del manual con
la jerarquia de clases que propone: arboles, `expression`, `eval`, MSE, `grow_random`,
`grow_full`, clonacion, crossover, mutacion, versiones protegidas por profundidad,
GP completo, benchmarks y pruebas.

A no compite en el torneo; su papel es ser la version **mas legible** y servir de
referencia para validar a B y C. El codigo sigue el pseudocodigo del manual casi linea por linea.

## Diseno

```
Node (abstracta)            expression(precision), eval(vector), eval_columns(columnas),
 |                          clone(), node_count(), depth()
 |- NodeInternal            operator, left, right      (sqrt usa solo left)
 |- NodeVariable            index                      (0 -> x1)
 |- NodeConstant            value
```

- **Operadores como objetos** (`Operator`: simbolo, aridad, funcion escalar y vectorial).
  Para agregar uno nuevo basta crear una instancia y anadirla a `OPERATORS`.
- **Crossover por intercambio de referencias**, como dice el manual. Cada punto de corte
  es un *slot* `(dueno, atributo)`. La raiz cuelga de un contenedor (`_Holder`), asi que
  intercambiar dos subarboles, incluso dos raices, es intercambiar dos referencias.
- **Protecciones por profundidad en O(1) por intento.** Se calculan una vez la
  profundidad y la altura de cada nodo. El intercambio `i <-> j` es valido si
  `prof_A(i) + altura_B(j) <= max` y `prof_B(j) + altura_A(i) <= max`. Es la misma
  regla que B: hasta 128 intentos y, si ninguno sirve, intercambio de raices.
- **Dos motores de fitness**, con resultados identicos bit a bit:
  - `scalar`: el pseudocodigo del manual, `eval(X[i])` fila por fila. Solo usa Python.
  - `numpy` (por defecto si numpy esta instalado): `eval_columns` recorre el arbol una
    vez por dataset y cada nodo opera sobre todas las filas a la vez. La suma del error
    es secuencial (`np.cumsum`), en el mismo orden que B, por eso el MSE es identico.

### Reglas del manual respetadas (las mismas que B y C)

- Raiz a profundidad 0; `grow_full(d)` pone todas las hojas exactamente a profundidad d.
- Terminales: variables repetibles y constantes uniformes en [-1, 1].
- Operadores: +, -, *, division protegida `a / (|b| < 1e-12 ? 1e-12 : b)` y raiz
  protegida `sqrt(max(x, 0))`.
- Mismas probabilidades de generacion que B: operador 1/5 cada uno, hoja constante o
  variable 1/2, y `grow_random` corta con probabilidad 1/4.
- Crossover y mutacion clonan (copia profunda) y no alteran a los padres.
- GP: 100% `grow_full`, torneo binario repetido, 50% crossover / 50% mutacion,
  reemplazo con el mejor 50% actual + el mejor 50% de los descendientes.
- MSE: menor es mejor; un resultado no finito vale +inf.

### Diferencia con B y C: el generador aleatorio

A usa `random.Random` de Python (Mersenne Twister de 32 bits). B y C usan
`std::mt19937_64` con distribuciones de C++, cuyo algoritmo ademas cambia entre MSVC
y GCC. Por eso, **con la misma semilla A no produce los mismos arboles que B**. Las
distribuciones si son las mismas (las pruebas comparan el tamano promedio contra el
valor teorico). La equivalencia se verifica con el formato de texto comun; ver abajo.

## Instalar y ejecutar (Windows)

```powershell
py -m pip install numpy        # opcional pero recomendado (acelera el fitness)
cd GP_PY_A
py main.py --test              # pruebas internas
py main.py --demo
```

Sin numpy todo funciona con el motor escalar (`--engine=scalar`); solo es mas lento.

## Uso

```powershell
py main.py --bench-crossover --trees=10000 --depths=7,9,10,12,15 --variables=2
py main.py --bench-gp --data=data\DATASET_OFICIAL.csv --population=5000 --depths=7,9,12 --generations=10
py main.py --eval-file --file=arboles.txt --data=data\ejemplo.csv
py main.py --export-trees --count=100 --depth=6 --variables=2 --seed=1 [--full]
```

| Opcion | Significado |
|---|---|
| `--batch-pairs=K` | Parejas por lote en el benchmark de crossover (4 = protocolo de B) |
| `--engine=numpy\|scalar` | Motor de evaluacion del fitness (mismo resultado, distinta velocidad) |
| `--eval-file` | Misma salida que `gp_c --eval-file`: `MSE<TAB>profundidad<TAB>nodos<TAB>expresion` |
| `--export-trees` | Arboles sin perdida (17 digitos), uno por linea |

La salida CSV de los benchmarks tiene las mismas columnas que B y C.

## Verificacion

`py main.py --test` hace **1,120 comprobaciones**, entre ellas:

- `expression`, `eval` y el MSE de la tabla del manual; clonacion profunda (los hijos
  no comparten ningun nodo con los padres); division y raiz protegidas; MSE = +inf al desbordar.
- `grow_full`: todas las hojas a la profundidad exacta. `grow_random`: profundidad maxima.
  Constantes en [-1, 1] y todas las variables usadas.
- Tamano promedio de `grow_full` y `grow_random` igual al teorico
  (`sum 1.8^k` y `E(d) = 1 + 1.35 E(d-1)`), lo que confirma las mismas probabilidades que B.
- Ida y vuelta `expression(17)` -> `parse_expression` sin perdida.
- MSE con numpy igual bit a bit al escalar (400 arboles en un dataset con ceros y extremos).
- Crossover: conserva el total de nodos. Crossover y mutacion: padres intactos.
  Versiones protegidas: respetan `max_depth`, incluso al limite.
- GP: elitismo (el mejor MSE nunca sube), misma semilla da la misma corrida, y el
  motor de evaluacion no cambia el resultado.

### Verificacion cruzada con C (y por transitividad con B)

C ya se verifico bit a bit contra B con `cross_check`. El script de A cubre el resto:

```powershell
py verificar_con_c.py --gp-c=..\GP_CPP20_C\out\build\x64-Release\gp_c.exe
```

1. A genera arboles y C los evalua con `--eval-file`.
2. C genera arboles con `--export-trees` y A los evalua.
3. En tres datasets (`ejemplo.csv`, uno con ceros, negativos y magnitudes extremas, y
   uno que desborda) compara **MSE bit a bit (con ambos motores de A), profundidad y
   numero de nodos**.

Resultado (g++ 13.3 en Linux): **4,320 arboles, 0 diferencias**. De ellos, 1,007 dieron
MSE = +inf en el dataset de desborde.

## Resultados de referencia

Medidos en un contenedor Linux con 2 nucleos, Python 3.13 + numpy 2.5, un solo hilo.
B y C se compilaron en la misma maquina con g++ 13.3 `-O2 -ffp-contract=off`.
**Solo son orientativos**: el manual exige medir todo en el mismo hardware.

### Crossover (10,000 arboles, 5,000 cruces, lotes de 4 parejas)

| Profundidad | Nodos promedio | A Python (us/cruce) | B (us/cruce) | C (us/cruce) |
|---|---|---|---|---|
| 7 | 136 | 63 | 8.8 | 0.22 |
| 9 | 447 | 204 | 31 | 0.33 |
| 10 | 801 | 378 | 50 | 0.47 |
| 12 | 2,605 | 1,295 | 210 | 1.15 |
| 15 | 15,170 | 8,507 | 1,589 | 12.0 |

A es unas 5 a 8 veces mas lento que B. El costo viene de crear un objeto de Python por
cada nodo al clonar. Para profundidad 15 el benchmark completo tarda unos 1.5 minutos,
porque generar los padres (no cronometrado) cuesta mas que los cruces.

### GP completo (poblacion 5,000, 10 generaciones, `ejemplo.csv`, semilla 42)

| Profundidad | A Python (s) | B (s) | C (s) | Mejor MSE de A |
|---|---|---|---|---|
| 7 | 10.4 | 1.21 | 0.19 | 0.0104 |
| 9 | 33.8 | 3.68 | 0.55 | 0.134 |
| 12 | 212.7 | 37.9 | 3.49 | 0.25 |

Con la misma semilla el mejor MSE no coincide con el de B porque los generadores
aleatorios son distintos. Para compararlos hay que correr varias semillas y comparar
las distribuciones.

Motor `numpy` contra `scalar` (poblacion 1,000, profundidad 7, 3 generaciones, mismo
resultado en ambos): con `ejemplo.csv` (49 filas) tarda 0.51 s contra 1.53 s, unas 3
veces mas rapido. Con un dataset de 1,000 filas tarda 0.71 s contra 25.0 s, unas 35
veces mas rapido. La ventaja crece con el numero de filas del dataset oficial.

## Archivos

- `gp.py`: clases, operadores, generacion, fitness, crossover, mutacion, GP, benchmark y pruebas.
- `main.py`: interfaz de terminal (mismas opciones que B y C).
- `verificar_con_c.py`: verificacion cruzada A <-> C.
- `data/ejemplo.csv`: datos ilustrativos (no oficiales).

## Limitaciones / mejoras posibles

1. Python crea un objeto por nodo; clonar arboles de profundidad 15 es caro. Se
   priorizo la claridad. PyPy (sin numpy, con `--engine=scalar`) podria acelerar
   bastante el benchmark de crossover.
2. La recursion de Python esta limitada a unas 1,000 llamadas, muy por encima de las
   profundidades del manual (maximo 15).
3. El recolector de ciclos se apaga durante el benchmark de crossover, porque los
   arboles no tienen ciclos y sus pausas meterian ruido en el cronometro.
4. Validar con el profesor el protocolo final: generaciones, dataset, semillas y repeticiones.
