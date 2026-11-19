"""Programacion Genetica (GP) - Implementacion A (alto nivel), Python 3.10+.

Version CONVENCIONAL orientada a objetos, tal como la describe el manual:

    Node (abstracta)        expression(), eval(vector)
     |- NodeInternal        left, right, operator   (sqrt usa solo left)
     |- NodeVariable        index
     |- NodeConstant        value

Reglas compartidas con B y C (para que las tres sean comparables):
  * La raiz esta a profundidad 0; una hoja sola tiene profundidad 0.
  * Terminales: variables x1..xn (repetibles) y constantes uniformes en [-1, 1].
  * Operadores: +, -, *, division protegida y raiz protegida (eps = 1e-12).
  * Mismas probabilidades de generacion que B (operador 1/5 cada uno,
    hoja = constante o variable 1/2, grow_random corta con prob. 1/4).
  * Crossover / mutacion clonan los padres (copia profunda) y no los alteran.
  * Las versiones protegidas usan la misma regla que B: hasta 128 intentos
    al azar y, si ninguno sirve, intercambio de raices.
  * GP: 100% grow_full, torneo binario repetido, 50% crossover / 50% mutacion,
    reemplazo = mejor 50% actual + mejor 50% de descendientes.
  * MSE: menor es mejor; un resultado no finito vale +inf.

El generador aleatorio es el de Python (Mersenne Twister de 32 bits), no el
std::mt19937_64 de C++, asi que con la misma semilla A NO produce los mismos
arboles que B. La equivalencia se verifica con el formato de texto comun:
expression(17) es sin perdida y el MSE de un mismo arbol es identico bit a bit
al de B y C (ver verificar_con_c.py).
"""

from __future__ import annotations

import gc
import math
import operator as _op
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:  # numpy es opcional: solo acelera la evaluacion del fitness.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

EPSILON = 1e-12
PROTECTED_ATTEMPTS = 128  # igual que B y C


# ---------------------------------------------------------------------------
# Operadores protegidos (version escalar y version vectorial con numpy)
# ---------------------------------------------------------------------------
def protected_div(a: float, b: float) -> float:
    """a / b, sustituyendo b por eps cuando |b| < eps."""
    return a / (EPSILON if abs(b) < EPSILON else b)


def protected_sqrt(a: float) -> float:
    """sqrt(max(a, 0)). Igual que std::max(a, 0.0): un NaN se conserva."""
    return math.sqrt(0.0 if a < 0.0 else a)


if np is not None:
    def _vec_div(a, b):
        return np.divide(a, np.where(np.abs(b) < EPSILON, EPSILON, b))

    def _vec_sqrt(a):
        return np.sqrt(np.where(a < 0.0, 0.0, a))
else:  # pragma: no cover
    _vec_div = _vec_sqrt = None


@dataclass(frozen=True)
class Operator:
    """Descripcion inmutable de un operador interno.

    Agregar un operador nuevo solo requiere crear otra instancia y anadirla a
    OPERATORS: el resto del codigo (eval, clone, crossover...) no cambia.
    """
    symbol: str
    arity: int
    scalar: Callable
    vector: Callable | None

    def __repr__(self) -> str:
        return f"Operator({self.symbol!r})"


ADD = Operator("+", 2, _op.add, np.add if np is not None else None)
SUB = Operator("-", 2, _op.sub, np.subtract if np is not None else None)
MUL = Operator("*", 2, _op.mul, np.multiply if np is not None else None)
DIV = Operator("/", 2, protected_div, _vec_div)
SQRT = Operator("sqrt", 1, protected_sqrt, _vec_sqrt)

# El orden importa: es el mismo indice 0..4 que usa B al sortear un operador.
OPERATORS: tuple[Operator, ...] = (ADD, SUB, MUL, DIV, SQRT)
BINARY_BY_SYMBOL = {op.symbol: op for op in OPERATORS if op.arity == 2}


def format_number(value: float, precision: int = 8) -> str:
    """Igual que `ostream << setprecision(p)` de C++ (formato %g)."""
    return format(value, f".{precision}g")


# ---------------------------------------------------------------------------
# A) Clases del arbol
# ---------------------------------------------------------------------------
class Node(ABC):
    """Nodo abstracto. Una llamada a la raiz procesa el arbol completo."""
    __slots__ = ()

    @abstractmethod
    def expression(self, precision: int = 8) -> str:
        """Texto infijo. precision=8 = formato de B; 17 = sin perdida."""

    @abstractmethod
    def eval(self, variables: Sequence[float]) -> float:
        """Evalua el arbol en un vector de entrada [x1, x2, ...]."""

    @abstractmethod
    def eval_columns(self, columns):
        """Evalua todas las filas a la vez (columns[i] = arreglo de x_{i+1})."""

    @abstractmethod
    def clone(self) -> "Node":
        """Copia profunda: crea una estructura nueva, nodo por nodo."""

    @abstractmethod
    def node_count(self) -> int: ...

    @abstractmethod
    def depth(self) -> int:
        """Profundidad del arbol (hoja sola = 0)."""

    def __str__(self) -> str:
        return self.expression()

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.expression()}>"


class NodeInternal(Node):
    __slots__ = ("operator", "left", "right")

    def __init__(self, operator: Operator, left: Node, right: Node | None = None):
        if (operator.arity == 1) != (right is None):
            raise ValueError(f"'{operator.symbol}' requiere {operator.arity} hijo(s)")
        self.operator = operator
        self.left = left
        self.right = right

    def expression(self, precision: int = 8) -> str:
        if self.operator.arity == 1:
            return f"{self.operator.symbol}({self.left.expression(precision)})"
        return (f"({self.left.expression(precision)} {self.operator.symbol} "
                f"{self.right.expression(precision)})")

    def eval(self, variables: Sequence[float]) -> float:
        if self.right is None:
            return self.operator.scalar(self.left.eval(variables))
        return self.operator.scalar(self.left.eval(variables), self.right.eval(variables))

    def eval_columns(self, columns):
        if self.right is None:
            return self.operator.vector(self.left.eval_columns(columns))
        return self.operator.vector(self.left.eval_columns(columns),
                                    self.right.eval_columns(columns))

    def clone(self) -> Node:
        right = self.right
        return NodeInternal(self.operator, self.left.clone(),
                            None if right is None else right.clone())

    def node_count(self) -> int:
        right = self.right
        return 1 + self.left.node_count() + (0 if right is None else right.node_count())

    def depth(self) -> int:
        right = self.right
        left_depth = self.left.depth()
        if right is None:
            return left_depth + 1
        right_depth = right.depth()
        return (left_depth if left_depth > right_depth else right_depth) + 1


class NodeVariable(Node):
    __slots__ = ("index",)

    def __init__(self, index: int):
        if index < 0:
            raise ValueError("El indice de variable debe ser >= 0")
        self.index = index  # 0 -> x1

    def expression(self, precision: int = 8) -> str:
        return f"x{self.index + 1}"

    def eval(self, variables: Sequence[float]) -> float:
        if self.index >= len(variables):
            raise IndexError("Indice de variable fuera del vector")
        return variables[self.index]

    def eval_columns(self, columns):
        if self.index >= len(columns):
            raise IndexError("Indice de variable fuera del vector")
        return columns[self.index]

    def clone(self) -> Node:
        return NodeVariable(self.index)

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 0


class NodeConstant(Node):
    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = float(value)

    def expression(self, precision: int = 8) -> str:
        return format_number(self.value, precision)

    def eval(self, variables: Sequence[float]) -> float:
        return self.value

    def eval_columns(self, columns):
        return self.value

    def clone(self) -> Node:
        return NodeConstant(self.value)

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 0


# Funciones con los nombres del manual.
def clone(node: Node) -> Node:
    return node.clone()


def node_count(node: Node) -> int:
    return node.node_count()


def tree_depth(node: Node) -> int:
    return node.depth()


# ---------------------------------------------------------------------------
# Lectura del formato de texto comun (el de expression() de B y C)
# ---------------------------------------------------------------------------
def parse_expression(text: str) -> Node:
    """Convierte "((x1 + -0.25) * sqrt(x2))" en un arbol. Inverso de expression()."""
    pos = 0
    n = len(text)

    def fail(why: str):
        raise ValueError(f"Expresion invalida ({why}) en posicion {pos}: {text!r}")

    def skip():
        nonlocal pos
        while pos < n and text[pos] in " \t":
            pos += 1

    def expect(char: str):
        nonlocal pos
        skip()
        if pos >= n or text[pos] != char:
            fail(f"se esperaba '{char}'")
        pos += 1

    def parse() -> Node:
        nonlocal pos
        skip()
        if pos >= n:
            fail("fin inesperado")
        c = text[pos]
        if c == "(":
            pos += 1
            left = parse()
            skip()
            if pos >= n or text[pos] not in BINARY_BY_SYMBOL:
                fail("operador desconocido")
            op = BINARY_BY_SYMBOL[text[pos]]
            pos += 1
            right = parse()
            expect(")")
            return NodeInternal(op, left, right)
        if text.startswith("sqrt(", pos):
            pos += 5
            child = parse()
            expect(")")
            return NodeInternal(SQRT, child)
        if c == "x":
            pos += 1
            start = pos
            while pos < n and text[pos].isdigit():
                pos += 1
            if start == pos:
                fail("variable sin indice")
            index = int(text[start:pos])
            if index == 0:
                fail("las variables empiezan en x1")
            return NodeVariable(index - 1)
        start = pos
        while pos < n and text[pos] not in " \t()":
            pos += 1
        try:
            return NodeConstant(float(text[start:pos]))
        except ValueError:
            pos = start
            fail("numero invalido")

    tree = parse()
    skip()
    if pos != n:
        fail("texto sobrante")
    return tree


# ---------------------------------------------------------------------------
# Dataset y fitness (MSE)
# ---------------------------------------------------------------------------
class Dataset:
    """Filas X (para eval escalar) y, si hay numpy, columnas (para eval por lotes)."""

    def __init__(self, x: Sequence[Sequence[float]], y: Sequence[float]):
        if not x or len(x) != len(y):
            raise ValueError("Dataset vacio o inconsistente")
        width = len(x[0])
        if any(len(row) != width for row in x):
            raise ValueError("Dimensiones inconsistentes del dataset")
        self.x = [tuple(map(float, row)) for row in x]
        self.y = [float(v) for v in y]
        if np is not None:
            self.columns = [np.array(col, dtype=np.float64) for col in zip(*self.x)]
            self.y_array = np.array(self.y, dtype=np.float64)

    @property
    def rows(self) -> int:
        return len(self.y)

    @property
    def variables(self) -> int:
        return len(self.x[0])

    @staticmethod
    def from_csv(filename: str) -> "Dataset":
        """Mismo formato que B: x1,...,xn,y; encabezado opcional en la 1a linea."""
        x, y = [], []
        columns = 0
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    values = [float(field.strip()) for field in line.split(",")]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError("numero no finito")
                    if len(values) < 2:
                        raise ValueError("faltan columnas")
                    columns = columns or len(values)
                    if len(values) != columns:
                        raise ValueError("columnas inconsistentes")
                except ValueError as error:
                    if line_number == 1:
                        continue  # encabezado, por ejemplo: x1,x2,y
                    raise ValueError(f"CSV linea {line_number}: {error}") from None
                y.append(values.pop())
                x.append(values)
        if not x:
            raise ValueError("Dataset sin filas numericas")
        return Dataset(x, y)


def fitness_scalar(tree: Node, data: Dataset) -> float:
    """MSE con el pseudocodigo del manual: eval(X[i]) fila por fila."""
    total = 0.0
    for row, target in zip(data.x, data.y):
        difference = target - tree.eval(row)
        squared = difference * difference
        if not math.isfinite(squared):
            return math.inf
        total += squared
        if not math.isfinite(total):
            return math.inf
    return total / data.rows


def fitness_numpy(tree: Node, data: Dataset) -> float:
    """MSE evaluando todas las filas a la vez con numpy.

    Cada nodo se recorre UNA vez por dataset (no una vez por fila). La suma
    de errores es secuencial (np.cumsum), en el mismo orden que la version
    escalar, de modo que el resultado es identico bit a bit.
    """
    with np.errstate(all="ignore"):
        predicted = tree.eval_columns(data.columns)
        difference = data.y_array - predicted  # difunde si el arbol es constante
        squared = difference * difference
        if not np.isfinite(squared).all():
            return math.inf
        total = float(np.cumsum(squared)[-1])
    return total / data.rows if math.isfinite(total) else math.inf


def fitness(tree: Node, data: Dataset, engine: str = "auto") -> float:
    """MSE (menor es mejor). engine: 'auto', 'numpy' o 'scalar'."""
    if engine == "scalar" or (engine == "auto" and np is None):
        return fitness_scalar(tree, data)
    if np is None:
        raise RuntimeError("engine='numpy' requiere tener numpy instalado")
    return fitness_numpy(tree, data)


# ---------------------------------------------------------------------------
# Parte 2 A) Creacion aleatoria de arboles
# ---------------------------------------------------------------------------
def _below(rng: random.Random, n: int) -> int:
    """Entero uniforme en [0, n). Mas rapido que randrange para este uso."""
    return int(rng.random() * n)


def _random_leaf(variable_count: int, rng: random.Random) -> Node:
    if variable_count == 0 or _below(rng, 2) == 0:
        return NodeConstant(rng.uniform(-1.0, 1.0))
    return NodeVariable(_below(rng, variable_count))


def _grow(remaining: int, variable_count: int, rng: random.Random, full: bool) -> Node:
    if remaining == 0 or (not full and _below(rng, 4) == 0):
        return _random_leaf(variable_count, rng)
    op = OPERATORS[_below(rng, len(OPERATORS))]
    left = _grow(remaining - 1, variable_count, rng, full)
    if op.arity == 1:
        return NodeInternal(op, left)
    return NodeInternal(op, left, _grow(remaining - 1, variable_count, rng, full))


def _check_depth(depth: int) -> None:
    if depth < 0:
        raise ValueError("La profundidad debe ser >= 0")


def grow_random(depth: int, no_variables: int, rng: random.Random) -> Node:
    """Arbol con profundidad de A LO MAS `depth`."""
    _check_depth(depth)
    return _grow(depth, no_variables, rng, full=False)


def grow_full(depth: int, no_variables: int, rng: random.Random) -> Node:
    """Arbol cuyas hojas estan EXACTAMENTE a profundidad `depth`."""
    _check_depth(depth)
    return _grow(depth, no_variables, rng, full=True)


# ---------------------------------------------------------------------------
# Puntos de corte. Un "slot" es (dueno, atributo): el lugar donde cuelga un
# subarbol. La raiz cuelga de un contenedor _Holder, asi que intercambiar dos
# subarboles (incluso raices) es intercambiar dos referencias con setattr.
# ---------------------------------------------------------------------------
class _Holder:
    __slots__ = ("left",)

    def __init__(self, root: Node):
        self.left = root


def _collect_slots(owner, attr: str, out: list) -> None:
    """Slots en preorden (raiz, izquierdo, derecho), como collect_slots de B."""
    out.append((owner, attr))
    node = getattr(owner, attr)
    if isinstance(node, NodeInternal):
        _collect_slots(node, "left", out)
        if node.right is not None:
            _collect_slots(node, "right", out)


def _collect_shape(owner, attr: str, depth: int, out: list) -> int:
    """Slots en preorden con (dueno, atributo, profundidad, altura). Devuelve la altura."""
    index = len(out)
    out.append(None)
    node = getattr(owner, attr)
    height = 0
    if isinstance(node, NodeInternal):
        height = _collect_shape(node, "left", depth + 1, out) + 1
        if node.right is not None:
            height = max(height, _collect_shape(node, "right", depth + 1, out) + 1)
    out[index] = (owner, attr, depth, height)
    return height


def _swap(slot_a, slot_b) -> None:
    owner_a, attr_a = slot_a[0], slot_a[1]
    owner_b, attr_b = slot_b[0], slot_b[1]
    sub_a = getattr(owner_a, attr_a)
    setattr(owner_a, attr_a, getattr(owner_b, attr_b))
    setattr(owner_b, attr_b, sub_a)


# ---------------------------------------------------------------------------
# Parte 2 C) Crossover y Parte 3) mutacion y versiones protegidas
# ---------------------------------------------------------------------------
def crossover(a: Node, b: Node, rng: random.Random) -> tuple[Node, Node]:
    """Clona ambos padres, elige un nodo en cada clon e intercambia subarboles."""
    holder_a, holder_b = _Holder(a.clone()), _Holder(b.clone())
    slots_a, slots_b = [], []
    _collect_slots(holder_a, "left", slots_a)
    _collect_slots(holder_b, "left", slots_b)
    _swap(slots_a[_below(rng, len(slots_a))], slots_b[_below(rng, len(slots_b))])
    return holder_a.left, holder_b.left


def mutation(parent: Node, no_variables: int, rng: random.Random,
             new_subtree_max_depth: int = 3) -> Node:
    """Clona el padre y reemplaza un subarbol por uno nuevo de grow_random."""
    _check_depth(new_subtree_max_depth)
    holder = _Holder(parent.clone())
    slots = []
    _collect_slots(holder, "left", slots)
    replacement = grow_random(_below(rng, new_subtree_max_depth + 1), no_variables, rng)
    owner, attr = slots[_below(rng, len(slots))]
    setattr(owner, attr, replacement)
    return holder.left


def depth_protected_crossover(a: Node, b: Node, max_depth: int,
                              rng: random.Random) -> tuple[Node, Node]:
    """Crossover cuyos dos hijos no superan max_depth.

    Si se cambia el subarbol i de A por el j de B, el hijo A mide
    max(resto de A, prof_A(i) + altura_B(j)). El resto de A ya cabe (el padre
    cabe), asi que basta comprobar prof_A(i) + altura_B(j) <= max y lo
    simetrico. Profundidad y altura se calculan una vez: cada intento es O(1).
    """
    _check_depth(max_depth)
    holder_a, holder_b = _Holder(a.clone()), _Holder(b.clone())
    shape_a, shape_b = [], []
    if (_collect_shape(holder_a, "left", 0, shape_a) > max_depth
            or _collect_shape(holder_b, "left", 0, shape_b) > max_depth):
        raise ValueError("Un padre supera max_depth")
    for _ in range(PROTECTED_ATTEMPTS):
        slot_a = shape_a[_below(rng, len(shape_a))]
        slot_b = shape_b[_below(rng, len(shape_b))]
        if slot_a[2] + slot_b[3] <= max_depth and slot_b[2] + slot_a[3] <= max_depth:
            _swap(slot_a, slot_b)
            return holder_a.left, holder_b.left
    # Igual que B: intercambiar las raices siempre respeta el limite.
    return holder_b.left, holder_a.left


def depth_protected_mutation(parent: Node, no_variables: int, max_depth: int,
                             rng: random.Random) -> Node:
    """Mutacion cuyo subarbol nuevo cabe en el presupuesto de profundidad restante."""
    _check_depth(max_depth)
    holder = _Holder(parent.clone())
    shape = []
    if _collect_shape(holder, "left", 0, shape) > max_depth:
        raise ValueError("El padre supera max_depth")
    owner, attr, depth, _height = shape[_below(rng, len(shape))]
    budget = max_depth - depth
    setattr(owner, attr, grow_random(_below(rng, budget + 1), no_variables, rng))
    return holder.left


# ---------------------------------------------------------------------------
# Parte 4 A) GP completo
# ---------------------------------------------------------------------------
@dataclass
class GPOptions:
    population_size: int = 40  # para probar; el benchmark pide 5000
    max_depth: int = 5
    generations: int = 5       # el manual NO fija las generaciones
    seed: int = 42
    engine: str = "auto"       # 'auto' | 'numpy' | 'scalar'


@dataclass
class GPResult:
    best: Node
    best_fitness: float
    best_fitness_per_generation: list[float] = field(default_factory=list)


def full_gp(data: Dataset, options: GPOptions) -> GPResult:
    """GP completo del manual (pagina 5). population_size multiplo de 4."""
    N = options.population_size
    if data.variables == 0:
        raise ValueError("Se necesita un dataset con variables")
    if N < 4 or N % 4 != 0:
        raise ValueError("population_size debe ser multiplo de 4 y >= 4")
    _check_depth(options.max_depth)
    rng = random.Random(options.seed)
    nvars, max_depth = data.variables, options.max_depth

    def score(tree: Node) -> float:
        return fitness(tree, data, options.engine)

    # 1-2) Poblacion inicial 100% grow_full y su fitness. Individuo = (MSE, arbol).
    population = []
    for _ in range(N):
        tree = grow_full(max_depth, nvars, rng)
        population.append((score(tree), tree))

    history: list[float] = []
    for generation in range(options.generations + 1):
        history.append(min(population, key=_by_score)[0])
        if generation == options.generations:
            break

        # 3) Torneo binario repetido: pool de N ganadores (empate -> el primero).
        pool = []
        for _ in range(N):
            a = population[_below(rng, N)]
            b = population[_below(rng, N)]
            pool.append(a[1] if a[0] <= b[0] else b[1])

        # 4-6) Parejas al azar del pool; 50% hijos por crossover, 50% por mutacion.
        offspring = []
        for _ in range(N // 4):
            p1 = pool[_below(rng, N)]
            p2 = pool[_below(rng, N)]
            c1, c2 = depth_protected_crossover(p1, p2, max_depth, rng)
            offspring.append((score(c1), c1))
            offspring.append((score(c2), c2))
        for _ in range(N // 2):
            child = depth_protected_mutation(pool[_below(rng, N)], nvars, max_depth, rng)
            offspring.append((score(child), child))

        # 7) Reemplazo 50/50 de estado estable.
        population.sort(key=_by_score)
        offspring.sort(key=_by_score)
        population = population[: N // 2] + offspring[: N // 2]

    best_score, best_tree = min(population, key=_by_score)
    return GPResult(best_tree.clone(), best_score, history)


def _by_score(individual) -> float:
    return individual[0]


# ---------------------------------------------------------------------------
# Parte 4 B) Benchmark de crossover
# ---------------------------------------------------------------------------
@dataclass
class CrossoverTiming:
    depth: int
    trees: int
    crossovers: int
    seconds: float
    mean_seconds_per_crossover: float
    mean_nodes_per_tree: float


def benchmark_crossover(trees: int, depth: int, variables: int, seed: int = 42,
                        batch_pairs: int = 4) -> CrossoverTiming:
    """Mismo protocolo que B y C.

    Se generan `trees` padres con grow_full en lotes de `batch_pairs` parejas
    (4 = protocolo de B; trees/2 = todos en memoria a la vez). Se cronometra
    SOLO el crossover (clonacion de ambos padres + intercambio), no la
    generacion de padres ni la destruccion de los hijos.

    El recolector de ciclos de Python se apaga durante el benchmark: los
    arboles no tienen ciclos y sus pausas meterian ruido en el cronometro.
    """
    if trees < 2 or trees % 2 or variables < 1 or batch_pairs < 1:
        raise ValueError("trees debe ser par >= 2, variables >= 1 y batch_pairs >= 1")
    _check_depth(depth)
    rng = random.Random(seed)
    pairs = trees // 2
    measured = 0.0
    total_nodes = 0
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for offset in range(0, pairs, batch_pairs):
            count = min(batch_pairs, pairs - offset)
            parents = [(grow_full(depth, variables, rng), grow_full(depth, variables, rng))
                       for _ in range(count)]
            total_nodes += sum(a.node_count() + b.node_count() for a, b in parents)
            start = time.perf_counter()
            children = [crossover(a, b, rng) for a, b in parents]
            measured += time.perf_counter() - start
            del children, parents  # destruccion fuera del intervalo cronometrado
    finally:
        if was_enabled:
            gc.enable()
    return CrossoverTiming(depth, trees, pairs, measured, measured / pairs,
                           total_nodes / trees)


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
def run_tests(verbose: bool = False) -> int:
    """Pruebas deterministas de todas las piezas. Devuelve cuantas paso."""
    passed = 0

    def check(condition: bool, description: str) -> None:
        nonlocal passed
        if not condition:
            raise AssertionError(f"Fallo en prueba: {description}")
        passed += 1
        if verbose:
            print(f"  ok  {description}")

    def same_bits(a: float, b: float) -> bool:
        return math.isnan(a) and math.isnan(b) or a.hex() == b.hex()

    rng = random.Random(42)
    engines = ["scalar"] + (["numpy"] if np is not None else [])

    # --- Parte 1: clases, expression, eval, MSE -----------------------------
    formula = NodeInternal(ADD, NodeVariable(0),
                           NodeInternal(MUL, NodeVariable(1), NodeConstant(0.5)))
    check(formula.expression() == "(x1 + (x2 * 0.5))", "expression")
    check(formula.eval([2.0, 3.0]) == 3.5, "eval (x1 + x2*0.5) en [2, 3] = 3.5")
    check(formula.node_count() == 5, "node_count")
    check(formula.depth() == 2, "profundidad con raiz = 0")
    check(NodeConstant(0.3).depth() == 0, "hoja sola = profundidad 0")

    manual = Dataset([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.5, 1.0]], [2.5, 1.0, 3.5])
    exact = Dataset([[2.0, 3.0], [1.0, 2.0]], [3.5, 2.0])
    for engine in engines:
        check(fitness(formula, exact, engine) == 0.0, f"MSE cero ({engine})")
        # (x1 + x2*0.5) sobre la tabla del manual: predicciones 2, 0.5, 2.25
        expected = ((2.5 - 2.0) ** 2 + (1.0 - 0.5) ** 2 + (3.5 - 2.25) ** 2) / 3
        check(math.isclose(fitness(formula, manual, engine), expected),
              f"MSE de la tabla del manual ({engine})")

    copy = formula.clone()
    copy.right.right.value = 0.7
    copy.left.index = 1
    check(formula.expression() == "(x1 + (x2 * 0.5))" and copy is not formula
          and copy.right is not formula.right, "clone es copia profunda")

    division = NodeInternal(DIV, NodeConstant(1.0), NodeConstant(0.0))
    check(division.eval([]) == 1e12, "division protegida 1/0 = 1/eps")
    check(NodeInternal(DIV, NodeConstant(1.0), NodeConstant(-1e-13)).eval([]) == 1e12,
          "division protegida |b| < eps")
    check(NodeInternal(SQRT, NodeConstant(-5.0)).eval([]) == 0.0, "sqrt protegida")
    huge = parse_expression("((x1 * 1e300) * 1e300)")
    for engine in engines:
        check(fitness(huge, exact, engine) == math.inf, f"MSE no finito = +inf ({engine})")
    try:
        NodeInternal(ADD, NodeConstant(1.0))
        check(False, "binario sin hijo derecho debe fallar")
    except ValueError:
        check(True, "binario sin hijo derecho es rechazado")

    # Formato de texto: ida y vuelta.
    check(parse_expression("(x1 + (x2 * 0.5))").expression() == formula.expression(),
          "parse_expression")
    check(parse_expression("sqrt((x3 / -0.25))").expression() == "sqrt((x3 / -0.25))",
          "parse con constante negativa")
    check(parse_expression("(x1 - -1e-05)").expression() == "(x1 - -1e-05)",
          "parse con exponente")

    # --- Parte 2: generacion ------------------------------------------------
    for depth in range(0, 9):
        for nvars in (1, 2, 5):
            full = grow_full(depth, nvars, rng)
            rnd = grow_random(depth, nvars, rng)
            check(full.depth() == depth, f"grow_full({depth}) profundidad exacta")
            check(rnd.depth() <= depth, f"grow_random({depth}) profundidad <= {depth}")
            back = parse_expression(full.expression(17))
            check(back.expression(17) == full.expression(17)
                  and back.node_count() == full.node_count(),
                  "expression(17) -> parse sin perdida")

    def leaves(node):
        if isinstance(node, NodeInternal):
            yield from leaves(node.left)
            if node.right is not None:
                yield from leaves(node.right)
        else:
            yield node

    def leaf_depths(node, d=0):
        if isinstance(node, NodeInternal):
            yield from leaf_depths(node.left, d + 1)
            if node.right is not None:
                yield from leaf_depths(node.right, d + 1)
        else:
            yield d

    tree = grow_full(6, 3, rng)
    check(set(leaf_depths(tree)) == {6}, "grow_full: TODAS las hojas a profundidad exacta")
    consts = [l.value for _ in range(50) for l in leaves(grow_full(5, 3, rng))
              if isinstance(l, NodeConstant)]
    check(all(-1.0 <= v <= 1.0 for v in consts), "constantes en [-1, 1]")
    used = {l.index for _ in range(50) for l in leaves(grow_full(5, 4, rng))
            if isinstance(l, NodeVariable)}
    check(used == {0, 1, 2, 3}, "variables x1..xn aparecen y se respetan los indices")

    # Tamano promedio vs. el valor teorico (operador binario 4/5, sqrt 1/5):
    #   grow_full(d):   E = sum_{k<=d} 1.8^k       grow_random: E(d) = 1 + 1.35 E(d-1)
    def mean_size(generator, depth, samples=4000):
        return sum(generator(depth, 2, rng).node_count() for _ in range(samples)) / samples
    theory_full = sum(1.8 ** k for k in range(6))
    theory_rand = 1.0
    for _ in range(6):
        theory_rand = 1 + 1.35 * theory_rand
    check(abs(mean_size(grow_full, 5) / theory_full - 1) < 0.03,
          "grow_full: tamano promedio = teorico (mismas probabilidades que B)")
    check(abs(mean_size(grow_random, 6) / theory_rand - 1) < 0.05,
          "grow_random: tamano promedio = teorico (mismas probabilidades que B)")

    # Evaluacion vectorial == escalar, bit a bit (incluye ceros, negativos y extremos).
    if np is not None:
        r2 = random.Random(7)
        x = [[0.0 if i % 11 == 0 else r2.uniform(-3, 3) * 10 ** r2.choice([-8, 0, 0, 3])
              for _ in range(3)] for i in range(120)]
        y = [r2.uniform(-3, 3) for _ in range(120)]
        hard = Dataset(x, y)
        mismatches = 0
        for t in range(400):
            p = grow_random(1 + t % 9, 3, rng)
            if not same_bits(fitness_scalar(p, hard), fitness_numpy(p, hard)):
                mismatches += 1
        check(mismatches == 0, "MSE numpy == MSE escalar, bit a bit (400 arboles)")

    # --- Parte 2 C y Parte 3: crossover y mutacion ---------------------------
    a, b = grow_full(4, 2, rng), grow_full(3, 2, rng)
    old_a, old_b = a.expression(17), b.expression(17)
    for _ in range(200):
        c, d = crossover(a, b, rng)
        check(c.node_count() + d.node_count() == a.node_count() + b.node_count(),
              "crossover conserva el total de nodos")
        m = mutation(a, 2, rng)
        check(m is not a, "mutacion devuelve un arbol nuevo")
        pc, pd = depth_protected_crossover(a, b, 4, rng)
        check(pc.depth() <= 4 and pd.depth() <= 4, "crossover protegido respeta max_depth")
        pm = depth_protected_mutation(a, 2, 4, rng)
        check(pm.depth() <= 4, "mutacion protegida respeta max_depth")
    check(a.expression(17) == old_a and b.expression(17) == old_b,
          "los operadores no alteran a los padres")

    # Hijos independientes de los padres (no comparten nodos).
    c, d = crossover(a, b, rng)
    ids_parents = {id(n) for n in _all_nodes(a)} | {id(n) for n in _all_nodes(b)}
    check(not ({id(n) for n in _all_nodes(c)} | {id(n) for n in _all_nodes(d)}) & ids_parents,
          "los hijos no comparten nodos con los padres")

    # Con limite justo (max = profundidad de los padres), sigue siendo valido.
    for _ in range(100):
        p1, p2 = grow_full(5, 2, rng), grow_full(5, 2, rng)
        c1, c2 = depth_protected_crossover(p1, p2, 5, rng)
        check(c1.depth() <= 5 and c2.depth() <= 5, "crossover protegido al limite")
        check(depth_protected_mutation(p1, 2, 5, rng).depth() <= 5,
              "mutacion protegida al limite")
    try:
        depth_protected_crossover(grow_full(6, 2, rng), b, 4, rng)
        check(False, "padre demasiado profundo debe fallar")
    except ValueError:
        check(True, "padre que supera max_depth es rechazado")

    # --- Parte 4: GP completo -------------------------------------------------
    for engine in engines:
        run = full_gp(exact, GPOptions(population_size=24, max_depth=3,
                                       generations=3, engine=engine))
        check(len(run.best_fitness_per_generation) == 4, f"historial de fitness ({engine})")
        check(all(later <= earlier for earlier, later in
                  zip(run.best_fitness_per_generation, run.best_fitness_per_generation[1:])),
              f"elitismo: el mejor MSE nunca aumenta ({engine})")
        check(run.best.depth() <= 3, f"el mejor arbol respeta max_depth ({engine})")
        check(same_bits(run.best_fitness, fitness_scalar(run.best, exact)),
              f"best_fitness coincide con el arbol devuelto ({engine})")
    if np is not None:
        o = GPOptions(population_size=40, max_depth=5, generations=4, seed=3)
        r_scalar = full_gp(exact, GPOptions(**{**o.__dict__, "engine": "scalar"}))
        r_numpy = full_gp(exact, GPOptions(**{**o.__dict__, "engine": "numpy"}))
        check(r_scalar.best_fitness_per_generation == r_numpy.best_fitness_per_generation
              and r_scalar.best.expression(17) == r_numpy.best.expression(17),
              "el motor de evaluacion no cambia el GP")
    r1 = full_gp(exact, GPOptions(seed=11))
    r2 = full_gp(exact, GPOptions(seed=11))
    check(r1.best.expression(17) == r2.best.expression(17), "misma semilla = misma corrida")

    timing = benchmark_crossover(40, 4, 2, seed=1)
    check(timing.crossovers == 20 and timing.seconds > 0, "benchmark de crossover")
    return passed


def _all_nodes(node: Node):
    yield node
    if isinstance(node, NodeInternal):
        yield from _all_nodes(node.left)
        if node.right is not None:
            yield from _all_nodes(node.right)
