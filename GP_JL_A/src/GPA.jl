"""
Programacion Genetica (GP) - Implementacion A (alto nivel, convencional), Julia 1.9+.

Jerarquia de tipos del manual:

    Node (abstracto)        expression(nodo), evaluate(nodo, x)  [eval del manual]
     |- NodeInternal        op, left, right      (sqrt usa solo left)
     |- NodeVariable        index                (1 -> x1)
     |- NodeConstant        value

`eval` es una funcion reservada de Julia (cada modulo tiene la suya), por eso el
metodo del manual se llama `evaluate`.

Reglas compartidas con B y C:
  * La raiz esta a profundidad 0; una hoja sola tiene profundidad 0.
  * Terminales: variables x1..xn (repetibles) y constantes uniformes en [-1, 1).
  * Operadores: +, -, *, division protegida y raiz protegida (eps = 1e-12).
  * Mismas probabilidades de generacion que B (operador 1/5 cada uno; hoja
    constante o variable 1/2; grow_random corta con probabilidad 1/4).
  * Crossover y mutacion clonan (copia profunda) y no alteran a los padres.
  * Versiones protegidas: hasta 128 intentos al azar y, si ninguno sirve,
    intercambio de raices (igual que B).
  * GP: 100% grow_full, torneo binario repetido, 50% crossover / 50% mutacion,
    reemplazo = mejor 50% actual + mejor 50% de descendientes.
  * MSE: menor es mejor; un resultado no finito vale +Inf.

Generador aleatorio: Random.Xoshiro. A y C (en Julia) consumen el generador
EXACTAMENTE en el mismo orden, asi que con la misma semilla producen los mismos
arboles y, con C en modo PREORDER, la misma corrida del GP (ver
GP_JL_C/tools/cross_check.jl). Con B (std::mt19937_64) la equivalencia se
verifica con el formato de texto comun: expression(arbol, 17) es sin perdida.
"""
module GPA

using Random
using Printf

export Node, NodeInternal, NodeVariable, NodeConstant, AnyNode, Op, ADD, SUB, MUL, DIV, SQRT,
       expression, evaluate, clone, node_count, tree_depth, parse_expression, format_number,
       Dataset, read_csv, nvariables, nrows, fitness,
       grow_random, grow_full, crossover, mutation,
       depth_protected_crossover, depth_protected_mutation,
       GPOptions, GPResult, full_gp, CrossoverTiming, benchmark_crossover, run_tests

const EPS = 1e-12
const PROTECTED_ATTEMPTS = 128  # igual que B y C
const CHILDREN_CHUNK = 16       # benchmark: hijos liberados cada 16 cruces (igual que B y C)

# ---------------------------------------------------------------------------
# Operadores
# ---------------------------------------------------------------------------
@enum Op::UInt8 ADD SUB MUL DIV SQRT

# El orden importa: es el indice 1..5 que se sortea al elegir un operador (igual que B).
const OPERATORS = (ADD, SUB, MUL, DIV, SQRT)
const BINARY_BY_CHAR = Dict('+' => ADD, '-' => SUB, '*' => MUL, '/' => DIV)

arity(op::Op) = op == SQRT ? 1 : 2
opsymbol(op::Op) = op == ADD ? "+" : op == SUB ? "-" : op == MUL ? "*" : op == DIV ? "/" : "sqrt"

"Division protegida: a / b, sustituyendo b por eps cuando |b| < eps."
@inline protected_div(a::Float64, b::Float64) = a / (abs(b) < EPS ? EPS : b)

"Raiz protegida: sqrt(max(a, 0)). Igual que std::max(a, 0.0): un NaN se conserva."
@inline protected_sqrt(a::Float64) = sqrt(a < 0.0 ? 0.0 : a)

@inline function apply(op::Op, a::Float64, b::Float64)
    op == ADD && return a + b
    op == SUB && return a - b
    op == MUL && return a * b
    return protected_div(a, b)
end

const FMT8 = Printf.Format("%.8g")
const FMT17 = Printf.Format("%.17g")

"Igual que `ostream << setprecision(p)` de C++ (formato %g). 17 digitos = sin perdida."
function format_number(x::Float64, precision::Integer=8)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    fmt = precision == 8 ? FMT8 : precision == 17 ? FMT17 : Printf.Format("%.$(precision)g")
    return Printf.format(fmt, x)
end

# ---------------------------------------------------------------------------
# Parte 1 A) Tipos del arbol
# ---------------------------------------------------------------------------
abstract type Node end

struct NodeConstant <: Node
    value::Float64
end

struct NodeVariable <: Node
    index::Int  # 1 -> x1
    function NodeVariable(index::Integer)
        index >= 1 || throw(ArgumentError("las variables empiezan en x1 (indice >= 1)"))
        return new(Int(index))
    end
end

# Los hijos se declaran con una union concreta (y no con el tipo abstracto Node):
# asi Julia resuelve en compilacion a que metodo llamar y la recursion es rapida.
mutable struct NodeInternal <: Node
    op::Op
    left::Union{NodeInternal, NodeVariable, NodeConstant}
    right::Union{NodeInternal, NodeVariable, NodeConstant, Nothing}  # nothing en sqrt
    function NodeInternal(op::Op, left::Node, right::Union{Node, Nothing}=nothing)
        (arity(op) == 1) == (right === nothing) ||
            throw(ArgumentError("'$(opsymbol(op))' requiere $(arity(op)) hijo(s)"))
        return new(op, left, right)
    end
end

const AnyNode = Union{NodeInternal, NodeVariable, NodeConstant}

# --- expression ---------------------------------------------------------------
expression(n::NodeConstant, precision::Integer=8) = format_number(n.value, precision)
expression(n::NodeVariable, precision::Integer=8) = "x$(n.index)"
function expression(n::NodeInternal, precision::Integer=8)
    left = expression(n.left, precision)
    n.op == SQRT && return "sqrt(" * left * ")"
    return "(" * left * " " * opsymbol(n.op) * " " * expression(n.right::AnyNode, precision) * ")"
end

Base.show(io::IO, n::Node) = print(io, expression(n))

# --- evaluate (eval del manual): una llamada a la raiz recorre todo el arbol ---
evaluate(n::NodeConstant, x::AbstractVector{Float64})::Float64 = n.value

function evaluate(n::NodeVariable, x::AbstractVector{Float64})::Float64
    n.index <= length(x) || throw(ArgumentError("Indice de variable fuera del vector"))
    return @inbounds x[n.index]
end

function evaluate(n::NodeInternal, x::AbstractVector{Float64})::Float64
    a = evaluate(n.left, x)
    n.op == SQRT && return protected_sqrt(a)
    return apply(n.op, a, evaluate(n.right::AnyNode, x))
end

evaluate(n::Node, x::AbstractVector{<:Real}) = evaluate(n, Float64.(x))

# --- clone: copia profunda, crea cada nodo de nuevo ------------------------------
clone(n::NodeConstant) = NodeConstant(n.value)
clone(n::NodeVariable) = NodeVariable(n.index)
function clone(n::NodeInternal)
    right = n.right
    return NodeInternal(n.op, clone(n.left), right === nothing ? nothing : clone(right))
end

# --- tamano y profundidad --------------------------------------------------------
node_count(::Union{NodeConstant, NodeVariable})::Int = 1
function node_count(n::NodeInternal)::Int
    right = n.right
    return 1 + node_count(n.left) + (right === nothing ? 0 : node_count(right))
end

"Profundidad del arbol: una hoja sola = 0."
tree_depth(::Union{NodeConstant, NodeVariable})::Int = 0
function tree_depth(n::NodeInternal)::Int
    d = tree_depth(n.left)
    right = n.right
    right === nothing || (d = max(d, tree_depth(right)))
    return d + 1
end

# ---------------------------------------------------------------------------
# Lectura del formato de texto comun (el de expression() de A, B y C)
# ---------------------------------------------------------------------------
mutable struct _Parser
    s::String
    pos::Int
end

_fail(p::_Parser, why) = throw(ArgumentError("Expresion invalida ($why) en posicion $(p.pos): $(p.s)"))
_peek(p::_Parser) = p.pos <= ncodeunits(p.s) ? Char(codeunit(p.s, p.pos)) : '\0'

function _skip!(p::_Parser)
    while _peek(p) == ' ' || _peek(p) == '\t'
        p.pos += 1
    end
end

function _expect!(p::_Parser, c::Char)
    _skip!(p)
    _peek(p) == c || _fail(p, "se esperaba '$c'")
    p.pos += 1
end

function _matches(p::_Parser, word::String)
    n = ncodeunits(word)
    p.pos + n - 1 <= ncodeunits(p.s) || return false
    for k in 1:n
        codeunit(p.s, p.pos + k - 1) == codeunit(word, k) || return false
    end
    return true
end

function _token!(p::_Parser)
    start = p.pos
    while p.pos <= ncodeunits(p.s) && !(_peek(p) in (' ', '\t', '(', ')'))
        p.pos += 1
    end
    return String(codeunits(p.s)[start:p.pos-1])
end

function _parse_node!(p::_Parser)::AnyNode
    _skip!(p)
    c = _peek(p)
    c == '\0' && _fail(p, "fin inesperado")
    if c == '('
        p.pos += 1
        left = _parse_node!(p)
        _skip!(p)
        op = get(BINARY_BY_CHAR, _peek(p), nothing)
        op === nothing && _fail(p, "operador desconocido")
        p.pos += 1
        right = _parse_node!(p)
        _expect!(p, ')')
        return NodeInternal(op, left, right)
    elseif _matches(p, "sqrt(")
        p.pos += 5
        child = _parse_node!(p)
        _expect!(p, ')')
        return NodeInternal(SQRT, child)
    elseif c == 'x'
        p.pos += 1
        index = tryparse(Int, _token!(p))
        (index === nothing || index < 1) && _fail(p, "variable invalida")
        return NodeVariable(index)
    end
    value = tryparse(Float64, _token!(p))
    value === nothing && _fail(p, "numero invalido")
    return NodeConstant(value)
end

"Convierte \"((x1 + -0.25) * sqrt(x2))\" en un arbol. Inverso de expression()."
function parse_expression(text::AbstractString)::AnyNode
    p = _Parser(String(text), 1)
    tree = _parse_node!(p)
    _skip!(p)
    p.pos <= ncodeunits(p.s) && _fail(p, "texto sobrante")
    return tree
end

# ---------------------------------------------------------------------------
# Parte 1 C) Dataset y fitness (MSE)
# ---------------------------------------------------------------------------
struct Dataset
    rows::Vector{Vector{Float64}}  # X[i] = vector de entrada de la fila i
    y::Vector{Float64}
    function Dataset(rows::AbstractVector, y::AbstractVector)
        (isempty(rows) || length(rows) != length(y)) &&
            throw(ArgumentError("Dataset vacio o inconsistente"))
        width = length(rows[1])
        all(r -> length(r) == width, rows) ||
            throw(ArgumentError("Dimensiones inconsistentes del dataset"))
        return new([Vector{Float64}(r) for r in rows], Vector{Float64}(y))
    end
end

nvariables(d::Dataset) = length(d.rows[1])
nrows(d::Dataset) = length(d.y)

"Mismo formato que B: x1,...,xn,y; encabezado opcional en la primera linea."
function read_csv(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    y = Float64[]
    columns = 0
    for (line_number, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        values = Float64[]
        why = ""
        for field in split(line, ',')
            v = tryparse(Float64, strip(field))
            if v === nothing || !isfinite(v)
                why = "numero invalido: $(strip(field))"
                break
            end
            push!(values, v)
        end
        if isempty(why) && length(values) < 2
            why = "faltan columnas"
        end
        if isempty(why)
            columns == 0 && (columns = length(values))
            length(values) == columns || (why = "columnas inconsistentes")
        end
        if !isempty(why)
            line_number == 1 && continue  # encabezado, por ejemplo: x1,x2,y
            error("CSV linea $line_number: $why")
        end
        push!(y, pop!(values))
        push!(rows, values)
    end
    isempty(rows) && error("Dataset sin filas numericas")
    return Dataset(rows, y)
end

"MSE con el pseudocodigo del manual: evaluate(arbol, X[i]) fila por fila."
function fitness(tree::Node, data::Dataset)::Float64
    total = 0.0
    @inbounds for i in eachindex(data.y)
        difference = data.y[i] - evaluate(tree, data.rows[i])
        squared = difference * difference
        isfinite(squared) || return Inf
        total += squared
        isfinite(total) || return Inf
    end
    return total / length(data.y)
end

# ---------------------------------------------------------------------------
# Parte 2 A) Creacion aleatoria de arboles
# ---------------------------------------------------------------------------
"Constante uniforme en [-1, 1). C la calcula con la misma expresion."
@inline random_constant(rng::AbstractRNG) = 2.0 * rand(rng) - 1.0

function _random_leaf(nvars::Int, rng::AbstractRNG)::AnyNode
    if nvars == 0 || rand(rng, 0:1) == 0
        return NodeConstant(random_constant(rng))
    end
    return NodeVariable(rand(rng, 1:nvars))
end

function _grow(remaining::Int, nvars::Int, rng::AbstractRNG, full::Bool)::AnyNode
    if remaining == 0 || (!full && rand(rng, 0:3) == 0)
        return _random_leaf(nvars, rng)
    end
    op = OPERATORS[rand(rng, 1:5)]
    left = _grow(remaining - 1, nvars, rng, full)
    op == SQRT && return NodeInternal(op, left)
    right = _grow(remaining - 1, nvars, rng, full)
    return NodeInternal(op, left, right)
end

_check_depth(depth::Integer) = depth >= 0 || throw(ArgumentError("La profundidad debe ser >= 0"))

"Arbol con profundidad de A LO MAS `depth`."
function grow_random(depth::Integer, no_variables::Integer, rng::AbstractRNG)::AnyNode
    _check_depth(depth)
    return _grow(Int(depth), Int(no_variables), rng, false)
end

"Arbol cuyas hojas estan TODAS exactamente a profundidad `depth`."
function grow_full(depth::Integer, no_variables::Integer, rng::AbstractRNG)::AnyNode
    _check_depth(depth)
    return _grow(Int(depth), Int(no_variables), rng, true)
end

# ---------------------------------------------------------------------------
# Puntos de corte. Un Slot es (dueno, lado): el lugar donde cuelga un subarbol.
# La raiz cuelga de un contenedor Holder, asi que intercambiar dos subarboles
# (incluso dos raices) es intercambiar dos referencias.
# ---------------------------------------------------------------------------
mutable struct Holder
    left::AnyNode
end

const Owner = Union{Holder, NodeInternal}

struct Slot
    owner::Owner
    isleft::Bool
    depth::Int   # profundidad del nodo que cuelga aqui (raiz = 0)
    height::Int  # altura del subarbol que cuelga aqui (hoja = 0)
end

@inline getchild(o::Owner, isleft::Bool)::AnyNode =
    isleft ? o.left : (o::NodeInternal).right::AnyNode

@inline function setchild!(o::Owner, isleft::Bool, child::AnyNode)
    if isleft
        o.left = child
    else
        (o::NodeInternal).right = child
    end
    return nothing
end

# Slots en PREORDEN (raiz, subarbol izquierdo, derecho), como collect_slots de B.
function _collect!(out::Vector{Slot}, owner::Owner, isleft::Bool, depth::Int)::Int
    push!(out, Slot(owner, isleft, depth, 0))
    index = length(out)
    node = getchild(owner, isleft)
    height = 0
    if node isa NodeInternal
        height = _collect!(out, node, true, depth + 1) + 1
        if node.right !== nothing
            height = max(height, _collect!(out, node, false, depth + 1) + 1)
        end
    end
    out[index] = Slot(owner, isleft, depth, height)
    return height
end

"Devuelve (slots en preorden, altura del arbol)."
function _slots(holder::Holder)
    out = Slot[]
    height = _collect!(out, holder, true, 0)
    return out, height
end

function _swap!(a::Slot, b::Slot)
    sub_a = getchild(a.owner, a.isleft)
    setchild!(a.owner, a.isleft, getchild(b.owner, b.isleft))
    setchild!(b.owner, b.isleft, sub_a)
    return nothing
end

# ---------------------------------------------------------------------------
# Parte 2 C) Crossover y Parte 3) mutacion y versiones protegidas
# ---------------------------------------------------------------------------
"Clona ambos padres, elige un nodo en cada clon e intercambia los subarboles."
function crossover(a::Node, b::Node, rng::AbstractRNG)
    holder_a, holder_b = Holder(clone(a)), Holder(clone(b))
    slots_a, _ = _slots(holder_a)
    slots_b, _ = _slots(holder_b)
    i = rand(rng, 1:length(slots_a))
    j = rand(rng, 1:length(slots_b))
    _swap!(slots_a[i], slots_b[j])
    return holder_a.left, holder_b.left
end

"Clona el padre y reemplaza un subarbol por uno nuevo de grow_random."
function mutation(parent::Node, no_variables::Integer, rng::AbstractRNG,
                  new_subtree_max_depth::Integer=3)
    _check_depth(new_subtree_max_depth)
    holder = Holder(clone(parent))
    slots, _ = _slots(holder)
    replacement = grow_random(rand(rng, 0:Int(new_subtree_max_depth)), no_variables, rng)
    slot = slots[rand(rng, 1:length(slots))]
    setchild!(slot.owner, slot.isleft, replacement)
    return holder.left
end

"""
Crossover cuyos dos hijos no superan max_depth.

Al cambiar el subarbol i de A por el j de B, el hijo A mide
max(resto de A, prof_A(i) + altura_B(j)). El resto de A ya cabe (el padre
cabe), asi que basta comprobar prof_A(i) + altura_B(j) <= max y lo simetrico.
"""
function depth_protected_crossover(a::Node, b::Node, max_depth::Integer, rng::AbstractRNG)
    _check_depth(max_depth)
    holder_a, holder_b = Holder(clone(a)), Holder(clone(b))
    slots_a, height_a = _slots(holder_a)
    slots_b, height_b = _slots(holder_b)
    (height_a > max_depth || height_b > max_depth) &&
        throw(ArgumentError("Un padre supera max_depth"))
    for _ in 1:PROTECTED_ATTEMPTS
        x = slots_a[rand(rng, 1:length(slots_a))]
        y = slots_b[rand(rng, 1:length(slots_b))]
        if x.depth + y.height <= max_depth && y.depth + x.height <= max_depth
            _swap!(x, y)
            return holder_a.left, holder_b.left
        end
    end
    # Igual que B: intercambiar las raices siempre respeta el limite.
    return holder_b.left, holder_a.left
end

"Mutacion cuyo subarbol nuevo cabe en el presupuesto de profundidad restante."
function depth_protected_mutation(parent::Node, no_variables::Integer, max_depth::Integer,
                                  rng::AbstractRNG)
    _check_depth(max_depth)
    holder = Holder(clone(parent))
    slots, height = _slots(holder)
    height > max_depth && throw(ArgumentError("El padre supera max_depth"))
    slot = slots[rand(rng, 1:length(slots))]
    budget = Int(max_depth) - slot.depth
    setchild!(slot.owner, slot.isleft, grow_random(rand(rng, 0:budget), no_variables, rng))
    return holder.left
end

# ---------------------------------------------------------------------------
# Parte 4 A) GP completo
# ---------------------------------------------------------------------------
Base.@kwdef struct GPOptions
    population_size::Int = 40  # para probar; el benchmark pide 5000
    max_depth::Int = 5
    generations::Int = 5       # el manual NO fija las generaciones
    seed::Int = 42
end

struct GPResult
    best::AnyNode
    best_fitness::Float64
    best_fitness_per_generation::Vector{Float64}
end

struct Individual
    score::Float64
    tree::AnyNode
end

_score(ind::Individual) = ind.score

"GP completo del manual (pagina 5). population_size debe ser multiplo de 4."
function full_gp(data::Dataset, options::GPOptions)
    N = options.population_size
    nvariables(data) > 0 || throw(ArgumentError("Se necesita un dataset con variables"))
    (N >= 4 && N % 4 == 0) ||
        throw(ArgumentError("population_size debe ser multiplo de 4 y >= 4"))
    _check_depth(options.max_depth)
    rng = Xoshiro(options.seed)
    nvars, max_depth = nvariables(data), options.max_depth

    # 1-2) Poblacion inicial 100% grow_full y su fitness.
    population = Individual[]
    for _ in 1:N
        tree = grow_full(max_depth, nvars, rng)
        push!(population, Individual(fitness(tree, data), tree))
    end

    history = Float64[]
    pool = Vector{AnyNode}(undef, N)
    offspring = Individual[]
    for generation in 0:options.generations
        push!(history, minimum(_score, population))
        generation == options.generations && break

        # 3) Torneo binario repetido: pool de N ganadores (empate -> el primero).
        for i in 1:N
            a = population[rand(rng, 1:N)]
            b = population[rand(rng, 1:N)]
            pool[i] = a.score <= b.score ? a.tree : b.tree
        end

        # 4-6) Parejas al azar del pool; 50% hijos por crossover, 50% por mutacion.
        empty!(offspring)
        for _ in 1:(N ÷ 4)
            p1 = pool[rand(rng, 1:N)]
            p2 = pool[rand(rng, 1:N)]
            c1, c2 = depth_protected_crossover(p1, p2, max_depth, rng)
            push!(offspring, Individual(fitness(c1, data), c1))
            push!(offspring, Individual(fitness(c2, data), c2))
        end
        for _ in 1:(N ÷ 2)
            child = depth_protected_mutation(pool[rand(rng, 1:N)], nvars, max_depth, rng)
            push!(offspring, Individual(fitness(child, data), child))
        end

        # 7) Reemplazo 50/50 de estado estable (orden estable: mismos empates que C).
        sort!(population; by=_score, alg=MergeSort)
        sort!(offspring; by=_score, alg=MergeSort)
        population = vcat(population[1:N÷2], offspring[1:N÷2])
    end
    best = population[argmin(map(_score, population))]
    return GPResult(clone(best.tree), best.score, history)
end

# ---------------------------------------------------------------------------
# Parte 4 B) Benchmark de crossover
# ---------------------------------------------------------------------------
struct CrossoverTiming
    depth::Int
    trees::Int
    resident_trees::Int  # padres en memoria al mismo tiempo
    crossovers::Int
    seconds::Float64
    mean_seconds_per_crossover::Float64
    mean_nodes_per_tree::Float64
end

"""
Protocolo acordado con el profesor (igual que B y C): los `trees` padres
(grow_full) estan en memoria AL MISMO TIEMPO. batch_pairs = 0 -> todos
(trees/2 parejas); con poca RAM, batch_pairs = 2500 deja 5000 a la vez.
Se cronometra SOLO el crossover (clonacion + intercambio). Los hijos se
sueltan cada 16 cruces y la memoria se recolecta fuera del cronometro
(equivale a la destruccion de los hijos fuera del cronometro en B).
"""
function benchmark_crossover(trees::Integer, depth::Integer, no_variables::Integer;
                             seed::Integer=42, batch_pairs::Integer=0)
    (trees >= 2 && iseven(trees) && no_variables >= 1 && batch_pairs >= 0) ||
        throw(ArgumentError("trees debe ser par >= 2, variables >= 1 y batch_pairs >= 0"))
    _check_depth(depth)
    rng = Xoshiro(seed)
    pairs = trees ÷ 2
    per_batch = (batch_pairs == 0 || batch_pairs > pairs) ? pairs : Int(batch_pairs)
    measured = 0.0
    total_nodes = 0
    parents = Tuple{AnyNode, AnyNode}[]
    children = Tuple{AnyNode, AnyNode}[]
    offset = 0
    while offset < pairs
        count = min(per_batch, pairs - offset)
        empty!(parents)
        for _ in 1:count
            p1 = grow_full(depth, no_variables, rng)
            p2 = grow_full(depth, no_variables, rng)
            total_nodes += node_count(p1) + node_count(p2)
            push!(parents, (p1, p2))
        end
        GC.gc()  # fuera del cronometro
        for start in 1:CHILDREN_CHUNK:count
            stop = min(count, start + CHILDREN_CHUNK - 1)
            t0 = time_ns()
            for i in start:stop
                push!(children, crossover(parents[i][1], parents[i][2], rng))
            end
            measured += (time_ns() - t0) / 1e9
            empty!(children)
            GC.gc(false)  # recoleccion rapida de los hijos, fuera del cronometro
        end
        offset += count
    end
    return CrossoverTiming(depth, trees, 2 * per_batch, pairs, measured, measured / pairs,
                           total_nodes / trees)
end

# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
_all_internal(n::Union{NodeConstant, NodeVariable}, out) = out
function _all_internal(n::NodeInternal, out)
    push!(out, n)
    _all_internal(n.left, out)
    n.right === nothing || _all_internal(n.right::AnyNode, out)
    return out
end

_leaves(n::Union{NodeConstant, NodeVariable}, out) = push!(out, n)
function _leaves(n::NodeInternal, out)
    _leaves(n.left, out)
    n.right === nothing || _leaves(n.right::AnyNode, out)
    return out
end

_leaf_depths(n::Union{NodeConstant, NodeVariable}, d, out) = push!(out, d)
function _leaf_depths(n::NodeInternal, d, out)
    _leaf_depths(n.left, d + 1, out)
    n.right === nothing || _leaf_depths(n.right::AnyNode, d + 1, out)
    return out
end

function _throws(f)
    try
        f()
        return false
    catch err
        return err isa ArgumentError
    end
end

"Pruebas deterministas de todas las piezas. Devuelve cuantas comprobaciones pasaron."
function run_tests(; verbose::Bool=false)
    passed = Ref(0)
    function check(condition::Bool, description::AbstractString)
        condition || error("Fallo en prueba: $description")
        passed[] += 1
        verbose && println("  ok  ", description)
    end
    rng = Xoshiro(42)

    # --- Parte 1: tipos, expression, evaluate, MSE ---------------------------
    formula = NodeInternal(ADD, NodeVariable(1),
                           NodeInternal(MUL, NodeVariable(2), NodeConstant(0.5)))
    check(expression(formula) == "(x1 + (x2 * 0.5))", "expression")
    check(evaluate(formula, [2.0, 3.0]) == 3.5, "evaluate (x1 + x2*0.5) en [2, 3] = 3.5")
    check(node_count(formula) == 5, "node_count")
    check(tree_depth(formula) == 2, "profundidad con raiz = 0")
    check(tree_depth(NodeConstant(0.3)) == 0, "hoja sola = profundidad 0")

    exact = Dataset([[2.0, 3.0], [1.0, 2.0]], [3.5, 2.0])
    manual = Dataset([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.5, 1.0]], [2.5, 1.0, 3.5])
    check(fitness(formula, exact) == 0.0, "MSE cero")
    # Tabla del manual: predicciones 2, 0.5 y 2.25.
    expected = ((2.5 - 2.0)^2 + (1.0 - 0.5)^2 + (3.5 - 2.25)^2) / 3
    check(isapprox(fitness(formula, manual), expected), "MSE de la tabla del manual")

    twin = clone(formula)
    twin.right.right = NodeConstant(0.7)
    twin.left = NodeVariable(2)
    check(expression(formula) == "(x1 + (x2 * 0.5))" && twin !== formula &&
          twin.right !== formula.right, "clone es copia profunda")

    check(evaluate(NodeInternal(DIV, NodeConstant(1.0), NodeConstant(0.0)), Float64[]) == 1e12,
          "division protegida 1/0 = 1/eps")
    check(evaluate(NodeInternal(DIV, NodeConstant(1.0), NodeConstant(-1e-13)), Float64[]) == 1e12,
          "division protegida |b| < eps")
    check(evaluate(NodeInternal(SQRT, NodeConstant(-5.0)), Float64[]) == 0.0, "sqrt protegida")
    check(fitness(parse_expression("((x1 * 1e300) * 1e300)"), exact) == Inf,
          "MSE no finito = +Inf")
    check(_throws(() -> NodeInternal(ADD, NodeConstant(1.0))),
          "operador binario sin hijo derecho es rechazado")

    check(expression(parse_expression("(x1 + (x2 * 0.5))")) == expression(formula),
          "parse_expression")
    check(expression(parse_expression("sqrt((x3 / -0.25))")) == "sqrt((x3 / -0.25))",
          "parse con constante negativa")
    check(expression(parse_expression("(x1 - -1e-05)")) == "(x1 - -1e-05)",
          "parse con exponente (formato %g de C++)")

    # --- Parte 2: generacion ------------------------------------------------
    for depth in 0:8, nvars in (1, 2, 5)
        full = grow_full(depth, nvars, rng)
        rnd = grow_random(depth, nvars, rng)
        check(tree_depth(full) == depth, "grow_full($depth) profundidad exacta")
        check(tree_depth(rnd) <= depth, "grow_random($depth) profundidad <= $depth")
        back = parse_expression(expression(full, 17))
        check(expression(back, 17) == expression(full, 17) &&
              node_count(back) == node_count(full), "expression(17) -> parse sin perdida")
    end
    check(unique(_leaf_depths(grow_full(6, 3, rng), 0, Int[])) == [6],
          "grow_full: TODAS las hojas a profundidad exacta")
    leaves = Node[]
    for _ in 1:50
        _leaves(grow_full(5, 4, rng), leaves)
    end
    consts = [l.value for l in leaves if l isa NodeConstant]
    check(all(v -> -1.0 <= v <= 1.0, consts), "constantes en [-1, 1]")
    check(Set(l.index for l in leaves if l isa NodeVariable) == Set(1:4),
          "variables x1..xn aparecen y respetan el rango")

    # Tamano promedio vs. valor teorico (operador binario 4/5, sqrt 1/5):
    #   grow_full(d): E = sum_{k<=d} 1.8^k;   grow_random: E(d) = 1 + 1.35 E(d-1)
    mean_size(grow, depth) = sum(node_count(grow(depth, 2, rng)) for _ in 1:4000) / 4000
    theory_full = sum(1.8^k for k in 0:5)
    theory_rand = 1.0
    for _ in 1:6
        theory_rand = 1 + 1.35 * theory_rand
    end
    check(abs(mean_size(grow_full, 5) / theory_full - 1) < 0.03,
          "grow_full: tamano promedio = teorico (mismas probabilidades que B)")
    check(abs(mean_size(grow_random, 6) / theory_rand - 1) < 0.05,
          "grow_random: tamano promedio = teorico (mismas probabilidades que B)")

    # --- Parte 2 C y Parte 3: crossover y mutacion ---------------------------
    a, b = grow_full(4, 2, rng), grow_full(3, 2, rng)
    old_a, old_b = expression(a, 17), expression(b, 17)
    for _ in 1:200
        c, d = crossover(a, b, rng)
        check(node_count(c) + node_count(d) == node_count(a) + node_count(b),
              "crossover conserva el total de nodos")
        m = mutation(a, 2, rng)
        check(m !== a, "mutacion devuelve un arbol nuevo")
        pc, pd = depth_protected_crossover(a, b, 4, rng)
        check(tree_depth(pc) <= 4 && tree_depth(pd) <= 4, "crossover protegido respeta max_depth")
        check(tree_depth(depth_protected_mutation(a, 2, 4, rng)) <= 4,
              "mutacion protegida respeta max_depth")
    end
    check(expression(a, 17) == old_a && expression(b, 17) == old_b,
          "los operadores no alteran a los padres")
    c, d = crossover(a, b, rng)
    parent_nodes = Set{UInt}(objectid(n) for n in vcat(_all_internal(a, Any[]), _all_internal(b, Any[])))
    child_nodes = Set{UInt}(objectid(n) for n in vcat(_all_internal(c, Any[]), _all_internal(d, Any[])))
    check(isempty(intersect(parent_nodes, child_nodes)),
          "los hijos no comparten nodos con los padres")
    for _ in 1:100
        p1, p2 = grow_full(5, 2, rng), grow_full(5, 2, rng)
        c1, c2 = depth_protected_crossover(p1, p2, 5, rng)
        check(tree_depth(c1) <= 5 && tree_depth(c2) <= 5, "crossover protegido al limite")
        check(tree_depth(depth_protected_mutation(p1, 2, 5, rng)) <= 5,
              "mutacion protegida al limite")
    end
    deep = grow_full(6, 2, rng)
    check(_throws(() -> depth_protected_crossover(deep, b, 4, rng)),
          "padre que supera max_depth es rechazado")

    # --- Parte 4: GP completo ---------------------------------------------------
    run = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3))
    h = run.best_fitness_per_generation
    check(length(h) == 4, "historial de fitness")
    check(all(h[i] <= h[i-1] for i in 2:length(h)), "elitismo: el mejor MSE nunca aumenta")
    check(tree_depth(run.best) <= 3, "el mejor arbol respeta max_depth")
    check(run.best_fitness === fitness(run.best, exact), "best_fitness coincide con el arbol")
    r1 = full_gp(exact, GPOptions(seed=11))
    r2 = full_gp(exact, GPOptions(seed=11))
    check(expression(r1.best, 17) == expression(r2.best, 17) &&
          r1.best_fitness_per_generation == r2.best_fitness_per_generation,
          "misma semilla = misma corrida")

    t = benchmark_crossover(40, 4, 2; seed=1)
    check(t.crossovers == 20 && t.resident_trees == 40 && t.seconds > 0, "benchmark de crossover")
    t = benchmark_crossover(40, 4, 2; seed=1, batch_pairs=5)
    check(t.resident_trees == 10, "benchmark de crossover por lotes")
    return passed[]
end

end # module
