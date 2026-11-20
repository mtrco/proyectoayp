"""
Programacion Genetica (GP) - Implementacion C (alternativa, no convencional), Julia 1.9+.

Idea: los arboles son INMUTABLES y comparten subarboles. Ambas cosas las aprobo el
profesor ("compartir subarboles en vez de copiarlos" y "toda la poblacion en un solo
grafo con cache de resultados"). La evaluacion por lotes NO esta permitida, asi que
se evalua fila por fila, como en el manual.

1. Nodos inmutables con metadatos. Cada nodo interno guarda su operador, sus hijos,
   su tamano (numero de nodos) y su altura, calculados al crearlo en O(1). Un nodo
   nunca se modifica, asi que los padres no pueden alterarse: compartir es seguro.
   Las variables x1..xn existen una sola vez y se reutilizan.

2. Copia de camino (path copying). El k-esimo nodo en preorden se encuentra bajando
   por los tamanos, en O(profundidad). Un hijo de crossover o de mutacion crea solo
   los nodos del camino de la raiz al punto de corte; el resto se COMPARTE con el
   padre. En profundidad 15 son unos 16 nodos nuevos por hijo, no unos 15,000.
   La poblacion entera es un grafo aciclico (DAG) de subarboles compartidos.

3. Evaluacion fila por fila con cache. Para cada fila i se recorre el arbol
   recursivamente. Los nodos con tamano >= S guardan su resultado en cada fila (su
   cache). Un nodo cuyo cache esta completo para el dataset en uso devuelve el valor
   guardado sin bajar mas. Como un hijo comparte casi todo con sus padres, ya
   evaluados, solo se calculan los nodos nuevos del camino y los subarboles pequenos
   (tamano < S) que cuelgan de el. El cache de un nodo se libera cuando el nodo deja
   de usarse (lo recoge el recolector de basura de Julia). S se elige segun un
   presupuesto de memoria (cache_mb).

4. Subarboles constantes precalculados. Un nodo cuyo subarbol solo tiene constantes
   (por ejemplo sqrt(0.25) + 0.1) calcula su valor UNA vez al crearse, con los mismos
   operadores protegidos, y la evaluacion lo devuelve sin recorrer el subarbol en cada
   fila. Los nodos originales se conservan: expression, profundidad, tamano y puntos de
   corte no cambian, y el MSE es identico bit a bit.

Mismas reglas que A y B, y MISMO consumo del generador aleatorio que A, con el punto
de corte elegido uniforme en preorden como en A. Con la misma semilla, C produce
EXACTAMENTE los mismos arboles, los mismos hijos y la misma corrida del GP que A
(ver tools/cross_check.jl). Un solo hilo: el profesor no permite paralelismo.
"""
module GPC

using Random
using Printf

export Node, OpNode, VarNode, ConstNode, AnyNode, Dataset,
       expression, parse_expression, format_number, node_count, tree_depth, clone,
       eval_row, fitness, read_csv, dataset_from_rows, nvariables, nrows, variable,
       grow_random, grow_full, crossover, mutation,
       depth_protected_crossover, depth_protected_mutation, auto_cache_min_size,
       GPOptions, GPResult, full_gp, CrossoverTiming, benchmark_crossover, run_tests

const EPS = 1e-12
const PROTECTED_ATTEMPTS = 128   # igual que A y B
const CHILDREN_CHUNK = 16        # benchmark: hijos liberados cada 16 cruces
const DEFAULT_CACHE_MIN_SIZE = 16
const NO_CACHE = typemax(Int)    # cache_min_size que desactiva el cache

# Codigos de operacion de los nodos internos (mismo orden que A y B al sortear).
const OP_ADD = 0x02
const OP_SUB = 0x03
const OP_MUL = 0x04
const OP_DIV = 0x05
const OP_SQRT = 0x06
const RANDOM_OPS = (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT)
const BINARY_BY_CHAR = Dict('+' => OP_ADD, '-' => OP_SUB, '*' => OP_MUL, '/' => OP_DIV)

_symbol(op::UInt8) = op == OP_ADD ? "+" : op == OP_SUB ? "-" : op == OP_MUL ? "*" :
                     op == OP_DIV ? "/" : "sqrt"

@inline protected_div(a::Float64, b::Float64) = a / (abs(b) < EPS ? EPS : b)
@inline protected_sqrt(a::Float64) = sqrt(a < 0.0 ? 0.0 : a)

@inline function apply(op::UInt8, a::Float64, b::Float64)
    op == OP_ADD && return a + b
    op == OP_SUB && return a - b
    op == OP_MUL && return a * b
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
# Representacion
# ---------------------------------------------------------------------------
abstract type Node end

struct ConstNode <: Node
    value::Float64
end

# mutable (con campo constante) para que cada variable sea UN objeto compartido.
mutable struct VarNode <: Node
    const index::Int  # 1 -> x1
    function VarNode(index::Integer)
        index >= 1 || throw(ArgumentError("las variables empiezan en x1 (indice >= 1)"))
        return new(Int(index))
    end
end

const EMPTY_CACHE = Float64[]  # compartido; nunca se modifica (se reemplaza el campo)

mutable struct OpNode <: Node
    const size::Int32     # nodos del subarbol (incluido este)
    const height::Int16   # altura del subarbol (hoja = 0)
    const op::UInt8
    const isconst::Bool   # el subarbol solo contiene constantes (ninguna variable)
    const left::Union{OpNode, VarNode, ConstNode}
    const right::Union{OpNode, VarNode, ConstNode, Nothing}  # nothing en sqrt
    const cvalue::Float64 # valor del subarbol si isconst (calculado una vez al crearlo)
    cache::Vector{Float64}  # resultado en cada fila del dataset `cache_id`
    cache_id::Int           # id del dataset; 0 = sin cache; -id = llenandose
    function OpNode(op::UInt8, left::Node, right::Union{Node, Nothing}=nothing)
        (OP_ADD <= op <= OP_SQRT) || throw(ArgumentError("codigo de operacion invalido: $op"))
        (op == OP_SQRT) == (right === nothing) ||
            throw(ArgumentError("'$(_symbol(op))' requiere $(op == OP_SQRT ? 1 : 2) hijo(s)"))
        s = 1 + nsize(left) + (right === nothing ? 0 : nsize(right))
        h = 1 + max(nheight(left), right === nothing ? 0 : nheight(right))
        # Plegado de constantes: mismas operaciones protegidas que la evaluacion, una sola
        # vez. El valor es bit a bit el que daria recorrer el subarbol en cualquier fila.
        left_const, left_value = _const_info(left)
        if right === nothing
            isc = left_const
            value = isc ? protected_sqrt(left_value) : 0.0
        else
            right_const, right_value = _const_info(right)
            isc = left_const && right_const
            value = isc ? apply(op, left_value, right_value) : 0.0
        end
        return new(Int32(s), Int16(h), op, isc, left, right, value, EMPTY_CACHE, 0)
    end
end

const AnyNode = Union{OpNode, VarNode, ConstNode}

# (es constante, valor) de un hijo, para el plegado de constantes de OpNode.
@inline _const_info(n::ConstNode) = (true, n.value)
@inline _const_info(::VarNode) = (false, 0.0)
@inline _const_info(n::OpNode) = (n.isconst, n.cvalue)

@inline nsize(::Union{VarNode, ConstNode}) = 1
@inline nsize(n::OpNode) = Int(n.size)
@inline nheight(::Union{VarNode, ConstNode}) = 0
@inline nheight(n::OpNode) = Int(n.height)

"Numero de nodos (O(1): se guarda en cada nodo)."
node_count(n::Node) = nsize(n)
"Profundidad del arbol, hoja sola = 0 (O(1): se guarda en cada nodo)."
tree_depth(n::Node) = nheight(n)

"""
Clonacion. Los nodos son inmutables, asi que compartirlos es seguro: ningun operador
puede alterar a un padre. Por eso `clone` devuelve el mismo arbol (aprobado por el
profesor). `deep_copy_tree` hace una copia nodo por nodo, solo para pruebas.
"""
clone(n::Node) = n
deep_copy_tree(n::Union{VarNode, ConstNode}) = n isa ConstNode ? ConstNode(n.value) : n
deep_copy_tree(n::OpNode) = OpNode(n.op, deep_copy_tree(n.left),
                                      n.right === nothing ? nothing : deep_copy_tree(n.right))

# Variables compartidas: una sola instancia por indice.
const VARIABLES = VarNode[]
function variable(index::Integer)
    index >= 1 || throw(ArgumentError("las variables empiezan en x1 (indice >= 1)"))
    while length(VARIABLES) < index
        push!(VARIABLES, VarNode(length(VARIABLES) + 1))
    end
    return @inbounds VARIABLES[index]
end

expression(n::ConstNode, precision::Integer=8) = format_number(n.value, precision)
expression(n::VarNode, precision::Integer=8) = "x$(n.index)"
function expression(n::OpNode, precision::Integer=8)
    left = expression(n.left, precision)
    n.op == OP_SQRT && return "sqrt(" * left * ")"
    return "(" * left * " " * _symbol(n.op) * " " * expression(n.right::AnyNode, precision) * ")"
end
Base.show(io::IO, n::Node) = print(io, expression(n))

# ---------------------------------------------------------------------------
# Lectura del formato de texto comun
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
        return OpNode(op, left, right)
    elseif _matches(p, "sqrt(")
        p.pos += 5
        child = _parse_node!(p)
        _expect!(p, ')')
        return OpNode(OP_SQRT, child)
    elseif c == 'x'
        p.pos += 1
        index = tryparse(Int, _token!(p))
        (index === nothing || index < 1) && _fail(p, "variable invalida")
        return variable(index)
    end
    value = tryparse(Float64, _token!(p))
    value === nothing && _fail(p, "numero invalido")
    return ConstNode(value)
end

"Lee el formato de expression(): \"((x1 + -0.25) * sqrt(x2))\"."
function parse_expression(text::AbstractString)::AnyNode
    p = _Parser(String(text), 1)
    tree = _parse_node!(p)
    _skip!(p)
    p.pos <= ncodeunits(p.s) && _fail(p, "texto sobrante")
    return tree
end

# ---------------------------------------------------------------------------
# Dataset (por filas: la evaluacion es fila por fila)
# ---------------------------------------------------------------------------
const DATASET_COUNTER = Ref(0)

# IMPORTANTE: un Dataset NO debe modificarse despues de crearse. El cache de los nodos
# se identifica con `id`; si se cambian valores de `rows` o `y`, los nodos con cache
# devolverian resultados del dataset anterior. Para datos distintos, crea otro Dataset
# (recibe un id nuevo). El constructor copia los datos, asi que modificar los vectores
# originales del llamador no afecta al Dataset. El programa lee el CSV una vez y no lo
# modifica. Con cache_min_size = NO_CACHE no se lee ni se escribe ningun cache.
struct Dataset
    rows::Vector{Vector{Float64}}
    y::Vector{Float64}
    id::Int  # identifica el dataset en el cache de los nodos
    function Dataset(rows::AbstractVector, y::AbstractVector)
        (isempty(rows) || length(rows) != length(y)) &&
            throw(ArgumentError("Dataset vacio o inconsistente"))
        width = length(rows[1])
        all(r -> length(r) == width, rows) ||
            throw(ArgumentError("Dimensiones inconsistentes del dataset"))
        DATASET_COUNTER[] += 1
        return new([Vector{Float64}(r) for r in rows], Vector{Float64}(y), DATASET_COUNTER[])
    end
end

nvariables(d::Dataset) = length(d.rows[1])
nrows(d::Dataset) = length(d.y)
dataset_from_rows(rows::AbstractVector, y::AbstractVector) = Dataset(rows, y)

"Mismo formato que B: x1,...,xn,y; encabezado opcional en la primera linea."
function read_csv(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    y = Float64[]
    columns = 0
    # readlines lee y CIERRA el archivo antes de procesarlo: si una linea es invalida y se
    # lanza un error, el archivo no queda abierto (en Windows impediria borrarlo).
    for (line_number, line) in enumerate(readlines(path))
        isempty(strip(line)) && continue
        fields = [strip(f) for f in split(line, ',')]
        parsed = [tryparse(Float64, f) for f in fields]
        # Encabezado solo si NINGUN campo es numerico (p. ej. x1,x2,y), igual que A y B.
        line_number == 1 && all(isnothing, parsed) && continue
        values = Float64[]
        why = ""
        for (field, v) in zip(fields, parsed)
            if v === nothing || !isfinite(v)
                why = isempty(field) ? "campo vacio" : "numero invalido: $field"
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
        isempty(why) || error("CSV linea $line_number: $why")
        push!(y, pop!(values))
        push!(rows, values)
    end
    isempty(rows) && error("Dataset sin filas numericas")
    return Dataset(rows, y)
end

# ---------------------------------------------------------------------------
# Evaluacion fila por fila
# ---------------------------------------------------------------------------
"Referencia escalar sin cache: evalua el arbol en un vector de entrada."
function eval_row(n::AnyNode, x::AbstractVector{Float64})::Float64
    if n isa ConstNode
        return n.value
    elseif n isa VarNode
        1 <= n.index <= length(x) || throw(ArgumentError("Indice de variable fuera del vector"))
        return @inbounds x[n.index]
    end
    m = n::OpNode
    a = eval_row(m.left, x)
    m.op == OP_SQRT && return protected_sqrt(a)
    return apply(m.op, a, eval_row(m.right::AnyNode, x))
end

const FILLING = OpNode[]  # nodos cuyo cache se esta llenando en la evaluacion actual

# Valor del nodo en la fila i. Usa y llena el cache de los nodos con tamano >= S.
function _eval(n::AnyNode, x::Vector{Float64}, i::Int, id::Int, rows::Int, S::Int)::Float64
    if n isa ConstNode
        return n.value
    elseif n isa VarNode
        n.index <= length(x) || throw(ArgumentError("Indice de variable fuera del vector"))
        return @inbounds x[n.index]
    end
    m = n::OpNode
    m.isconst && return m.cvalue  # subarbol sin variables: valor precalculado
    cid = m.cache_id
    # Con NO_CACHE no se usa ningun cache, ni siquiera uno que ya existiera.
    (cid == id && S != NO_CACHE) && return @inbounds m.cache[i]
    a = _eval(m.left, x, i, id, rows, S)
    v = m.op == OP_SQRT ? protected_sqrt(a) :
                          apply(m.op, a, _eval(m.right::AnyNode, x, i, id, rows, S))
    if m.size >= S
        if cid == -id                # se esta llenando en esta evaluacion
            @inbounds m.cache[i] = v
        elseif i == 1                # empieza a llenarse en la primera fila
            m.cache = Vector{Float64}(undef, rows)
            @inbounds m.cache[1] = v
            m.cache_id = -id
            push!(FILLING, m)
        end
    end
    return v
end

function _abandon_filling!()
    for m in FILLING
        m.cache = EMPTY_CACHE
        m.cache_id = 0
    end
    empty!(FILLING)
end

"""
MSE evaluando fila por fila (como el pseudocodigo del manual), con cache de
resultados en los nodos de tamano >= cache_min_size (NO_CACHE lo desactiva).
Mismas operaciones y mismo orden de suma que A y B: el MSE es identico bit a bit.
"""
function fitness(tree::AnyNode, data::Dataset, cache_min_size::Int=DEFAULT_CACHE_MIN_SIZE)::Float64
    rows = nrows(data)
    id = data.id
    empty!(FILLING)
    total = 0.0
    try
        @inbounds for i in 1:rows
            difference = data.y[i] - _eval(tree, data.rows[i], i, id, rows, cache_min_size)
            squared = difference * difference
            if !isfinite(squared)
                _abandon_filling!()
                return Inf
            end
            total += squared
            if !isfinite(total)
                _abandon_filling!()
                return Inf
            end
        end
    catch
        _abandon_filling!()
        rethrow()
    end
    for m in FILLING
        m.cache_id = id  # cache completo para todas las filas
    end
    empty!(FILLING)
    return total / rows
end

"""
Tamano minimo de subarbol que guarda cache, para que el cache de la poblacion quepa
aproximadamente en `cache_mb` megabytes (estimacion conservadora: poblacion +
descendientes, arboles grow_full de profundidad max_depth).
"""
function auto_cache_min_size(population::Integer, max_depth::Integer, rows::Integer,
                             cache_mb::Integer)
    cache_mb <= 0 && return NO_CACHE
    budget = cache_mb * 2.0^20
    expected_nodes = sum(1.8^k for k in 0:max_depth)
    S = 8
    while S < 2^20 && 2.0 * population * expected_nodes / S * rows * 8.0 > budget
        S *= 2
    end
    return S
end

# ---------------------------------------------------------------------------
# Generacion (mismas reglas y mismo consumo del generador que A)
# ---------------------------------------------------------------------------
@inline random_constant(rng::AbstractRNG) = 2.0 * rand(rng) - 1.0  # igual que A

function _grow(remaining::Int, nvars::Int, rng::AbstractRNG, full::Bool)::AnyNode
    if remaining == 0 || (!full && rand(rng, 0:3) == 0)
        if nvars == 0 || rand(rng, 0:1) == 0
            return ConstNode(random_constant(rng))
        end
        return variable(rand(rng, 1:nvars))
    end
    op = RANDOM_OPS[rand(rng, 1:5)]
    left = _grow(remaining - 1, nvars, rng, full)
    op == OP_SQRT && return OpNode(op, left)
    right = _grow(remaining - 1, nvars, rng, full)
    return OpNode(op, left, right)
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
# Navegacion y copia de camino
# ---------------------------------------------------------------------------
"k-esimo nodo en PREORDEN (1 = raiz; luego subarbol izquierdo y derecho, como A) y su profundidad."
function _node_at(root::AnyNode, k::Int)
    n::AnyNode = root
    depth = 0
    while k > 1
        m = n::OpNode
        k -= 1
        left_size = nsize(m.left)
        if k <= left_size
            n = m.left
        else
            k -= left_size
            n = m.right::AnyNode
        end
        depth += 1
    end
    return n, depth
end

"Arbol igual a `n` con el k-esimo nodo (preorden) reemplazado por `sub`. Solo crea el camino."
function _replace(n::AnyNode, k::Int, sub::AnyNode)::AnyNode
    k == 1 && return sub
    m = n::OpNode
    k -= 1
    left_size = nsize(m.left)
    if k <= left_size
        return OpNode(m.op, _replace(m.left, k, sub), m.right)
    end
    return OpNode(m.op, m.left, _replace(m.right::AnyNode, k - left_size, sub))
end

# ---------------------------------------------------------------------------
# Operadores geneticos (mismo consumo del generador que A)
# ---------------------------------------------------------------------------
"Intercambia un subarbol de cada padre. Los padres no cambian (son inmutables)."
function crossover(a::AnyNode, b::AnyNode, rng::AbstractRNG)
    i = rand(rng, 1:nsize(a))
    j = rand(rng, 1:nsize(b))
    sub_a, _ = _node_at(a, i)
    sub_b, _ = _node_at(b, j)
    return _replace(a, i, sub_b), _replace(b, j, sub_a)
end

"Reemplaza un subarbol por uno nuevo de grow_random."
function mutation(parent::AnyNode, no_variables::Integer, rng::AbstractRNG,
                  new_subtree_max_depth::Integer=3)
    _check_depth(new_subtree_max_depth)
    replacement = grow_random(rand(rng, 0:Int(new_subtree_max_depth)), no_variables, rng)
    k = rand(rng, 1:nsize(parent))
    return _replace(parent, k, replacement)
end

"""
Crossover cuyos hijos no superan max_depth: el intercambio i <-> j es valido si
prof_A(i) + altura_B(j) <= max y prof_B(j) + altura_A(i) <= max (misma regla que A
y B: 128 intentos y, si ninguno sirve, intercambio de raices).
"""
function depth_protected_crossover(a::AnyNode, b::AnyNode, max_depth::Integer, rng::AbstractRNG)
    _check_depth(max_depth)
    (nheight(a) > max_depth || nheight(b) > max_depth) &&
        throw(ArgumentError("Un padre supera max_depth"))
    for _ in 1:PROTECTED_ATTEMPTS
        i = rand(rng, 1:nsize(a))
        j = rand(rng, 1:nsize(b))
        sub_a, depth_a = _node_at(a, i)
        sub_b, depth_b = _node_at(b, j)
        if depth_a + nheight(sub_b) <= max_depth && depth_b + nheight(sub_a) <= max_depth
            return _replace(a, i, sub_b), _replace(b, j, sub_a)
        end
    end
    return b, a
end

"Mutacion cuyo subarbol nuevo cabe en el presupuesto de profundidad restante."
function depth_protected_mutation(parent::AnyNode, no_variables::Integer, max_depth::Integer,
                                  rng::AbstractRNG)
    _check_depth(max_depth)
    nheight(parent) > max_depth && throw(ArgumentError("El padre supera max_depth"))
    k = rand(rng, 1:nsize(parent))
    _, depth = _node_at(parent, k)
    budget = Int(max_depth) - depth
    replacement = grow_random(rand(rng, 0:budget), no_variables, rng)
    return _replace(parent, k, replacement)
end

# ---------------------------------------------------------------------------
# GP completo (mismo algoritmo que A y B)
# ---------------------------------------------------------------------------
Base.@kwdef struct GPOptions
    population_size::Int = 40
    max_depth::Int = 5
    generations::Int = 5
    seed::Int = 42
    cache_mb::Int = 1024       # presupuesto aproximado del cache de resultados
    cache_min_size::Int = 0    # 0 = automatico segun cache_mb; NO_CACHE = sin cache
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

function full_gp(data::Dataset, options::GPOptions)
    N = options.population_size
    nvariables(data) > 0 || throw(ArgumentError("Se necesita un dataset con variables"))
    (N >= 4 && N % 4 == 0) ||
        throw(ArgumentError("population_size debe ser multiplo de 4 y >= 4"))
    _check_depth(options.max_depth)
    options.generations >= 0 || throw(ArgumentError("generations debe ser >= 0"))
    rng = Xoshiro(options.seed)
    nvars, max_depth = nvariables(data), options.max_depth
    S = options.cache_min_size > 0 ? options.cache_min_size :
        auto_cache_min_size(N, max_depth, nrows(data), options.cache_mb)

    population = Individual[]
    sizehint!(population, N)
    for _ in 1:N
        tree = grow_full(max_depth, nvars, rng)
        push!(population, Individual(fitness(tree, data, S), tree))
    end

    history = Float64[]
    pool = Vector{AnyNode}(undef, N)
    offspring = Individual[]
    sizehint!(offspring, N)
    for generation in 0:options.generations
        push!(history, minimum(_score, population))
        generation == options.generations && break

        for i in 1:N  # torneo binario repetido
            a = population[rand(rng, 1:N)]
            b = population[rand(rng, 1:N)]
            pool[i] = a.score <= b.score ? a.tree : b.tree
        end

        empty!(offspring)
        for _ in 1:(N ÷ 4)  # N/2 hijos por crossover
            p1 = pool[rand(rng, 1:N)]
            p2 = pool[rand(rng, 1:N)]
            c1, c2 = depth_protected_crossover(p1, p2, max_depth, rng)
            push!(offspring, Individual(fitness(c1, data, S), c1))
            push!(offspring, Individual(fitness(c2, data, S), c2))
        end
        for _ in 1:(N ÷ 2)  # N/2 hijos por mutacion
            m = depth_protected_mutation(pool[rand(rng, 1:N)], nvars, max_depth, rng)
            push!(offspring, Individual(fitness(m, data, S), m))
        end

        # Reemplazo 50/50 (orden estable: mismos empates que A).
        sort!(population; by=_score, alg=MergeSort)
        sort!(offspring; by=_score, alg=MergeSort)
        population = vcat(population[1:N÷2], offspring[1:N÷2])
    end
    best = population[argmin(map(_score, population))]
    return GPResult(best.tree, best.score, history)
end

# ---------------------------------------------------------------------------
# Benchmark de crossover (mismo protocolo que A y B)
# ---------------------------------------------------------------------------
struct CrossoverTiming
    depth::Int
    trees::Int
    resident_trees::Int
    crossovers::Int
    seconds::Float64
    mean_seconds_per_crossover::Float64
    mean_nodes_per_tree::Float64
end

"""
Los `trees` padres (grow_full) estan en memoria AL MISMO TIEMPO (batch_pairs = 0;
con poca RAM, batch_pairs = 2500 deja 5000 a la vez). Se cronometra SOLO el
crossover; los hijos se sueltan cada 16 cruces y se recolectan fuera del cronometro.
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
    sizehint!(children, CHILDREN_CHUNK)
    offset = 0
    while offset < pairs
        count = min(per_batch, pairs - offset)
        empty!(parents)
        for _ in 1:count
            p1 = grow_full(depth, no_variables, rng)
            p2 = grow_full(depth, no_variables, rng)
            total_nodes += nsize(p1) + nsize(p2)
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
            GC.gc(false)  # recoleccion de los hijos fuera del cronometro
        end
        offset += count
    end
    return CrossoverTiming(depth, trees, 2 * per_batch, pairs, measured, measured / pairs,
                           total_nodes / trees)
end

# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
function _throws(f)
    try
        f()
        return false
    catch err
        return err isa ArgumentError
    end
end

function _throws_any(f)
    try
        f()
        return false
    catch
        return true
    end
end

# Nodos internos alcanzables (sin repetir) y comprobacion de metadatos.
function _internal_nodes!(n::AnyNode, out::IdDict{Any, Nothing})
    n isa OpNode || return out
    haskey(out, n) && return out
    out[n] = nothing
    _internal_nodes!(n.left, out)
    n.right === nothing || _internal_nodes!(n.right::AnyNode, out)
    return out
end

function _metadata_ok(n::AnyNode)
    n isa OpNode || return true
    s = 1 + nsize(n.left) + (n.right === nothing ? 0 : nsize(n.right::AnyNode))
    h = 1 + max(nheight(n.left), n.right === nothing ? 0 : nheight(n.right::AnyNode))
    return nsize(n) == s && nheight(n) == h && _metadata_ok(n.left) &&
           (n.right === nothing || _metadata_ok(n.right::AnyNode))
end

_count_nodes(n::Union{VarNode, ConstNode}) = 1
_count_nodes(n::OpNode) = 1 + _count_nodes(n.left) +
                            (n.right === nothing ? 0 : _count_nodes(n.right::AnyNode))

_leaf_depths(n::Union{VarNode, ConstNode}, d, out) = push!(out, d)
function _leaf_depths(n::OpNode, d, out)
    _leaf_depths(n.left, d + 1, out)
    n.right === nothing || _leaf_depths(n.right::AnyNode, d + 1, out)
    return out
end

function _reference_mse(tree, xs, ys)
    total = 0.0
    for r in eachindex(ys)
        d = ys[r] - eval_row(tree, xs[r])
        total += d * d
        isfinite(total) || return Inf
    end
    return total / length(ys)
end

function run_tests(; verbose::Bool=false)
    passed = Ref(0)
    function check(condition::Bool, description::AbstractString)
        condition || error("Fallo en prueba: $description")
        passed[] += 1
        verbose && println("  ok  ", description)
    end
    rng = Xoshiro(42)

    # --- Parte 1: representacion, expression, eval, MSE ---------------------
    formula = OpNode(OP_ADD, variable(1), OpNode(OP_MUL, variable(2), ConstNode(0.5)))
    check(expression(formula) == "(x1 + (x2 * 0.5))", "expression")
    check(eval_row(formula, [2.0, 3.0]) == 3.5, "eval_row (x1 + x2*0.5) en [2, 3] = 3.5")
    check(node_count(formula) == 5 && tree_depth(formula) == 2, "tamano y profundidad")
    check(variable(2) === variable(2), "las variables se comparten")
    check(_throws(() -> OpNode(OP_ADD, ConstNode(1.0))), "binario sin hijo derecho es rechazado")
    check(_throws(() -> OpNode(0x07, ConstNode(1.0), ConstNode(2.0))), "codigo de operacion invalido")
    check(_throws(() -> VarNode(0)), "indice de variable 0 rechazado")

    exact = Dataset([[2.0, 3.0], [1.0, 2.0]], [3.5, 2.0])
    manual = Dataset([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.5, 1.0]], [2.5, 1.0, 3.5])
    check(fitness(formula, exact) == 0.0, "MSE cero")
    expected = ((2.5 - 2.0)^2 + (1.0 - 0.5)^2 + (3.5 - 2.25)^2) / 3
    check(isapprox(fitness(formula, manual), expected), "MSE de la tabla del manual")
    check(eval_row(parse_expression("(1 / 0)"), Float64[]) == 1e12, "division protegida")
    check(eval_row(parse_expression("sqrt(-5)"), Float64[]) == 0.0, "sqrt protegida")
    check(fitness(parse_expression("((x1 * 1e300) * 1e300)"), exact) == Inf,
          "MSE no finito = +Inf")
    check(_throws(() -> fitness(parse_expression("(x3 + 1)"), exact, 1)), "variable fuera de rango")
    check(expression(parse_expression("sqrt((x3 / -0.25))")) == "sqrt((x3 / -0.25))",
          "parse con constante negativa")
    check(_throws(() -> Dataset([[1.0, 2.0], [1.0]], [1.0, 2.0])), "Dataset inconsistente")
    for (text, header_ok) in (("x1,x2,y\n1,2,3\n", true), ("1,abc,3\n4,5,6\n", false),
                              ("x1,x2,y\n1,2,\n", false))
        mktemp() do path, io
            write(io, text)
            close(io)
            check(header_ok ? nrows(read_csv(path)) == 1 : _throws_any(() -> read_csv(path)),
                  "lector CSV: $(repr(text))")
        end
    end

    # --- Parte 2: generacion -----------------------------------------------
    for depth in 0:8, nvars in (1, 2, 5)
        full = grow_full(depth, nvars, rng)
        rnd = grow_random(depth, nvars, rng)
        check(tree_depth(full) == depth && _metadata_ok(full), "grow_full profundidad exacta")
        check(tree_depth(rnd) <= depth && _metadata_ok(rnd), "grow_random profundidad maxima")
        check(node_count(full) == _count_nodes(full), "tamano guardado = tamano real")
        back = parse_expression(expression(full, 17))
        check(expression(back, 17) == expression(full, 17), "expression(17) -> parse sin perdida")
    end
    check(unique(_leaf_depths(grow_full(6, 3, rng), 0, Int[])) == [6],
          "grow_full: TODAS las hojas a profundidad exacta")

    # --- Cache: mismo MSE bit a bit con y sin cache, en frio y en caliente ---
    r2 = Xoshiro(7)
    xs = [[(i % 17 == 0) ? 0.0 : (6.0 * rand(r2) - 3.0) * 10.0^rand(r2, (-8, 0, 0, 3))
           for _ in 1:3] for i in 1:150]
    ys = [6.0 * rand(r2) - 3.0 for _ in 1:150]
    hard = Dataset(xs, ys)
    other = Dataset(xs[1:40], ys[41:80])
    mismatches = 0
    trees = [grow_random(1 + t % 9, 3, rng) for t in 1:200]
    for t in trees
        ref = _reference_mse(t, xs, ys)
        ref_other = _reference_mse(t, xs[1:40], ys[41:80])
        for S in (NO_CACHE, 1, 4, 16)
            fitness(t, hard, S) === ref || (mismatches += 1)   # frio o caliente
            fitness(t, hard, S) === ref || (mismatches += 1)   # caliente
            fitness(t, other, S) === ref_other || (mismatches += 1)  # otro dataset
        end
    end
    check(mismatches == 0, "MSE con cache == sin cache == referencia, bit a bit")
    # Hijos evaluados con los padres ya en cache (se calienta el cache primero).
    foreach(t -> fitness(t, hard, 1), trees)
    mismatches = 0
    for _ in 1:200
        a, b = trees[rand(rng, 1:200)], trees[rand(rng, 1:200)]
        c, d = crossover(a, b, rng)
        m = mutation(a, 3, rng)
        for t in (c, d, m)
            fitness(t, hard, 1) === _reference_mse(t, xs, ys) || (mismatches += 1)
        end
    end
    check(mismatches == 0, "hijos con subarboles en cache: MSE identico")

    # --- Plegado de constantes ------------------------------------------------
    folded = parse_expression("((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))")
    check(folded.left.isconst && folded.left.cvalue === 0.6 && !folded.isconst &&
          !folded.right.isconst && folded.right.right.isconst && folded.right.right.cvalue === 1e12,
          "plegado de constantes: valores y marcas")
    check(expression(folded) == "((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))" && node_count(folded) == 10,
          "el plegado no cambia expression ni tamano")
    function _check_folding(n::AnyNode)
        n isa OpNode || return true
        n.isconst && !(n.cvalue === eval_row(n, Float64[])) && return false
        n.isconst == !occursin('x', expression(n)) || return false
        return _check_folding(n.left) && (n.right === nothing || _check_folding(n.right::AnyNode))
    end
    folding_ok = true
    for t in 1:300
        tree = grow_random(1 + t % 10, t % 4, rng)  # 0 variables = arbol solo de constantes
        folding_ok &= _check_folding(tree)
    end
    check(folding_ok, "cada subarbol constante vale lo mismo, bit a bit, que recorrerlo")
    big = parse_expression("((1e300 * 1e300) - (1e300 * 1e300))")  # Inf - Inf = NaN
    check(big.isconst && isnan(big.cvalue) && fitness(big, exact) == Inf,
          "constante no finita: MSE = +Inf como sin plegado")

    # --- NO_CACHE ignora caches previos ----------------------------------------
    mutable_data = Dataset([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]], [1.0, 2.0, 3.0])
    cached_tree = parse_expression("((x1 + x2) * (x1 - (x2 / 0.5)))")
    fitness(cached_tree, mutable_data, 1)  # llena el cache de sus nodos
    mutable_data.rows[2][1] = 7.0         # prohibido en uso normal: solo para la prueba
    check(fitness(cached_tree, mutable_data, NO_CACHE) ===
          _reference_mse(cached_tree, mutable_data.rows, mutable_data.y),
          "NO_CACHE no usa caches de evaluaciones anteriores")

    # --- Crossover y mutacion: copia de camino --------------------------------
    a, b = grow_full(4, 2, rng), grow_full(3, 2, rng)
    old_a, old_b = expression(a, 17), expression(b, 17)
    parents_nodes = _internal_nodes!(b, _internal_nodes!(a, IdDict{Any, Nothing}()))
    for _ in 1:200
        c, d = crossover(a, b, rng)
        check(node_count(c) + node_count(d) == node_count(a) + node_count(b),
              "crossover conserva el total de nodos")
        check(_metadata_ok(c) && _metadata_ok(d), "metadatos correctos en los hijos")
        new_c = count(k -> !haskey(parents_nodes, k), keys(_internal_nodes!(c, IdDict{Any, Nothing}())))
        check(new_c <= tree_depth(a) + 1, "copia de camino: solo se crean los nodos del camino")
        m = mutation(a, 2, rng)
        check(_metadata_ok(m), "mutacion con metadatos correctos")
        x, y = depth_protected_crossover(a, b, 4, rng)
        check(tree_depth(x) <= 4 && tree_depth(y) <= 4, "crossover protegido")
        check(tree_depth(depth_protected_mutation(a, 2, 4, rng)) <= 4, "mutacion protegida")
    end
    check(expression(a, 17) == old_a && expression(b, 17) == old_b,
          "los operadores no alteran a los padres")
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
    check(expression(deep_copy_tree(deep), 17) == expression(deep, 17), "copia profunda de prueba")

    # --- GP completo --------------------------------------------------------
    check(_throws(() -> full_gp(exact, GPOptions(generations=-1))), "generaciones negativas rechazadas")
    run = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3))
    h = run.best_fitness_per_generation
    check(length(h) == 4, "historial de fitness")
    check(all(h[i] <= h[i-1] for i in 2:length(h)), "elitismo: el mejor MSE no aumenta")
    o1 = GPOptions(population_size=40, max_depth=5, generations=5, cache_min_size=1)
    o2 = GPOptions(population_size=40, max_depth=5, generations=5, cache_min_size=NO_CACHE)
    r1, r2b = full_gp(exact, o1), full_gp(exact, o2)
    check(r1.best_fitness_per_generation == r2b.best_fitness_per_generation &&
          expression(r1.best, 17) == expression(r2b.best, 17), "el cache no altera el GP")
    check(auto_cache_min_size(5000, 12, 49, 1024) >= 8 &&
          auto_cache_min_size(5000, 12, 49, 0) == NO_CACHE, "tamano automatico del cache")

    t = benchmark_crossover(40, 4, 2; seed=1)
    check(t.crossovers == 20 && t.resident_trees == 40 && t.seconds > 0, "benchmark de crossover")
    return passed[]
end

end # module
