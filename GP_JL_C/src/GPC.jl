"""
Programacion Genetica (GP) - Implementacion C (alternativa, no convencional), Julia 1.9+.

Idea: TODA la poblacion vive en UN SOLO GRAFO guardado en arreglos, con subarboles
compartidos y cache de resultados. Ambas cosas las aprobo el profesor ("compartir
subarboles en vez de copiarlos" y "toda la poblacion en un solo grafo con cache de
resultados"). La evaluacion por lotes NO esta permitida: se evalua fila por fila, como
en el manual. No hay paralelismo: un solo hilo.

1. Un grafo en arreglos. Cada nodo es un registro de 24 bytes (`NodeRec`) dentro de un
   `Vector{NodeRec}`; sus hijos son INDICES de 32 bits, no punteros a objetos. Un arbol
   es (grafo, indice de su raiz). No hay un objeto de Julia por nodo, asi que el
   recolector de basura no tiene millones de objetos que revisar (antes, en profundidad
   12, la mitad del tiempo de la poblacion inicial era recoleccion de basura).
   Los nodos nunca se modifican despues de creados: compartirlos es seguro.

2. Arboles en preorden. Al generar un arbol, el registro del padre se reserva antes que
   los de sus hijos, asi que un arbol nuevo queda en memoria en el orden en que se
   recorre (raiz, izquierdo, derecho): el hijo izquierdo esta justo despues del padre.

3. Copia de camino. Un hijo de crossover o de mutacion crea solo los nodos del camino de
   la raiz al punto de corte (a lo mas uno por nivel); todo lo demas se comparte.

4. Evaluacion fila por fila con cache. Los nodos con tamano >= S guardan su resultado en
   cada fila en un bloque grande de memoria. Un hijo comparte casi todo con sus padres
   (ya evaluados): solo se calculan los nodos nuevos del camino y los subarboles pequenos
   que cuelgan de el. Debajo de S se usa un evaluador sin cache y sin saltos para elegir
   el operador (se calculan + - * / y se elige con `ifelse`).

5. Plegado de constantes. Un nodo cuyo subarbol no tiene variables guarda su valor al
   crearse (mismas operaciones protegidas): la evaluacion lo lee sin recorrerlo.

6. Corte temprano exacto. Solo sobrevive el mejor 50% de los hijos; un hijo cuyo MSE
   parcial ya supera al (N/2)-esimo mejor deja de evaluarse (nunca sobreviviria).

Todo esto da EXACTAMENTE el mismo resultado: mismas operaciones IEEE en el mismo orden.
C consume el generador aleatorio en el mismo orden que A y elige el punto de corte
uniforme en preorden como A, asi que con la misma semilla produce los mismos arboles,
los mismos hijos y la misma corrida del GP (ver tools/cross_check.jl).
"""
module GPC

using Random
using Printf

export Tree, Graph, Dataset,
       expression, parse_expression, format_number, node_count, tree_depth, clone,
       eval_row, fitness, read_csv, dataset_from_rows, nvariables, nrows,
       grow_random, grow_full, crossover, mutation,
       depth_protected_crossover, depth_protected_mutation, auto_cache_min_size,
       GPOptions, GPResult, full_gp, CrossoverTiming, benchmark_crossover, run_tests

const EPS = 1e-12
const PROTECTED_ATTEMPTS = 128   # igual que A y B
const CHILDREN_CHUNK = 16        # benchmark: hijos liberados cada 16 cruces
const DEFAULT_CACHE_MIN_SIZE = 16
const NO_CACHE = typemax(Int)    # cache_min_size que desactiva el cache
const BLOCK_VALUES = 1 << 20     # valores Float64 por bloque de cache (8 MB)

# Tipos de nodo. Los operadores tienen el mismo orden que A y B al sortearlos.
const K_CONST = 0x00
const K_VAR = 0x01
const OP_ADD = 0x02
const OP_SUB = 0x03
const OP_MUL = 0x04
const OP_DIV = 0x05
const OP_SQRT = 0x06
const RANDOM_OPS = (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT)
const BINARY_BY_CHAR = Dict('+' => OP_ADD, '-' => OP_SUB, '*' => OP_MUL, '/' => OP_DIV)
const FOLDED = 0x01              # bandera: subarbol sin variables, valor ya calculado
# Hijo de un operador protegido que es copia exacta de un padre (mismo arbol => mismo MSE).
const COPY_NONE = 0x00           # hijo(s) nuevo(s)
const COPY_PARENTS = 0x01        # hijo1 = padre a, hijo2 = padre b (subarboles iguales)
const COPY_SWAPPED = 0x02        # hijo1 = padre b, hijo2 = padre a (intercambio de raices)

_symbol(op::UInt8) = op == OP_ADD ? "+" : op == OP_SUB ? "-" : op == OP_MUL ? "*" :
                     op == OP_DIV ? "/" : "sqrt"

@inline protected_div(a::Float64, b::Float64) = a / (abs(b) < EPS ? EPS : b)
@inline protected_sqrt(a::Float64) = sqrt(a < 0.0 ? 0.0 : a)

"Operador binario con saltos (referencia, plegado y eval_row)."
@inline function apply(op::UInt8, a::Float64, b::Float64)
    op == OP_ADD && return a + b
    op == OP_SUB && return a - b
    op == OP_MUL && return a * b
    return protected_div(a, b)
end

"""
Operador binario SIN saltos: se calculan las cuatro operaciones y se elige una con
`ifelse` (instruccion de seleccion, no un salto). En un arbol grande el procesador no
puede adivinar la secuencia de operadores y cada fallo cuesta ~15-20 ciclos; calcular
las cuatro cuesta menos. Mismas operaciones IEEE: resultado identico a `apply`.
"""
@inline function apply_select(op::UInt8, a::Float64, b::Float64)
    d = a / ifelse(abs(b) < EPS, EPS, b)
    r = ifelse(op == OP_ADD, a + b, a - b)
    r = ifelse(op == OP_MUL, a * b, r)
    return ifelse(op == OP_DIV, d, r)
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
# Representacion: un grafo en arreglos
# ---------------------------------------------------------------------------
"""
Registro de un nodo (24 bytes, sin punteros: `isbits`).

- `left`, `right`: indices de los hijos (0 = no hay). En una variable, `left` es su
  indice (1 -> x1).
- `size`: nodos del subarbol (incluido este). `height`: altura (hoja = 0).
- `kind`: K_CONST, K_VAR u operador. `flags`: FOLDED si el subarbol no tiene variables
  y su valor ya esta calculado.
- `value`: la constante; el valor plegado; o, en los demas nodos internos, el mayor
  indice de variable del subarbol (para validar el dataset una vez por arbol, en O(1)).
"""
struct NodeRec
    left::Int32
    right::Int32
    size::Int32
    height::Int16
    kind::UInt8
    flags::UInt8
    value::Float64
end

"""
Grafo que contiene muchos arboles que comparten nodos. Los nodos solo se agregan al
final y nunca cambian. El cache de resultados (opcional) vive en arreglos paralelos y en
bloques grandes de Float64, no en un objeto por nodo.
"""
mutable struct Graph
    nodes::Vector{NodeRec}
    vars::Vector{Int32}             # nodo compartido de cada variable (0 = aun no existe)
    fold::Bool                      # plegado de constantes al crear nodos
    track::Bool                     # hay arreglos de cache por nodo
    cache_id::Vector{Int32}         # dataset del cache: 0 = ninguno, -id = llenandose
    cache_off::Vector{Int64}        # (bloque << 32) | inicio dentro del bloque
    blocks::Vector{Vector{Float64}}
    block_used::Int
    cache_values::Int               # valores de cache reservados (para medir memoria)
    filling::Vector{Int32}          # nodos cuyo cache se llena en la evaluacion actual
end

Graph(; fold::Bool=true) = Graph(NodeRec[], Int32[], fold, false, Int32[], Int64[],
                                 Vector{Float64}[], 0, 0, Int32[])

"Grafo por omision (arboles creados fuera de full_gp y benchmark_crossover)."
const DEFAULT_GRAPH = Graph()

"Un arbol: su grafo y el indice de su raiz. Copiarlo no copia nodos."
struct Tree
    g::Graph
    i::Int32
end

@inline _isdirect(r::NodeRec) = r.kind <= K_VAR || (r.flags & FOLDED) != 0
@inline _isconst(r::NodeRec) = r.kind == K_CONST || (r.flags & FOLDED) != 0

"Mayor indice de variable del subarbol (0 = sin variables), en O(1)."
@inline function _maxvar(r::NodeRec)::Int
    r.kind == K_VAR && return Int(r.left)
    _isconst(r) && return 0
    return Int(r.value)
end

function _track!(g::Graph)
    if !g.track
        n = length(g.nodes)
        g.cache_id = zeros(Int32, n)
        g.cache_off = zeros(Int64, n)
        g.track = true
    end
    return g
end

@inline function _push!(g::Graph, r::NodeRec)::Int32
    push!(g.nodes, r)
    if g.track
        push!(g.cache_id, Int32(0))
        push!(g.cache_off, Int64(0))
    end
    return Int32(length(g.nodes))
end

"Registro de un operador a partir de sus hijos: tamano, altura, plegado y variables."
@inline function _op_record(g::Graph, op::UInt8, l::Int32, r::Int32)::NodeRec
    nodes = g.nodes
    L = @inbounds nodes[l]
    if r == 0
        folded = g.fold && _isconst(L)
        value = folded ? protected_sqrt(L.value) : Float64(_maxvar(L))
        return NodeRec(l, Int32(0), L.size + Int32(1), L.height + Int16(1), op,
                       folded ? FOLDED : 0x00, value)
    end
    R = @inbounds nodes[r]
    folded = g.fold && _isconst(L) && _isconst(R)
    value = folded ? apply(op, L.value, R.value) : Float64(max(_maxvar(L), _maxvar(R)))
    return NodeRec(l, r, L.size + R.size + Int32(1), max(L.height, R.height) + Int16(1), op,
                   folded ? FOLDED : 0x00, value)
end

const_node(g::Graph, v::Float64) = _push!(g, NodeRec(Int32(0), Int32(0), Int32(1), Int16(0), K_CONST, 0x00, v))

"Nodo de la variable x`index`: hay una sola instancia por grafo, compartida."
function var_node(g::Graph, index::Integer)
    index >= 1 || throw(ArgumentError("las variables empiezan en x1 (indice >= 1)"))
    while length(g.vars) < index
        push!(g.vars, Int32(0))
    end
    n = @inbounds g.vars[index]
    if n == 0
        n = _push!(g, NodeRec(Int32(index), Int32(0), Int32(1), Int16(0), K_VAR, 0x00, 0.0))
        g.vars[index] = n
    end
    return n
end

"Nodo operador validado (para el lector de texto y las pruebas)."
function op_node(g::Graph, op::UInt8, l::Integer, r::Integer=0)
    (OP_ADD <= op <= OP_SQRT) || throw(ArgumentError("codigo de operacion invalido: $op"))
    (op == OP_SQRT) == (r == 0) ||
        throw(ArgumentError("'$(_symbol(op))' requiere $(op == OP_SQRT ? 1 : 2) hijo(s)"))
    n = length(g.nodes)
    (1 <= l <= n && 0 <= r <= n) || throw(ArgumentError("hijo inexistente en el grafo"))
    return _push!(g, _op_record(g, op, Int32(l), Int32(r)))
end

"Suelta los nodos creados despues de `mark` (por ejemplo, los hijos del benchmark)."
function _truncate!(g::Graph, mark::Int)
    resize!(g.nodes, mark)
    if g.track
        resize!(g.cache_id, mark)
        resize!(g.cache_off, mark)
    end
    for k in eachindex(g.vars)
        g.vars[k] > mark && (g.vars[k] = 0)
    end
    return g
end

@inline _rec(t::Tree) = t.g.nodes[t.i]
"Numero de nodos (O(1): se guarda en cada nodo)."
node_count(t::Tree) = Int(_rec(t).size)
"Profundidad del arbol, hoja sola = 0 (O(1): se guarda en cada nodo)."
tree_depth(t::Tree) = Int(_rec(t).height)

"""
Clonacion. Los nodos nunca cambian, asi que compartirlos es seguro: ningun operador
puede alterar a un padre. Por eso `clone` devuelve el mismo arbol (aprobado por el
profesor). `deep_copy_tree` hace una copia nodo por nodo, solo para pruebas.
"""
clone(t::Tree) = t
deep_copy_tree(t::Tree) = Tree(t.g, _deep_copy!(t.g, t.i))
function _deep_copy!(g::Graph, n::Int32)::Int32
    r = g.nodes[n]
    r.kind == K_CONST && return const_node(g, r.value)
    r.kind == K_VAR && return n
    l = _deep_copy!(g, r.left)
    rr = r.right == 0 ? Int32(0) : _deep_copy!(g, r.right)
    return _push!(g, _op_record(g, r.kind, l, rr))
end

function _same_graph(a::Tree, b::Tree)
    a.g === b.g || throw(ArgumentError("los dos arboles deben estar en el mismo grafo"))
    return a.g
end

function expression(t::Tree, precision::Integer=8)
    io = IOBuffer()
    _expression(io, t.g.nodes, t.i, Int(precision))
    return String(take!(io))
end

function _expression(io::IO, nodes::Vector{NodeRec}, n::Int32, precision::Int)
    r = nodes[n]
    if r.kind == K_CONST
        print(io, format_number(r.value, precision))
    elseif r.kind == K_VAR
        print(io, 'x', r.left)
    elseif r.kind == OP_SQRT
        print(io, "sqrt(")
        _expression(io, nodes, r.left, precision)
        print(io, ')')
    else
        print(io, '(')
        _expression(io, nodes, r.left, precision)
        print(io, ' ', _symbol(r.kind), ' ')
        _expression(io, nodes, r.right, precision)
        print(io, ')')
    end
end

Base.show(io::IO, t::Tree) = print(io, expression(t))

# ---------------------------------------------------------------------------
# Lectura del formato de texto comun
# ---------------------------------------------------------------------------
mutable struct _Parser
    s::String
    pos::Int
    g::Graph
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

function _parse_node!(p::_Parser)::Int32
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
        return op_node(p.g, op, left, right)
    elseif _matches(p, "sqrt(")
        p.pos += 5
        child = _parse_node!(p)
        _expect!(p, ')')
        return op_node(p.g, OP_SQRT, child)
    elseif c == 'x'
        p.pos += 1
        index = tryparse(Int, _token!(p))
        (index === nothing || index < 1) && _fail(p, "variable invalida")
        return var_node(p.g, index)
    end
    value = tryparse(Float64, _token!(p))
    value === nothing && _fail(p, "numero invalido")
    return const_node(p.g, value)
end

"Lee el formato de expression(): \"((x1 + -0.25) * sqrt(x2))\"."
function parse_expression(text::AbstractString; graph::Graph=DEFAULT_GRAPH)::Tree
    p = _Parser(String(text), 1, graph)
    root = _parse_node!(p)
    _skip!(p)
    p.pos <= ncodeunits(p.s) && _fail(p, "texto sobrante")
    return Tree(graph, root)
end

# ---------------------------------------------------------------------------
# Dataset (por filas: la evaluacion es fila por fila)
# ---------------------------------------------------------------------------
const DATASET_COUNTER = Ref(0)

# Un Dataset NO debe modificarse despues de crearse: el cache de los nodos se identifica
# con `id`. El constructor copia los datos.
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
"Referencia escalar sin cache, sin plegado y con saltos: evalua el arbol en un vector."
eval_row(t::Tree, x::AbstractVector{Float64})::Float64 = _eval_row(t.g.nodes, t.i, x)

function _eval_row(nodes::Vector{NodeRec}, n::Int32, x::AbstractVector{Float64})::Float64
    r = nodes[n]
    r.kind == K_CONST && return r.value
    if r.kind == K_VAR
        1 <= r.left <= length(x) || throw(ArgumentError("Indice de variable fuera del vector"))
        return @inbounds x[r.left]
    end
    a = _eval_row(nodes, r.left, x)
    r.kind == OP_SQRT && return protected_sqrt(a)
    return apply(r.kind, a, _eval_row(nodes, r.right, x))
end

const ZERO_ROW = [0.0]  # fila de relleno para datasets sin variables

# Valor de un nodo que se lee directo (constante, variable o plegado), sin saltos: se
# leen las dos posibilidades y se elige una con `ifelse`. x tiene al menos un elemento.
@inline function _direct(r::NodeRec, x::Vector{Float64})::Float64
    isvar = r.kind == K_VAR
    v = @inbounds x[ifelse(isvar, r.left, Int32(1))]
    return ifelse(isvar, v, r.value)
end

# Evaluador sin cache para subarboles pequenos (tamano < S): ninguno de sus nodos guarda
# cache, asi que no hay nada que revisar. `n` es un operador que no se lee directo.
# Se pasa el INDICE del nodo y no su registro: pasar el registro de 24 bytes obliga a
# copiarlo en la pila en cada llamada (medido: 22% mas lento en profundidad 12).
function _eval_small(nodes::Vector{NodeRec}, n::Int32, x::Vector{Float64})::Float64
    r = @inbounds nodes[n]
    L = @inbounds nodes[r.left]
    a = _isdirect(L) ? _direct(L, x) : _eval_small(nodes, r.left, x)
    r.kind == OP_SQRT && return protected_sqrt(a)
    R = @inbounds nodes[r.right]
    b = _isdirect(R) ? _direct(R, x) : _eval_small(nodes, r.right, x)
    return apply_select(r.kind, a, b)
end

@inline function _cache_ref(g::Graph, n::Int32)
    off = @inbounds g.cache_off[n]
    return @inbounds(g.blocks[off >> 32]), Int(off & 0xffffffff)
end

function _new_slot!(g::Graph, rows::Int)::Int64
    if isempty(g.blocks) || g.block_used + rows > length(g.blocks[end])
        push!(g.blocks, Vector{Float64}(undef, max(BLOCK_VALUES, rows)))
        g.block_used = 0
    end
    start = g.block_used
    g.block_used += rows
    g.cache_values += rows
    return (Int64(length(g.blocks)) << 32) | Int64(start)
end

# Valor del nodo `n` (un operador que no se lee directo) en la fila i.
# Usa y llena el cache de los nodos con tamano >= S.
function _eval(g::Graph, n::Int32, x::Vector{Float64}, i::Int, id::Int32,
               rows::Int, S::Int)::Float64
    nodes = g.nodes
    r = @inbounds nodes[n]
    r.size < S && return _eval_small(nodes, n, x)
    cid = @inbounds g.cache_id[n]
    if cid == id  # cache completo para este dataset
        block, start = _cache_ref(g, n)
        return @inbounds block[start + i]
    end
    L = @inbounds nodes[r.left]
    a = _isdirect(L) ? _direct(L, x) : _eval(g, r.left, x, i, id, rows, S)
    if r.kind == OP_SQRT
        v = protected_sqrt(a)
    else
        R = @inbounds nodes[r.right]
        b = _isdirect(R) ? _direct(R, x) : _eval(g, r.right, x, i, id, rows, S)
        v = apply_select(r.kind, a, b)
    end
    if cid == -id                # se esta llenando en esta evaluacion
        block, start = _cache_ref(g, n)
        @inbounds block[start + i] = v
    elseif i == 1                # empieza a llenarse en la primera fila
        @inbounds g.cache_off[n] = _new_slot!(g, rows)
        block, start = _cache_ref(g, n)
        @inbounds block[start + 1] = v
        @inbounds g.cache_id[n] = -id
        push!(g.filling, n)
    end
    return v
end

function _abandon_filling!(g::Graph)
    for n in g.filling
        @inbounds g.cache_id[n] = Int32(0)
    end
    empty!(g.filling)
end

"""
MSE evaluando fila por fila (como el pseudocodigo del manual), con cache de resultados
en los nodos de tamano >= cache_min_size (NO_CACHE lo desactiva). Mismas operaciones y
mismo orden de suma que A y B: el MSE es identico bit a bit.

Con `cut` finito, deja de evaluar en cuanto el MSE parcial (suma/n) supera `cut`: las
filas restantes solo pueden sumar, asi que el MSE final seria mayor; devuelve +Inf.
"""
function fitness(t::Tree, data::Dataset, cache_min_size::Integer=DEFAULT_CACHE_MIN_SIZE;
                 cut::Float64=Inf)::Float64
    g = _track!(t.g)
    root = g.nodes[t.i]
    width = nvariables(data)
    # Indices de variable validados UNA vez por arbol (el mayor esta en la raiz).
    _maxvar(root) <= width || throw(ArgumentError("Indice de variable fuera del vector"))
    _abandon_filling!(g)  # por si una evaluacion anterior se interrumpio
    rows = nrows(data)
    id = Int32(data.id)
    S = Int(min(cache_min_size, typemax(Int32) + 1))
    direct = _isdirect(root)
    total = 0.0
    @inbounds for i in 1:rows
        x = width == 0 ? ZERO_ROW : data.rows[i]
        predicted = direct ? _direct(root, x) : _eval(g, t.i, x, i, id, rows, S)
        difference = data.y[i] - predicted
        squared = difference * difference
        if !isfinite(squared)
            _abandon_filling!(g)
            return Inf
        end
        total += squared
        if !isfinite(total) || total / rows > cut
            _abandon_filling!(g)
            return Inf
        end
    end
    for n in g.filling
        @inbounds g.cache_id[n] = id  # cache completo para todas las filas
    end
    empty!(g.filling)
    return total / rows
end

"""
Tamano minimo de subarbol que guarda cache. Llenar el cache de la poblacion inicial
cuesta (todos sus nodos son nuevos), y un S chico solo ahorra en las generaciones. El
punto mas rapido medido esta en S = 16 (profundidades chicas) o 32 (profundidad >= 11),
y S crece si la estimacion del cache pasa de 4 veces `cache_mb` megabytes. La estimacion
(poblacion + descendientes, arboles grow_full de profundidad max_depth) es muy
conservadora, y la memoria real la acota la compactacion del grafo (cuando el cache pasa
de 2 * cache_mb). Medido con dataset3a (961 filas, 5000 individuos, profundidad 12): con
el limite de 1 vez el presupuesto elegia S = 256 (53.6 s); con 4 veces elige S = 64
(39.1 s, pico de unos 4 GB de RAM), mismo resultado.
"""
function auto_cache_min_size(population::Integer, max_depth::Integer, rows::Integer,
                             cache_mb::Integer)
    cache_mb <= 0 && return NO_CACHE
    budget = cache_mb * 2.0^20
    expected_nodes = sum(1.8^k for k in 0:max_depth)
    S = max_depth >= 11 ? 32 : 16
    while S < 2^20 && 2.0 * population * expected_nodes / S * rows * 8.0 > 4.0 * budget
        S *= 2
    end
    return S
end

# ---------------------------------------------------------------------------
# Generacion (mismas reglas y mismo consumo del generador que A)
# ---------------------------------------------------------------------------
@inline random_constant(rng::AbstractRNG) = 2.0 * rand(rng) - 1.0  # igual que A

# El padre se reserva ANTES que sus hijos: el arbol queda en memoria en preorden. El
# generador se consume igual que en A: operador, hijo izquierdo, hijo derecho.
function _grow!(g::Graph, remaining::Int, nvars::Int, rng::AbstractRNG, full::Bool)::Int32
    if remaining == 0 || (!full && rand(rng, 0:3) == 0)
        if nvars == 0 || rand(rng, 0:1) == 0
            return const_node(g, random_constant(rng))
        end
        return var_node(g, rand(rng, 1:nvars))
    end
    op = RANDOM_OPS[rand(rng, 1:5)]
    me = _push!(g, NodeRec(Int32(0), Int32(0), Int32(0), Int16(0), op, 0x00, 0.0))
    left = _grow!(g, remaining - 1, nvars, rng, full)
    right = op == OP_SQRT ? Int32(0) : _grow!(g, remaining - 1, nvars, rng, full)
    @inbounds g.nodes[me] = _op_record(g, op, left, right)
    return me
end

_check_depth(depth::Integer) = depth >= 0 || throw(ArgumentError("La profundidad debe ser >= 0"))

"Arbol con profundidad de A LO MAS `depth`."
function grow_random(depth::Integer, no_variables::Integer, rng::AbstractRNG;
                     graph::Graph=DEFAULT_GRAPH)::Tree
    _check_depth(depth)
    return Tree(graph, _grow!(graph, Int(depth), Int(no_variables), rng, false))
end

"Arbol cuyas hojas estan TODAS exactamente a profundidad `depth`."
function grow_full(depth::Integer, no_variables::Integer, rng::AbstractRNG;
                   graph::Graph=DEFAULT_GRAPH)::Tree
    _check_depth(depth)
    return Tree(graph, _grow!(graph, Int(depth), Int(no_variables), rng, true))
end

"Nodos esperados de grow_full(depth): sum 1.8^k (operador binario 4/5, sqrt 1/5)."
_expected_full_size(depth::Integer) = sum(1.8^k for k in 0:depth)

# ---------------------------------------------------------------------------
# Navegacion y copia de camino
# ---------------------------------------------------------------------------
"k-esimo nodo en PREORDEN (1 = raiz; luego subarbol izquierdo y derecho, como A) y su profundidad."
function _node_at(nodes::Vector{NodeRec}, root::Int32, k::Int)
    n = root
    depth = 0
    while k > 1
        r = @inbounds nodes[n]
        k -= 1
        left_size = Int(@inbounds(nodes[r.left]).size)
        if k <= left_size
            n = r.left
        else
            k -= left_size
            n = r.right
        end
        depth += 1
    end
    return n, depth
end

"Arbol igual a `n` con el k-esimo nodo (preorden) reemplazado por `sub`. Solo crea el camino."
function _replace!(g::Graph, n::Int32, k::Int, sub::Int32)::Int32
    k == 1 && return sub
    r = @inbounds g.nodes[n]
    k -= 1
    left_size = Int(@inbounds(g.nodes[r.left]).size)
    if k <= left_size
        new_left = _replace!(g, r.left, k, sub)
        return _push!(g, _op_record(g, r.kind, new_left, r.right))
    end
    new_right = _replace!(g, r.right, k - left_size, sub)
    return _push!(g, _op_record(g, r.kind, r.left, new_right))
end

# ---------------------------------------------------------------------------
# Operadores geneticos (mismo consumo del generador que A)
# ---------------------------------------------------------------------------
"Intercambia un subarbol de cada padre. Los padres no cambian (sus nodos son inmutables)."
function crossover(a::Tree, b::Tree, rng::AbstractRNG)
    g = _same_graph(a, b)
    i = rand(rng, 1:node_count(a))
    j = rand(rng, 1:node_count(b))
    sub_a, _ = _node_at(g.nodes, a.i, i)
    sub_b, _ = _node_at(g.nodes, b.i, j)
    return Tree(g, _replace!(g, a.i, i, sub_b)), Tree(g, _replace!(g, b.i, j, sub_a))
end

"Reemplaza un subarbol por uno nuevo de grow_random."
function mutation(parent::Tree, no_variables::Integer, rng::AbstractRNG,
                  new_subtree_max_depth::Integer=3)
    _check_depth(new_subtree_max_depth)
    g = parent.g
    replacement = _grow!(g, rand(rng, 0:Int(new_subtree_max_depth)), Int(no_variables), rng, false)
    k = rand(rng, 1:node_count(parent))
    return Tree(g, _replace!(g, parent.i, k, replacement))
end

"Dos subarboles son iguales: misma forma, mismos operadores, mismas variables y constantes con los mismos bits."
function _same_subtree(nodes::Vector{NodeRec}, x::Int32, y::Int32)::Bool
    x == y && return true
    a = @inbounds nodes[x]
    b = @inbounds nodes[y]
    (a.kind == b.kind && a.size == b.size) || return false
    a.kind == K_CONST && return reinterpret(UInt64, a.value) == reinterpret(UInt64, b.value)
    a.kind == K_VAR && return a.left == b.left
    _same_subtree(nodes, a.left, b.left) || return false
    return a.right == 0 || _same_subtree(nodes, a.right, b.right)
end

# Crossover protegido que ademas, con detect = true, avisa si un hijo es copia exacta de un
# padre (COPY_PARENTS o COPY_SWAPPED) y en ese caso devuelve los padres mismos, sin crear
# nodos. Solo compara: consume el generador igual y los hijos son los mismos arboles.
function _protected_crossover(a::Tree, b::Tree, max_depth::Integer, rng::AbstractRNG,
                              detect::Bool)
    _check_depth(max_depth)
    (tree_depth(a) > max_depth || tree_depth(b) > max_depth) &&
        throw(ArgumentError("Un padre supera max_depth"))
    g = _same_graph(a, b)
    nodes = g.nodes
    size_a, size_b = node_count(a), node_count(b)
    for _ in 1:PROTECTED_ATTEMPTS
        i = rand(rng, 1:size_a)
        j = rand(rng, 1:size_b)
        sub_a, depth_a = _node_at(nodes, a.i, i)
        sub_b, depth_b = _node_at(nodes, b.i, j)
        if depth_a + Int(@inbounds(nodes[sub_b]).height) <= max_depth &&
           depth_b + Int(@inbounds(nodes[sub_a]).height) <= max_depth
            if detect
                (i == 1 && j == 1) && return (b, a, COPY_SWAPPED)
                _same_subtree(nodes, sub_a, sub_b) && return (a, b, COPY_PARENTS)
            end
            return (Tree(g, _replace!(g, a.i, i, sub_b)), Tree(g, _replace!(g, b.i, j, sub_a)), COPY_NONE)
        end
    end
    return (b, a, detect ? COPY_SWAPPED : COPY_NONE)
end

"""
Crossover cuyos hijos no superan max_depth: el intercambio i <-> j es valido si
prof_A(i) + altura_B(j) <= max y prof_B(j) + altura_A(i) <= max (misma regla que A
y B: 128 intentos y, si ninguno sirve, intercambio de raices).
"""
function depth_protected_crossover(a::Tree, b::Tree, max_depth::Integer, rng::AbstractRNG)
    c1, c2, _ = _protected_crossover(a, b, max_depth, rng, false)
    return c1, c2
end

# Mutacion protegida que, con detect = true, avisa si el subarbol nuevo es igual al que
# reemplaza; en ese caso devuelve el padre mismo (hijo identico, mismo MSE).
function _protected_mutation(parent::Tree, no_variables::Integer, max_depth::Integer,
                             rng::AbstractRNG, detect::Bool)
    _check_depth(max_depth)
    tree_depth(parent) > max_depth && throw(ArgumentError("El padre supera max_depth"))
    g = parent.g
    k = rand(rng, 1:node_count(parent))
    old, depth = _node_at(g.nodes, parent.i, k)
    budget = Int(max_depth) - depth
    replacement = _grow!(g, rand(rng, 0:budget), Int(no_variables), rng, false)
    (detect && _same_subtree(g.nodes, old, replacement)) && return (parent, true)
    return (Tree(g, _replace!(g, parent.i, k, replacement)), false)
end

"Mutacion cuyo subarbol nuevo cabe en el presupuesto de profundidad restante."
function depth_protected_mutation(parent::Tree, no_variables::Integer, max_depth::Integer,
                                  rng::AbstractRNG)
    return first(_protected_mutation(parent, no_variables, max_depth, rng, false))
end

# ---------------------------------------------------------------------------
# GP completo (mismo algoritmo que A y B)
# ---------------------------------------------------------------------------
Base.@kwdef struct GPOptions
    population_size::Int = 40
    max_depth::Int = 5
    generations::Int = 5
    seed::Int = 42
    cache_mb::Int = 1024         # presupuesto aproximado del cache de resultados
    cache_min_size::Int = 0      # 0 = automatico segun cache_mb; NO_CACHE = sin cache
    fold_constants::Bool = true  # plegado de constantes (--no-fold lo apaga)
    early_cut::Bool = true       # corte temprano exacto (--no-cut lo apaga)
    order_by_parent::Bool = true # evaluar primero los hijos de mejores padres (--no-order lo apaga)
    skip_duplicates::Bool = true # un hijo identico a su padre recibe su MSE sin evaluarse (--no-dup)
end

struct GPResult
    best::Tree
    best_fitness::Float64
    best_fitness_per_generation::Vector{Float64}
end

struct Individual
    score::Float64
    tree::Int32  # raiz dentro del grafo de la corrida
end

_score(ind::Individual) = ind.score

# Monticulo de maximos con los `cap` mejores MSE de los hijos ya evaluados. h[1] es el
# (cap)-esimo mejor: solo puede bajar conforme llegan hijos.
function _record_best!(h::Vector{Float64}, s::Float64, cap::Int)
    if length(h) < cap
        push!(h, s)
        k = length(h)
        @inbounds while k > 1
            parent = k >> 1
            h[parent] >= h[k] && break
            h[parent], h[k] = h[k], h[parent]
            k = parent
        end
    elseif s < h[1]
        h[1] = s
        n = length(h)
        k = 1
        @inbounds while true
            c = 2k
            c > n && break
            (c < n && h[c + 1] > h[c]) && (c += 1)
            h[k] >= h[c] && break
            h[k], h[c] = h[c], h[k]
            k = c
        end
    end
    return h
end

function _score_child!(heap::Vector{Float64}, t::Tree, data::Dataset, S::Int, half::Int,
                       early_cut::Bool)
    cut = (early_cut && length(heap) == half) ? @inbounds(heap[1]) : Inf
    s = fitness(t, data, S; cut=cut)
    _record_best!(heap, s, half)
    return s
end

# Hijo que es copia exacta de un padre: su MSE es el del padre, sin evaluar las filas. Se
# registra igual que si se hubiera evaluado: con el corte temprano, fitness habria devuelto
# +Inf justo cuando el MSE pasa del corte.
function _score_copy!(heap::Vector{Float64}, parent_mse::Float64, half::Int, early_cut::Bool)
    cut = (early_cut && length(heap) == half) ? @inbounds(heap[1]) : Inf
    s = parent_mse > cut ? Inf : parent_mse
    _record_best!(heap, s, half)
    return s
end

# Copia los arboles vivos (y su cache completo) a un grafo nuevo, conservando lo que
# comparten. Se usa en corridas largas para que la memoria no crezca sin limite.
function _compact(g::Graph, population::Vector{Individual}, data::Dataset)
    newg = _track!(Graph(fold=g.fold))
    map = zeros(Int32, length(g.nodes))
    rows = nrows(data)
    id = Int32(data.id)
    for k in eachindex(population)
        ind = population[k]
        population[k] = Individual(ind.score, _copy_into!(newg, g, ind.tree, map, rows, id))
    end
    return newg
end

function _copy_into!(newg::Graph, g::Graph, n::Int32, map::Vector{Int32}, rows::Int, id::Int32)::Int32
    m = map[n]
    m != 0 && return m
    r = g.nodes[n]
    if r.kind == K_CONST
        m = const_node(newg, r.value)
    elseif r.kind == K_VAR
        m = var_node(newg, r.left)
    else
        m = _push!(newg, r)  # reservado en preorden; los hijos se corrigen abajo
        l = _copy_into!(newg, g, r.left, map, rows, id)
        rr = r.right == 0 ? Int32(0) : _copy_into!(newg, g, r.right, map, rows, id)
        newg.nodes[m] = NodeRec(l, rr, r.size, r.height, r.kind, r.flags, r.value)
        if g.cache_id[n] == id
            newg.cache_off[m] = _new_slot!(newg, rows)
            dst, dstart = _cache_ref(newg, m)
            src, sstart = _cache_ref(g, n)
            copyto!(dst, dstart + 1, src, sstart + 1, rows)
            newg.cache_id[m] = id
        end
    end
    map[n] = m
    return m
end

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
    half = N ÷ 2
    # Grafo propio de la corrida; se reserva memoria para que no se copie al crecer.
    g = _track!(Graph(fold=options.fold_constants))
    expected = ceil(Int, 1.1 * N * _expected_full_size(max_depth)) +
               options.generations * N * (2 * max_depth + 16) + 1024
    sizehint!(g.nodes, expected)
    sizehint!(g.cache_id, expected)
    sizehint!(g.cache_off, expected)
    # Corridas largas: compactar cuando el cache ocupa el doble del presupuesto.
    compact_values = options.cache_mb > 0 ? 2 * options.cache_mb * 2^20 ÷ 8 : typemax(Int)

    population = Individual[]
    sizehint!(population, N)
    for _ in 1:N
        root = _grow!(g, max_depth, nvars, rng, true)
        push!(population, Individual(fitness(Tree(g, root), data, S), root))
    end

    history = Float64[]
    pool = Vector{Int32}(undef, N)
    pool_score = Vector{Float64}(undef, N)  # MSE de cada ganador del torneo
    offspring = Individual[]
    sizehint!(offspring, N)
    heap = Float64[]
    sizehint!(heap, half)
    children = Vector{Int32}(undef, N)        # hijos en orden de creacion
    parent_score = Vector{Float64}(undef, N)  # MSE del padre del que hereda la raiz
    child_score = Vector{Float64}(undef, N)
    copy_score = Vector{Float64}(undef, N)    # MSE del padre si el hijo es su copia (NaN = evaluar)
    detect = options.skip_duplicates
    evaluation_order = Vector{Int}(undef, N)
    for generation in 0:options.generations
        push!(history, minimum(_score, population))
        generation == options.generations && break

        for i in 1:N  # torneo binario repetido
            a = population[rand(rng, 1:N)]
            b = population[rand(rng, 1:N)]
            winner = a.score <= b.score ? a : b
            pool[i] = winner.tree
            pool_score[i] = winner.score
        end

        # Primero se CREAN todos los hijos (mismo orden y mismo consumo del generador que
        # antes) y despues se evaluan empezando por los de mejores padres: el umbral del
        # corte temprano baja antes y se corta mas. El orden de evaluacion no cambia el MSE
        # de ningun sobreviviente ni el orden de creacion (que decide los empates).
        empty!(offspring)
        empty!(heap)
        k = 0
        for _ in 1:(N ÷ 4)  # N/2 hijos por crossover
            j1 = rand(rng, 1:N)
            j2 = rand(rng, 1:N)
            c1, c2, copied = _protected_crossover(Tree(g, pool[j1]), Tree(g, pool[j2]), max_depth,
                                                  rng, detect)
            k += 1
            children[k] = c1.i
            parent_score[k] = pool_score[j1]  # c1 conserva la raiz de p1
            copy_score[k] = copied == COPY_PARENTS ? pool_score[j1] :
                            copied == COPY_SWAPPED ? pool_score[j2] : NaN
            k += 1
            children[k] = c2.i
            parent_score[k] = pool_score[j2]
            copy_score[k] = copied == COPY_PARENTS ? pool_score[j2] :
                            copied == COPY_SWAPPED ? pool_score[j1] : NaN
        end
        for _ in 1:(N ÷ 2)  # N/2 hijos por mutacion
            j = rand(rng, 1:N)
            m, same = _protected_mutation(Tree(g, pool[j]), nvars, max_depth, rng, detect)
            k += 1
            children[k] = m.i
            parent_score[k] = pool_score[j]
            copy_score[k] = same ? pool_score[j] : NaN
        end
        if options.order_by_parent
            sortperm!(evaluation_order, parent_score; alg=MergeSort)  # estable
        else
            evaluation_order .= 1:N  # orden de creacion
        end
        for i in evaluation_order
            child_score[i] = isnan(copy_score[i]) ?
                _score_child!(heap, Tree(g, children[i]), data, S, half, options.early_cut) :
                _score_copy!(heap, copy_score[i], half, options.early_cut)
        end
        for i in 1:N
            push!(offspring, Individual(child_score[i], children[i]))
        end

        # Reemplazo 50/50 (orden estable: mismos empates que A).
        sort!(population; by=_score, alg=MergeSort)
        sort!(offspring; by=_score, alg=MergeSort)
        population = vcat(population[1:half], offspring[1:half])
        if g.cache_values > compact_values
            g = _compact(g, population, data)
        end
    end
    best = population[argmin(map(_score, population))]
    return GPResult(Tree(g, best.tree), best.score, history)
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
crossover; los hijos se sueltan cada 16 cruces, fuera del cronometro.
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
    parents = Tuple{Int32, Int32}[]
    offset = 0
    while offset < pairs
        count = min(per_batch, pairs - offset)
        empty!(parents)
        # Grafo nuevo por tanda: la tanda anterior se libera (fuera del cronometro).
        g = Graph()
        sizehint!(g.nodes, ceil(Int, 2.1 * count * _expected_full_size(depth)) +
                           2 * CHILDREN_CHUNK * (depth + 2) + 1024)
        for _ in 1:count
            p1 = _grow!(g, Int(depth), Int(no_variables), rng, true)
            p2 = _grow!(g, Int(depth), Int(no_variables), rng, true)
            total_nodes += Int(g.nodes[p1].size) + Int(g.nodes[p2].size)
            push!(parents, (p1, p2))
        end
        GC.gc()  # fuera del cronometro
        mark = length(g.nodes)
        for start in 1:CHILDREN_CHUNK:count
            stop = min(count, start + CHILDREN_CHUNK - 1)
            t0 = time_ns()
            for i in start:stop
                p1, p2 = @inbounds parents[i]
                crossover(Tree(g, p1), Tree(g, p2), rng)
            end
            measured += (time_ns() - t0) / 1e9
            _truncate!(g, mark)  # hijos soltados fuera del cronometro
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

# Metadatos guardados == recalculados, en todo el subarbol.
function _metadata_ok(g::Graph, n::Int32)
    r = g.nodes[n]
    r.kind <= K_VAR && return r.size == 1 && r.height == 0
    L = g.nodes[r.left]
    size = 1 + Int(L.size)
    height = Int(L.height)
    maxv = _maxvar(L)
    isc = _isconst(L)
    if r.right != 0
        R = g.nodes[r.right]
        size += Int(R.size)
        height = max(height, Int(R.height))
        maxv = max(maxv, _maxvar(R))
        isc &= _isconst(R)
    end
    folded = (r.flags & FOLDED) != 0
    ok = r.size == size && r.height == height + 1 && folded == (g.fold && isc) &&
         (folded || _maxvar(r) == maxv)
    return ok && _metadata_ok(g, r.left) && (r.right == 0 || _metadata_ok(g, r.right))
end
_metadata_ok(t::Tree) = _metadata_ok(t.g, t.i)

function _count_nodes(g::Graph, n::Int32)
    r = g.nodes[n]
    r.kind <= K_VAR && return 1
    return 1 + _count_nodes(g, r.left) + (r.right == 0 ? 0 : _count_nodes(g, r.right))
end

function _leaf_depths(g::Graph, n::Int32, d, out)
    r = g.nodes[n]
    if r.kind <= K_VAR
        push!(out, d)
    else
        _leaf_depths(g, r.left, d + 1, out)
        r.right == 0 || _leaf_depths(g, r.right, d + 1, out)
    end
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

# Cada subarbol plegado vale, bit a bit, lo mismo que recorrerlo; y esta plegado si y
# solo si no tiene variables.
function _folding_ok(g::Graph, n::Int32)
    r = g.nodes[n]
    r.kind <= K_VAR && return true
    sub = Tree(g, n)
    folded = (r.flags & FOLDED) != 0
    folded == !occursin('x', expression(sub)) || return false
    folded && !(r.value === eval_row(sub, Float64[])) && return false
    return _folding_ok(g, r.left) && (r.right == 0 || _folding_ok(g, r.right))
end

# Los operadores de un arbol nuevo estan en preorden: cada operador va despues de su
# padre y su hijo izquierdo (si no es una variable compartida) va justo despues de el.
function _preorder_ok(g::Graph, n::Int32)
    r = g.nodes[n]
    r.kind <= K_VAR && return true
    L = g.nodes[r.left]
    (L.kind == K_VAR || r.left == n + 1) || return false
    r.right == 0 || g.nodes[r.right].kind == K_VAR || r.right > r.left || return false
    return _preorder_ok(g, r.left) && (r.right == 0 || _preorder_ok(g, r.right))
end

function run_tests(; verbose::Bool=false)
    passed = Ref(0)
    function check(condition::Bool, description::AbstractString)
        condition || error("Fallo en prueba: $description")
        passed[] += 1
        verbose && println("  ok  ", description)
    end
    rng = Xoshiro(42)
    g = DEFAULT_GRAPH

    # --- Parte 1: representacion, expression, eval, MSE ---------------------
    check(sizeof(NodeRec) == 24 && isbitstype(NodeRec), "registro de nodo de 24 bytes sin punteros")
    formula = Tree(g, op_node(g, OP_ADD, var_node(g, 1),
                              op_node(g, OP_MUL, var_node(g, 2), const_node(g, 0.5))))
    check(expression(formula) == "(x1 + (x2 * 0.5))", "expression")
    check(eval_row(formula, [2.0, 3.0]) == 3.5, "eval_row (x1 + x2*0.5) en [2, 3] = 3.5")
    check(node_count(formula) == 5 && tree_depth(formula) == 2, "tamano y profundidad")
    check(var_node(g, 2) == var_node(g, 2), "las variables se comparten")
    check(_throws(() -> op_node(g, OP_ADD, const_node(g, 1.0))), "binario sin hijo derecho es rechazado")
    check(_throws(() -> op_node(g, 0x07, const_node(g, 1.0), const_node(g, 2.0))),
          "codigo de operacion invalido")
    check(_throws(() -> var_node(g, 0)), "indice de variable 0 rechazado")

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

    # Operador sin saltos == operador con saltos, bit a bit (incluye Inf, NaN, ceros).
    specials = [0.0, -0.0, 1e-13, -1e-13, 1.0, -2.5, 3e300, -Inf, Inf, NaN, 1e-300]
    same_ops = true
    for a in specials, b in specials, op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV)
        same_ops &= apply_select(op, a, b) === apply(op, a, b)
    end
    for _ in 1:10000
        a, b = (rand(rng) - 0.5) * 10.0^rand(rng, -14:5), (rand(rng) - 0.5) * 10.0^rand(rng, -14:5)
        op = RANDOM_OPS[rand(rng, 1:4)]
        same_ops &= apply_select(op, a, b) === apply(op, a, b)
    end
    check(same_ops, "operador sin saltos identico al operador con saltos")

    # --- Parte 2: generacion -----------------------------------------------
    for depth in 0:8, nvars in (1, 2, 5)
        full = grow_full(depth, nvars, rng)
        rnd = grow_random(depth, nvars, rng)
        check(tree_depth(full) == depth && _metadata_ok(full), "grow_full profundidad exacta")
        check(tree_depth(rnd) <= depth && _metadata_ok(rnd), "grow_random profundidad maxima")
        check(node_count(full) == _count_nodes(g, full.i), "tamano guardado = tamano real")
        check(_preorder_ok(g, full.i) && _preorder_ok(g, rnd.i), "arboles nuevos en preorden")
        back = parse_expression(expression(full, 17))
        check(expression(back, 17) == expression(full, 17), "expression(17) -> parse sin perdida")
    end
    deep6 = grow_full(6, 3, rng)
    check(unique(_leaf_depths(g, deep6.i, 0, Int[])) == [6],
          "grow_full: TODAS las hojas a profundidad exacta")

    # --- Plegado de constantes ------------------------------------------------
    folded = parse_expression("((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))")
    fr = g.nodes[folded.i]
    L, R = g.nodes[fr.left], g.nodes[fr.right]
    check((L.flags & FOLDED) != 0 && L.value === 0.6 && (fr.flags & FOLDED) == 0 &&
          (R.flags & FOLDED) == 0 && (g.nodes[R.right].flags & FOLDED) != 0 &&
          g.nodes[R.right].value === 1e12, "plegado de constantes: valores y marcas")
    check(expression(folded) == "((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))" && node_count(folded) == 10,
          "el plegado no cambia expression ni tamano")
    folding_ok = true
    for t in 1:300
        tree = grow_random(1 + t % 10, t % 4, rng)  # 0 variables = arbol solo de constantes
        folding_ok &= _folding_ok(g, tree.i)
    end
    check(folding_ok, "cada subarbol constante vale lo mismo, bit a bit, que recorrerlo")
    big = parse_expression("((1e300 * 1e300) - (1e300 * 1e300))")  # Inf - Inf = NaN
    check((g.nodes[big.i].flags & FOLDED) != 0 && isnan(g.nodes[big.i].value) &&
          fitness(big, exact) == Inf, "constante no finita: MSE = +Inf como sin plegado")
    nofold = Graph(fold=false)
    plain = parse_expression("((sqrt(0.25) + 0.1) * x1)"; graph=nofold)
    check((nofold.nodes[nofold.nodes[plain.i].left].flags & FOLDED) == 0 && _metadata_ok(plain),
          "grafo sin plegado")

    # --- Cache: mismo MSE bit a bit con y sin cache, en frio y en caliente ---
    r2 = Xoshiro(7)
    xs = [[(i % 17 == 0) ? 0.0 : (6.0 * rand(r2) - 3.0) * 10.0^rand(r2, (-8, 0, 0, 3))
           for _ in 1:3] for i in 1:150]
    ys = [6.0 * rand(r2) - 3.0 for _ in 1:150]
    hard = Dataset(xs, ys)
    other = Dataset(xs[1:40], ys[41:80])
    mismatches = 0
    for graph in (g, nofold)
        trees = [grow_random(1 + t % 9, 3, rng; graph=graph) for t in 1:200]
        for t in trees
            ref = _reference_mse(t, xs, ys)
            ref_other = _reference_mse(t, xs[1:40], ys[41:80])
            for S in (NO_CACHE, 1, 4, 16)
                fitness(t, hard, S) === ref || (mismatches += 1)   # frio o caliente
                fitness(t, hard, S) === ref || (mismatches += 1)   # caliente
                fitness(t, other, S) === ref_other || (mismatches += 1)  # otro dataset
            end
        end
        # Hijos evaluados con los padres ya en cache (se calienta el cache primero).
        foreach(t -> fitness(t, hard, 1), trees)
        for _ in 1:200
            a, b = trees[rand(rng, 1:200)], trees[rand(rng, 1:200)]
            c, d = crossover(a, b, rng)
            m = mutation(a, 3, rng)
            for t in (c, d, m)
                fitness(t, hard, 1) === _reference_mse(t, xs, ys) || (mismatches += 1)
            end
        end
    end
    check(mismatches == 0, "MSE con cache == sin cache == referencia, bit a bit (con y sin plegado)")

    # --- Corte temprano: si devuelve un numero, es el MSE exacto ---------------
    cut_ok = true
    for t in 1:300
        tree = grow_random(1 + t % 8, 3, rng)
        exact_mse = _reference_mse(tree, xs, ys)
        cut = 10.0^rand(rng, -2:4)
        s = fitness(tree, hard, 4; cut=cut)
        cut_ok &= isinf(s) ? (exact_mse > cut || isinf(exact_mse)) : (s === exact_mse && s <= cut)
    end
    check(cut_ok, "corte temprano: MSE exacto o +Inf solo si el MSE real supera el corte")

    # --- Crossover y mutacion: copia de camino --------------------------------
    a, b = grow_full(4, 2, rng), grow_full(3, 2, rng)
    old_a, old_b = expression(a, 17), expression(b, 17)
    for _ in 1:200
        before = length(g.nodes)
        c, d = crossover(a, b, rng)
        check(node_count(c) + node_count(d) == node_count(a) + node_count(b),
              "crossover conserva el total de nodos")
        check(_metadata_ok(c) && _metadata_ok(d), "metadatos correctos en los hijos")
        check(length(g.nodes) - before <= tree_depth(a) + tree_depth(b),
              "copia de camino: solo se crean los nodos del camino")
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
    check(_throws(() -> crossover(a, grow_full(3, 2, rng; graph=Graph()), rng)),
          "no se mezclan arboles de grafos distintos")

    # Deteccion de hijos que copian a un padre: mismos hijos (mismo texto), mismo consumo
    # del generador, y cuando se reporta una copia, el hijo es el padre que dice.
    let r1 = Xoshiro(11), r2 = Xoshiro(11), copies = 0, ok = true
        for t in 1:3000
            p1 = grow_random(1 + t % 4, 2, r1)
            p2 = grow_random(1 + (t ÷ 4) % 4, 2, r1)
            q1 = grow_random(1 + t % 4, 2, r2)
            q2 = grow_random(1 + (t ÷ 4) % 4, 2, r2)
            c1, c2, copied = _protected_crossover(p1, p2, 4, r1, true)
            d1, d2 = depth_protected_crossover(q1, q2, 4, r2)
            ok &= expression(c1, 17) == expression(d1, 17) && expression(c2, 17) == expression(d2, 17)
            if copied == COPY_PARENTS
                copies += 1
                ok &= c1.i == p1.i && c2.i == p2.i
            elseif copied == COPY_SWAPPED
                copies += 1
                ok &= c1.i == p2.i && c2.i == p1.i
            end
            m, same = _protected_mutation(p1, 2, 4, r1, true)
            n = depth_protected_mutation(q1, 2, 4, r2)
            ok &= expression(m, 17) == expression(n, 17)
            if same
                copies += 1
                ok &= m.i == p1.i
            end
        end
        ok &= rand(r1, UInt64) == rand(r2, UInt64)
        check(ok && copies > 0, "deteccion de hijos que copian a un padre")
    end
    check(!_same_subtree(g.nodes, parse_expression("(x1 + 0)").i, parse_expression("(x1 + -0)").i),
          "constantes iguales solo con los mismos bits")

    # --- GP completo --------------------------------------------------------
    check(_throws(() -> full_gp(exact, GPOptions(generations=-1))), "generaciones negativas rechazadas")
    run = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3))
    h = run.best_fitness_per_generation
    check(length(h) == 4, "historial de fitness")
    check(all(h[i] <= h[i-1] for i in 2:length(h)), "elitismo: el mejor MSE no aumenta")
    same_run(r1, r2) = r1.best_fitness_per_generation == r2.best_fitness_per_generation &&
                       expression(r1.best, 17) == expression(r2.best, 17) &&
                       r1.best_fitness === r2.best_fitness
    for seed in 1:3
        base = GPOptions(population_size=200, max_depth=6, generations=6, seed=seed)
        ref = full_gp(hard, base)
        check(same_run(ref, full_gp(hard, GPOptions(population_size=200, max_depth=6, generations=6,
                                                    seed=seed, cache_min_size=NO_CACHE))),
              "el cache no altera el GP")
        check(same_run(ref, full_gp(hard, GPOptions(population_size=200, max_depth=6, generations=6,
                                                    seed=seed, fold_constants=false, early_cut=false))),
              "el plegado y el corte temprano no alteran el GP")
        check(same_run(ref, full_gp(hard, GPOptions(population_size=200, max_depth=6, generations=6,
                                                    seed=seed, order_by_parent=false))),
              "evaluar primero los hijos de mejores padres no altera el GP")
        check(same_run(ref, full_gp(hard, GPOptions(population_size=200, max_depth=6, generations=6,
                                                    seed=seed, skip_duplicates=false))),
              "reutilizar el MSE de los hijos que copian a un padre no altera el GP")
        # cache_mb=1 con S=1 fuerza la compactacion del grafo en cada generacion.
        check(same_run(ref, full_gp(hard, GPOptions(population_size=200, max_depth=6, generations=6,
                                                    seed=seed, cache_mb=1, cache_min_size=1))),
              "la compactacion del grafo no altera el GP")
    end
    check(auto_cache_min_size(5000, 12, 49, 1024) >= 16 &&
          auto_cache_min_size(5000, 12, 49, 0) == NO_CACHE, "tamano automatico del cache")
    # Con el dataset oficial (961 filas) lo mas rapido medido fue S = 16, 16 y 64.
    check(auto_cache_min_size(5000, 7, 961, 1024) == 16 &&
          auto_cache_min_size(5000, 9, 961, 1024) == 16 &&
          auto_cache_min_size(5000, 12, 961, 1024) == 64, "tamano automatico con 961 filas")

    t = benchmark_crossover(40, 4, 2; seed=1)
    check(t.crossovers == 20 && t.resident_trees == 40 && t.seconds > 0, "benchmark de crossover")
    t = benchmark_crossover(40, 4, 2; seed=1, batch_pairs=5)
    check(t.resident_trees == 10, "benchmark de crossover por lotes")
    return passed[]
end

end # module
