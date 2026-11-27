"""
A_v6 experimental: evaluacion posfija SIMD por bloques con reduccion MSE
secuencial en orden original. Equivalencia pendiente de validar en Julia.
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
# Parte 1 C) Dataset y fitness (MSE) con almacenamiento columnar contiguo (SIMD)
# ---------------------------------------------------------------------------
struct Dataset
    rows::Vector{Vector{Float64}}  # Compatibilidad con API existente
    cols::Matrix{Float64}          # N_filas x N_vars contiguo en memoria para SIMD
    y::Vector{Float64}
    function Dataset(rows::AbstractVector, y::AbstractVector)
        (isempty(rows) || length(rows) != length(y)) &&
            throw(ArgumentError("Dataset vacio o inconsistente"))
        width = length(rows[1])
        all(r -> length(r) == width, rows) ||
            throw(ArgumentError("Dimensiones inconsistentes del dataset"))
        n = length(y)
        r_vec = [Vector{Float64}(r) for r in rows]
        cols = Matrix{Float64}(undef, n, width)
        @inbounds for i in 1:n, j in 1:width
            cols[i, j] = r_vec[i][j]
        end
        return new(r_vec, cols, Vector{Float64}(y))
    end
end

nvariables(d::Dataset) = size(d.cols, 2)
nrows(d::Dataset) = length(d.y)

"Mismo formato que B: x1,...,xn,y; encabezado opcional en la primera linea."
function read_csv(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    y = Float64[]
    columns = 0
    for (line_number, line) in enumerate(readlines(path))
        isempty(strip(line)) && continue
        fields = [strip(f) for f in split(line, ',')]
        parsed = [tryparse(Float64, f) for f in fields]
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
# Motor de Evaluacion Vectorizada por Bloques en Cache L1 (Tile SIMD)
# ---------------------------------------------------------------------------
const TILE_SIZE = 512

const IC_ADD       = 0x01
const IC_SUB       = 0x02
const IC_MUL       = 0x03
const IC_DIV       = 0x04
const IC_SQRT      = 0x05
const IC_CONST     = 0x06
const IC_VAR       = 0x07
const IC_ADD_C     = 0x08
const IC_SUB_C     = 0x09
const IC_C_SUB     = 0x0a
const IC_MUL_C     = 0x0b
const IC_DIV_C     = 0x0c
const IC_C_DIV     = 0x0d
const IC_ADD_VAR   = 0x0e
const IC_SUB_VAR   = 0x0f
const IC_MUL_VAR   = 0x10
const IC_DIV_VAR   = 0x11

mutable struct EvalWorkspace
    code::Vector{UInt8}
    arg_int::Vector{Int}
    arg_flt::Vector{Float64}
    len::Int
    reg_stack::Matrix{Float64} # TILE_SIZE x max_stack_depth (cabe 100% en cache L1)
end

function EvalWorkspace(n_rows::Int, max_depth::Int)
    cap = 2^(max_depth + 1) + 4
    stack_cols = max_depth + 4
    tile = min(n_rows, TILE_SIZE)
    return EvalWorkspace(
        Vector{UInt8}(undef, cap),
        Vector{Int}(undef, cap),
        Vector{Float64}(undef, cap),
        0,
        Matrix{Float64}(undef, max(tile, 1), stack_cols)
    )
end

@inline function _emit!(ws::EvalWorkspace, op::UInt8, iv::Int, fv::Float64)
    idx = ws.len + 1
    if idx > length(ws.code)
        resize!(ws.code, idx * 2)
        resize!(ws.arg_int, idx * 2)
        resize!(ws.arg_flt, idx * 2)
    end
    ws.len = idx
    @inbounds begin
        ws.code[idx] = op
        ws.arg_int[idx] = iv
        ws.arg_flt[idx] = fv
    end
    return nothing
end

function _compile_node!(ws::EvalWorkspace, n::NodeConstant)::Tuple{Bool, Float64}
    return (true, n.value)
end

function _compile_node!(ws::EvalWorkspace, n::NodeVariable)::Tuple{Bool, Float64}
    _emit!(ws, IC_VAR, n.index, 0.0)
    return (false, 0.0)
end

function _compile_node!(ws::EvalWorkspace, n::NodeInternal)::Tuple{Bool, Float64}
    if n.op == SQRT
        lc, lv = _compile_node!(ws, n.left)
        if lc
            return (true, protected_sqrt(lv))
        end
        _emit!(ws, IC_SQRT, 0, 0.0)
        return (false, 0.0)
    end

    left = n.left
    right = n.right::AnyNode

    if left isa NodeConstant && right isa NodeConstant
        return (true, apply(n.op, left.value, right.value))
    end

    lc, lv = _compile_node!(ws, left)
    if lc
        rc, rv = _compile_node!(ws, right)
        if rc
            return (true, apply(n.op, lv, rv))
        end
        op_code = n.op == ADD ? IC_ADD_C :
                  n.op == SUB ? IC_C_SUB :
                  n.op == MUL ? IC_MUL_C : IC_C_DIV
        _emit!(ws, op_code, 0, lv)
        return (false, 0.0)
    else
        if right isa NodeVariable
            op_code = n.op == ADD ? IC_ADD_VAR :
                      n.op == SUB ? IC_SUB_VAR :
                      n.op == MUL ? IC_MUL_VAR : IC_DIV_VAR
            _emit!(ws, op_code, right.index, 0.0)
            return (false, 0.0)
        end
        rc, rv = _compile_node!(ws, right)
        if rc
            op_code = n.op == ADD ? IC_ADD_C :
                      n.op == SUB ? IC_SUB_C :
                      n.op == MUL ? IC_MUL_C : IC_DIV_C
            _emit!(ws, op_code, 0, rv)
            return (false, 0.0)
        else
            _emit!(ws, UInt8(n.op) + 0x01, 0, 0.0)
            return (false, 0.0)
        end
    end
end

function _fitness_fast(tree::AnyNode, data::Dataset, ws::EvalWorkspace)::Float64
    n = length(data.y)
    ws.len = 0
    is_const, cval = _compile_node!(ws, tree)

    y = data.y
    if is_const
        total = 0.0
        @inbounds for i in 1:n
            d = y[i] - cval
            squared = d * d
            isfinite(squared) || return Inf
            total += squared
            isfinite(total) || return Inf
        end
        return isfinite(total) ? (total / n) : Inf
    end

    cols = data.cols
    stack = ws.reg_stack
    code = ws.code
    arg_int = ws.arg_int
    arg_flt = ws.arg_flt
    n_inst = ws.len
    total = 0.0

    @inbounds for row_start in 1:TILE_SIZE:n
        row_end = min(n, row_start + TILE_SIZE - 1)
        m = row_end - row_start + 1
        off = row_start - 1
        sp = 0

        for k in 1:n_inst
            op = code[k]
            if op == IC_VAR
                sp += 1
                col = arg_int[k]
                @simd for i in 1:m
                    stack[i, sp] = cols[off + i, col]
                end
            elseif op == IC_ADD_VAR
                col = arg_int[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] + cols[off + i, col]
                end
            elseif op == IC_SUB_VAR
                col = arg_int[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] - cols[off + i, col]
                end
            elseif op == IC_MUL_VAR
                col = arg_int[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] * cols[off + i, col]
                end
            elseif op == IC_DIV_VAR
                col = arg_int[k]
                @simd for i in 1:m
                    b = cols[off + i, col]
                    denom = abs(b) < EPS ? EPS : b
                    stack[i, sp] = stack[i, sp] / denom
                end
            elseif op == IC_ADD
                sp -= 1
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] + stack[i, sp + 1]
                end
            elseif op == IC_SUB
                sp -= 1
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] - stack[i, sp + 1]
                end
            elseif op == IC_MUL
                sp -= 1
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] * stack[i, sp + 1]
                end
            elseif op == IC_DIV
                sp -= 1
                @simd for i in 1:m
                    b = stack[i, sp + 1]
                    denom = abs(b) < EPS ? EPS : b
                    stack[i, sp] = stack[i, sp] / denom
                end
            elseif op == IC_SQRT
                @simd for i in 1:m
                    a = stack[i, sp]
                    stack[i, sp] = sqrt(a < 0.0 ? 0.0 : a)
                end
            elseif op == IC_ADD_C
                c = arg_flt[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] + c
                end
            elseif op == IC_SUB_C
                c = arg_flt[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] - c
                end
            elseif op == IC_C_SUB
                c = arg_flt[k]
                @simd for i in 1:m
                    stack[i, sp] = c - stack[i, sp]
                end
            elseif op == IC_MUL_C
                c = arg_flt[k]
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] * c
                end
            elseif op == IC_DIV_C
                c = arg_flt[k]
                denom = abs(c) < EPS ? EPS : c
                @simd for i in 1:m
                    stack[i, sp] = stack[i, sp] / denom
                end
            elseif op == IC_C_DIV
                c = arg_flt[k]
                @simd for i in 1:m
                    b = stack[i, sp]
                    denom = abs(b) < EPS ? EPS : b
                    stack[i, sp] = c / denom
                end
            elseif op == IC_CONST
                sp += 1
                c = arg_flt[k]
                @simd for i in 1:m
                    stack[i, sp] = c
                end
            end
        end

        # Una sola reduccion, en el orden de las filas de fitness().
        for i in 1:m
            d = y[off + i] - stack[i, 1]
            squared = d * d
            isfinite(squared) || return Inf
            total += squared
            isfinite(total) || return Inf
        end
    end

    return isfinite(total) ? (total / n) : Inf
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
# Puntos de corte (Slot = dueno, lado, profundidad, altura)
# ---------------------------------------------------------------------------
mutable struct Holder
    left::AnyNode
end

const Owner = Union{Holder, NodeInternal}

struct Slot
    owner::Owner
    isleft::Bool
    depth::Int
    height::Int
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

function _slots(holder::Holder)
    out = Slot[]
    height = _collect!(out, holder, true, 0)
    return out, height
end

function _clone_collect!(out::Vector{Slot}, owner::Owner, isleft::Bool,
                         depth::Int)::Int
    node = getchild(owner, isleft)
    push!(out, Slot(owner, isleft, depth, 0))
    index = length(out)
    height = 0
    if node isa NodeInternal
        copy_node = NodeInternal(node.op, node.left, node.right)
        setchild!(owner, isleft, copy_node)
        height = _clone_collect!(out, copy_node, true, depth + 1) + 1
        if copy_node.right !== nothing
            height = max(height, _clone_collect!(out, copy_node, false, depth + 1) + 1)
        end
        out[index] = Slot(owner, isleft, depth, height)
    end
    return height
end

function _clone_slots!(slots::Vector{Slot}, tree::Node)
    empty!(slots)
    holder = Holder(tree)
    height = _clone_collect!(slots, holder, true, 0)
    return holder, height
end

function _clone_slots(tree::Node)
    slots = Slot[]
    holder, height = _clone_slots!(slots, tree)
    return holder, slots, height
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
function crossover(a::Node, b::Node, rng::AbstractRNG)
    holder_a, slots_a, _ = _clone_slots(a)
    holder_b, slots_b, _ = _clone_slots(b)
    i = rand(rng, 1:length(slots_a))
    j = rand(rng, 1:length(slots_b))
    _swap!(slots_a[i], slots_b[j])
    return holder_a.left, holder_b.left
end

function mutation(parent::Node, no_variables::Integer, rng::AbstractRNG,
                  new_subtree_max_depth::Integer=3)
    _check_depth(new_subtree_max_depth)
    holder, slots, _ = _clone_slots(parent)
    replacement = grow_random(rand(rng, 0:Int(new_subtree_max_depth)), no_variables, rng)
    slot = slots[rand(rng, 1:length(slots))]
    setchild!(slot.owner, slot.isleft, replacement)
    return holder.left
end

function _depth_protected_crossover!(slots_a::Vector{Slot}, slots_b::Vector{Slot},
                                     a::Node, b::Node, max_depth::Integer, rng::AbstractRNG)
    holder_a, height_a = _clone_slots!(slots_a, a)
    holder_b, height_b = _clone_slots!(slots_b, b)
    (height_a > max_depth || height_b > max_depth) &&
        throw(ArgumentError("Un padre supera max_depth"))
    na = length(slots_a)
    nb = length(slots_b)
    for _ in 1:PROTECTED_ATTEMPTS
        x = @inbounds slots_a[rand(rng, 1:na)]
        y = @inbounds slots_b[rand(rng, 1:nb)]
        if x.depth + y.height <= max_depth && y.depth + x.height <= max_depth
            _swap!(x, y)
            return holder_a.left, holder_b.left
        end
    end
    return holder_b.left, holder_a.left
end

function depth_protected_crossover(a::Node, b::Node, max_depth::Integer, rng::AbstractRNG)
    _check_depth(max_depth)
    return _depth_protected_crossover!(Slot[], Slot[], a, b, max_depth, rng)
end

function _depth_protected_mutation!(slots::Vector{Slot}, parent::Node,
                                    no_variables::Integer, max_depth::Integer,
                                    rng::AbstractRNG)
    holder, height = _clone_slots!(slots, parent)
    height > max_depth && throw(ArgumentError("El padre supera max_depth"))
    slot = @inbounds slots[rand(rng, 1:length(slots))]
    budget = Int(max_depth) - slot.depth
    setchild!(slot.owner, slot.isleft, grow_random(rand(rng, 0:budget), no_variables, rng))
    return holder.left
end

function depth_protected_mutation(parent::Node, no_variables::Integer, max_depth::Integer,
                                  rng::AbstractRNG)
    _check_depth(max_depth)
    return _depth_protected_mutation!(Slot[], parent, no_variables, max_depth, rng)
end

# ---------------------------------------------------------------------------
# Parte 4 A) GP completo
# ---------------------------------------------------------------------------
Base.@kwdef struct GPOptions
    population_size::Int = 40
    max_depth::Int = 5
    generations::Int = 5
    seed::Int = 42
    on_generation::Any = nothing
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
    options.generations >= 0 || throw(ArgumentError("generations debe ser >= 0"))
    rng = Xoshiro(options.seed)
    nvars, max_depth = nvariables(data), options.max_depth

    ws = EvalWorkspace(nrows(data), max_depth)
    slots_a = Vector{Slot}(undef, 0)
    slots_b = Vector{Slot}(undef, 0)
    sizehint!(slots_a, 2^(max_depth + 1))
    sizehint!(slots_b, 2^(max_depth + 1))

    # 1-2) Poblacion inicial 100% grow_full y su fitness.
    population = Vector{Individual}(undef, N)
    for i in 1:N
        tree = grow_full(max_depth, nvars, rng)
        population[i] = Individual(_fitness_fast(tree, data, ws), tree)
    end

    history = Float64[]
    pool = Vector{AnyNode}(undef, N)
    offspring = Vector{Individual}(undef, N)
    next_pop = Vector{Individual}(undef, N)
    half_N = N ÷ 2
    quarter_N = N ÷ 4

    for generation in 0:options.generations
        push!(history, minimum(_score, population))
        options.on_generation === nothing || options.on_generation(generation, history[end])
        generation == options.generations && break

        # 3) Torneo binario repetido: pool de N ganadores (empate -> el primero).
        @inbounds for i in 1:N
            a = population[rand(rng, 1:N)]
            b = population[rand(rng, 1:N)]
            pool[i] = a.score <= b.score ? a.tree : b.tree
        end

        # 4-6) Parejas al azar del pool; 50% hijos por crossover, 50% por mutacion.
        idx = 1
        @inbounds for _ in 1:quarter_N
            p1 = pool[rand(rng, 1:N)]
            p2 = pool[rand(rng, 1:N)]
            c1, c2 = _depth_protected_crossover!(slots_a, slots_b, p1, p2, max_depth, rng)
            offspring[idx] = Individual(_fitness_fast(c1, data, ws), c1)
            offspring[idx + 1] = Individual(_fitness_fast(c2, data, ws), c2)
            idx += 2
        end
        @inbounds for _ in 1:half_N
            child = _depth_protected_mutation!(slots_a, pool[rand(rng, 1:N)], nvars, max_depth, rng)
            offspring[idx] = Individual(_fitness_fast(child, data, ws), child)
            idx += 1
        end

        # 7) Reemplazo 50/50 de estado estable (orden estable: mismos empates que C).
        sort!(population; by=_score, alg=MergeSort)
        sort!(offspring; by=_score, alg=MergeSort)
        @inbounds for i in 1:half_N
            next_pop[i] = population[i]
            next_pop[half_N + i] = offspring[i]
        end
        population, next_pop = next_pop, population
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
    resident_trees::Int
    crossovers::Int
    seconds::Float64
    mean_seconds_per_crossover::Float64
    mean_nodes_per_tree::Float64
end

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
    slots_a = Slot[]
    slots_b = Slot[]
    sizehint!(slots_a, 2^(depth + 1))
    sizehint!(slots_b, 2^(depth + 1))
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
        GC.gc()
        for start in 1:CHILDREN_CHUNK:count
            stop = min(count, start + CHILDREN_CHUNK - 1)
            t0 = time_ns()
            for i in start:stop
                ha, _ = _clone_slots!(slots_a, parents[i][1])
                hb, _ = _clone_slots!(slots_b, parents[i][2])
                ia = rand(rng, 1:length(slots_a))
                ib = rand(rng, 1:length(slots_b))
                _swap!(slots_a[ia], slots_b[ib])
                push!(children, (ha.left, hb.left))
            end
            measured += (time_ns() - t0) / 1e9
            empty!(children)
            GC.gc(false)
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

function _throws_any(f)
    try
        f()
        return false
    catch
        return true
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

    fold_rng = Xoshiro(2026)
    fold_data = Dataset([[0.0, -1.0], [1.0, 2.0], [-2.0, 0.0]], [0.0, 1.0, -1.0])
    ws = EvalWorkspace(nrows(fold_data), 7)
    for depth in 0:7, _ in 1:20
        tree = grow_full(depth, 2, fold_rng)
        before = expression(tree, 17)
        ref = fitness(tree, fold_data)
        fast = _fitness_fast(tree, fold_data, ws)
        check(isequal(ref, fast),
              "A_v6: MSE equivalente depth=$depth ref=$ref fast=$fast arbol=$before")
        check(expression(tree, 17) == before, "A_v6: arbol genetico intacto")
    end
    for tree in (parse_expression("sqrt(-5)"), parse_expression("(1 / 0)"),
                 parse_expression("((1e300 * 1e300) - (1e300 * 1e300))"),
                 parse_expression("sqrt(-0.0)"))
        check(isequal(fitness(tree, fold_data), _fitness_fast(tree, fold_data, ws)),
              "A_v6: caso numerico especial")
    end

    # Regresion: limites de bloque y dataset de 961 filas.
    for count in (511, 512, 513, 961, 1025)
        block_data = Dataset([[sin(i / 7), cos(i / 11)] for i in 1:count],
                             [sin(i / 13) for i in 1:count])
        block_ws = EvalWorkspace(count, 7)
        for tree in (NodeConstant(0.123), NodeVariable(1),
                     parse_expression("(x1 / x2)"),
                     parse_expression("(0.3 - sqrt(x1))"),
                     parse_expression("((x1 * x2) + (x2 / 0.0))"),
                     grow_full(7, 2, fold_rng))
            ref = fitness(tree, block_data)
            fast = _fitness_fast(tree, block_data, block_ws)
            check(isequal(ref, fast),
                  "A_v6: filas=$count ref=$ref fast=$fast arbol=$(expression(tree, 17))")
        end
    end

    fusion_rng = Xoshiro(2027)
    for depth in 0:7, _ in 1:20
        tree = grow_full(depth, 2, fusion_rng)
        before = expression(tree, 17)
        reference = Holder(clone(tree))
        old_slots, old_height = _slots(reference)
        holder, slots, height = _clone_slots(tree)
        check(height == old_height && length(slots) == length(old_slots),
              "A_v3: altura y cantidad de slots")
        check([(s.depth, s.height, s.isleft) for s in slots] ==
              [(s.depth, s.height, s.isleft) for s in old_slots], "A_v3: preorden")
        check(expression(holder.left, 17) == before, "A_v3: clon equivalente")
        original_slots, _ = _slots(Holder(tree))
        check(all(!(getchild(a.owner, a.isleft) isa NodeInternal) ||
                  getchild(a.owner, a.isleft) !== getchild(b.owner, b.isleft)
                  for (a,b) in zip(original_slots, slots)), "A_v3: nodos internos independientes")
        cut = slots[rand(fusion_rng, 1:length(slots))]
        setchild!(cut.owner, cut.isleft, NodeConstant(0.123))
        check(expression(tree, 17) == before, "A_v3: padre intacto")
    end

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

    check(_throws(() -> full_gp(exact, GPOptions(generations=-1))), "generaciones negativas rechazadas")
    for (text, header_ok) in (("x1,x2,y\n1,2,3\n", true), ("1,abc,3\n4,5,6\n", false),
                              ("x1,x2,y\n1,2,\n", false))
        mktemp() do path, io
            write(io, text)
            close(io)
            check(header_ok ? nrows(read_csv(path)) == 1 : _throws_any(() -> read_csv(path)),
                  "lector CSV: $(repr(text))")
        end
    end
    run = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3))
    h = run.best_fitness_per_generation
    check(length(h) == 4, "historial de fitness")
    check(all(h[i] <= h[i-1] for i in 2:length(h)), "elitismo: el mejor MSE nunca aumenta")
    check(tree_depth(run.best) <= 3, "el mejor arbol respeta max_depth")
    check(run.best_fitness === fitness(run.best, exact), "best_fitness coincide con el arbol")
    seen = Tuple{Int,Float64}[]
    again = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3,
                                     on_generation=(g, mse) -> push!(seen, (g, mse))))
    check(first.(seen) == 0:3 && last.(seen) == h, "on_generation da el historial en orden")
    check(again.best_fitness_per_generation == h &&
          expression(again.best, 17) == expression(run.best, 17), "on_generation no cambia el GP")
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