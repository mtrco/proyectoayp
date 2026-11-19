"""
Programacion Genetica (GP) - Implementacion C (alternativa, no convencional), Julia 1.9+.

Representacion: cada individuo es un ARREGLO PLANO en notacion POSTFIJA
(hijo izquierdo, hijo derecho, operador), sin punteros entre nodos y en
"struct of arrays": un vector de codigos (1 byte por nodo) y un vector de
valores (8 bytes por nodo).

    x1 + (x2 * 0.5)   ->   ops : VAR  VAR  CONST  MUL  ADD
                           vals: 1    2    0.5    0    0

Un subarbol siempre ocupa un SEGMENTO CONTIGUO que termina en su raiz, asi que:
  * clonar = copiar dos vectores (dos memcpy);
  * crossover y mutacion = pegar segmentos (tres copias contiguas por hijo);
  * con tamano, altura y profundidad precalculados, cada intento del crossover
    protegido se valida en O(1).
La evaluacion es POR LOTES: el programa se recorre una vez por bloque de filas
y cada instruccion opera sobre el bloque completo (datos por columnas).

Mismas reglas que A y B, y mismo consumo del generador aleatorio que A: con la
misma semilla, C produce los mismos arboles que A. Con pick = PREORDER, el
punto de corte se elige en preorden como en A, y C reproduce EXACTAMENTE las
corridas de A (ver tools/cross_check.jl). Con FAST (por defecto) se elige
directamente en el arreglo: misma distribucion, sin calcular el preorden.
Un solo hilo: el profesor no permite paralelismo.
"""
module GPC

using Random
using Printf

export Program, Dataset, Workspace, Pick, FAST, PREORDER,
       expression, postfix, parse_expression, format_number, is_well_formed,
       node_count, tree_depth, subtree_start, eval_row, fitness, clone,
       read_csv, dataset_from_rows, nvariables, nrows,
       grow_random, grow_full, crossover, mutation,
       depth_protected_crossover, depth_protected_mutation,
       GPOptions, GPResult, full_gp, CrossoverTiming, benchmark_crossover, run_tests

const EPS = 1e-12
const PROTECTED_ATTEMPTS = 128  # igual que A y B
const CHILDREN_CHUNK = 16       # benchmark: hijos liberados cada 16 cruces

# Codigos de operacion (1 byte).
const OP_CONST = 0x00
const OP_VAR = 0x01
const OP_ADD = 0x02
const OP_SUB = 0x03
const OP_MUL = 0x04
const OP_DIV = 0x05
const OP_SQRT = 0x06
const RANDOM_OPS = (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT)  # mismo orden que A y B

@inline arity(op::UInt8) = op <= OP_VAR ? 0 : (op == OP_SQRT ? 1 : 2)
_symbol(op::UInt8) = op == OP_ADD ? "+" : op == OP_SUB ? "-" : op == OP_MUL ? "*" : "/"

# Raiz IEEE sin la comprobacion de dominio de `sqrt` (el argumento ya es >= 0,
# -0.0 o NaN): permite vectorizar el ciclo. Mismo resultado bit a bit.
@static if isdefined(Base, :sqrt_llvm)
    @inline _sqrt(x::Float64) = Base.sqrt_llvm(x)
else
    @inline _sqrt(x::Float64) = sqrt(x)
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

@enum Pick FAST PREORDER

# ---------------------------------------------------------------------------
# Representacion
# ---------------------------------------------------------------------------
"Arreglo postfijo. Para OP_VAR, vals guarda el indice de la variable (1 = x1)."
struct Program
    ops::Vector{UInt8}
    vals::Vector{Float64}
end

Base.length(p::Program) = length(p.ops)
node_count(p::Program) = length(p.ops)
"Clonacion profunda: dos copias contiguas."
clone(p::Program) = Program(copy(p.ops), copy(p.vals))
Base.copy(p::Program) = clone(p)
Base.show(io::IO, p::Program) = print(io, expression(p))

# Buffer de trabajo reutilizable para generar arboles sin reservar memoria cada vez.
struct Scratch
    ops::Vector{UInt8}
    vals::Vector{Float64}
end
Scratch() = Scratch(UInt8[], Float64[])
const SCRATCH = Scratch()

@inline function _push!(s::Scratch, op::UInt8, value::Float64)
    push!(s.ops, op)
    push!(s.vals, value)
    return nothing
end
_clear!(s::Scratch) = (empty!(s.ops); empty!(s.vals); s)
_program(s::Scratch) = Program(copy(s.ops), copy(s.vals))

function expression(p::Program, precision::Integer=8)
    stack = String[]
    @inbounds for i in eachindex(p.ops)
        op = p.ops[i]
        if op == OP_CONST
            push!(stack, format_number(p.vals[i], precision))
        elseif op == OP_VAR
            push!(stack, "x$(Int(p.vals[i]))")
        elseif op == OP_SQRT
            stack[end] = "sqrt(" * stack[end] * ")"
        else
            right = pop!(stack)
            stack[end] = "(" * stack[end] * " " * _symbol(op) * " " * right * ")"
        end
    end
    return isempty(stack) ? "" : stack[end]
end

"Notacion postfija legible: \"x1 x2 0.5 * +\"."
function postfix(p::Program, precision::Integer=17)
    parts = String[]
    for i in eachindex(p.ops)
        op = p.ops[i]
        push!(parts, op == OP_CONST ? format_number(p.vals[i], precision) :
                     op == OP_VAR ? "x$(Int(p.vals[i]))" :
                     op == OP_SQRT ? "sqrt" : _symbol(op))
    end
    return join(parts, ' ')
end

"Disciplina de pila correcta: cada operador tiene sus operandos y queda un solo valor."
function is_well_formed(p::Program)
    length(p.ops) == length(p.vals) && !isempty(p.ops) || return false
    sp = 0
    for i in eachindex(p.ops)
        op = p.ops[i]
        op > OP_SQRT && return false
        a = arity(op)
        sp < a && return false
        if op == OP_VAR
            v = p.vals[i]
            (v >= 1 && isinteger(v)) || return false
        end
        sp = sp - a + 1
    end
    return sp == 1
end

"Profundidad (una hoja sola = 0)."
function tree_depth(p::Program)
    stack = Int[]
    for op in p.ops
        a = arity(op)
        if a == 0
            push!(stack, 0)
        elseif a == 1
            stack[end] += 1
        else
            r = pop!(stack)
            stack[end] = max(stack[end], r) + 1
        end
    end
    return isempty(stack) ? -1 : stack[end]
end

"Inicio del subarbol cuya raiz esta en `root`: cuenta operandos hacia atras."
function subtree_start(p::Program, root::Int)
    need = 1
    j = root + 1
    @inbounds while true
        j -= 1
        need += arity(p.ops[j]) - 1
        need <= 0 && return j
    end
end

# ---------------------------------------------------------------------------
# Lectura del formato de texto comun (genera el arreglo postfijo directamente)
# ---------------------------------------------------------------------------
mutable struct _Parser
    s::String
    pos::Int
    out::Scratch
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

const _BINARY_BY_CHAR = Dict('+' => OP_ADD, '-' => OP_SUB, '*' => OP_MUL, '/' => OP_DIV)

function _parse_node!(p::_Parser)
    _skip!(p)
    c = _peek(p)
    c == '\0' && _fail(p, "fin inesperado")
    if c == '('
        p.pos += 1
        _parse_node!(p)
        _skip!(p)
        op = get(_BINARY_BY_CHAR, _peek(p), nothing)
        op === nothing && _fail(p, "operador desconocido")
        p.pos += 1
        _parse_node!(p)
        _expect!(p, ')')
        _push!(p.out, op, 0.0)
    elseif _matches(p, "sqrt(")
        p.pos += 5
        _parse_node!(p)
        _expect!(p, ')')
        _push!(p.out, OP_SQRT, 0.0)
    elseif c == 'x'
        p.pos += 1
        index = tryparse(Int, _token!(p))
        (index === nothing || index < 1) && _fail(p, "variable invalida")
        _push!(p.out, OP_VAR, Float64(index))
    else
        value = tryparse(Float64, _token!(p))
        value === nothing && _fail(p, "numero invalido")
        _push!(p.out, OP_CONST, value)
    end
    return nothing
end

"Lee el formato de expression(): \"((x1 + -0.25) * sqrt(x2))\"."
function parse_expression(text::AbstractString)::Program
    p = _Parser(String(text), 1, Scratch())
    _parse_node!(p)
    _skip!(p)
    p.pos <= ncodeunits(p.s) && _fail(p, "texto sobrante")
    return _program(p.out)
end

# ---------------------------------------------------------------------------
# Dataset por COLUMNAS
# ---------------------------------------------------------------------------
struct Dataset
    columns::Vector{Vector{Float64}}  # columns[v][fila]
    y::Vector{Float64}
end

nrows(d::Dataset) = length(d.y)
nvariables(d::Dataset) = length(d.columns)

function dataset_from_rows(rows::AbstractVector, y::AbstractVector)
    (isempty(rows) || length(rows) != length(y)) &&
        throw(ArgumentError("Dataset vacio o inconsistente"))
    width = length(rows[1])
    columns = [Vector{Float64}(undef, length(rows)) for _ in 1:width]
    for (r, row) in enumerate(rows)
        length(row) == width || throw(ArgumentError("Dimensiones inconsistentes del dataset"))
        for v in 1:width
            columns[v][r] = row[v]
        end
    end
    return Dataset(columns, Vector{Float64}(y))
end

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
            line_number == 1 && continue  # encabezado
            error("CSV linea $line_number: $why")
        end
        push!(y, pop!(values))
        push!(rows, values)
    end
    isempty(rows) && error("Dataset sin filas numericas")
    return dataset_from_rows(rows, y)
end

# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
"Referencia escalar: evalua el programa en un solo vector de entrada."
function eval_row(p::Program, x::AbstractVector{Float64})::Float64
    stack = Float64[]
    @inbounds for i in eachindex(p.ops)
        op = p.ops[i]
        if op == OP_CONST
            push!(stack, p.vals[i])
        elseif op == OP_VAR
            v = Int(p.vals[i])
            v <= length(x) || throw(ArgumentError("Indice de variable fuera del vector"))
            push!(stack, x[v])
        elseif op == OP_SQRT
            a = stack[end]
            stack[end] = sqrt(a < 0.0 ? 0.0 : a)
        else
            b = pop!(stack)
            a = stack[end]
            stack[end] = op == OP_ADD ? a + b :
                         op == OP_SUB ? a - b :
                         op == OP_MUL ? a * b : a / (abs(b) < EPS ? EPS : b)
        end
    end
    return stack[end]
end

"Espacio de trabajo de la evaluacion por lotes: pila de vectores de `block` filas."
mutable struct Workspace
    block::Int
    stack::Matrix{Float64}  # stack[:, k] = k-esimo vector de la pila
end
Workspace(block::Integer=64) = Workspace(Int(block), Matrix{Float64}(undef, Int(block), 0))

"""
MSE por lotes. Para cada bloque de filas, el programa se recorre UNA vez y cada
instruccion se aplica al bloque completo. La suma del error es SECUENCIAL por
fila (mismo orden que A y B), de modo que el MSE es identico bit a bit.
"""
function fitness(p::Program, data::Dataset, ws::Workspace)::Float64
    rows = nrows(data)
    rows > 0 || throw(ArgumentError("Dataset vacio"))
    ws.block >= 1 || throw(ArgumentError("block debe ser >= 1"))
    ops, vals = p.ops, p.vals
    n = length(ops)
    nv = nvariables(data)

    # Pasada previa O(n): valida el programa y calcula la profundidad maxima de pila.
    sp = 0
    max_sp = 0
    @inbounds for i in 1:n
        a = arity(ops[i])
        if a == 0
            if ops[i] == OP_VAR
                v = vals[i]
                (v >= 1 && v <= nv) || throw(ArgumentError("Indice de variable fuera del vector"))
            end
            sp += 1
            max_sp = max(max_sp, sp)
        else
            sp < a && throw(ArgumentError("Programa mal formado"))
            sp -= a - 1
        end
    end
    sp == 1 || throw(ArgumentError("Programa mal formado"))

    B = ws.block
    if size(ws.stack, 1) != B || size(ws.stack, 2) < max_sp
        ws.stack = Matrix{Float64}(undef, B, max(max_sp, 2 * size(ws.stack, 2)))
    end
    st = ws.stack
    y = data.y
    columns = data.columns
    total = 0.0
    r0 = 1
    @inbounds while r0 <= rows
        L = min(B, rows - r0 + 1)
        off = r0 - 1
        top = 0
        for i in 1:n
            op = ops[i]
            if op == OP_CONST
                top += 1
                v = vals[i]
                @simd for k in 1:L
                    st[k, top] = v
                end
            elseif op == OP_VAR
                top += 1
                col = columns[unsafe_trunc(Int, vals[i])]
                @simd for k in 1:L
                    st[k, top] = col[off + k]
                end
            elseif op == OP_SQRT
                @simd for k in 1:L
                    a = st[k, top]
                    st[k, top] = _sqrt(ifelse(a < 0.0, 0.0, a))
                end
            elseif op == OP_ADD
                @simd for k in 1:L
                    st[k, top - 1] = st[k, top - 1] + st[k, top]
                end
                top -= 1
            elseif op == OP_SUB
                @simd for k in 1:L
                    st[k, top - 1] = st[k, top - 1] - st[k, top]
                end
                top -= 1
            elseif op == OP_MUL
                @simd for k in 1:L
                    st[k, top - 1] = st[k, top - 1] * st[k, top]
                end
                top -= 1
            else  # OP_DIV
                @simd for k in 1:L
                    b = st[k, top]
                    st[k, top - 1] = st[k, top - 1] / ifelse(abs(b) < EPS, EPS, b)
                end
                top -= 1
            end
        end
        # Suma SECUENCIAL (sin @simd, que podria reordenarla): resultados bit a bit iguales.
        for k in 1:L
            d = y[off + k] - st[k, 1]
            total += d * d
        end
        isfinite(total) || return Inf
        r0 += B
    end
    return total / rows
end

const DEFAULT_WORKSPACE = Workspace(64)
fitness(p::Program, data::Dataset) = fitness(p, data, DEFAULT_WORKSPACE)

# ---------------------------------------------------------------------------
# Generacion (mismas reglas y mismo consumo del generador que A)
# ---------------------------------------------------------------------------
@inline random_constant(rng::AbstractRNG) = 2.0 * rand(rng) - 1.0  # igual que A

function _emit_leaf!(s::Scratch, nvars::Int, rng::AbstractRNG)
    if nvars == 0 || rand(rng, 0:1) == 0
        _push!(s, OP_CONST, random_constant(rng))
    else
        _push!(s, OP_VAR, Float64(rand(rng, 1:nvars)))
    end
    return nothing
end

# Genera directamente en postorden: hijos primero, operador al final. El operador
# se sortea ANTES que los hijos, en el mismo orden que A.
function _grow!(s::Scratch, remaining::Int, nvars::Int, rng::AbstractRNG, full::Bool)
    if remaining == 0 || (!full && rand(rng, 0:3) == 0)
        _emit_leaf!(s, nvars, rng)
        return nothing
    end
    op = RANDOM_OPS[rand(rng, 1:5)]
    _grow!(s, remaining - 1, nvars, rng, full)
    op != OP_SQRT && _grow!(s, remaining - 1, nvars, rng, full)
    _push!(s, op, 0.0)
    return nothing
end

_check_depth(depth::Integer) = depth >= 0 || throw(ArgumentError("La profundidad debe ser >= 0"))

"Arbol con profundidad de A LO MAS `depth`."
function grow_random(depth::Integer, no_variables::Integer, rng::AbstractRNG)
    _check_depth(depth)
    s = _clear!(SCRATCH)
    _grow!(s, Int(depth), Int(no_variables), rng, false)
    return _program(s)
end

"Arbol cuyas hojas estan TODAS exactamente a profundidad `depth`."
function grow_full(depth::Integer, no_variables::Integer, rng::AbstractRNG)
    _check_depth(depth)
    s = _clear!(SCRATCH)
    _grow!(s, Int(depth), Int(no_variables), rng, true)
    return _program(s)
end

# ---------------------------------------------------------------------------
# Metadatos bajo demanda: solo los operadores que los necesitan los calculan.
# ---------------------------------------------------------------------------
struct Shape
    size::Vector{Int}      # nodos del subarbol con raiz en i
    height::Vector{Int}    # altura del subarbol (hoja = 0)
    parent::Vector{Int}
    depth::Vector{Int}     # profundidad del nodo (raiz = 0)
    pre2post::Vector{Int}  # k-esimo nodo en preorden -> indice postfijo
    stack::Vector{Int}
end
Shape() = Shape(Int[], Int[], Int[], Int[], Int[], Int[])
const SHAPE_A = Shape()
const SHAPE_B = Shape()

function _analyze!(s::Shape, p::Program, want_depth::Bool, want_preorder::Bool)
    n = length(p)
    ops = p.ops
    resize!(s.size, n)
    resize!(s.height, n)
    resize!(s.parent, n)
    empty!(s.stack)
    @inbounds for i in 1:n
        a = arity(ops[i])
        if a == 0
            s.size[i] = 1
            s.height[i] = 0
        elseif a == 1
            c = pop!(s.stack)
            s.size[i] = 1 + s.size[c]
            s.height[i] = s.height[c] + 1
            s.parent[c] = i
        else
            r = pop!(s.stack)
            l = pop!(s.stack)
            s.size[i] = 1 + s.size[l] + s.size[r]
            s.height[i] = max(s.height[l], s.height[r]) + 1
            s.parent[l] = i
            s.parent[r] = i
        end
        push!(s.stack, i)
    end
    if want_depth
        resize!(s.depth, n)
        s.depth[n] = 0
        @inbounds for i in n-1:-1:1
            s.depth[i] = s.depth[s.parent[i]] + 1  # el padre siempre va despues
        end
    end
    if want_preorder
        # Mismo orden que A: raiz, subarbol izquierdo, subarbol derecho.
        empty!(s.pre2post)
        empty!(s.stack)
        push!(s.stack, n)
        @inbounds while !isempty(s.stack)
            i = pop!(s.stack)
            push!(s.pre2post, i)
            a = arity(ops[i])
            if a == 1
                push!(s.stack, i - 1)
            elseif a == 2
                right = i - 1
                left = right - s.size[right]
                push!(s.stack, right)
                push!(s.stack, left)
            end
        end
    end
    return s
end

# base[1:start-1] + donante + base[stop+1:end]: la "clonacion + intercambio" del
# manual hecha en un solo paso con tres copias contiguas por arreglo.
function _splice(base::Program, start::Int, stop::Int,
                 donor::Program, dstart::Int, dlen::Int)
    n = length(base)
    tail = n - stop
    m = (start - 1) + dlen + tail
    ops = Vector{UInt8}(undef, m)
    vals = Vector{Float64}(undef, m)
    copyto!(ops, 1, base.ops, 1, start - 1)
    copyto!(ops, start, donor.ops, dstart, dlen)
    copyto!(ops, start + dlen, base.ops, stop + 1, tail)
    copyto!(vals, 1, base.vals, 1, start - 1)
    copyto!(vals, start, donor.vals, dstart, dlen)
    copyto!(vals, start + dlen, base.vals, stop + 1, tail)
    return Program(ops, vals)
end

_splice(base::Program, start::Int, stop::Int, donor::Scratch) =
    _splice(base, start, stop, Program(donor.ops, donor.vals), 1, length(donor.ops))

# ---------------------------------------------------------------------------
# Operadores geneticos
# ---------------------------------------------------------------------------
function crossover(a::Program, b::Program, rng::AbstractRNG; pick::Pick=FAST)
    na, nb = length(a), length(b)
    if pick == FAST
        ia = rand(rng, 1:na)
        ib = rand(rng, 1:nb)
        sa = subtree_start(a, ia)
        sb = subtree_start(b, ib)
    else
        A = _analyze!(SHAPE_A, a, false, true)
        Bs = _analyze!(SHAPE_B, b, false, true)
        ia = A.pre2post[rand(rng, 1:na)]
        ib = Bs.pre2post[rand(rng, 1:nb)]
        sa = ia - A.size[ia] + 1
        sb = ib - Bs.size[ib] + 1
    end
    return (_splice(a, sa, ia, b, sb, ib - sb + 1), _splice(b, sb, ib, a, sa, ia - sa + 1))
end

function mutation(parent::Program, no_variables::Integer, rng::AbstractRNG,
                  new_subtree_max_depth::Integer=3; pick::Pick=FAST)
    _check_depth(new_subtree_max_depth)
    # Mismo orden de consumo del generador que A: profundidad, subarbol nuevo, punto.
    d = rand(rng, 0:Int(new_subtree_max_depth))
    s = _clear!(SCRATCH)
    _grow!(s, d, Int(no_variables), rng, false)
    n = length(parent)
    if pick == FAST
        idx = rand(rng, 1:n)
        start = subtree_start(parent, idx)
    else
        A = _analyze!(SHAPE_A, parent, false, true)
        idx = A.pre2post[rand(rng, 1:n)]
        start = idx - A.size[idx] + 1
    end
    return _splice(parent, start, idx, s)
end

function depth_protected_crossover(a::Program, b::Program, max_depth::Integer,
                                   rng::AbstractRNG; pick::Pick=FAST)
    _check_depth(max_depth)
    pre = pick == PREORDER
    A = _analyze!(SHAPE_A, a, true, pre)
    Bs = _analyze!(SHAPE_B, b, true, pre)
    na, nb = length(a), length(b)
    (A.height[na] > max_depth || Bs.height[nb] > max_depth) &&
        throw(ArgumentError("Un padre supera max_depth"))
    # hijo 1 valido <=> depth_A(i) + height_B(j) <= max;  hijo 2: lo simetrico.
    for _ in 1:PROTECTED_ATTEMPTS
        ka = rand(rng, 1:na)
        kb = rand(rng, 1:nb)
        ia = pre ? A.pre2post[ka] : ka
        ib = pre ? Bs.pre2post[kb] : kb
        if A.depth[ia] + Bs.height[ib] <= max_depth && Bs.depth[ib] + A.height[ia] <= max_depth
            sa = ia - A.size[ia] + 1
            sb = ib - Bs.size[ib] + 1
            return (_splice(a, sa, ia, b, sb, ib - sb + 1),
                    _splice(b, sb, ib, a, sa, ia - sa + 1))
        end
    end
    return (clone(b), clone(a))  # igual que A y B: intercambio de raices
end

function depth_protected_mutation(parent::Program, no_variables::Integer, max_depth::Integer,
                                  rng::AbstractRNG; pick::Pick=FAST)
    _check_depth(max_depth)
    pre = pick == PREORDER
    A = _analyze!(SHAPE_A, parent, true, pre)
    n = length(parent)
    A.height[n] > max_depth && throw(ArgumentError("El padre supera max_depth"))
    k = rand(rng, 1:n)
    idx = pre ? A.pre2post[k] : k
    budget = Int(max_depth) - A.depth[idx]
    d = rand(rng, 0:budget)
    s = _clear!(SCRATCH)
    _grow!(s, d, Int(no_variables), rng, false)
    return _splice(parent, idx - A.size[idx] + 1, idx, s)
end

# ---------------------------------------------------------------------------
# GP completo (mismo algoritmo que A y B)
# ---------------------------------------------------------------------------
Base.@kwdef struct GPOptions
    population_size::Int = 40
    max_depth::Int = 5
    generations::Int = 5
    seed::Int = 42
    block::Int = 64
    pick::Pick = FAST
end

struct GPResult
    best::Program
    best_fitness::Float64
    best_fitness_per_generation::Vector{Float64}
end

struct Individual
    score::Float64
    tree::Program
end

_score(ind::Individual) = ind.score

function full_gp(data::Dataset, options::GPOptions)
    N = options.population_size
    (nvariables(data) > 0 && nrows(data) > 0) ||
        throw(ArgumentError("Se necesita un dataset con variables"))
    (N >= 4 && N % 4 == 0) ||
        throw(ArgumentError("population_size debe ser multiplo de 4 y >= 4"))
    _check_depth(options.max_depth)
    rng = Xoshiro(options.seed)
    nvars, max_depth, pick = nvariables(data), options.max_depth, options.pick
    ws = Workspace(options.block)

    population = Individual[]
    sizehint!(population, N)
    for _ in 1:N
        tree = grow_full(max_depth, nvars, rng)
        push!(population, Individual(fitness(tree, data, ws), tree))
    end

    history = Float64[]
    pool = Vector{Program}(undef, N)
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
            c1, c2 = depth_protected_crossover(p1, p2, max_depth, rng; pick=pick)
            push!(offspring, Individual(fitness(c1, data, ws), c1))
            push!(offspring, Individual(fitness(c2, data, ws), c2))
        end
        for _ in 1:(N ÷ 2)  # N/2 hijos por mutacion
            m = depth_protected_mutation(pool[rand(rng, 1:N)], nvars, max_depth, rng; pick=pick)
            push!(offspring, Individual(fitness(m, data, ws), m))
        end

        # Reemplazo 50/50 (orden estable: mismos empates que A).
        sort!(population; by=_score, alg=MergeSort)
        sort!(offspring; by=_score, alg=MergeSort)
        population = vcat(population[1:N÷2], offspring[1:N÷2])
    end
    best = population[argmin(map(_score, population))]
    return GPResult(clone(best.tree), best.score, history)
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
                             seed::Integer=42, batch_pairs::Integer=0, pick::Pick=FAST)
    (trees >= 2 && iseven(trees) && no_variables >= 1 && batch_pairs >= 0) ||
        throw(ArgumentError("trees debe ser par >= 2, variables >= 1 y batch_pairs >= 0"))
    _check_depth(depth)
    rng = Xoshiro(seed)
    pairs = trees ÷ 2
    per_batch = (batch_pairs == 0 || batch_pairs > pairs) ? pairs : Int(batch_pairs)
    measured = 0.0
    total_nodes = 0
    parents = Tuple{Program, Program}[]
    children = Tuple{Program, Program}[]
    sizehint!(children, CHILDREN_CHUNK)
    offset = 0
    while offset < pairs
        count = min(per_batch, pairs - offset)
        empty!(parents)
        for _ in 1:count
            p1 = grow_full(depth, no_variables, rng)
            p2 = grow_full(depth, no_variables, rng)
            total_nodes += length(p1) + length(p2)
            push!(parents, (p1, p2))
        end
        GC.gc()  # fuera del cronometro
        for start in 1:CHILDREN_CHUNK:count
            stop = min(count, start + CHILDREN_CHUNK - 1)
            t0 = time_ns()
            for i in start:stop
                push!(children, crossover(parents[i][1], parents[i][2], rng; pick=pick))
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

function run_tests(; verbose::Bool=false)
    passed = Ref(0)
    function check(condition::Bool, description::AbstractString)
        condition || error("Fallo en prueba: $description")
        passed[] += 1
        verbose && println("  ok  ", description)
    end
    rng = Xoshiro(42)

    # --- Parte 1: representacion, expression, eval, MSE ---------------------
    formula = Program([OP_VAR, OP_VAR, OP_CONST, OP_MUL, OP_ADD], [1.0, 2.0, 0.5, 0.0, 0.0])
    check(is_well_formed(formula), "programa bien formado")
    check(expression(formula) == "(x1 + (x2 * 0.5))", "expression")
    check(postfix(formula) == "x1 x2 0.5 * +", "postfix")
    check(eval_row(formula, [2.0, 3.0]) == 3.5, "eval_row (x1 + x2*0.5) en [2, 3] = 3.5")
    check(node_count(formula) == 5 && tree_depth(formula) == 2, "tamano y profundidad")
    check(subtree_start(formula, 4) == 2 && subtree_start(formula, 5) == 1, "subtree_start")
    check(!is_well_formed(Program([OP_VAR, OP_ADD], [1.0, 0.0])), "programa mal formado")

    exact = dataset_from_rows([[2.0, 3.0], [1.0, 2.0]], [3.5, 2.0])
    manual = dataset_from_rows([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.5, 1.0]], [2.5, 1.0, 3.5])
    check(fitness(formula, exact) == 0.0, "MSE cero")
    expected = ((2.5 - 2.0)^2 + (1.0 - 0.5)^2 + (3.5 - 2.25)^2) / 3
    check(isapprox(fitness(formula, manual), expected), "MSE de la tabla del manual")

    twin = clone(formula)
    twin.vals[3] = 0.7
    check(eval_row(formula, [2.0, 3.0]) == 3.5 && twin.ops !== formula.ops,
          "clonacion profunda")
    check(eval_row(parse_expression("(1 / 0)"), Float64[]) == 1e12, "division protegida")
    check(eval_row(parse_expression("sqrt(-5)"), Float64[]) == 0.0, "sqrt protegida")
    check(fitness(parse_expression("((x1 * 1e300) * 1e300)"), exact) == Inf,
          "MSE no finito = +Inf")
    check(postfix(parse_expression("(x1 + (x2 * 0.5))")) == postfix(formula), "parse_expression")
    check(expression(parse_expression("sqrt((x3 / -0.25))")) == "sqrt((x3 / -0.25))",
          "parse con constante negativa")
    check(_throws(() -> fitness(parse_expression("x3"), exact)), "variable fuera de rango")

    # --- Parte 2: generacion -----------------------------------------------
    for depth in 0:8
        full = grow_full(depth, 3, rng)
        rnd = grow_random(depth, 3, rng)
        check(is_well_formed(full) && tree_depth(full) == depth, "grow_full profundidad exacta")
        check(is_well_formed(rnd) && tree_depth(rnd) <= depth, "grow_random profundidad maxima")
        back = parse_expression(expression(full, 17))
        check(back.ops == full.ops && back.vals == full.vals,
              "ida y vuelta expression(17) -> parse_expression")
    end

    # Evaluacion por lotes == evaluacion escalar fila por fila, bit a bit.
    r2 = Xoshiro(7)
    xs = [[(i % 17 == 0) ? 0.0 : (6.0 * rand(r2) - 3.0) * 10.0^rand(r2, (-8, 0, 0, 3))
           for _ in 1:3] for i in 1:150]
    ys = [6.0 * rand(r2) - 3.0 for _ in 1:150]
    hard = dataset_from_rows(xs, ys)
    mismatches = 0
    for t in 1:300
        p = grow_random(1 + t % 9, 3, rng)
        ref = 0.0
        for r in 1:150
            d = ys[r] - eval_row(p, xs[r])
            ref += d * d
            isfinite(ref) || break
        end
        ref = isfinite(ref) ? ref / 150 : Inf
        for block in (1, 7, 64, 256)
            fitness(p, hard, Workspace(block)) === ref || (mismatches += 1)
        end
    end
    check(mismatches == 0, "evaluacion por lotes == escalar, bit a bit (300 arboles x 4 bloques)")

    for _ in 1:50
        p = grow_random(7, 2, rng)
        s = _analyze!(SHAPE_A, p, true, true)
        check(all(subtree_start(p, i) == i - s.size[i] + 1 for i in 1:length(p)),
              "subtree_start coincide con los tamanos")
        check(length(s.pre2post) == length(p) && s.pre2post[1] == length(p), "preorden")
        check(s.height[length(p)] == tree_depth(p), "altura de la raiz")
    end

    # --- Crossover y mutacion ---------------------------------------------
    a, b = grow_full(4, 2, rng), grow_full(3, 2, rng)
    old_a, old_b = expression(a, 17), expression(b, 17)
    for pick in (FAST, PREORDER), _ in 1:100
        c, d = crossover(a, b, rng; pick=pick)
        check(is_well_formed(c) && is_well_formed(d), "crossover produce arboles validos")
        check(length(c) + length(d) == length(a) + length(b), "crossover conserva nodos")
        check(is_well_formed(mutation(a, 2, rng; pick=pick)), "mutacion valida")
        x, y = depth_protected_crossover(a, b, 4, rng; pick=pick)
        check(tree_depth(x) <= 4 && tree_depth(y) <= 4, "crossover protegido")
        z = depth_protected_mutation(a, 2, 4, rng; pick=pick)
        check(is_well_formed(z) && tree_depth(z) <= 4, "mutacion protegida")
    end
    check(expression(a, 17) == old_a && expression(b, 17) == old_b,
          "los operadores no alteran a los padres")
    deep = grow_full(6, 2, rng)
    check(_throws(() -> depth_protected_crossover(deep, b, 4, rng)),
          "padre que supera max_depth es rechazado")

    # --- GP completo --------------------------------------------------------
    run = full_gp(exact, GPOptions(population_size=24, max_depth=3, generations=3))
    h = run.best_fitness_per_generation
    check(length(h) == 4, "historial de fitness")
    check(all(h[i] <= h[i-1] for i in 2:length(h)), "elitismo: el mejor MSE no aumenta")
    o1 = GPOptions(population_size=40, max_depth=5, generations=5, block=1)
    o2 = GPOptions(population_size=40, max_depth=5, generations=5, block=64)
    r1, r2b = full_gp(exact, o1), full_gp(exact, o2)
    check(r1.best_fitness_per_generation == r2b.best_fitness_per_generation &&
          expression(r1.best, 17) == expression(r2b.best, 17), "el bloque no altera el GP")

    t = benchmark_crossover(40, 4, 2; seed=1)
    check(t.crossovers == 20 && t.resident_trees == 40 && t.seconds > 0, "benchmark de crossover")
    return passed[]
end

end # module
