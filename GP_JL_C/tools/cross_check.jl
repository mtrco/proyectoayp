#!/usr/bin/env julia
# Verificacion cruzada A <-> C en un solo proceso de Julia.
#
#   julia tools/cross_check.jl [ruta/a/GP_JL_A] [dataset.csv]
#
# Por defecto busca A en ../GP_JL_A (carpetas hermanas) y usa data/ejemplo.csv.
# A y C consumen el generador aleatorio en el mismo orden, asi que con la misma
# semilla deben producir EXACTAMENTE los mismos arboles y, con C en modo
# PREORDER, los mismos hijos de crossover/mutacion y la misma corrida del GP.
# Comprueba:
#   1. Mismos arboles (grow_full y grow_random) para 40 semillas, profundidades
#      0-9 y 1-4 variables, y el mismo estado final del generador.
#   2. Conversion A -> texto -> C -> texto sin perdida; mismos nodos y profundidad.
#   3. evaluate/eval_row y MSE identicos bit a bit en tres datasets (ejemplo, uno
#      con ceros, negativos y magnitudes extremas y uno que desborda), y para
#      varios tamanos de bloque de C.
#   4. crossover, mutation y sus versiones protegidas: mismos hijos (C PREORDER).
#   5. GP completo: mismo historial del mejor MSE y mismo mejor arbol (C PREORDER).

const ROOT_C = normpath(joinpath(@__DIR__, ".."))
const ROOT_A = length(ARGS) >= 1 ? ARGS[1] : normpath(joinpath(ROOT_C, "..", "GP_JL_A"))
const DATA = length(ARGS) >= 2 ? ARGS[2] : joinpath(ROOT_C, "data", "ejemplo.csv")

include(joinpath(ROOT_A, "src", "GPA.jl"))
include(joinpath(ROOT_C, "src", "GPC.jl"))
using Random
const A = GPA
const C = GPC

const checks = Ref(0)
const failures = Ref(0)
function check(condition::Bool, description::AbstractString)
    checks[] += 1
    if !condition
        failures[] += 1
        failures[] <= 20 && println("  FALLO: ", description)
    end
    return condition
end

# --- datasets ---------------------------------------------------------------
function synthetic(seed, rows, scale_choices)
    rng = Xoshiro(seed)
    xs = [[(i + k) % 7 == 0 ? 0.0 : (4.0 * rand(rng) - 2.0) * 10.0^rand(rng, scale_choices)
           for k in 1:4] for i in 1:rows]
    ys = [10.0 * rand(rng) - 5.0 for _ in 1:rows]
    return xs, ys
end

function load_rows(path)
    d = A.read_csv(path)
    return d.rows, d.y
end

datasets = [("principal", load_rows(DATA)...),
            ("extremos", synthetic(2026, 97, (-13, -6, 0, 0, 4))...),
            ("desborde", synthetic(2027, 40, (0, 80, 160))...)]

println("Verificacion cruzada A (", ROOT_A, ") <-> C (", ROOT_C, ")")

# --- 1. generacion ----------------------------------------------------------
for seed in 1:40, depth in 0:9, nv in 1:4
    ra, rc = Xoshiro(seed), Xoshiro(seed)
    for _ in 1:3
        check(A.expression(A.grow_full(depth, nv, ra), 17) ==
              C.expression(C.grow_full(depth, nv, rc), 17),
              "grow_full seed=$seed depth=$depth vars=$nv")
        check(A.expression(A.grow_random(depth, nv, ra), 17) ==
              C.expression(C.grow_random(depth, nv, rc), 17),
              "grow_random seed=$seed depth=$depth vars=$nv")
    end
    check(rand(ra, UInt64) == rand(rc, UInt64), "mismo consumo del generador seed=$seed")
end
println("[1] generacion: ", checks[], " comprobaciones, ", failures[], " fallos")

# --- 2 y 3. texto, evaluacion y MSE ------------------------------------------
before = checks[]
rng = Xoshiro(99)
for (name, xs, ys) in datasets
    nv = length(xs[1])
    da = A.Dataset(xs, ys)
    dc = C.dataset_from_rows(xs, ys)
    workspaces = [C.Workspace(b) for b in (1, 7, 64, 256)]
    infinite = 0
    for t in 1:400
        tree = isodd(t) ? A.grow_random(t % 11, rand(rng, 1:nv), rng) :
                          A.grow_full(t % 9, rand(rng, 1:nv), rng)
        text = A.expression(tree, 17)
        p = C.parse_expression(text)
        check(C.expression(p, 17) == text, "[$name] texto sin perdida")
        check(C.node_count(p) == A.node_count(tree) && C.tree_depth(p) == A.tree_depth(tree),
              "[$name] mismos nodos y profundidad")
        check(A.expression(A.parse_expression(C.expression(p, 17)), 17) == text,
              "[$name] C -> texto -> A")
        for r in 1:min(length(xs), 25)
            check(A.evaluate(tree, xs[r]) === C.eval_row(p, xs[r]), "[$name] eval fila $r")
        end
        mse = A.fitness(tree, da)
        infinite += isinf(mse)
        for ws in workspaces
            check(mse === C.fitness(p, dc, ws), "[$name] MSE bit a bit, bloque=$(ws.block)")
        end
    end
    println("[2-3] $name: 400 arboles, $infinite con MSE = +Inf")
end
println("[2-3] texto, eval y MSE: ", checks[] - before, " comprobaciones, ", failures[], " fallos")

# --- 4. operadores geneticos (C en modo PREORDER) ------------------------------
same(x, y) = A.expression(x, 17) == C.expression(y, 17)
before = checks[]
for seed in 1:60
    ra, rc = Xoshiro(seed), Xoshiro(seed)
    depth = 2 + seed % 6
    a1, a2 = A.grow_full(depth, 2, ra), A.grow_random(depth, 2, ra)
    c1, c2 = C.grow_full(depth, 2, rc), C.grow_random(depth, 2, rc)
    check(same(a1, c1) && same(a2, c2), "padres iguales seed=$seed")
    for _ in 1:5
        x1, x2 = A.crossover(a1, a2, ra)
        y1, y2 = C.crossover(c1, c2, rc; pick=C.PREORDER)
        check(same(x1, y1) && same(x2, y2), "crossover seed=$seed")
        check(same(A.mutation(a1, 2, ra), C.mutation(c1, 2, rc; pick=C.PREORDER)),
              "mutation seed=$seed")
        md = depth + 1
        x1, x2 = A.depth_protected_crossover(a1, a2, md, ra)
        y1, y2 = C.depth_protected_crossover(c1, c2, md, rc; pick=C.PREORDER)
        check(same(x1, y1) && same(x2, y2), "crossover protegido seed=$seed")
        check(same(A.depth_protected_mutation(a1, 2, md, ra),
                   C.depth_protected_mutation(c1, 2, md, rc; pick=C.PREORDER)),
              "mutacion protegida seed=$seed")
    end
    check(rand(ra, UInt64) == rand(rc, UInt64), "mismo consumo del generador (operadores)")
end
println("[4] operadores: ", checks[] - before, " comprobaciones, ", failures[], " fallos")

# --- 5. GP completo -------------------------------------------------------------
before = checks[]
xs, ys = load_rows(DATA)
da, dc = A.Dataset(xs, ys), C.dataset_from_rows(xs, ys)
for seed in 1:4, depth in (3, 5, 7)
    ra = A.full_gp(da, A.GPOptions(population_size=200, max_depth=depth, generations=6, seed=seed))
    rc = C.full_gp(dc, C.GPOptions(population_size=200, max_depth=depth, generations=6, seed=seed,
                                   pick=C.PREORDER))
    check(ra.best_fitness_per_generation == rc.best_fitness_per_generation,
          "GP historial seed=$seed depth=$depth")
    check(A.expression(ra.best, 17) == C.expression(rc.best, 17) &&
          ra.best_fitness === rc.best_fitness, "GP mejor arbol seed=$seed depth=$depth")
    fast = C.full_gp(dc, C.GPOptions(population_size=200, max_depth=depth, generations=6,
                                     seed=seed))
    check(C.tree_depth(fast.best) <= depth && issorted(fast.best_fitness_per_generation; rev=true),
          "GP modo FAST valido seed=$seed depth=$depth")
end
println("[5] GP completo: ", checks[] - before, " comprobaciones, ", failures[], " fallos")

println("\nComprobaciones: ", checks[], ", fallos: ", failures[])
if failures[] == 0
    println("OK: A y C se comportan de forma identica.")
else
    exit(1)
end
