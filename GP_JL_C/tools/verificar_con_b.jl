#!/usr/bin/env julia
# Verificacion cruzada A, C (Julia) <-> B (C++) con el formato de texto comun.
#
#   julia tools/verificar_con_b.jl ruta\a\gp_b.exe [ruta/a/GP_JL_A] [dataset.csv]
#
# Los generadores aleatorios de Julia y de C++ son distintos, asi que aqui no se
# comparan corridas con la misma semilla, sino EL MISMO ARBOL evaluado por las tres:
#   1. B genera arboles (--export-trees) y A y C los generan en Julia.
#   2. B evalua todos con --eval-file (MSE en 17 digitos, profundidad, nodos).
#   3. A y C evaluan los mismos arboles; se compara el MSE bit a bit, la
#      profundidad y el numero de nodos, en tres datasets (el principal, uno con
#      ceros, negativos y magnitudes extremas y uno que desborda).

length(ARGS) >= 1 || (println("Uso: julia tools/verificar_con_b.jl gp_b.exe [GP_JL_A] [dataset.csv]"); exit(2))
const GP_B = ARGS[1]
const ROOT_C = normpath(joinpath(@__DIR__, ".."))
const ROOT_A = length(ARGS) >= 2 ? ARGS[2] : normpath(joinpath(ROOT_C, "..", "GP_JL_A"))
const DATA = length(ARGS) >= 3 ? ARGS[3] : joinpath(ROOT_C, "data", "ejemplo.csv")

include(joinpath(ROOT_A, "src", "GPA.jl"))
include(joinpath(ROOT_C, "src", "GPC.jl"))
using Random
const A = GPA
const C = GPC

parse_mse(s) = s == "inf" ? Inf : s == "-inf" ? -Inf : s in ("nan", "-nan") ? NaN : parse(Float64, s)

function write_csv(path, xs, ys)
    open(path, "w") do io
        println(io, join(["x$i" for i in 1:length(xs[1])], ',') * ",y")
        for (x, y) in zip(xs, ys)
            println(io, join([repr(v) for v in x], ',') * "," * repr(y))
        end
    end
end

function synthetic(seed, rows, scale_choices)
    rng = Xoshiro(seed)
    xs = [[(i + k) % 7 == 0 ? 0.0 : (4.0 * rand(rng) - 2.0) * 10.0^rand(rng, scale_choices)
           for k in 1:4] for i in 1:rows]
    ys = [10.0 * rand(rng) - 5.0 for _ in 1:rows]
    return xs, ys
end

function main()
    help = read(`$GP_B --help`, String)
    if !occursin("--export-trees", help) || !occursin("--eval-file", help)
        println("Este gp_b no tiene --export-trees / --eval-file. Recompila B (x64-Release) con")
        println("la version actual de src/main.cpp, src/gp.cpp e include/gp.hpp y vuelve a correr esto.")
        exit(2)
    end
    folder = mktempdir()
    paths = [("principal", DATA)]
    for (name, seed, rows, scales) in (("extremos", 2026, 97, (-13, -6, 0, 0, 4)),
                                       ("desborde", 2027, 40, (0, 80, 160)))
        path = joinpath(folder, "$name.csv")
        write_csv(path, synthetic(seed, rows, scales)...)
        push!(paths, (name, path))
    end

    total = 0
    failures = 0
    for (name, path) in paths
        da = A.read_csv(path)
        dc = C.read_csv(path)
        nv = A.nvariables(da)
        trees = String[]
        for seed in 1:3, depth in (0, 3, 6, 9), v in 1:nv
            out = read(`$GP_B --export-trees --count=20 --depth=$depth --variables=$v --seed=$seed`, String)
            append!(trees, filter(!isempty, split(out, '\n')))
        end
        rng = Xoshiro(1234)
        for depth in 0:10, v in 1:nv, _ in 1:4
            push!(trees, A.expression(A.grow_random(depth, v, rng), 17))
            push!(trees, C.expression(C.grow_full(min(depth, 8), v, rng), 17))
        end
        file = joinpath(folder, "arboles_$name.txt")
        write(file, join(trees, '\n') * "\n")
        lines = filter(!isempty, split(read(`$GP_B --eval-file --file=$file --data=$path`, String), '\n'))
        if length(lines) != length(trees)
            println("[$name] B devolvio $(length(lines)) lineas; se esperaban $(length(trees))")
            exit(1)
        end
        infinite = 0
        for line in lines
            fields = split(rstrip(line, '\r'), '\t'; limit=4)
            b_mse, b_depth, b_size, text = parse_mse(fields[1]), parse(Int, fields[2]),
                                           parse(Int, fields[3]), fields[4]
            tree = A.parse_expression(text)
            p = C.parse_expression(text)
            ok = A.expression(tree, 17) == text && C.expression(p, 17) == text
            ok &= A.tree_depth(tree) == b_depth && C.tree_depth(p) == b_depth
            ok &= A.node_count(tree) == b_size && C.node_count(p) == b_size
            ok &= A.fitness(tree, da) === b_mse && C.fitness(p, dc) === b_mse
            infinite += isinf(b_mse)
            total += 1
            if !ok
                failures += 1
                failures <= 10 && println("[$name] DIFERENCIA con B: $line\n   A: ",
                    A.format_number(A.fitness(tree, da), 17), "  C: ",
                    C.format_number(C.fitness(p, dc), 17))
            end
        end
        println("[$name] $(length(lines)) arboles comparados, $infinite con MSE = +Inf")
    end
    println("\nTotal: $total arboles comparados contra B, $failures diferencias.")
    failures == 0 ? println("OK: A, B y C evaluan igual.") : exit(1)
end

main()
