#!/usr/bin/env julia
# Interfaz de terminal de la implementacion A (mismas opciones que B y C).
#
#   julia main.jl --test [-v]
#   julia main.jl --demo [--data=data/ejemplo.csv]
#   julia main.jl --bench-crossover [--trees=10000] [--depths=7,9,10,12,15]
#                                   [--variables=2] [--seed=42] [--batch-pairs=0]
#   julia main.jl --bench-gp [--data=dataset.csv] [--population=5000]
#                            [--depths=7,9,12] [--generations=10] [--seed=42]
#   julia main.jl --eval-file --file=arboles.txt [--data=dataset.csv]
#        Una expresion por linea -> MSE(17 digitos)<TAB>profundidad<TAB>nodos<TAB>expresion
#   julia main.jl --export-trees [--count=100] [--depth=5] [--variables=2] [--seed=42] [--full]
#
# Antes de cronometrar se ejecuta una corrida pequena de calentamiento, para que
# el tiempo de compilacion JIT de Julia no se cuente en los benchmarks.

include(joinpath(@__DIR__, "src", "GPA.jl"))
using .GPA
using Printf
using Random

const HELP = """
GP A - Julia, implementacion convencional de alto nivel

  julia main.jl --test [-v]
  julia main.jl --demo [--data=data/ejemplo.csv]
  julia main.jl --bench-crossover [--trees=10000] [--depths=7,9,10,12,15]
                                  [--variables=2] [--seed=42] [--batch-pairs=0]
  julia main.jl --bench-gp [--data=dataset.csv] [--population=5000]
                           [--depths=7,9,12] [--generations=10] [--seed=42]
  julia main.jl --eval-file --file=arboles.txt [--data=dataset.csv]
  julia main.jl --export-trees [--count=100] [--depth=5] [--variables=2] [--seed=42] [--full]

--batch-pairs: parejas de padres en memoria a la vez en el benchmark de crossover
               (0 = los 10000 arboles a la vez, protocolo acordado; 2500 = 5000).
El manual NO fija el numero de generaciones (10 por defecto, igual que B y C).
"""

const INT_OPTIONS = Set(["trees", "population", "generations", "variables", "seed",
                         "batch-pairs", "count", "depth"])
const STRING_OPTIONS = Set(["data", "depths", "file"])

function parse_args(args)
    mode = isempty(args) ? "--help" : args[1]
    o = Dict{String, Any}(
        "data" => joinpath(@__DIR__, "data", "ejemplo.csv"), "depths" => "", "file" => "",
        "trees" => 10000, "population" => 5000, "generations" => 10, "variables" => 2,
        "seed" => 42, "batch-pairs" => 0, "count" => 100, "depth" => 5,
        "full" => false, "verbose" => false)
    for arg in args[2:end]
        if arg == "--full"
            o["full"] = true
        elseif arg == "-v"
            o["verbose"] = true
        elseif startswith(arg, "--") && occursin('=', arg)
            key, value = split(arg[3:end], '='; limit=2)
            if key in INT_OPTIONS
                n = tryparse(Int, value)
                (n === nothing || n < 0) && error("--$key requiere un entero no negativo: $value")
                o[key] = n
            elseif key in STRING_OPTIONS
                o[key] = String(value)
            else
                error("Opcion desconocida: --$key")
            end
        else
            error("Se esperaba --opcion=valor: $arg")
        end
    end
    return mode, o
end

parse_depths(text, default) = [parse(Int, strip(s)) for s in split(isempty(text) ? default : text, ',')
                               if !isempty(strip(s))]

function main(args)
    mode, o = parse_args(args)
    if mode in ("--help", "-h")
        print(HELP)
    elseif mode == "--test"
        count = run_tests(; verbose=o["verbose"])
        println("OK: $count comprobaciones correctas.")
    elseif mode == "--demo"
        data = read_csv(o["data"])
        result = full_gp(data, GPOptions(population_size=40, max_depth=4, generations=5,
                                         seed=o["seed"]))
        @printf("MSE final: %.6g\n", result.best_fitness)
        println("Expresion: ", expression(result.best))
        println("Mejor MSE por generacion:")
        for (i, v) in enumerate(result.best_fitness_per_generation)
            @printf("  Generacion %d: %.6g\n", i - 1, v)
        end
    elseif mode == "--bench-crossover"
        benchmark_crossover(40, 3, o["variables"]; seed=1)  # calentamiento (JIT)
        println("depth,trees,resident_trees,crossovers,elapsed_seconds,",
                "mean_seconds_per_crossover,mean_nodes")
        for depth in parse_depths(o["depths"], "7,9,10,12,15")
            t = benchmark_crossover(o["trees"], depth, o["variables"]; seed=o["seed"],
                                    batch_pairs=o["batch-pairs"])
            @printf("%d,%d,%d,%d,%.9g,%.9g,%.6g\n", t.depth, t.trees, t.resident_trees,
                    t.crossovers, t.seconds, t.mean_seconds_per_crossover, t.mean_nodes_per_tree)
            flush(stdout)
        end
    elseif mode == "--bench-gp"
        data = read_csv(o["data"])
        full_gp(data, GPOptions(population_size=8, max_depth=3, generations=1))  # calentamiento
        println("depth,population,generations,elapsed_seconds,best_mse,best_expression")
        for depth in parse_depths(o["depths"], "7,9,12")
            options = GPOptions(population_size=o["population"], max_depth=depth,
                                generations=o["generations"], seed=o["seed"])
            t0 = time_ns()
            result = full_gp(data, options)
            elapsed = (time_ns() - t0) / 1e9
            @printf("%d,%d,%d,%.9g,%.9g,\"%s\"\n", depth, options.population_size,
                    options.generations, elapsed, result.best_fitness, expression(result.best))
            flush(stdout)
        end
    elseif mode == "--eval-file"
        isempty(o["file"]) && error("Falta --file=arboles.txt")
        data = read_csv(o["data"])
        for line in eachline(o["file"])
            line = rstrip(line, '\r')
            isempty(strip(line)) && continue
            tree = parse_expression(line)
            println(format_number(fitness(tree, data), 17), '\t', tree_depth(tree), '\t',
                    node_count(tree), '\t', line)
        end
    elseif mode == "--export-trees"
        rng = Xoshiro(o["seed"])
        grow = o["full"] ? grow_full : grow_random
        for _ in 1:o["count"]
            println(expression(grow(o["depth"], o["variables"], rng), 17))
        end
    else
        error("Modo no reconocido: $mode")
    end
    return 0
end

try
    main(ARGS)
catch err
    err isa InterruptException && rethrow()
    println(stderr, "ERROR: ", sprint(showerror, err))
    exit(1)
end
