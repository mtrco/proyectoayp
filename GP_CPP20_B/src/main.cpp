#include "gp.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Args {
    std::string mode = "--help";
    std::string data = "data/ejemplo.csv";
    std::string depths;
    std::size_t trees = 10000;
    std::size_t population = 5000;
    std::size_t generations = 10; // El manual no especifica este valor.
    std::size_t variables = 2;
    std::uint64_t seed = 42;
};

Args parse_args(int argc, char** argv) {
    Args opts;
    if (argc > 1) opts.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const std::size_t equal = arg.find('=');
        if (equal == std::string::npos) {
            throw std::invalid_argument("Se esperaba --opcion=valor: " + arg);
        }
        const std::string key = arg.substr(0, equal);
        const std::string value = arg.substr(equal + 1);
        if (key == "--data") opts.data = value;
        else if (key == "--depths") opts.depths = value;
        else if (key == "--trees") opts.trees = std::stoull(value);
        else if (key == "--population") opts.population = std::stoull(value);
        else if (key == "--generations") opts.generations = std::stoull(value);
        else if (key == "--variables") opts.variables = std::stoull(value);
        else if (key == "--seed") opts.seed = std::stoull(value);
        else throw std::invalid_argument("Opcion desconocida: " + key);
    }
    return opts;
}

std::vector<int> parse_depths(std::string list) {
    std::vector<int> result;
    std::stringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ',')) {
        result.push_back(std::stoi(item));
    }
    if (result.empty()) throw std::invalid_argument("Lista de profundidades vacia");
    return result;
}

void print_help() {
    std::cout << R"(GP B - C++20, implementacion convencional

  gp_b --test
  gp_b --demo [--data=data/ejemplo.csv]
  gp_b --bench-crossover [--trees=10000] [--depths=7,9,10,12,15]
                          [--variables=2] [--seed=42]
  gp_b --bench-gp [--data=dataset.csv] [--population=5000]
                  [--depths=7,9,12] [--generations=10] [--seed=42]

El CSV debe tener cabecera opcional, x1,x2,...,y, y como ultima columna.
El manual NO especifica el numero de generaciones; acordarlo con el docente.
En el benchmark de crossover se mide solo el crossover, no generar padres.
Se generan 10000 arboles EN LOTES para no llenar la memoria RAM.
)";
}

void run_gp_benchmark(const gp::Dataset& data, const std::vector<int>& depths,
                      const Args& args) {
    std::cout << "depth,population,generations,elapsed_seconds,best_mse,best_expression\n";
    for (int depth : depths) {
        gp::GPOptions opts;
        opts.population_size = args.population;
        opts.max_depth = depth;
        opts.generations = args.generations;
        opts.seed = args.seed;
        const auto start = std::chrono::steady_clock::now();
        auto result = gp::full_gp(data, opts);
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        std::cout << depth << ',' << opts.population_size << ',' << opts.generations << ','
                  << std::setprecision(9) << seconds << ',' << result.best_fitness << ",\""
                  << result.best->expression() << "\"\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.mode == "--help" || args.mode == "-h") {
            print_help();
        } else if (args.mode == "--test") {
            gp::run_tests();
            std::cout << "OK: todas las pruebas terminaron correctamente.\n";
        } else if (args.mode == "--demo") {
            const gp::Dataset data = gp::Dataset::from_csv(args.data);
            gp::GPOptions options;
            options.population_size = 40;
            options.max_depth = 4;
            options.generations = 5;
            options.seed = args.seed;
            gp::GPResult result = gp::full_gp(data, options);
            std::cout << "MSE final: " << result.best_fitness << '\n';
            std::cout << "Expresion: " << result.best->expression() << '\n';
            std::cout << "Mejor MSE por generacion:\n";
            for (std::size_t i = 0; i < result.best_fitness_per_generation.size(); ++i) {
                std::cout << "  Generacion " << i << ": "
                          << result.best_fitness_per_generation[i] << '\n';
            }
        } else if (args.mode == "--bench-crossover") {
            const auto depths = parse_depths(args.depths.empty()
                ? "7,9,10,12,15" : args.depths);
            std::cout << "depth,trees,crossovers,elapsed_seconds,mean_seconds_per_crossover\n";
            for (int depth : depths) {
                const auto time = gp::benchmark_crossover(args.trees, depth,
                                                            args.variables, args.seed);
                std::cout << time.depth << ',' << time.trees << ',' << time.crossovers
                          << ',' << std::setprecision(9) << time.seconds << ','
                          << time.mean_seconds_per_crossover << '\n';
            }
        } else if (args.mode == "--bench-gp") {
            const gp::Dataset data = gp::Dataset::from_csv(args.data);
            const auto depths = parse_depths(args.depths.empty() ? "7,9,12" : args.depths);
            run_gp_benchmark(data, depths, args);
        } else {
            throw std::invalid_argument("Modo no reconocido: " + args.mode);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
