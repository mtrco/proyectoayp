#include "gpc.hpp"

#include <chrono>
#include <exception>
#include <fstream>
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
    std::string file;
    std::size_t trees = 10000;
    std::size_t population = 5000;
    std::size_t generations = 10;  // El manual no especifica este valor.
    std::size_t variables = 2;
    std::size_t block = 64;
    std::size_t batch_pairs = 4;
    std::size_t count = 100;
    int depth = 5;
    std::uint64_t seed = 42;
    bool preorder = false;
};

Args parse_args(int argc, char** argv) {
    Args opts;
    if (argc > 1) opts.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--preorder") { opts.preorder = true; continue; }
        const std::size_t equal = arg.find('=');
        if (equal == std::string::npos) throw std::invalid_argument("Se esperaba --opcion=valor: " + arg);
        const std::string key = arg.substr(0, equal);
        const std::string value = arg.substr(equal + 1);
        if (key == "--data") opts.data = value;
        else if (key == "--depths") opts.depths = value;
        else if (key == "--file") opts.file = value;
        else if (key == "--trees") opts.trees = std::stoull(value);
        else if (key == "--population") opts.population = std::stoull(value);
        else if (key == "--generations") opts.generations = std::stoull(value);
        else if (key == "--variables") opts.variables = std::stoull(value);
        else if (key == "--block") opts.block = std::stoull(value);
        else if (key == "--batch-pairs") opts.batch_pairs = std::stoull(value);
        else if (key == "--count") opts.count = std::stoull(value);
        else if (key == "--depth") opts.depth = std::stoi(value);
        else if (key == "--seed") opts.seed = std::stoull(value);
        else throw std::invalid_argument("Opcion desconocida: " + key);
    }
    return opts;
}

std::vector<int> parse_depths(const std::string& list) {
    std::vector<int> result;
    std::stringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ',')) result.push_back(std::stoi(item));
    if (result.empty()) throw std::invalid_argument("Lista de profundidades vacia");
    return result;
}

void print_help() {
    std::cout << R"HELP(GP C - C++20, implementacion alternativa (arreglo postfijo + evaluacion por lotes)

  gp_c --test
  gp_c --demo [--data=data/ejemplo.csv]
  gp_c --bench-crossover [--trees=10000] [--depths=7,9,10,12,15] [--variables=2]
                         [--seed=42] [--batch-pairs=4] [--preorder]
  gp_c --bench-gp [--data=dataset.csv] [--population=5000] [--depths=7,9,12]
                  [--generations=10] [--seed=42] [--block=64] [--preorder]
  gp_c --eval-file --file=arboles.txt [--data=dataset.csv]
       Lee una expresion por linea (formato de B: "(x1 + sqrt(x2))") y escribe
       su MSE con 17 digitos. Sirve para comparar con A y B.
  gp_c --export-trees [--count=100] [--depth=5] [--variables=2] [--seed=42]
       Escribe arboles grow_random sin perdida (17 digitos), uno por linea.

--batch-pairs: parejas generadas por lote en el benchmark de crossover
               (4 = mismo protocolo que B; 5000 = los 10000 arboles a la vez).
--block:       filas del dataset por lote en la evaluacion.
--preorder:    elige nodos en preorden como B (misma distribucion; con la misma
               semilla reproduce exactamente los resultados de B).
)HELP";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const gpc::Pick pick = args.preorder ? gpc::Pick::Preorder : gpc::Pick::Fast;
        if (args.mode == "--help" || args.mode == "-h") {
            print_help();
        } else if (args.mode == "--test") {
            gpc::run_tests();
            std::cout << "OK: todas las pruebas terminaron correctamente.\n";
        } else if (args.mode == "--demo") {
            const gpc::Dataset data = gpc::Dataset::from_csv(args.data);
            gpc::GPOptions options;
            options.population_size = 40;
            options.max_depth = 4;
            options.generations = 5;
            options.seed = args.seed;
            options.pick = pick;
            const gpc::GPResult result = gpc::full_gp(data, options);
            std::cout << "MSE final: " << result.best_fitness << '\n';
            std::cout << "Expresion: " << result.best.expression() << '\n';
            std::cout << "Postfijo:  " << result.best.postfix(8) << '\n';
            std::cout << "Mejor MSE por generacion:\n";
            for (std::size_t i = 0; i < result.best_fitness_per_generation.size(); ++i) {
                std::cout << "  Generacion " << i << ": " << result.best_fitness_per_generation[i] << '\n';
            }
        } else if (args.mode == "--bench-crossover") {
            const auto depths = parse_depths(args.depths.empty() ? "7,9,10,12,15" : args.depths);
            std::cout << "depth,trees,crossovers,elapsed_seconds,mean_seconds_per_crossover,mean_nodes\n";
            for (int depth : depths) {
                const auto t = gpc::benchmark_crossover(args.trees, depth, args.variables, args.seed,
                                                        args.batch_pairs, pick);
                std::cout << t.depth << ',' << t.trees << ',' << t.crossovers << ','
                          << std::setprecision(9) << t.seconds << ',' << t.mean_seconds_per_crossover
                          << ',' << std::setprecision(6) << t.mean_nodes_per_tree << '\n';
            }
        } else if (args.mode == "--bench-gp") {
            const gpc::Dataset data = gpc::Dataset::from_csv(args.data);
            const auto depths = parse_depths(args.depths.empty() ? "7,9,12" : args.depths);
            std::cout << "depth,population,generations,elapsed_seconds,best_mse,best_expression\n";
            for (int depth : depths) {
                gpc::GPOptions opts;
                opts.population_size = args.population;
                opts.max_depth = depth;
                opts.generations = args.generations;
                opts.seed = args.seed;
                opts.block = args.block;
                opts.pick = pick;
                const auto start = std::chrono::steady_clock::now();
                const auto result = gpc::full_gp(data, opts);
                const auto stop = std::chrono::steady_clock::now();
                std::cout << depth << ',' << opts.population_size << ',' << opts.generations << ','
                          << std::setprecision(9)
                          << std::chrono::duration<double>(stop - start).count() << ','
                          << result.best_fitness << ",\"" << result.best.expression() << "\"\n";
            }
        } else if (args.mode == "--eval-file") {
            if (args.file.empty()) throw std::invalid_argument("Falta --file=arboles.txt");
            const gpc::Dataset data = gpc::Dataset::from_csv(args.data);
            std::ifstream in(args.file);
            if (!in) throw std::runtime_error("No se pudo abrir: " + args.file);
            std::string line;
            std::cout << std::setprecision(17);
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.find_first_not_of(" \t") == std::string::npos) continue;
                const gpc::Program p = gpc::parse_infix(line);
                std::cout << gpc::fitness(p, data) << '\t' << gpc::tree_depth(p) << '\t'
                          << p.size() << '\t' << line << '\n';
            }
        } else if (args.mode == "--export-trees") {
            std::mt19937_64 rng(args.seed);
            for (std::size_t i = 0; i < args.count; ++i) {
                std::cout << gpc::grow_random(args.depth, args.variables, rng).expression(17) << '\n';
            }
        } else {
            throw std::invalid_argument("Modo no reconocido: " + args.mode);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
