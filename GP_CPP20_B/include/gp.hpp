#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gp {

constexpr double kEpsilon = 1e-12;

enum class Kind : std::uint8_t {
    Constant, Variable, Add, Subtract, Multiply, Divide, Sqrt
};

// B: arbol CONVENCIONAL con nodos independientes y punteros propietarios.
// La raiz se encuentra a profundidad cero.
struct Node {
    Kind kind = Kind::Constant;
    double value = 0.0;
    std::size_t variable_index = 0;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    static std::unique_ptr<Node> constant(double value);
    static std::unique_ptr<Node> variable(std::size_t index);
    static std::unique_ptr<Node> unary(Kind kind, std::unique_ptr<Node> child);
    static std::unique_ptr<Node> binary(Kind kind, std::unique_ptr<Node> a,
                                        std::unique_ptr<Node> b);

    double eval(std::span<const double> variables) const;
    std::string expression() const;
};

using Tree = std::unique_ptr<Node>;

struct Dataset {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    std::size_t variables() const;
    static Dataset from_csv(const std::string& filename);
};

Tree clone(const Node& root);
std::size_t node_count(const Node& root);
int tree_depth(const Node& root); // Una hoja sola tiene profundidad 0.

Tree grow_random(int max_depth, std::size_t variable_count,
                 std::mt19937_64& rng);
Tree grow_full(int exact_depth, std::size_t variable_count,
               std::mt19937_64& rng);

double fitness(const Node& root, const Dataset& data);

std::pair<Tree, Tree> crossover(const Node& a, const Node& b,
                                std::mt19937_64& rng);
Tree mutation(const Node& parent, std::size_t variable_count,
              std::mt19937_64& rng, int new_subtree_max_depth = 3);

std::pair<Tree, Tree> depth_protected_crossover(
    const Node& a, const Node& b, int max_depth, std::mt19937_64& rng);
Tree depth_protected_mutation(const Node& parent, std::size_t variable_count,
                              int max_depth, std::mt19937_64& rng);

struct GPOptions {
    std::size_t population_size = 40; // Para probar; el benchmark pide 5000.
    int max_depth = 5;
    std::size_t generations = 5; // El manual NO fija generaciones.
    std::uint64_t seed = 42;
};

struct GPResult {
    Tree best;
    double best_fitness = 0.0;
    std::vector<double> best_fitness_per_generation;
};

// La inicializacion es 100% grow_full; variacion y reemplazo 50/50.
// population_size debe ser divisible entre 4 para un 50/50 exacto.
GPResult full_gp(const Dataset& data, const GPOptions& options);

struct CrossoverTiming {
    int depth = 0;
    std::size_t trees = 0;
    std::size_t crossovers = 0;
    double seconds = 0.0;
    double mean_seconds_per_crossover = 0.0;
};

// Genera 'trees' padres con grow_full en lotes para limitar RAM.
// El cronometro mide SOLO crossover, no la generacion de padres.
CrossoverTiming benchmark_crossover(std::size_t trees, int depth,
                                     std::size_t variables,
                                     std::uint64_t seed = 42);

// Pruebas deterministas de todas las piezas importantes.
void run_tests();

} // namespace gp
