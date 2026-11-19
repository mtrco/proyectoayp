#pragma once
// Programacion Genetica - Implementacion C (alternativa, no convencional), C++20.
//
// Representacion: cada individuo es un ARREGLO PLANO en notacion POSTFIJA
// (postorden: hijo izquierdo, hijo derecho, operador). No hay punteros entre
// nodos. Los datos se guardan como "struct of arrays" en UN solo bloque de
// memoria contiguo:  [ valores (n doubles) | codigos de operacion (n bytes) ]
//
//   x1 + (x2 * 0.5)   ->   ops:  Var  Var  Const  Mul  Add
//                          vals: 0    1    0.5    -    -
//
// Un subarbol siempre ocupa un SEGMENTO CONTIGUO del arreglo que termina en su
// raiz. Por eso clonar es un memcpy y el crossover/mutacion es pegar segmentos.
//
// La evaluacion es POR LOTES: el programa se recorre una vez por bloque de filas
// del dataset y cada instruccion opera sobre el bloque completo (columnas),
// lo que amortiza el costo de interpretar y permite vectorizacion (SIMD).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gpc {

constexpr double kEpsilon = 1e-12;

enum class Op : std::uint8_t { Constant, Variable, Add, Subtract, Multiply, Divide, Sqrt };

constexpr int arity(Op op) {
    switch (op) {
        case Op::Constant:
        case Op::Variable: return 0;
        case Op::Sqrt: return 1;
        default: return 2;
    }
}

// Arreglo postfijo con un unico buffer. Para Variable, value guarda el indice.
class Program {
public:
    Program() = default;
    explicit Program(std::size_t n);  // contenido sin inicializar
    Program(const Program& other);    // clonacion profunda: un solo memcpy
    Program& operator=(const Program& other);
    Program(Program&&) noexcept = default;
    Program& operator=(Program&&) noexcept = default;

    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    double* values() { return reinterpret_cast<double*>(buf_.get()); }
    const double* values() const { return reinterpret_cast<const double*>(buf_.get()); }
    Op* ops() { return reinterpret_cast<Op*>(buf_.get() + n_ * sizeof(double)); }
    const Op* ops() const { return reinterpret_cast<const Op*>(buf_.get() + n_ * sizeof(double)); }

    // Representacion textual identica a la de B: "(x1 + (x2 * 0.5))".
    // precision=8 reproduce el formato de B; precision=17 es sin perdida
    // (formato de intercambio para la verificacion cruzada).
    std::string expression(int precision = 8) const;
    std::string postfix(int precision = 17) const;  // "x1 x2 0.5 * +"

private:
    std::unique_ptr<std::byte[]> buf_;
    std::size_t n_ = 0;
};

// Dataset por COLUMNAS: columns[v][fila].
struct Dataset {
    std::vector<std::vector<double>> columns;
    std::vector<double> y;
    std::size_t rows() const { return y.size(); }
    std::size_t variables() const { return columns.size(); }
    static Dataset from_csv(const std::string& filename);
    static Dataset from_rows(const std::vector<std::vector<double>>& x,
                             const std::vector<double>& y);
};

// Construccion e inspeccion
Program make_program(std::span<const Op> ops, std::span<const double> values);
Program parse_infix(const std::string& text);  // lee el formato de expression()
bool is_well_formed(const Program& p);         // disciplina de pila correcta
std::size_t node_count(const Program& p);
int tree_depth(const Program& p);              // una hoja sola = profundidad 0
std::size_t subtree_start(const Program& p, std::size_t root);

// Generacion (mismas reglas y mismo consumo del generador aleatorio que B)
Program grow_random(int max_depth, std::size_t variable_count, std::mt19937_64& rng);
Program grow_full(int exact_depth, std::size_t variable_count, std::mt19937_64& rng);

// Evaluacion
double eval_row(const Program& p, std::span<const double> variables);  // referencia escalar

struct EvalWorkspace {
    std::size_t block = 64;     // filas por lote
    std::vector<double> stack;  // pila de vectores: (profundidad de pila) x block
};
double fitness(const Program& p, const Dataset& data, EvalWorkspace& ws);
double fitness(const Program& p, const Dataset& data);

// Seleccion de nodos:
//  Fast     -> indice uniforme directamente en el arreglo postfijo (mas rapido).
//  Preorder -> indice uniforme en PREORDEN, igual que B. Misma distribucion que
//              Fast, pero consume el RNG exactamente como B, de modo que con la
//              misma semilla B y C producen los MISMOS arboles (verificacion).
enum class Pick { Fast, Preorder };

std::pair<Program, Program> crossover(const Program& a, const Program& b,
                                      std::mt19937_64& rng, Pick pick = Pick::Fast);
Program mutation(const Program& parent, std::size_t variable_count,
                 std::mt19937_64& rng, int new_subtree_max_depth = 3,
                 Pick pick = Pick::Fast);
std::pair<Program, Program> depth_protected_crossover(
    const Program& a, const Program& b, int max_depth, std::mt19937_64& rng,
    Pick pick = Pick::Fast);
Program depth_protected_mutation(const Program& parent, std::size_t variable_count,
                                 int max_depth, std::mt19937_64& rng,
                                 Pick pick = Pick::Fast);

struct GPOptions {
    std::size_t population_size = 40;
    int max_depth = 5;
    std::size_t generations = 5;
    std::uint64_t seed = 42;
    std::size_t block = 64;
    Pick pick = Pick::Fast;
};

struct GPResult {
    Program best;
    double best_fitness = 0.0;
    std::vector<double> best_fitness_per_generation;
};

// Mismo algoritmo que B: 100% grow_full, torneo binario repetido para el pool,
// 50% descendientes por crossover / 50% por mutacion, reemplazo 50/50.
GPResult full_gp(const Dataset& data, const GPOptions& options);

struct CrossoverTiming {
    int depth = 0;
    std::size_t trees = 0;
    std::size_t crossovers = 0;
    double seconds = 0.0;
    double mean_seconds_per_crossover = 0.0;
    double mean_nodes_per_tree = 0.0;
};

// Mismo protocolo que B: padres grow_full en lotes de batch_pairs parejas; se
// cronometra SOLO el crossover (incluida la creacion de los hijos).
CrossoverTiming benchmark_crossover(std::size_t trees, int depth, std::size_t variables,
                                    std::uint64_t seed = 42, std::size_t batch_pairs = 4,
                                    Pick pick = Pick::Fast);

void run_tests();

}  // namespace gpc
