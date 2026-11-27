#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace std; 
//unico cambio, para no tener std::cout y diversas cosas

namespace gp {

constexpr double kEpsilon = 1e-12;

enum class Kind :  uint8_t {
    Constant, Variable, Add, Subtract, Multiply, Divide, Sqrt
};

namespace detail {

// Asignador propio para los nodos. Los arboles siguen siendo CONVENCIONALES (cada
// nodo es un objeto independiente enlazado por punteros y clonar copia nodo por
// nodo); solo cambia de donde sale la memoria. En vez de pedir y liberar cada nodo
// al heap del sistema (lento, sobre todo en Windows), se reservan bloques grandes y
// los nodos liberados se reutilizan (lista libre LIFO: los ultimos liberados, aun en
// cache, son los primeros en reutilizarse). La memoria reservada no se devuelve al
// sistema hasta que termina el programa. Un solo hilo (el torneo no permite paralelismo).
class NodePool {
public:
    static constexpr  size_t kBlockBytes = 32;

    static void* allocate() {
        if (free_list_ != nullptr) {
            FreeBlock* block = free_list_;
            free_list_ = block->next;
            return block;
        }
        if (next_ == end_) refill();
        void* block = next_;
        next_ += kBlockBytes;
        return block;
    }

    static void deallocate(void* p) noexcept {
        auto* block = static_cast<FreeBlock*>(p);
        block->next = free_list_;
        free_list_ = block;
    }

private:
    struct FreeBlock {
        FreeBlock* next;
    };
    static constexpr  size_t kBlocksPerChunk =  size_t{1} << 16;  // 2 MB por bloque

    static void refill() {
        next_ = static_cast< byte*>(::operator new(kBlockBytes * kBlocksPerChunk));
        end_ = next_ + kBlockBytes * kBlocksPerChunk;
    }

    inline static FreeBlock* free_list_ = nullptr;
    inline static  byte* next_ = nullptr;
    inline static  byte* end_ = nullptr;
};

} // namespace detail

// B: arbol CONVENCIONAL con nodos independientes y punteros propietarios.
// La raiz se encuentra a profundidad cero.
struct Node {
    Kind kind = Kind::Constant;
    // true si el valor del nodo se LEE directamente, sin recorrer hijos: constantes,
    // variables y subarboles sin variables ya plegados (su valor queda en `value`).
    // Lo recalcula fitness en cada evaluacion; ocupa un byte del relleno (el nodo sigue
    // midiendo 32 bytes).
    bool direct = true;
    // Un nodo es constante O variable, nunca ambas: la union deja el nodo en 32 bytes
    // (antes 40), asi que hay menos memoria que copiar al clonar y caben mas nodos en cache.
    // Los nodos internos no usan la union salvo para guardar su valor plegado.
    union {
        double value = 0.0;          // Kind::Constant (o valor plegado de un nodo interno)
         size_t variable_index;  // Kind::Variable (0 -> x1)
    };
     unique_ptr<Node> left;
     unique_ptr<Node> right;

    // new/delete de Node (los usan make_unique y unique_ptr) van al asignador propio.
    static void* operator new( size_t size) {
        return size == sizeof(Node) ? detail::NodePool::allocate() : ::operator new(size);
    }
    static void operator delete(void* p,  size_t size) noexcept {
        if (size == sizeof(Node)) {
            detail::NodePool::deallocate(p);
        } else {
            ::operator delete(p);
        }
    }

    // Copia el valor o el indice (la union completa, sin importar el tipo de nodo).
    void copy_payload_from(const Node& other) noexcept {
        direct = other.direct;
         memcpy(&value, &other.value, sizeof(value));
    }

    static  unique_ptr<Node> constant(double value);
    static  unique_ptr<Node> variable( size_t index);
    static  unique_ptr<Node> unary(Kind kind,  unique_ptr<Node> child);
    static  unique_ptr<Node> binary(Kind kind,  unique_ptr<Node> a,
                                         unique_ptr<Node> b);

    double eval( span<const double> variables) const;
    // precision=8: formato habitual; precision=17: sin perdida (intercambio con A y C).
     string expression(int precision = 8) const;
};

static_assert(sizeof(Node) <= detail::NodePool::kBlockBytes, "Node no cabe en el bloque del asignador");
static_assert(sizeof(double) == sizeof( size_t), "la union value/variable_index asume 8 bytes");

using Tree =  unique_ptr<Node>;

struct Dataset {
     vector< vector<double>> x;
     vector<double> y;
     size_t variables() const;
    static Dataset from_csv(const  string& filename);
};

// Lee el formato de expression(): "((x1 + -0.25) * sqrt(x2))".
Tree parse_expression(const  string& text);

Tree clone(const Node& root);
 size_t node_count(const Node& root);
int tree_depth(const Node& root); // Una hoja sola tiene profundidad 0.

Tree grow_random(int max_depth,  size_t variable_count,
                  mt19937_64& rng);
Tree grow_full(int exact_depth,  size_t variable_count,
                mt19937_64& rng);

// MSE fila por fila. fold_constants = true pliega los subarboles sin variables antes de
// evaluar (mismo resultado bit a bit). fitness no cambia el significado del arbol: solo
// actualiza los campos `direct` y el valor plegado de sus nodos.
double fitness(const Node& root, const Dataset& data, bool fold_constants = true);

 pair<Tree, Tree> crossover(const Node& a, const Node& b,
                                 mt19937_64& rng);
Tree mutation(const Node& parent,  size_t variable_count,
               mt19937_64& rng, int new_subtree_max_depth = 3);

 pair<Tree, Tree> depth_protected_crossover(
    const Node& a, const Node& b, int max_depth,  mt19937_64& rng);
Tree depth_protected_mutation(const Node& parent,  size_t variable_count,
                              int max_depth,  mt19937_64& rng);

struct GPOptions {
     size_t population_size = 40; // Para probar; el benchmark pide 5000.
    int max_depth = 5;
     size_t generations = 5; // El manual NO fija generaciones.
     uint64_t seed = 42;
    // Optimizaciones exactas (el resultado del GP no cambia); --no-fold y --no-cut las apagan.
    bool fold_constants = true; // plegar subarboles sin variables antes de evaluar
    bool early_cut = true;      // dejar de evaluar un hijo que ya no puede quedar en el mejor 50%
    // Evaluar primero los hijos de mejores padres (--no-order lo apaga): el umbral del corte
    // temprano baja antes y se corta mas. Tampoco cambia el resultado.
    bool order_by_parent = true;
    // Un hijo identico a su padre (crossover que intercambia dos subarboles iguales, mutacion
    // que pone un subarbol igual al que quita, o intercambio de raices) recibe el MSE del
    // padre sin evaluarse (--no-dup lo apaga). Mismo arbol => mismo MSE: no cambia el GP.
    bool skip_duplicates = true;
    // Se llama al terminar cada generacion (0 = poblacion inicial) con su mejor MSE, para
    // irlo mostrando en vivo. No toca el generador aleatorio: el GP da lo mismo con o sin el.
     function<void( size_t generation, double best_mse)> on_generation;
};

struct GPResult {
    Tree best;
    double best_fitness = 0.0;
     vector<double> best_fitness_per_generation;
};

// La inicializacion es 100% grow_full; variacion y reemplazo 50/50.
// population_size debe ser divisible entre 4 para un 50/50 exacto.
GPResult full_gp(const Dataset& data, const GPOptions& options);

struct CrossoverTiming {
    int depth = 0;
     size_t trees = 0;
     size_t resident_trees = 0; // padres en memoria al mismo tiempo
     size_t crossovers = 0;
    double seconds = 0.0;
    double mean_seconds_per_crossover = 0.0;
    double mean_nodes_per_tree = 0.0;
};

// Protocolo acordado con el profesor: los 'trees' padres (grow_full) estan en
// memoria AL MISMO TIEMPO. batch_pairs = 0 significa todos (trees/2 parejas);
// si no hay RAM suficiente, batch_pairs = 2500 deja 5000 arboles a la vez.
// El cronometro mide SOLO crossover (clonacion + intercambio). Los hijos se
// liberan cada pocos cruces, fuera del intervalo cronometrado.
CrossoverTiming benchmark_crossover( size_t trees, int depth,
                                      size_t variables,
                                      uint64_t seed = 42,
                                      size_t batch_pairs = 0);

// Pruebas deterministas de todas las piezas importantes.
void run_tests();

} // namespace gp
