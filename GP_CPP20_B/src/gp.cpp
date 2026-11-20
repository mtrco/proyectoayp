#include "gp.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gp {
namespace {

template <typename Int>
Int random_integer(std::mt19937_64& rng, Int min_value, Int max_value) {
    std::uniform_int_distribution<Int> dist(min_value, max_value);
    return dist(rng);
}

double random_constant(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

Kind random_operator(std::mt19937_64& rng) {
    switch (random_integer<int>(rng, 0, 4)) {
        case 0: return Kind::Add;
        case 1: return Kind::Subtract;
        case 2: return Kind::Multiply;
        case 3: return Kind::Divide;
        default: return Kind::Sqrt;
    }
}

Tree random_leaf(std::size_t variable_count, std::mt19937_64& rng) {
    if (variable_count == 0 || random_integer<int>(rng, 0, 1) == 0) {
        return Node::constant(random_constant(rng));
    }
    return Node::variable(random_integer<std::size_t>(rng, 0, variable_count - 1));
}

Tree grow_impl(int remaining_depth, std::size_t variable_count,
               std::mt19937_64& rng, bool full) {
    if (remaining_depth == 0 || (!full && random_integer<int>(rng, 0, 3) == 0)) {
        return random_leaf(variable_count, rng);
    }
    const Kind kind = random_operator(rng);
    Tree left = grow_impl(remaining_depth - 1, variable_count, rng, full);
    if (kind == Kind::Sqrt) return Node::unary(kind, std::move(left));
    Tree right = grow_impl(remaining_depth - 1, variable_count, rng, full);
    return Node::binary(kind, std::move(left), std::move(right));
}

// Clona `src` en `dest` y, en la MISMA pasada, registra en preorden (raiz,
// izquierdo, derecho) cada lugar donde cuelga un subarbol del clon. Antes esto
// eran tres recorridos por arbol (clone, node_count para reservar y collect_slots).
void clone_collect(Tree& dest, const Node& src, std::vector<Tree*>& slots) {
    dest = std::make_unique<Node>();
    dest->kind = src.kind;
    dest->value = src.value;
    dest->variable_index = src.variable_index;
    slots.push_back(&dest);
    if (src.left) clone_collect(dest->left, *src.left, slots);
    if (src.right) clone_collect(dest->right, *src.right, slots);
}

// Slot con la profundidad del nodo que cuelga ahi (raiz = 0) y la altura de su
// subarbol (hoja = 0). Con ambos datos, cada intento del crossover protegido se
// valida en O(1) en lugar de recorrer los dos hijos completos.
struct ShapedSlot {
    Tree* slot;
    int depth;
    int height;
};

// Igual que clone_collect, pero guarda profundidad y altura. Devuelve la altura de src.
int clone_shape(Tree& dest, const Node& src, int depth, std::vector<ShapedSlot>& slots) {
    dest = std::make_unique<Node>();
    dest->kind = src.kind;
    dest->value = src.value;
    dest->variable_index = src.variable_index;
    const std::size_t index = slots.size();
    slots.push_back({&dest, depth, 0});
    int height = 0;
    if (src.left) height = clone_shape(dest->left, *src.left, depth + 1, slots) + 1;
    if (src.right) {
        height = std::max(height, clone_shape(dest->right, *src.right, depth + 1, slots) + 1);
    }
    slots[index].height = height;
    return height;
}

Tree* choose_slot(std::vector<Tree*>& slots, std::mt19937_64& rng) {
    return slots[random_integer<std::size_t>(rng, 0, slots.size() - 1)];
}

void ensure_depth_argument(int depth) {
    if (depth < 0) throw std::invalid_argument("La profundidad debe ser >= 0");
}

std::string double_to_string(double value, int precision) {
    std::ostringstream out;
    out << std::setprecision(precision) << value;
    return out.str();
}

// Lector del formato de expression(): "(a op b)", "sqrt(a)", "x3", "-0.25".
class InfixParser {
public:
    explicit InfixParser(const std::string& text) : text_(text) {}
    Tree run() {
        Tree tree = parse();
        skip_spaces();
        if (pos_ != text_.size()) fail("texto sobrante");
        return tree;
    }

private:
    void skip_spaces() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t')) ++pos_;
    }
    [[noreturn]] void fail(const std::string& why) const {
        throw std::invalid_argument("Expresion invalida (" + why + ") en posicion "
                                    + std::to_string(pos_));
    }
    void expect(char c) {
        skip_spaces();
        if (pos_ >= text_.size() || text_[pos_] != c) fail(std::string("se esperaba '") + c + "'");
        ++pos_;
    }
    Tree parse() {
        skip_spaces();
        if (pos_ >= text_.size()) fail("fin inesperado");
        const char c = text_[pos_];
        if (c == '(') {
            ++pos_;
            Tree left = parse();
            skip_spaces();
            if (pos_ >= text_.size()) fail("falta operador");
            Kind kind = Kind::Add;
            switch (text_[pos_]) {
                case '+': kind = Kind::Add; break;
                case '-': kind = Kind::Subtract; break;
                case '*': kind = Kind::Multiply; break;
                case '/': kind = Kind::Divide; break;
                default: fail("operador desconocido");
            }
            ++pos_;
            Tree right = parse();
            expect(')');
            return Node::binary(kind, std::move(left), std::move(right));
        }
        if (text_.compare(pos_, 5, "sqrt(") == 0) {
            pos_ += 5;
            Tree child = parse();
            expect(')');
            return Node::unary(Kind::Sqrt, std::move(child));
        }
        if (c == 'x') {
            ++pos_;
            const std::size_t begin = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (begin == pos_) fail("variable sin indice");
            const unsigned long index = std::stoul(text_.substr(begin, pos_ - begin));
            if (index == 0) fail("las variables empiezan en x1");
            return Node::variable(index - 1);
        }
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) fail("numero invalido");
        pos_ += static_cast<std::size_t>(end - begin);
        return Node::constant(value);
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

std::vector<double> parse_numeric_row(const std::string& line) {
    std::vector<double> result;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        const std::size_t first = field.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) throw std::invalid_argument("Campo CSV vacio");
        const std::size_t last = field.find_last_not_of(" \t\r\n");
        field = field.substr(first, last - first + 1);
        std::size_t pos = 0;
        double value = 0.0;
        try {
            value = std::stod(field, &pos);
        } catch (const std::exception&) {
            throw std::invalid_argument("Numero CSV invalido: " + field);
        }
        if (pos != field.size() || !std::isfinite(value)) {
            throw std::invalid_argument("Numero CSV invalido: " + field);
        }
        result.push_back(value);
    }
    // getline no reporta un ultimo campo vacio ("1,2,"): se detecta aparte.
    const std::size_t last = line.find_last_not_of(" \t\r\n");
    if (last != std::string::npos && line[last] == ',') {
        throw std::invalid_argument("Campo CSV vacio al final de la linea");
    }
    return result;
}

bool is_number_field(const std::string& raw) {
    const std::size_t first = raw.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return false;
    const std::size_t last = raw.find_last_not_of(" \t\r\n");
    const std::string field = raw.substr(first, last - first + 1);
    try {
        std::size_t pos = 0;
        std::stod(field, &pos);
        return pos == field.size();
    } catch (const std::exception&) {
        return false;
    }
}

// La primera linea se toma como encabezado solo si NINGUN campo es numerico
// (por ejemplo "x1,x2,y"). Una primera fila numerica mal formada es un error.
bool looks_like_header(const std::string& line) {
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        const std::string field = line.substr(start, comma == std::string::npos
                                                         ? std::string::npos : comma - start);
        if (is_number_field(field)) return false;
        if (comma == std::string::npos) return true;
        start = comma + 1;
    }
}

struct Individual {
    Tree tree;
    double score;
};

bool better(const Individual& a, const Individual& b) {
    return a.score < b.score;
}

} // namespace

std::unique_ptr<Node> Node::constant(double val) {
    auto node = std::make_unique<Node>();
    node->kind = Kind::Constant;
    node->value = val;
    return node;
}

std::unique_ptr<Node> Node::variable(std::size_t index) {
    auto node = std::make_unique<Node>();
    node->kind = Kind::Variable;
    node->variable_index = index;
    return node;
}

std::unique_ptr<Node> Node::unary(Kind kind, Tree child) {
    if (kind != Kind::Sqrt || !child) {
        throw std::invalid_argument("Operador unario invalido");
    }
    auto node = std::make_unique<Node>();
    node->kind = kind;
    node->left = std::move(child);
    return node;
}

std::unique_ptr<Node> Node::binary(Kind kind, Tree a, Tree b) {
    if (kind == Kind::Sqrt || kind == Kind::Constant || kind == Kind::Variable
        || !a || !b) {
        throw std::invalid_argument("Operador binario invalido");
    }
    auto node = std::make_unique<Node>();
    node->kind = kind;
    node->left = std::move(a);
    node->right = std::move(b);
    return node;
}

double Node::eval(std::span<const double> variables) const {
    switch (kind) {
        case Kind::Constant: return value;
        case Kind::Variable:
            if (variable_index >= variables.size()) {
                throw std::out_of_range("Indice de variable fuera del vector");
            }
            return variables[variable_index];
        case Kind::Sqrt: return std::sqrt(std::max(left->eval(variables), 0.0));
        case Kind::Add: return left->eval(variables) + right->eval(variables);
        case Kind::Subtract: return left->eval(variables) - right->eval(variables);
        case Kind::Multiply: return left->eval(variables) * right->eval(variables);
        case Kind::Divide: {
            const double a = left->eval(variables);
            const double b = right->eval(variables);
            return a / (std::abs(b) < kEpsilon ? kEpsilon : b);
        }
    }
    throw std::logic_error("Nodo desconocido");
}

std::string Node::expression(int precision) const {
    switch (kind) {
        case Kind::Constant: return double_to_string(value, precision);
        case Kind::Variable: return "x" + std::to_string(variable_index + 1);
        case Kind::Sqrt: return "sqrt(" + left->expression(precision) + ")";
        default: break;
    }
    const char* operation = "?";
    switch (kind) {
        case Kind::Add: operation = "+"; break;
        case Kind::Subtract: operation = "-"; break;
        case Kind::Multiply: operation = "*"; break;
        case Kind::Divide: operation = "/"; break;
        default: break;
    }
    return "(" + left->expression(precision) + " " + operation + " "
           + right->expression(precision) + ")";
}

Tree parse_expression(const std::string& text) { return InfixParser(text).run(); }

std::size_t Dataset::variables() const {
    return x.empty() ? 0 : x.front().size();
}

Dataset Dataset::from_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("No se pudo abrir: " + filename);
    Dataset result;
    std::string line;
    std::size_t line_number = 0;
    std::size_t columns = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        try {
            std::vector<double> values = parse_numeric_row(line);
            if (values.size() < 2) throw std::invalid_argument("Faltan columnas");
            if (columns == 0) columns = values.size();
            if (values.size() != columns) throw std::invalid_argument("Columnas inconsistentes");
            result.y.push_back(values.back());
            values.pop_back();
            result.x.push_back(std::move(values));
        } catch (const std::exception& e) {
            // Permitir un encabezado en la primera linea, por ejemplo: x1,x2,y.
            if (line_number == 1 && looks_like_header(line)) continue;
            throw std::runtime_error("CSV linea " + std::to_string(line_number)
                                     + ": " + e.what());
        }
    }
    if (result.x.empty()) throw std::runtime_error("Dataset sin filas numericas");
    return result;
}

Tree clone(const Node& root) {
    auto result = std::make_unique<Node>();
    result->kind = root.kind;
    result->value = root.value;
    result->variable_index = root.variable_index;
    if (root.left) result->left = clone(*root.left);
    if (root.right) result->right = clone(*root.right);
    return result;
}

std::size_t node_count(const Node& root) {
    return 1 + (root.left ? node_count(*root.left) : 0)
             + (root.right ? node_count(*root.right) : 0);
}

int tree_depth(const Node& root) {
    const int left = root.left ? tree_depth(*root.left) + 1 : 0;
    const int right = root.right ? tree_depth(*root.right) + 1 : 0;
    return std::max(left, right);
}

Tree grow_random(int max_depth, std::size_t variable_count, std::mt19937_64& rng) {
    ensure_depth_argument(max_depth);
    return grow_impl(max_depth, variable_count, rng, false);
}

Tree grow_full(int exact_depth, std::size_t variable_count, std::mt19937_64& rng) {
    ensure_depth_argument(exact_depth);
    return grow_impl(exact_depth, variable_count, rng, true);
}

double fitness(const Node& root, const Dataset& data) {
    if (data.x.empty() || data.x.size() != data.y.size()) {
        throw std::invalid_argument("Dataset vacio o inconsistente");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < data.x.size(); ++i) {
        if (data.x[i].size() != data.variables()) {
            throw std::invalid_argument("Dimensiones inconsistentes del dataset");
        }
        const double predicted = root.eval(data.x[i]);
        const double difference = data.y[i] - predicted;
        const double squared = difference * difference;
        if (!std::isfinite(squared)) return std::numeric_limits<double>::infinity();
        sum += squared;
        if (!std::isfinite(sum)) return std::numeric_limits<double>::infinity();
    }
    return sum / static_cast<double>(data.x.size());
}

std::pair<Tree, Tree> crossover(const Node& a, const Node& b,
                                std::mt19937_64& rng) {
    // Buffers reutilizados entre llamadas: sin reservar memoria para los slots.
    thread_local std::vector<Tree*> slots_a;
    thread_local std::vector<Tree*> slots_b;
    slots_a.clear();
    slots_b.clear();
    Tree child_a;
    Tree child_b;
    clone_collect(child_a, a, slots_a);
    clone_collect(child_b, b, slots_b);
    // Orden explicito: el orden de evaluacion de argumentos no esta definido en C++.
    Tree* selected_a = choose_slot(slots_a, rng);
    Tree* selected_b = choose_slot(slots_b, rng);
    std::swap(*selected_a, *selected_b);
    return {std::move(child_a), std::move(child_b)};
}

Tree mutation(const Node& parent, std::size_t variable_count,
              std::mt19937_64& rng, int new_subtree_max_depth) {
    ensure_depth_argument(new_subtree_max_depth);
    thread_local std::vector<Tree*> slots;
    slots.clear();
    Tree result;
    clone_collect(result, parent, slots);
    const int replacement_depth = random_integer<int>(rng, 0, new_subtree_max_depth);
    Tree replacement = grow_random(replacement_depth, variable_count, rng);
    *choose_slot(slots, rng) = std::move(replacement);
    return result;
}

std::pair<Tree, Tree> depth_protected_crossover(
    const Node& a, const Node& b, int max_depth, std::mt19937_64& rng) {
    ensure_depth_argument(max_depth);
    thread_local std::vector<ShapedSlot> slots_a;
    thread_local std::vector<ShapedSlot> slots_b;
    slots_a.clear();
    slots_b.clear();
    Tree child_a;
    Tree child_b;
    const int height_a = clone_shape(child_a, a, 0, slots_a);
    const int height_b = clone_shape(child_b, b, 0, slots_b);
    if (height_a > max_depth || height_b > max_depth) {
        throw std::invalid_argument("Un padre supera max_depth");
    }

    // Cambiar el subarbol i de A por el j de B deja al hijo A con profundidad
    // max(resto de A, prof_A(i) + altura_B(j)); el resto de A ya cabe porque el
    // padre cabe. Por eso esta condicion equivale EXACTAMENTE a intercambiar y
    // medir tree_depth de ambos hijos (lo que se hacia antes): mismos intentos,
    // mismo consumo del generador y mismos hijos, sin recorrer los arboles.
    for (int attempt = 0; attempt < 128; ++attempt) {
        const ShapedSlot& x = slots_a[random_integer<std::size_t>(rng, 0, slots_a.size() - 1)];
        const ShapedSlot& y = slots_b[random_integer<std::size_t>(rng, 0, slots_b.size() - 1)];
        if (x.depth + y.height <= max_depth && y.depth + x.height <= max_depth) {
            std::swap(*x.slot, *y.slot);
            return {std::move(child_a), std::move(child_b)};
        }
    }
    // Intercambiar las raices completas siempre respeta el limite de ambos padres.
    std::swap(child_a, child_b);
    return {std::move(child_a), std::move(child_b)};
}

Tree depth_protected_mutation(const Node& parent, std::size_t variable_count,
                              int max_depth, std::mt19937_64& rng) {
    ensure_depth_argument(max_depth);
    thread_local std::vector<ShapedSlot> slots;
    slots.clear();
    Tree result;
    if (clone_shape(result, parent, 0, slots) > max_depth) {
        throw std::invalid_argument("El padre supera max_depth");
    }
    const ShapedSlot& selected = slots[random_integer<std::size_t>(rng, 0, slots.size() - 1)];
    const int remaining = max_depth - selected.depth;
    const int replacement_depth = random_integer<int>(rng, 0, remaining);
    *selected.slot = grow_random(replacement_depth, variable_count, rng);
    return result;
}

GPResult full_gp(const Dataset& data, const GPOptions& options) {
    if (data.variables() == 0 || data.x.empty()) {
        throw std::invalid_argument("Se necesita un dataset con variables");
    }
    if (options.population_size < 4 || options.population_size % 4 != 0) {
        throw std::invalid_argument("population_size debe ser multiplo de 4 y >= 4");
    }
    ensure_depth_argument(options.max_depth);
    std::mt19937_64 rng(options.seed);
    const std::size_t N = options.population_size;

    std::vector<Individual> population;
    population.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        Tree tree = grow_full(options.max_depth, data.variables(), rng);
        double score = fitness(*tree, data);
        population.push_back({std::move(tree), score});
    }

    GPResult result;
    for (std::size_t generation = 0; generation <= options.generations; ++generation) {
        const Individual& best = *std::min_element(population.begin(), population.end(), better);
        result.best_fitness_per_generation.push_back(best.score);
        if (generation == options.generations) break;

        // Torneo binario repetido: llenar un pool de N punteros a los ganadores.
        std::vector<const Node*> parent_pool;
        parent_pool.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            const auto& a = population[random_integer<std::size_t>(rng, 0, N - 1)];
            const auto& b = population[random_integer<std::size_t>(rng, 0, N - 1)];
            parent_pool.push_back((a.score <= b.score ? a.tree : b.tree).get());
        }
        const auto pick_parent = [&]() -> const Node& {
            return *parent_pool[random_integer<std::size_t>(rng, 0, N - 1)];
        };

        std::vector<Individual> offspring;
        offspring.reserve(N);
        // N/4 cruces x 2 hijos = N/2 descendientes por crossover.
        for (std::size_t i = 0; i < N / 4; ++i) {
            const Node& p1 = pick_parent(); // orden explicito de seleccion
            const Node& p2 = pick_parent();
            auto [a, b] = depth_protected_crossover(p1, p2, options.max_depth, rng);
            const double fa = fitness(*a, data);
            const double fb = fitness(*b, data);
            offspring.push_back({std::move(a), fa});
            offspring.push_back({std::move(b), fb});
        }
        // N/2 descendientes por mutacion = 50% del total.
        for (std::size_t i = 0; i < N / 2; ++i) {
            Tree mutated = depth_protected_mutation(pick_parent(), data.variables(),
                                                     options.max_depth, rng);
            const double score = fitness(*mutated, data);
            offspring.push_back({std::move(mutated), score});
        }

        const auto keep_half = [](std::vector<Individual>& group) {
            const auto midpoint = group.begin() + static_cast<std::ptrdiff_t>(group.size() / 2);
            std::nth_element(group.begin(), midpoint, group.end(), better);
            group.erase(midpoint, group.end());
        };
        keep_half(population);
        keep_half(offspring);
        for (auto& individual : offspring) population.push_back(std::move(individual));
        assert(population.size() == N);
    }
    const Individual& best = *std::min_element(population.begin(), population.end(), better);
    result.best = clone(*best.tree);
    result.best_fitness = best.score;
    return result;
}

CrossoverTiming benchmark_crossover(std::size_t trees, int depth,
                                     std::size_t variables, std::uint64_t seed,
                                     std::size_t batch_pairs) {
    if (trees < 2 || trees % 2 != 0 || variables == 0) {
        throw std::invalid_argument("trees debe ser par >=2 y variables >=1");
    }
    ensure_depth_argument(depth);
    std::mt19937_64 rng(seed);
    using clock = std::chrono::steady_clock;
    double measured_seconds = 0.0;
    double total_nodes = 0.0;
    const std::size_t pairs = trees / 2;
    const std::size_t per_batch = (batch_pairs == 0 || batch_pairs > pairs) ? pairs : batch_pairs;
    // Hijos que se guardan antes de liberarlos (fuera del cronometro). Asi la RAM
    // la ocupan solo los padres y el costo del reloj se reparte entre 16 cruces.
    constexpr std::size_t kChildrenChunk = 16;

    std::vector<std::pair<Tree, Tree>> parents;
    std::vector<std::pair<Tree, Tree>> children;
    parents.reserve(per_batch);
    children.reserve(kChildrenChunk);
    for (std::size_t offset = 0; offset < pairs; offset += per_batch) {
        const std::size_t count = std::min(per_batch, pairs - offset);
        parents.clear(); // libera el lote anterior (fuera del cronometro)
        for (std::size_t i = 0; i < count; ++i) {
            Tree first = grow_full(depth, variables, rng); // orden explicito
            Tree second = grow_full(depth, variables, rng);
            total_nodes += static_cast<double>(node_count(*first) + node_count(*second));
            parents.emplace_back(std::move(first), std::move(second));
        }
        for (std::size_t begin = 0; begin < count; begin += kChildrenChunk) {
            const std::size_t end = std::min(count, begin + kChildrenChunk);
            const auto start = clock::now();
            for (std::size_t i = begin; i < end; ++i) {
                children.push_back(crossover(*parents[i].first, *parents[i].second, rng));
            }
            const auto stop = clock::now();
            measured_seconds += std::chrono::duration<double>(stop - start).count();
            children.clear(); // destruccion de los hijos fuera del cronometro
        }
    }
    return {depth, trees, 2 * per_batch, pairs, measured_seconds,
            measured_seconds / static_cast<double>(pairs),
            total_nodes / static_cast<double>(trees)};
}

void run_tests() {
    const auto check = [](bool condition, const std::string& description) {
        if (!condition) throw std::runtime_error("Fallo en prueba: " + description);
    };
    const auto approx = [](double a, double b) {
        return std::abs(a - b) <= 1e-10;
    };
    std::mt19937_64 rng(42);

    // x1 + (x2 * 0.5) con x1=2 y x2=3 -> 3.5
    Tree formula = Node::binary(Kind::Add,
        Node::variable(0),
        Node::binary(Kind::Multiply, Node::variable(1), Node::constant(0.5)));
    check(formula->expression() == "(x1 + (x2 * 0.5))", "expression");
    check(parse_expression("(x1 + (x2 * 0.5))")->expression() == formula->expression(),
          "parse_expression");
    check(parse_expression("sqrt((x3 / -0.25))")->expression() == "sqrt((x3 / -0.25))",
          "parse con constante negativa");
    for (int depth = 0; depth <= 8; ++depth) {
        Tree t = grow_full(depth, 3, rng);
        check(parse_expression(t->expression(17))->expression(17) == t->expression(17),
              "expression(17) -> parse_expression sin perdida");
    }
    const std::vector<double> sample{2.0, 3.0};
    check(approx(formula->eval(sample), 3.5), "eval");
    check(node_count(*formula) == 5, "node_count");
    check(tree_depth(*formula) == 2, "tree_depth raiz 0");

    Dataset data{{{2.0, 3.0}, {1.0, 2.0}}, {3.5, 2.0}};
    check(approx(fitness(*formula, data), 0.0), "MSE cero");

    Tree root_copy = clone(*formula);
    root_copy->right->right->value = 0.7;
    check(approx(formula->eval(sample), 3.5), "clone copia profunda");

    Tree safe_division = Node::binary(Kind::Divide, Node::constant(1.0),
                                      Node::constant(0.0));
    check(approx(safe_division->eval(sample), 1e12), "division protegida");
    Tree safe_sqrt = Node::unary(Kind::Sqrt, Node::constant(-5.0));
    check(approx(safe_sqrt->eval(sample), 0.0), "sqrt protegido");

    for (int depth = 0; depth <= 6; ++depth) {
        Tree full = grow_full(depth, 3, rng);
        check(tree_depth(*full) == depth, "grow_full profundidad exacta");
        Tree rnd = grow_random(depth, 3, rng);
        check(tree_depth(*rnd) <= depth, "grow_random profundidad maxima");
    }
    Tree a = grow_full(4, 2, rng);
    Tree b = grow_full(3, 2, rng);
    const std::string old_a = a->expression();
    const std::string old_b = b->expression();
    auto [c, d] = crossover(*a, *b, rng);
    check(c && d && a->expression() == old_a && b->expression() == old_b,
          "crossover no altera padres");
    Tree m = mutation(*a, 2, rng);
    check(m && a->expression() == old_a, "mutation no altera padre");
    for (int i = 0; i < 30; ++i) {
        auto [x, y] = depth_protected_crossover(*a, *b, 4, rng);
        check(tree_depth(*x) <= 4 && tree_depth(*y) <= 4,
              "crossover protegido");
        Tree z = depth_protected_mutation(*a, 2, 4, rng);
        check(tree_depth(*z) <= 4, "mutacion protegida");
    }
    GPOptions options;
    options.population_size = 24;
    options.max_depth = 3;
    options.generations = 3;
    GPResult run = full_gp(data, options);
    check(run.best != nullptr, "full_gp devuelve arbol");
    check(run.best_fitness_per_generation.size() == 4, "historial de fitness");
    for (std::size_t i = 1; i < run.best_fitness_per_generation.size(); ++i) {
        check(run.best_fitness_per_generation[i] <= run.best_fitness_per_generation[i - 1],
              "elitismo: mejor MSE no aumenta");
    }
}

} // namespace gp
