//Cambios, solo se borra  
#include "gp.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gp {
namespace {

template <typename Int>
Int random_integer( mt19937_64& rng, Int min_value, Int max_value) {
     uniform_int_distribution<Int> dist(min_value, max_value);
    return dist(rng);
}

double random_constant( mt19937_64& rng) {
     uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

Kind random_operator( mt19937_64& rng) {
    switch (random_integer<int>(rng, 0, 4)) {
        case 0: return Kind::Add;
        case 1: return Kind::Subtract;
        case 2: return Kind::Multiply;
        case 3: return Kind::Divide;
        default: return Kind::Sqrt;
    }
}

Tree random_leaf( size_t variable_count,  mt19937_64& rng) {
    if (variable_count == 0 || random_integer<int>(rng, 0, 1) == 0) {
        return Node::constant(random_constant(rng));
    }
    return Node::variable(random_integer< size_t>(rng, 0, variable_count - 1));
}

// El nodo padre se reserva ANTES que sus hijos, asi que el arbol queda en memoria en
// preorden (raiz, subarbol izquierdo, subarbol derecho): el mismo orden en que lo recorren
// clone, crossover y la evaluacion. Clonar un padre "frio" lee la memoria en secuencia y
// el procesador la adelanta (en el benchmark de crossover, 1.4 a 1.9 veces mas rapido).
// El generador se consume en el mismo orden que antes (operador, hijo izquierdo, hijo
// derecho), asi que los arboles son identicos.
Tree grow_impl(int remaining_depth,  size_t variable_count,
                mt19937_64& rng, bool full) {
    if (remaining_depth == 0 || (!full && random_integer<int>(rng, 0, 3) == 0)) {
        return random_leaf(variable_count, rng);
    }
    const Kind kind = random_operator(rng);
    auto node =  make_unique<Node>();
    node->kind = kind;
    node->direct = false;
    node->left = grow_impl(remaining_depth - 1, variable_count, rng, full);
    if (kind != Kind::Sqrt) {
        node->right = grow_impl(remaining_depth - 1, variable_count, rng, full);
    }
    return node;
}

// Clona `src` en `dest` y, en la MISMA pasada, registra en preorden (raiz,
// izquierdo, derecho) cada lugar donde cuelga un subarbol del clon. Antes esto
// eran tres recorridos por arbol (clone, node_count para reservar y collect_slots).
void clone_collect(Tree& dest, const Node& src,  vector<Tree*>& slots) {
    dest =  make_unique<Node>();
    dest->kind = src.kind;
    dest->copy_payload_from(src);
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
int clone_shape(Tree& dest, const Node& src, int depth,  vector<ShapedSlot>& slots) {
    dest =  make_unique<Node>();
    dest->kind = src.kind;
    dest->copy_payload_from(src);
    const  size_t index = slots.size();
    slots.push_back({&dest, depth, 0});
    int height = 0;
    if (src.left) height = clone_shape(dest->left, *src.left, depth + 1, slots) + 1;
    if (src.right) {
        height =  max(height, clone_shape(dest->right, *src.right, depth + 1, slots) + 1);
    }
    slots[index].height = height;
    return height;
}

Tree* choose_slot( vector<Tree*>& slots,  mt19937_64& rng) {
    return slots[random_integer< size_t>(rng, 0, slots.size() - 1)];
}

void ensure_depth_argument(int depth) {
    if (depth < 0) throw  invalid_argument("La profundidad debe ser >= 0");
}

 string double_to_string(double value, int precision) {
     ostringstream out;
    out <<  setprecision(precision) << value;
    return out.str();
}

// Lector del formato de expression(): "(a op b)", "sqrt(a)", "x3", "-0.25".
class InfixParser {
public:
    explicit InfixParser(const  string& text) : text_(text) {}
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
    [[noreturn]] void fail(const  string& why) const {
        throw  invalid_argument("Expresion invalida (" + why + ") en posicion "
                                    +  to_string(pos_));
    }
    void expect(char c) {
        skip_spaces();
        if (pos_ >= text_.size() || text_[pos_] != c) fail( string("se esperaba '") + c + "'");
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
            return Node::binary(kind,  move(left),  move(right));
        }
        if (text_.compare(pos_, 5, "sqrt(") == 0) {
            pos_ += 5;
            Tree child = parse();
            expect(')');
            return Node::unary(Kind::Sqrt,  move(child));
        }
        if (c == 'x') {
            ++pos_;
            const  size_t begin = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (begin == pos_) fail("variable sin indice");
            const unsigned long index =  stoul(text_.substr(begin, pos_ - begin));
            if (index == 0) fail("las variables empiezan en x1");
            return Node::variable(index - 1);
        }
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double value =  strtod(begin, &end);
        if (end == begin) fail("numero invalido");
        pos_ += static_cast< size_t>(end - begin);
        return Node::constant(value);
    }

    const  string& text_;
     size_t pos_ = 0;
};

 vector<double> parse_numeric_row(const  string& line) {
     vector<double> result;
     stringstream stream(line);
     string field;
    while ( getline(stream, field, ',')) {
        const  size_t first = field.find_first_not_of(" \t\r\n");
        if (first ==  string::npos) throw  invalid_argument("Campo CSV vacio");
        const  size_t last = field.find_last_not_of(" \t\r\n");
        field = field.substr(first, last - first + 1);
         size_t pos = 0;
        double value = 0.0;
        try {
            value =  stod(field, &pos);
        } catch (const  exception&) {
            throw  invalid_argument("Numero CSV invalido: " + field);
        }
        if (pos != field.size() || ! isfinite(value)) {
            throw  invalid_argument("Numero CSV invalido: " + field);
        }
        result.push_back(value);
    }
    // getline no reporta un ultimo campo vacio ("1,2,"): se detecta aparte.
    const  size_t last = line.find_last_not_of(" \t\r\n");
    if (last !=  string::npos && line[last] == ',') {
        throw  invalid_argument("Campo CSV vacio al final de la linea");
    }
    return result;
}

bool is_number_field(const  string& raw) {
    const  size_t first = raw.find_first_not_of(" \t\r\n");
    if (first ==  string::npos) return false;
    const  size_t last = raw.find_last_not_of(" \t\r\n");
    const  string field = raw.substr(first, last - first + 1);
    try {
         size_t pos = 0;
         stod(field, &pos);
        return pos == field.size();
    } catch (const  exception&) {
        return false;
    }
}

// La primera linea se toma como encabezado solo si NINGUN campo es numerico
// (por ejemplo "x1,x2,y"). Una primera fila numerica mal formada es un error.
bool looks_like_header(const  string& line) {
     size_t start = 0;
    while (true) {
        const  size_t comma = line.find(',', start);
        const  string field = line.substr(start, comma ==  string::npos
                                                         ?  string::npos : comma - start);
        if (is_number_field(field)) return false;
        if (comma ==  string::npos) return true;
        start = comma + 1;
    }
}

#if defined(_MSC_VER)
#define GP_FORCE_INLINE __forceinline
#else
#define GP_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Las cuatro operaciones binarias con el resultado elegido por INDICE, sin saltos
// condicionales. Un switch por nodo es un salto que el procesador debe adivinar; en un
// arbol grande (miles de nodos) no puede memorizar la secuencia y falla muy seguido
// (cada fallo cuesta ~15-20 ciclos). Calcular las cuatro y elegir una cuesta menos.
// Son las mismas operaciones IEEE sobre los mismos operandos: resultado identico bit a bit.


// Valor de un nodo `direct` (constante, variable o subarbol plegado) sin saltos: se
// elige la DIRECCION a leer (instruccion cmov) y se hace una sola lectura.


// Misma evaluacion que Node::eval (mismas operaciones, mismo resultado bit a bit) pero
// sin comprobar el indice de variable (fitness lo valida una vez por arbol), con un
// puntero simple en vez de  span, sin saltos para elegir el operador y con las hojas
// resueltas en el propio nodo padre (sin llamada recursiva para ellas). Sigue siendo una
// evaluacion recursiva, fila por fila, del arbol de punteros.
GP_FORCE_INLINE double safe_div(double a, double b) noexcept {
    const double denom = (abs(b) < kEpsilon) ? kEpsilon : b;
    return a / denom;
}

GP_FORCE_INLINE double apply_binary(Kind kind, double a, double b) noexcept {
    if (kind == Kind::Divide) [[unlikely]] {
        return safe_div(a, b);
    }
    double r[3];
    r[0] = a + b;
    r[1] = a - b;
    r[2] = a * b;
    return r[static_cast<int>(kind) - static_cast<int>(Kind::Add)];
}

GP_FORCE_INLINE double direct_value(const Node& c, const double* x) noexcept {
    const double* p = (c.kind == Kind::Variable) ? x + c.variable_index : &c.value;
    return *p;
}

double eval_unchecked(const Node& n, const double* x) noexcept;

GP_FORCE_INLINE double operand(const Node& c, const double* x) noexcept {
    if (c.direct) return direct_value(c, x);
    return eval_unchecked(c, x);
}

double eval_unchecked(const Node& n, const double* x) noexcept {
    if (n.direct) return direct_value(n, x);
    const double a = operand(*n.left, x);
    if (n.kind == Kind::Sqrt) return  sqrt( max(a, 0.0));
    const double b = operand(*n.right, x);
    return apply_binary(n.kind, a, b);
}

long long prepare_tree(Node& n, bool fold) {
    if (n.kind == Kind::Constant) { n.direct = true; return -1; }
    if (n.kind == Kind::Variable) {
        n.direct = true;
        return static_cast<long long>(n.variable_index);
    }
    n.direct = false;
    long long m = prepare_tree(*n.left, fold);
    if (n.right) m =  max(m, prepare_tree(*n.right, fold));
    if (fold && m < 0) {
        const double a = n.left->value;
        n.value = n.kind == Kind::Sqrt ?  sqrt( max(a, 0.0))
                                       : apply_binary(n.kind, a, n.right->value);
        n.direct = true;
    }
    return m;
}

constexpr size_t kEvalChunk = 16;

struct TransposedCache {
    const Dataset* source = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    vector<double> data;

    void sync(const Dataset& ds) {
        const size_t r = ds.x.size();
        const size_t c = ds.variables();
        if (source == &ds && rows == r && cols == c && data.size() == r * c) {
            return;
        }
        source = &ds;
        rows = r;
        cols = c;
        data.resize(r * c);
        for (size_t i = 0; i < r; ++i) {
            const double* row_ptr = ds.x[i].data();
            for ( size_t j = 0; j < c; ++j) {
                data[j * r + i] = row_ptr[j];
            }
        }
    }
};

void eval_chunk(const Node& n, const double* col_base,  size_t stride,
                 size_t row_offset,  size_t count,
                double* __restrict out) noexcept {
    if (n.direct) {
        if (n.kind == Kind::Variable) {
            const double* __restrict src = col_base + n.variable_index * stride + row_offset;
            for (size_t k = 0; k < count; ++k) {
                out[k] = src[k];
            }
        } else {
            const double v = n.value;
            for (size_t k = 0; k < count; ++k) {
                out[k] = v;
            }
        }
        return;
    }

    eval_chunk(*n.left, col_base, stride, row_offset, count, out);

    if (n.kind == Kind::Sqrt) {
        for ( size_t k = 0; k < count; ++k) {
            const double a = out[k];
            out[k] = sqrt(a > 0.0 ? a : 0.0);
        }
        return;
    }

    alignas(64) double rhs[kEvalChunk];
    const double* __restrict b_ptr = nullptr;

    if (n.right->direct) {
        if (n.right->kind == Kind::Variable) {
            b_ptr = col_base + n.right->variable_index * stride + row_offset;
        } else {
            const double v = n.right->value;
            for ( size_t k = 0; k < count; ++k) rhs[k] = v;
            b_ptr = rhs;
        }
    } else {
        eval_chunk(*n.right, col_base, stride, row_offset, count, rhs);
        b_ptr = rhs;
    }

    switch (n.kind) {
        case Kind::Add:
            for (size_t k = 0; k < count; ++k) out[k] += b_ptr[k];
            break;
        case Kind::Subtract:
            for (size_t k = 0; k < count; ++k) out[k] -= b_ptr[k];
            break;
        case Kind::Multiply:
            for (size_t k = 0; k < count; ++k) out[k] *= b_ptr[k];
            break;
        case Kind::Divide:
            for (size_t k = 0; k < count; ++k) {
                const double b = b_ptr[k];
                const double denom = ( abs(b) < kEpsilon) ? kEpsilon : b;
                out[k] /= denom;
            }
            break;
        default:
            break;
    }
}

double mse_impl(const Node& root, const Dataset& data, bool fold, double cut) {
    if (data.x.empty() || data.x.size() != data.y.size()) {
        throw invalid_argument("Dataset vacio o inconsistente");
    }
    const  size_t variables = data.variables();
    for (const auto& row : data.x) {
        if (row.size() != variables) {
            throw invalid_argument("Dimensiones inconsistentes del dataset");
        }
    }
    if (prepare_tree(const_cast<Node&>(root), fold) >= static_cast<long long>(variables)) {
        throw out_of_range("Indice de variable fuera del vector");
    }

    const  size_t total_rows = data.x.size();
    const double rows = static_cast<double>(total_rows);
    const double pre_cut =  isfinite(cut) ? (cut * rows) * 0.999999999999
                                              :  numeric_limits<double>::infinity();

    if (total_rows < 8) {
        double sum = 0.0;
        for ( size_t i = 0; i < total_rows; ++i) {
            const double predicted = eval_unchecked(root, data.x[i].data());
            const double difference = data.y[i] - predicted;
            const double squared = difference * difference;
            if (! isfinite(squared)) return  numeric_limits<double>::infinity();
            sum += squared;
            if (! isfinite(sum)) return  numeric_limits<double>::infinity();
            if (sum >= pre_cut && (sum / rows) > cut) {
                return  numeric_limits<double>::infinity();
            }
        }
        return sum / rows;
    }

    thread_local TransposedCache cache;
    cache.sync(data);
    const double* col_base = cache.data.data();
    const double* y_ptr = data.y.data();

    alignas(64) double preds[kEvalChunk];
    double sum = 0.0;

    for ( size_t offset = 0; offset < total_rows; offset += kEvalChunk) {
        const  size_t count =  min(kEvalChunk, total_rows - offset);
        eval_chunk(root, col_base, total_rows, offset, count, preds);

        for ( size_t k = 0; k < count; ++k) {
            const double difference = y_ptr[offset + k] - preds[k];
            const double squared = difference * difference;
            if (! isfinite(squared)) return  numeric_limits<double>::infinity();
            sum += squared;
            if (! isfinite(sum)) return  numeric_limits<double>::infinity();
            if (sum >= pre_cut && (sum / rows) > cut) {
                return  numeric_limits<double>::infinity();
            }
        }
    }
    return sum / rows;
}


// Una sola pasada por el arbol antes de evaluarlo (ya existia para validar los indices de
// variable). Devuelve el mayor indice de variable, -1 si no hay variables. Ademas marca
// `direct` en cada nodo y, con fold = true, PLIEGA los subarboles sin variables: su valor
// es el mismo en todas las filas, asi que se calcula una vez (con las mismas operaciones
// protegidas, mismo resultado bit a bit) y la evaluacion lo lee sin recorrerlos.



// MSE fila por fila. Si `cut` es finito, deja de evaluar en cuanto el MSE parcial
// (suma/n) lo supera: las filas restantes solo pueden sumar (cuadrados >= 0), asi que
// el MSE final seria aun mayor que `cut`, y se devuelve +infinito.


// Dos subarboles son iguales si tienen la misma forma, los mismos operadores, las mismas
// variables y constantes con los MISMOS bits (asi dan el mismo valor en cada fila).
bool equal_trees(const Node& a, const Node& b) noexcept {
    if (a.kind != b.kind) return false;
    if (a.kind == Kind::Constant) return  memcmp(&a.value, &b.value, sizeof a.value) == 0;
    if (a.kind == Kind::Variable) return a.variable_index == b.variable_index;
    if (!equal_trees(*a.left, *b.left)) return false;
    return !a.right || equal_trees(*a.right, *b.right);
}

// Que hijo de un operador protegido es copia exacta de un padre (mismo arbol => mismo MSE).
enum class Copy : unsigned char {
    None,     // hijo(s) nuevo(s)
    Parents,  // hijo a = padre a, hijo b = padre b (se intercambiaron subarboles iguales)
    Swapped   // hijo a = padre b, hijo b = padre a (se intercambiaron las raices)
};

struct Individual {
    Tree tree;
    double score;
};

bool better(const Individual& a, const Individual& b) {
    return a.score < b.score;
}

} // namespace

 unique_ptr<Node> Node::constant(double val) {
    auto node =  make_unique<Node>();
    node->kind = Kind::Constant;
    node->value = val;
    return node;
}

 unique_ptr<Node> Node::variable( size_t index) {
    auto node =  make_unique<Node>();
    node->kind = Kind::Variable;
    node->variable_index = index;
    return node;
}

 unique_ptr<Node> Node::unary(Kind kind, Tree child) {
    if (kind != Kind::Sqrt || !child) {
        throw  invalid_argument("Operador unario invalido");
    }
    auto node =  make_unique<Node>();
    node->kind = kind;
    node->direct = false;
    node->left =  move(child);
    return node;
}

 unique_ptr<Node> Node::binary(Kind kind, Tree a, Tree b) {
    if (kind == Kind::Sqrt || kind == Kind::Constant || kind == Kind::Variable
        || !a || !b) {
        throw  invalid_argument("Operador binario invalido");
    }
    auto node =  make_unique<Node>();
    node->kind = kind;
    node->direct = false;
    node->left =  move(a);
    node->right =  move(b);
    return node;
}

double Node::eval( span<const double> variables) const {
    switch (kind) {
        case Kind::Constant: return value;
        case Kind::Variable:
            if (variable_index >= variables.size()) {
                throw  out_of_range("Indice de variable fuera del vector");
            }
            return variables[variable_index];
        case Kind::Sqrt: return  sqrt( max(left->eval(variables), 0.0));
        case Kind::Add: return left->eval(variables) + right->eval(variables);
        case Kind::Subtract: return left->eval(variables) - right->eval(variables);
        case Kind::Multiply: return left->eval(variables) * right->eval(variables);
        case Kind::Divide: {
            const double a = left->eval(variables);
            const double b = right->eval(variables);
            return a / ( abs(b) < kEpsilon ? kEpsilon : b);
        }
    }
    throw  logic_error("Nodo desconocido");
}

 string Node::expression(int precision) const {
    switch (kind) {
        case Kind::Constant: return double_to_string(value, precision);
        case Kind::Variable: return "x" +  to_string(variable_index + 1);
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

Tree parse_expression(const  string& text) { return InfixParser(text).run(); }

 size_t Dataset::variables() const {
    return x.empty() ? 0 : x.front().size();
}

Dataset Dataset::from_csv(const  string& filename) {
     ifstream file(filename);
    if (!file) throw  runtime_error("No se pudo abrir: " + filename);
    Dataset result;
     string line;
     size_t line_number = 0;
     size_t columns = 0;
    while ( getline(file, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") ==  string::npos) continue;
        try {
             vector<double> values = parse_numeric_row(line);
            if (values.size() < 2) throw  invalid_argument("Faltan columnas");
            if (columns == 0) columns = values.size();
            if (values.size() != columns) throw  invalid_argument("Columnas inconsistentes");
            result.y.push_back(values.back());
            values.pop_back();
            result.x.push_back( move(values));
        } catch (const  exception& e) {
            // Permitir un encabezado en la primera linea, por ejemplo: x1,x2,y.
            if (line_number == 1 && looks_like_header(line)) continue;
            throw  runtime_error("CSV linea " +  to_string(line_number)
                                     + ": " + e.what());
        }
    }
    if (result.x.empty()) throw  runtime_error("Dataset sin filas numericas");
    return result;
}

Tree clone(const Node& root) {
    auto result =  make_unique<Node>();
    result->kind = root.kind;
    result->copy_payload_from(root);
    if (root.left) result->left = clone(*root.left);
    if (root.right) result->right = clone(*root.right);
    return result;
}

 size_t node_count(const Node& root) {
    return 1 + (root.left ? node_count(*root.left) : 0)
             + (root.right ? node_count(*root.right) : 0);
}

int tree_depth(const Node& root) {
    const int left = root.left ? tree_depth(*root.left) + 1 : 0;
    const int right = root.right ? tree_depth(*root.right) + 1 : 0;
    return  max(left, right);
}

Tree grow_random(int max_depth,  size_t variable_count,  mt19937_64& rng) {
    ensure_depth_argument(max_depth);
    return grow_impl(max_depth, variable_count, rng, false);
}

Tree grow_full(int exact_depth,  size_t variable_count,  mt19937_64& rng) {
    ensure_depth_argument(exact_depth);
    return grow_impl(exact_depth, variable_count, rng, true);
}

double fitness(const Node& root, const Dataset& data, bool fold_constants) {
    return mse_impl(root, data, fold_constants,  numeric_limits<double>::infinity());
}

 pair<Tree, Tree> crossover(const Node& a, const Node& b,
                                 mt19937_64& rng) {
    // Buffers reutilizados entre llamadas: sin reservar memoria para los slots.
    thread_local  vector<Tree*> slots_a;
    thread_local  vector<Tree*> slots_b;
    slots_a.clear();
    slots_b.clear();
    Tree child_a;
    Tree child_b;
    clone_collect(child_a, a, slots_a);
    clone_collect(child_b, b, slots_b);
    // Orden explicito: el orden de evaluacion de argumentos no esta definido en C++.
    Tree* selected_a = choose_slot(slots_a, rng);
    Tree* selected_b = choose_slot(slots_b, rng);
     swap(*selected_a, *selected_b);
    return { move(child_a),  move(child_b)};
}

Tree mutation(const Node& parent,  size_t variable_count,
               mt19937_64& rng, int new_subtree_max_depth) {
    ensure_depth_argument(new_subtree_max_depth);
    thread_local  vector<Tree*> slots;
    slots.clear();
    Tree result;
    clone_collect(result, parent, slots);
    const int replacement_depth = random_integer<int>(rng, 0, new_subtree_max_depth);
    Tree replacement = grow_random(replacement_depth, variable_count, rng);
    *choose_slot(slots, rng) =  move(replacement);
    return result;
}

namespace {
 pair<Tree, Tree> protected_crossover_impl(const Node& a, const Node& b, int max_depth,
                                                mt19937_64& rng, Copy* copy) {
    if (copy) *copy = Copy::None;
    ensure_depth_argument(max_depth);
    thread_local  vector<ShapedSlot> slots_a;
    thread_local  vector<ShapedSlot> slots_b;
    slots_a.clear();
    slots_b.clear();
    Tree child_a;
    Tree child_b;
    const int height_a = clone_shape(child_a, a, 0, slots_a);
    const int height_b = clone_shape(child_b, b, 0, slots_b);
    if (height_a > max_depth || height_b > max_depth) {
        throw  invalid_argument("Un padre supera max_depth");
    }

    // Cambiar el subarbol i de A por el j de B deja al hijo A con profundidad
    // max(resto de A, prof_A(i) + altura_B(j)); el resto de A ya cabe porque el
    // padre cabe. Por eso esta condicion equivale EXACTAMENTE a intercambiar y
    // medir tree_depth de ambos hijos (lo que se hacia antes): mismos intentos,
    // mismo consumo del generador y mismos hijos, sin recorrer los arboles.
    for (int attempt = 0; attempt < 128; ++attempt) {
        const ShapedSlot& x = slots_a[random_integer< size_t>(rng, 0, slots_a.size() - 1)];
        const ShapedSlot& y = slots_b[random_integer< size_t>(rng, 0, slots_b.size() - 1)];
        if (x.depth + y.height <= max_depth && y.depth + x.height <= max_depth) {
            if (copy) {  // solo compara; no consume el generador ni cambia los hijos
                if (x.depth == 0 && y.depth == 0) *copy = Copy::Swapped;
                else if (equal_trees(**x.slot, **y.slot)) *copy = Copy::Parents;
            }
             swap(*x.slot, *y.slot);
            return { move(child_a),  move(child_b)};
        }
    }
    // Intercambiar las raices completas siempre respeta el limite de ambos padres.
    if (copy) *copy = Copy::Swapped;
     swap(child_a, child_b);
    return { move(child_a),  move(child_b)};
}

Tree protected_mutation_impl(const Node& parent,  size_t variable_count, int max_depth,
                              mt19937_64& rng, bool* same) {
    if (same) *same = false;
    ensure_depth_argument(max_depth);
    thread_local  vector<ShapedSlot> slots;
    slots.clear();
    Tree result;
    if (clone_shape(result, parent, 0, slots) > max_depth) {
        throw  invalid_argument("El padre supera max_depth");
    }
    const ShapedSlot& selected = slots[random_integer< size_t>(rng, 0, slots.size() - 1)];
    const int remaining = max_depth - selected.depth;
    const int replacement_depth = random_integer<int>(rng, 0, remaining);
    Tree replacement = grow_random(replacement_depth, variable_count, rng);
    if (same) *same = equal_trees(*replacement, **selected.slot);
    *selected.slot =  move(replacement);
    return result;
}
} // namespace

 pair<Tree, Tree> depth_protected_crossover(
    const Node& a, const Node& b, int max_depth,  mt19937_64& rng) {
    return protected_crossover_impl(a, b, max_depth, rng, nullptr);
}

Tree depth_protected_mutation(const Node& parent,  size_t variable_count,
                              int max_depth,  mt19937_64& rng) {
    return protected_mutation_impl(parent, variable_count, max_depth, rng, nullptr);
}

GPResult full_gp(const Dataset& data, const GPOptions& options) {
    if (data.variables() == 0 || data.x.empty()) {
        throw  invalid_argument("Se necesita un dataset con variables");
    }
    if (options.population_size < 4 || options.population_size % 4 != 0) {
        throw  invalid_argument("population_size debe ser multiplo de 4 y >= 4");
    }
    ensure_depth_argument(options.max_depth);
     mt19937_64 rng(options.seed);
    const  size_t N = options.population_size;

     vector<Individual> population;
    population.reserve(N);
    for ( size_t i = 0; i < N; ++i) {
        Tree tree = grow_full(options.max_depth, data.variables(), rng);
        double score = fitness(*tree, data, options.fold_constants);
        population.push_back({ move(tree), score});
    }

    GPResult result;
    for ( size_t generation = 0; generation <= options.generations; ++generation) {
        const Individual& best = * min_element(population.begin(), population.end(), better);
        result.best_fitness_per_generation.push_back(best.score);
        if (options.on_generation) options.on_generation(generation, best.score);
        if (generation == options.generations) break;

        // Torneo binario repetido: llenar un pool de N punteros a los ganadores.
         vector<const Individual*> parent_pool;
        parent_pool.reserve(N);
        for ( size_t i = 0; i < N; ++i) {
            const auto& a = population[random_integer< size_t>(rng, 0, N - 1)];
            const auto& b = population[random_integer< size_t>(rng, 0, N - 1)];
            parent_pool.push_back(a.score <= b.score ? &a : &b);
        }
        const auto pick_parent = [&]() -> const Individual& {
            return *parent_pool[random_integer< size_t>(rng, 0, N - 1)];
        };

         vector<Individual> offspring;
        offspring.reserve(N);

        // Corte temprano EXACTO. Solo el mejor 50% de los hijos sobrevive. `best_half`
        // (monticulo de maximos) guarda los N/2 mejores MSE vistos hasta ahora; su tope
        // es el (N/2)-esimo mejor y solo puede bajar. Un hijo cuyo MSE parcial ya supera
        // ese tope terminaria peor que N/2 hijos, asi que nunca sobreviviria: se deja de
        // evaluar y recibe +infinito. Los sobrevivientes conservan su MSE exacto y, con el
        // orden estable de abajo, el GP da exactamente el mismo resultado con o sin corte.
         vector<double> best_half;
        best_half.reserve(N / 2);
        const auto record = [&](double score) {
            const bool full_heap = best_half.size() == N / 2;
            if (!full_heap) {
                best_half.push_back(score);
                 push_heap(best_half.begin(), best_half.end());
            } else if (score < best_half.front()) {
                 pop_heap(best_half.begin(), best_half.end());
                best_half.back() = score;
                 push_heap(best_half.begin(), best_half.end());
            }
            return score;
        };
        const auto cut_now = [&]() {
            return (options.early_cut && best_half.size() == N / 2)
                       ? best_half.front()
                       :  numeric_limits<double>::infinity();
        };
        const auto score_child = [&](const Node& tree) {
            return record(mse_impl(tree, data, options.fold_constants, cut_now()));
        };
        // Hijo que es copia exacta de un padre (--no-dup lo apaga): su MSE es el del padre,
        // sin evaluar las filas. Se registra igual que si se hubiera evaluado: con el corte
        // temprano, mse_impl habria devuelto +infinito justo cuando el MSE pasa del corte.
        const auto score_copy = [&](double parent_mse) {
            return record(parent_mse > cut_now() ?  numeric_limits<double>::infinity()
                                                 : parent_mse);
        };

        // Primero se CREAN todos los hijos (mismo orden y mismo consumo del generador que
        // antes) y despues se evaluan empezando por los de mejores padres: el umbral del
        // corte temprano baja antes y se corta mas. El orden de evaluacion no cambia ningun
        // MSE de los sobrevivientes ni el orden de creacion (que decide los empates).
         vector<double> parent_score;
        parent_score.reserve(N);
        // MSE conocido de los hijos que son copia exacta de un padre (NaN = hay que evaluar).
         vector<double> copy_score;
        copy_score.reserve(N);
        const double unknown =  numeric_limits<double>::quiet_NaN();
        Copy copy = Copy::None;
        bool same = false;
        Copy* const copy_out = options.skip_duplicates ? &copy : nullptr;
        bool* const same_out = options.skip_duplicates ? &same : nullptr;
        // N/4 cruces x 2 hijos = N/2 descendientes por crossover.
        for ( size_t i = 0; i < N / 4; ++i) {
            const Individual& p1 = pick_parent(); // orden explicito de seleccion
            const Individual& p2 = pick_parent();
            auto [a, b] = protected_crossover_impl(*p1.tree, *p2.tree, options.max_depth, rng,
                                                   copy_out);
            offspring.push_back({ move(a), 0.0});
            offspring.push_back({ move(b), 0.0});
            parent_score.push_back(p1.score);  // el hijo a conserva la raiz de p1
            parent_score.push_back(p2.score);
            copy_score.push_back(copy == Copy::Parents ? p1.score
                                 : copy == Copy::Swapped ? p2.score : unknown);
            copy_score.push_back(copy == Copy::Parents ? p2.score
                                 : copy == Copy::Swapped ? p1.score : unknown);
        }
        // N/2 descendientes por mutacion = 50% del total.
        for ( size_t i = 0; i < N / 2; ++i) {
            const Individual& p = pick_parent();
            Tree mutated = protected_mutation_impl(*p.tree, data.variables(), options.max_depth,
                                                   rng, same_out);
            offspring.push_back({ move(mutated), 0.0});
            parent_score.push_back(p.score);
            copy_score.push_back(same ? p.score : unknown);
        }
         vector< size_t> evaluation_order(offspring.size());
         iota(evaluation_order.begin(), evaluation_order.end(),  size_t{0});
        if (options.order_by_parent) {  // estable: a igual MSE del padre, orden de creacion
             stable_sort(evaluation_order.begin(), evaluation_order.end(),
                             [&]( size_t i,  size_t j) { return parent_score[i] < parent_score[j]; });
        }
        for (const  size_t i : evaluation_order) {
            offspring[i].score =  isnan(copy_score[i]) ? score_child(*offspring[i].tree)
                                                           : score_copy(copy_score[i]);
        }

        // Orden ESTABLE (como A y C): en empates conserva el orden de creacion. Con
        // nth_element el orden de los sobrevivientes dependia de la implementacion de la
        // biblioteca estandar y de TODOS los puntajes (incluso los descartados), y el torneo
        // elige por posicion; asi el resultado ya no depende de eso ni del corte temprano.
        const auto keep_half = []( vector<Individual>& group) {
             stable_sort(group.begin(), group.end(), better);
            group.erase(group.begin() + static_cast< ptrdiff_t>(group.size() / 2), group.end());
        };
        keep_half(population);
        keep_half(offspring);
        for (auto& individual : offspring) population.push_back( move(individual));
        assert(population.size() == N);
    }
    const Individual& best = * min_element(population.begin(), population.end(), better);
    result.best = clone(*best.tree);
    result.best_fitness = best.score;
    return result;
}

CrossoverTiming benchmark_crossover( size_t trees, int depth,
                                      size_t variables,  uint64_t seed,
                                      size_t batch_pairs) {
    if (trees < 2 || trees % 2 != 0 || variables == 0) {
        throw  invalid_argument("trees debe ser par >=2 y variables >=1");
    }
    ensure_depth_argument(depth);
     mt19937_64 rng(seed);
    using clock =  chrono::steady_clock;
    double measured_seconds = 0.0;
    double total_nodes = 0.0;
    const  size_t pairs = trees / 2;
    const  size_t per_batch = (batch_pairs == 0 || batch_pairs > pairs) ? pairs : batch_pairs;
    // Hijos que se guardan antes de liberarlos (fuera del cronometro). Asi la RAM
    // la ocupan solo los padres y el costo del reloj se reparte entre 16 cruces.
    constexpr  size_t kChildrenChunk = 16;

     vector< pair<Tree, Tree>> parents;
     vector< pair<Tree, Tree>> children;
    parents.reserve(per_batch);
    children.reserve(kChildrenChunk);
    for ( size_t offset = 0; offset < pairs; offset += per_batch) {
        const  size_t count =  min(per_batch, pairs - offset);
        parents.clear(); // libera el lote anterior (fuera del cronometro)
        for ( size_t i = 0; i < count; ++i) {
            Tree first = grow_full(depth, variables, rng); // orden explicito
            Tree second = grow_full(depth, variables, rng);
            total_nodes += static_cast<double>(node_count(*first) + node_count(*second));
            parents.emplace_back( move(first),  move(second));
        }
        for ( size_t begin = 0; begin < count; begin += kChildrenChunk) {
            const  size_t end =  min(count, begin + kChildrenChunk);
            const auto start = clock::now();
            for ( size_t i = begin; i < end; ++i) {
                children.push_back(crossover(*parents[i].first, *parents[i].second, rng));
            }
            const auto stop = clock::now();
            measured_seconds +=  chrono::duration<double>(stop - start).count();
            children.clear(); // destruccion de los hijos fuera del cronometro
        }
    }
    return {depth, trees, 2 * per_batch, pairs, measured_seconds,
            measured_seconds / static_cast<double>(pairs),
            total_nodes / static_cast<double>(trees)};
}

void run_tests() {
    const auto check = [](bool condition, const  string& description) {
        if (!condition) throw  runtime_error("Fallo en prueba: " + description);
    };
    const auto approx = [](double a, double b) {
        return  abs(a - b) <= 1e-10;
    };
     mt19937_64 rng(42);

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
    const  vector<double> sample{2.0, 3.0};
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
    const  string old_a = a->expression();
    const  string old_b = b->expression();
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
    for ( size_t i = 1; i < run.best_fitness_per_generation.size(); ++i) {
        check(run.best_fitness_per_generation[i] <= run.best_fitness_per_generation[i - 1],
              "elitismo: mejor MSE no aumenta");
    }
    {   // on_generation: una llamada por generacion con el mismo MSE del historial,
        // y la corrida no cambia por tenerlo.
        GPOptions watched = options;
         vector< size_t> seen_generation;
         vector<double> seen_mse;
        watched.on_generation = [&]( size_t g, double mse) {
            seen_generation.push_back(g);
            seen_mse.push_back(mse);
        };
        GPResult again = full_gp(data, watched);
        check(seen_mse == run.best_fitness_per_generation, "on_generation da el historial");
        check(seen_generation ==  vector< size_t>{0, 1, 2, 3},
              "on_generation en orden 0..generaciones");
        check(again.best_fitness_per_generation == run.best_fitness_per_generation &&
              again.best->expression(17) == run.best->expression(17),
              "on_generation no cambia el GP");
    }

    // --- Optimizaciones: mismo resultado bit a bit -----------------------------
    const auto same_bits = [](double a, double b) { return  memcmp(&a, &b, sizeof a) == 0; };
    // MSE de referencia con Node::eval (switch por nodo, sin plegado ni atajos).
    const auto reference_mse = [](const Node& tree, const Dataset& d) {
        double sum = 0.0;
        for ( size_t i = 0; i < d.x.size(); ++i) {
            const double diff = d.y[i] - tree.eval(d.x[i]);
            const double sq = diff * diff;
            if (! isfinite(sq)) return  numeric_limits<double>::infinity();
            sum += sq;
            if (! isfinite(sum)) return  numeric_limits<double>::infinity();
        }
        return sum / static_cast<double>(d.x.size());
    };
    // Datos con ceros, negativos y magnitudes extremas (division protegida, raiz de
    // negativos, desbordes).
    Dataset hard;
    {
         mt19937_64 r2(7);
         uniform_real_distribution<double> u(-3.0, 3.0);
        const double scales[] = {1e-13, 1e-8, 1.0, 1.0, 1e3, 1e150};
        for (int i = 0; i < 60; ++i) {
             vector<double> row;
            for (int k = 0; k < 3; ++k) {
                row.push_back((i + k) % 7 == 0 ? 0.0 : u(r2) * scales[(i * 3 + k) % 6]);
            }
            hard.x.push_back(row);
            hard.y.push_back(u(r2));
        }
    }
    int mismatches = 0;
    for (int t = 0; t < 400; ++t) {
        Tree tree = (t % 2 == 0) ? grow_random(1 + t % 10, 3, rng) : grow_full(t % 8, 3, rng);
        const double ref = reference_mse(*tree, hard);
        for (bool fold : {true, false}) {
            if (!same_bits(fitness(*tree, hard, fold), ref)) ++mismatches;
        }
        // Los hijos heredan marcas de plegado de sus padres; fitness debe recalcularlas.
        Tree other = grow_random(4, 3, rng);
        auto [c1, c2] = crossover(*tree, *other, rng);
        if (!same_bits(fitness(*c1, hard), reference_mse(*c1, hard))) ++mismatches;
        if (!same_bits(fitness(*c2, hard), reference_mse(*c2, hard))) ++mismatches;
        Tree m1 = mutation(*tree, 3, rng);
        if (!same_bits(fitness(*m1, hard), reference_mse(*m1, hard))) ++mismatches;
    }
    check(mismatches == 0, "evaluacion sin saltos y plegado: MSE identico a Node::eval");

    Tree folded = parse_expression("((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))");
    fitness(*folded, data);
    check(folded->left->direct && folded->left->value == 0.6 && !folded->direct
              && !folded->right->direct && folded->right->right->direct
              && folded->right->right->value == 1e12,
          "plegado de constantes: valores y marcas");
    check(folded->expression() == "((sqrt(0.25) + 0.1) * (x1 - (1 / 0)))" && node_count(*folded) == 10,
          "el plegado no cambia la expresion ni el tamano");
    fitness(*folded, data, false);
    check(!folded->left->direct, "sin plegado solo las hojas se leen directo");
    check(fitness(*parse_expression("((1e300 * 1e300) - (1e300 * 1e300))"), data) ==
               numeric_limits<double>::infinity(),
          "subarbol constante no finito: MSE = +Inf");
    bool threw = false;
    try {
        fitness(*parse_expression("(x3 + 1)"), data);
    } catch (const  out_of_range&) {
        threw = true;
    }
    check(threw, "variable fuera del dataset es rechazada");

    // Generacion en preorden: mismo consumo del generador que la version anterior.
    {
         function<Tree(int,  size_t,  mt19937_64&, bool)> old_grow =
            [&](int rem,  size_t vars,  mt19937_64& r, bool full) -> Tree {
            if (rem == 0 || (!full && random_integer<int>(r, 0, 3) == 0)) return random_leaf(vars, r);
            const Kind kind = random_operator(r);
            Tree left = old_grow(rem - 1, vars, r, full);
            if (kind == Kind::Sqrt) return Node::unary(kind,  move(left));
            Tree right = old_grow(rem - 1, vars, r, full);
            return Node::binary(kind,  move(left),  move(right));
        };
        bool same = true;
        for (int seed = 1; seed <= 20; ++seed) {
             mt19937_64 r_new(seed);
             mt19937_64 r_old(seed);
            for (int depth = 0; depth <= 7; ++depth) {
                same &= grow_full(depth, 3, r_new)->expression(17) ==
                        old_grow(depth, 3, r_old, true)->expression(17);
                same &= grow_random(depth, 2, r_new)->expression(17) ==
                        old_grow(depth, 2, r_old, false)->expression(17);
            }
            same &= r_new() == r_old();
        }
        check(same, "grow en preorden: mismos arboles y mismo consumo del generador");
    }

    // Deteccion de hijos que copian a un padre: mismos hijos, mismo consumo del generador,
    // y cuando se reporta una copia, el hijo es identico al padre que dice.
    {
         mt19937_64 r_flag(11);
         mt19937_64 r_plain(11);
        int copies = 0;
        bool ok = true;
        for (int t = 0; t < 3000 && ok; ++t) {
            Tree p1 = grow_random(1 + t % 4, 2, r_flag);
            Tree p2 = grow_random(1 + (t / 4) % 4, 2, r_flag);
            Tree q1 = grow_random(1 + t % 4, 2, r_plain);
            Tree q2 = grow_random(1 + (t / 4) % 4, 2, r_plain);
            Copy copy = Copy::None;
            auto [c1, c2] = protected_crossover_impl(*p1, *p2, 4, r_flag, &copy);
            auto [d1, d2] = depth_protected_crossover(*q1, *q2, 4, r_plain);
            ok &= c1->expression(17) == d1->expression(17) && c2->expression(17) == d2->expression(17);
            if (copy == Copy::Parents) {
                ++copies;
                ok &= equal_trees(*c1, *p1) && equal_trees(*c2, *p2);
            } else if (copy == Copy::Swapped) {
                ++copies;
                ok &= equal_trees(*c1, *p2) && equal_trees(*c2, *p1);
            }
            bool same = false;
            Tree m = protected_mutation_impl(*p1, 2, 4, r_flag, &same);
            Tree n = depth_protected_mutation(*q1, 2, 4, r_plain);
            ok &= m->expression(17) == n->expression(17);
            if (same) {
                ++copies;
                ok &= equal_trees(*m, *p1);
            }
        }
        ok &= r_flag() == r_plain();
        check(ok && copies > 0, "deteccion de hijos que copian a un padre");
        check(!equal_trees(*parse_expression("(x1 + 0)"), *parse_expression("(x1 + -0)")),
              "constantes iguales solo con los mismos bits");
    }

    // El corte temprano, el plegado y el orden de evaluacion no cambian el GP.
    for ( uint64_t seed : {1u, 2u, 3u}) {
        GPOptions o;
        o.population_size = 200;
        o.max_depth = 6;
        o.generations = 6;
        o.seed = seed;
        GPOptions plain = o;
        plain.early_cut = false;
        plain.fold_constants = false;
        plain.order_by_parent = false;
        plain.skip_duplicates = false;
        GPOptions creation_order = o;
        creation_order.order_by_parent = false;
        GPOptions evaluate_copies = o;
        evaluate_copies.skip_duplicates = false;
        GPResult with_copies = full_gp(hard, evaluate_copies);
        GPResult fast = full_gp(hard, o);
        GPResult slow = full_gp(hard, plain);
        GPResult unordered = full_gp(hard, creation_order);
        bool same_order = unordered.best_fitness_per_generation == fast.best_fitness_per_generation
                          && unordered.best->expression(17) == fast.best->expression(17);
        check(same_order, "evaluar primero los hijos de mejores padres: mismo GP");
        bool same_copies = with_copies.best_fitness_per_generation.size() ==
                           fast.best_fitness_per_generation.size();
        for ( size_t i = 0; same_copies && i < fast.best_fitness_per_generation.size(); ++i) {
            same_copies = same_bits(fast.best_fitness_per_generation[i],
                                    with_copies.best_fitness_per_generation[i]);
        }
        check(same_copies && with_copies.best->expression(17) == fast.best->expression(17),
              "reutilizar el MSE de los hijos que copian a un padre: mismo GP");
        bool same = fast.best_fitness_per_generation.size() == slow.best_fitness_per_generation.size();
        for ( size_t i = 0; same && i < fast.best_fitness_per_generation.size(); ++i) {
            same = same_bits(fast.best_fitness_per_generation[i], slow.best_fitness_per_generation[i]);
        }
        check(same && fast.best->expression(17) == slow.best->expression(17)
                  && same_bits(fast.best_fitness, slow.best_fitness),
              "corte temprano y plegado: mismo GP que sin ellos");
    }
}

} // namespace gp