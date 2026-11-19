#include "gpc.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gpc {
namespace {

constexpr std::size_t kBytesPerNode = sizeof(double) + sizeof(Op);
constexpr int kProtectedCrossoverAttempts = 128;  // igual que B

// ---------------------------------------------------------------------------
// Aleatoriedad: mismas distribuciones y mismos tipos que B, para que con la
// misma semilla ambas implementaciones consuman el generador de forma identica.
// ---------------------------------------------------------------------------
template <typename Int>
Int random_integer(std::mt19937_64& rng, Int min_value, Int max_value) {
    std::uniform_int_distribution<Int> dist(min_value, max_value);
    return dist(rng);
}

double random_constant(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

Op random_operator(std::mt19937_64& rng) {
    switch (random_integer<int>(rng, 0, 4)) {
        case 0: return Op::Add;
        case 1: return Op::Subtract;
        case 2: return Op::Multiply;
        case 3: return Op::Divide;
        default: return Op::Sqrt;
    }
}

void ensure_depth_argument(int depth) {
    if (depth < 0) throw std::invalid_argument("La profundidad debe ser >= 0");
}

// Buffers de trabajo reutilizables (evitan reservar memoria en cada operacion).
struct Scratch {
    std::vector<Op> ops;
    std::vector<double> values;
    void clear() { ops.clear(); values.clear(); }
};
Scratch& scratch() {
    thread_local Scratch s;
    return s;
}

// Generacion directa en postorden: hijos primero, operador al final.
void emit_leaf(std::size_t variable_count, std::mt19937_64& rng, Scratch& out) {
    if (variable_count == 0 || random_integer<int>(rng, 0, 1) == 0) {
        out.ops.push_back(Op::Constant);
        out.values.push_back(random_constant(rng));
    } else {
        out.ops.push_back(Op::Variable);
        out.values.push_back(static_cast<double>(
            random_integer<std::size_t>(rng, 0, variable_count - 1)));
    }
}

void grow_impl(int remaining_depth, std::size_t variable_count, std::mt19937_64& rng,
               bool full, Scratch& out) {
    if (remaining_depth == 0 || (!full && random_integer<int>(rng, 0, 3) == 0)) {
        emit_leaf(variable_count, rng, out);
        return;
    }
    const Op kind = random_operator(rng);
    grow_impl(remaining_depth - 1, variable_count, rng, full, out);
    if (kind != Op::Sqrt) grow_impl(remaining_depth - 1, variable_count, rng, full, out);
    out.ops.push_back(kind);
    out.values.push_back(0.0);
}

Program from_scratch(const Scratch& s) {
    Program p(s.ops.size());
    std::memcpy(p.values(), s.values.data(), s.values.size() * sizeof(double));
    std::memcpy(p.ops(), s.ops.data(), s.ops.size() * sizeof(Op));
    return p;
}

// ---------------------------------------------------------------------------
// Metadatos "bajo demanda": solo se calculan en los operadores que los
// necesitan (versiones protegidas o modo Preorder). El arreglo guarda lo
// minimo (9 bytes por nodo) para que el crossover copie lo menos posible.
// ---------------------------------------------------------------------------
struct Shape {
    std::vector<std::uint32_t> size;      // nodos del subarbol con raiz en i
    std::vector<std::int32_t> height;     // altura del subarbol (hoja = 0)
    std::vector<std::uint32_t> parent;
    std::vector<std::int32_t> depth;      // profundidad del nodo (raiz = 0)
    std::vector<std::uint32_t> pre2post;  // k-esimo nodo en preorden -> indice postfijo
    std::vector<std::uint32_t> stack;
};

Shape& shape(int which) {
    thread_local Shape shapes[2];
    return shapes[which];
}

void analyze(const Program& p, Shape& s, bool want_depth, bool want_preorder) {
    const std::size_t n = p.size();
    const Op* op = p.ops();
    s.size.resize(n);
    s.height.resize(n);
    s.parent.resize(n);
    s.stack.clear();
    for (std::size_t i = 0; i < n; ++i) {
        const auto idx = static_cast<std::uint32_t>(i);
        switch (arity(op[i])) {
            case 0:
                s.size[i] = 1;
                s.height[i] = 0;
                break;
            case 1: {
                const std::uint32_t c = s.stack.back();
                s.stack.pop_back();
                s.size[i] = 1 + s.size[c];
                s.height[i] = s.height[c] + 1;
                s.parent[c] = idx;
                break;
            }
            default: {
                const std::uint32_t r = s.stack.back();
                s.stack.pop_back();
                const std::uint32_t l = s.stack.back();
                s.stack.pop_back();
                s.size[i] = 1 + s.size[l] + s.size[r];
                s.height[i] = std::max(s.height[l], s.height[r]) + 1;
                s.parent[l] = idx;
                s.parent[r] = idx;
                break;
            }
        }
        s.stack.push_back(idx);
    }
    if (want_depth) {
        s.depth.resize(n);
        s.depth[n - 1] = 0;
        for (std::size_t i = n - 1; i-- > 0;) s.depth[i] = s.depth[s.parent[i]] + 1;
    }
    if (want_preorder) {
        // Mismo orden que collect_slots de B: raiz, subarbol izquierdo, derecho.
        s.pre2post.clear();
        s.stack.clear();
        s.stack.push_back(static_cast<std::uint32_t>(n - 1));
        while (!s.stack.empty()) {
            const std::uint32_t i = s.stack.back();
            s.stack.pop_back();
            s.pre2post.push_back(i);
            const int a = arity(op[i]);
            if (a == 1) {
                s.stack.push_back(i - 1);
            } else if (a == 2) {
                const std::uint32_t right = i - 1;
                const std::uint32_t left = right - s.size[right];
                s.stack.push_back(right);
                s.stack.push_back(left);
            }
        }
    }
}

std::size_t pick_node(std::size_t n, std::mt19937_64& rng, Pick pick, const Shape* s) {
    const std::size_t k = random_integer<std::size_t>(rng, 0, n - 1);
    return pick == Pick::Preorder ? s->pre2post[k] : k;
}

// base[0:start) + donor + base(end:]  -- la "clonacion + intercambio" del
// manual hecha en un solo paso con tres copias contiguas por arreglo.
Program splice(const Program& base, std::size_t start, std::size_t end,
               const double* donor_values, const Op* donor_ops, std::size_t donor_len) {
    const std::size_t n = base.size();
    const std::size_t tail = n - end - 1;
    Program out(start + donor_len + tail);
    double* v = out.values();
    Op* o = out.ops();
    const double* bv = base.values();
    const Op* bo = base.ops();
    std::memcpy(v, bv, start * sizeof(double));
    std::memcpy(v + start, donor_values, donor_len * sizeof(double));
    std::memcpy(v + start + donor_len, bv + end + 1, tail * sizeof(double));
    std::memcpy(o, bo, start * sizeof(Op));
    std::memcpy(o + start, donor_ops, donor_len * sizeof(Op));
    std::memcpy(o + start + donor_len, bo + end + 1, tail * sizeof(Op));
    return out;
}

std::string double_to_string(double value, int precision) {
    std::ostringstream out;
    out << std::setprecision(precision) << value;
    return out.str();
}

std::size_t variable_index(double stored) { return static_cast<std::size_t>(stored); }

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
        const double value = std::stod(field, &pos);
        if (pos != field.size() || !std::isfinite(value)) {
            throw std::invalid_argument("Numero CSV invalido: " + field);
        }
        result.push_back(value);
    }
    return result;
}

// Lector del formato textual de B: "(a op b)", "sqrt(a)", "x3", "-0.25".
class InfixParser {
public:
    explicit InfixParser(const std::string& text) : text_(text) {}
    Program run() {
        parse();
        skip_spaces();
        if (pos_ != text_.size()) fail("texto sobrante");
        return from_scratch(out_);
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
    void parse() {
        skip_spaces();
        if (pos_ >= text_.size()) fail("fin inesperado");
        const char c = text_[pos_];
        if (c == '(') {
            ++pos_;
            parse();
            skip_spaces();
            if (pos_ >= text_.size()) fail("falta operador");
            Op op = Op::Add;
            switch (text_[pos_]) {
                case '+': op = Op::Add; break;
                case '-': op = Op::Subtract; break;
                case '*': op = Op::Multiply; break;
                case '/': op = Op::Divide; break;
                default: fail("operador desconocido");
            }
            ++pos_;
            parse();
            expect(')');
            push(op, 0.0);
        } else if (text_.compare(pos_, 5, "sqrt(") == 0) {
            pos_ += 5;
            parse();
            expect(')');
            push(Op::Sqrt, 0.0);
        } else if (c == 'x') {
            ++pos_;
            const std::size_t begin = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (begin == pos_) fail("variable sin indice");
            const unsigned long index = std::stoul(text_.substr(begin, pos_ - begin));
            if (index == 0) fail("las variables empiezan en x1");
            push(Op::Variable, static_cast<double>(index - 1));
        } else {
            const char* begin = text_.c_str() + pos_;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin) fail("numero invalido");
            pos_ += static_cast<std::size_t>(end - begin);
            push(Op::Constant, value);
        }
    }
    void push(Op op, double value) {
        out_.ops.push_back(op);
        out_.values.push_back(value);
    }

    const std::string& text_;
    std::size_t pos_ = 0;
    Scratch out_;
};

struct Individual {
    Program tree;
    double score;
};

bool better(const Individual& a, const Individual& b) { return a.score < b.score; }

}  // namespace

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------
Program::Program(std::size_t n) : n_(n) {
    if (n_ > 0) buf_ = std::make_unique_for_overwrite<std::byte[]>(n_ * kBytesPerNode);
}

Program::Program(const Program& other) : Program(other.n_) {
    if (n_ > 0) std::memcpy(buf_.get(), other.buf_.get(), n_ * kBytesPerNode);
}

Program& Program::operator=(const Program& other) {
    if (this != &other) {
        Program copy(other);
        *this = std::move(copy);
    }
    return *this;
}

std::string Program::expression(int precision) const {
    std::vector<std::string> stack;
    const Op* op = ops();
    const double* val = values();
    for (std::size_t i = 0; i < n_; ++i) {
        switch (op[i]) {
            case Op::Constant: stack.push_back(double_to_string(val[i], precision)); break;
            case Op::Variable:
                stack.push_back("x" + std::to_string(variable_index(val[i]) + 1));
                break;
            case Op::Sqrt: stack.back() = "sqrt(" + stack.back() + ")"; break;
            default: {
                const char* symbol = op[i] == Op::Add ? " + "
                                   : op[i] == Op::Subtract ? " - "
                                   : op[i] == Op::Multiply ? " * " : " / ";
                std::string right = std::move(stack.back());
                stack.pop_back();
                stack.back() = "(" + stack.back() + symbol + right + ")";
                break;
            }
        }
    }
    return stack.empty() ? std::string() : stack.back();
}

std::string Program::postfix(int precision) const {
    std::string out;
    const Op* op = ops();
    const double* val = values();
    for (std::size_t i = 0; i < n_; ++i) {
        if (i > 0) out += ' ';
        switch (op[i]) {
            case Op::Constant: out += double_to_string(val[i], precision); break;
            case Op::Variable: out += "x" + std::to_string(variable_index(val[i]) + 1); break;
            case Op::Add: out += '+'; break;
            case Op::Subtract: out += '-'; break;
            case Op::Multiply: out += '*'; break;
            case Op::Divide: out += '/'; break;
            case Op::Sqrt: out += "sqrt"; break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------
Dataset Dataset::from_rows(const std::vector<std::vector<double>>& x,
                           const std::vector<double>& y) {
    if (x.empty() || x.size() != y.size()) {
        throw std::invalid_argument("Dataset vacio o inconsistente");
    }
    const std::size_t vars = x.front().size();
    Dataset d;
    d.columns.assign(vars, std::vector<double>(x.size()));
    for (std::size_t r = 0; r < x.size(); ++r) {
        if (x[r].size() != vars) throw std::invalid_argument("Dimensiones inconsistentes");
        for (std::size_t v = 0; v < vars; ++v) d.columns[v][r] = x[r][v];
    }
    d.y = y;
    return d;
}

Dataset Dataset::from_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("No se pudo abrir: " + filename);
    std::vector<std::vector<double>> x;
    std::vector<double> y;
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
            y.push_back(values.back());
            values.pop_back();
            x.push_back(std::move(values));
        } catch (const std::exception& e) {
            if (line_number == 1) continue;  // encabezado opcional
            throw std::runtime_error("CSV linea " + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (x.empty()) throw std::runtime_error("Dataset sin filas numericas");
    return from_rows(x, y);
}

// ---------------------------------------------------------------------------
// Construccion e inspeccion
// ---------------------------------------------------------------------------
Program make_program(std::span<const Op> ops, std::span<const double> values) {
    if (ops.size() != values.size()) throw std::invalid_argument("ops y values difieren");
    Program p(ops.size());
    std::memcpy(p.ops(), ops.data(), ops.size() * sizeof(Op));
    std::memcpy(p.values(), values.data(), values.size() * sizeof(double));
    if (!is_well_formed(p)) throw std::invalid_argument("Programa postfijo mal formado");
    return p;
}

Program parse_infix(const std::string& text) { return InfixParser(text).run(); }

bool is_well_formed(const Program& p) {
    if (p.empty()) return false;
    std::size_t sp = 0;
    const Op* op = p.ops();
    const double* val = p.values();
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (static_cast<std::uint8_t>(op[i]) > static_cast<std::uint8_t>(Op::Sqrt)) return false;
        const int a = arity(op[i]);
        if (sp < static_cast<std::size_t>(a)) return false;
        if (op[i] == Op::Variable && (val[i] < 0 || val[i] != std::floor(val[i]))) return false;
        sp = sp - static_cast<std::size_t>(a) + 1;
    }
    return sp == 1;
}

std::size_t node_count(const Program& p) { return p.size(); }

int tree_depth(const Program& p) {
    thread_local std::vector<int> stack;
    stack.clear();
    const Op* op = p.ops();
    for (std::size_t i = 0; i < p.size(); ++i) {
        switch (arity(op[i])) {
            case 0: stack.push_back(0); break;
            case 1: ++stack.back(); break;
            default: {
                const int r = stack.back();
                stack.pop_back();
                stack.back() = std::max(stack.back(), r) + 1;
                break;
            }
        }
    }
    return stack.empty() ? -1 : stack.back();
}

// Recorre hacia atras contando operandos pendientes: O(tamano del subarbol).
std::size_t subtree_start(const Program& p, std::size_t root) {
    const Op* op = p.ops();
    std::ptrdiff_t need = 1;
    std::size_t j = root + 1;
    do {
        --j;
        need += arity(op[j]) - 1;
    } while (need > 0);
    return j;
}

// ---------------------------------------------------------------------------
// Generacion
// ---------------------------------------------------------------------------
Program grow_random(int max_depth, std::size_t variable_count, std::mt19937_64& rng) {
    ensure_depth_argument(max_depth);
    Scratch& s = scratch();
    s.clear();
    grow_impl(max_depth, variable_count, rng, false, s);
    return from_scratch(s);
}

Program grow_full(int exact_depth, std::size_t variable_count, std::mt19937_64& rng) {
    ensure_depth_argument(exact_depth);
    Scratch& s = scratch();
    s.clear();
    grow_impl(exact_depth, variable_count, rng, true, s);
    return from_scratch(s);
}

// ---------------------------------------------------------------------------
// Evaluacion
// ---------------------------------------------------------------------------
double eval_row(const Program& p, std::span<const double> variables) {
    thread_local std::vector<double> stack;
    stack.clear();
    const Op* op = p.ops();
    const double* val = p.values();
    for (std::size_t i = 0; i < p.size(); ++i) {
        switch (op[i]) {
            case Op::Constant: stack.push_back(val[i]); break;
            case Op::Variable: {
                const std::size_t v = variable_index(val[i]);
                if (v >= variables.size()) throw std::out_of_range("Indice de variable fuera del vector");
                stack.push_back(variables[v]);
                break;
            }
            case Op::Sqrt: stack.back() = std::sqrt(std::max(stack.back(), 0.0)); break;
            default: {
                const double b = stack.back();
                stack.pop_back();
                double& a = stack.back();
                switch (op[i]) {
                    case Op::Add: a = a + b; break;
                    case Op::Subtract: a = a - b; break;
                    case Op::Multiply: a = a * b; break;
                    default: a = a / (std::abs(b) < kEpsilon ? kEpsilon : b); break;
                }
                break;
            }
        }
    }
    return stack.back();
}

double fitness(const Program& p, const Dataset& data, EvalWorkspace& ws) {
    const std::size_t rows = data.rows();
    if (rows == 0 || data.columns.empty()) throw std::invalid_argument("Dataset vacio");
    if (ws.block == 0) throw std::invalid_argument("block debe ser >= 1");
    const std::size_t n = p.size();
    const Op* op = p.ops();
    const double* val = p.values();

    // Pasada previa O(n): valida indices y calcula la profundidad maxima de pila.
    std::size_t sp = 0;
    std::size_t max_sp = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int a = arity(op[i]);
        if (a == 0) {
            if (op[i] == Op::Variable && variable_index(val[i]) >= data.variables()) {
                throw std::out_of_range("Indice de variable fuera del vector");
            }
            max_sp = std::max(max_sp, ++sp);
        } else {
            if (sp < static_cast<std::size_t>(a)) throw std::invalid_argument("Programa mal formado");
            sp -= static_cast<std::size_t>(a - 1);
        }
    }
    if (sp != 1) throw std::invalid_argument("Programa mal formado");

    const std::size_t B = ws.block;
    if (ws.stack.size() < max_sp * B) ws.stack.resize(max_sp * B);
    double* const st = ws.stack.data();
    const double* const y = data.y.data();

    double total = 0.0;
    for (std::size_t r0 = 0; r0 < rows; r0 += B) {
        const std::size_t L = std::min(B, rows - r0);
        std::size_t top = 0;  // numero de vectores en la pila
        for (std::size_t i = 0; i < n; ++i) {
            switch (op[i]) {
                case Op::Constant:
                    std::fill_n(st + top * B, L, val[i]);
                    ++top;
                    break;
                case Op::Variable:
                    std::copy_n(data.columns[variable_index(val[i])].data() + r0, L, st + top * B);
                    ++top;
                    break;
                case Op::Sqrt: {
                    double* __restrict a = st + (top - 1) * B;
                    for (std::size_t k = 0; k < L; ++k) a[k] = std::sqrt(std::max(a[k], 0.0));
                    break;
                }
                case Op::Add: {
                    double* __restrict a = st + (top - 2) * B;
                    const double* __restrict b = a + B;
                    for (std::size_t k = 0; k < L; ++k) a[k] = a[k] + b[k];
                    --top;
                    break;
                }
                case Op::Subtract: {
                    double* __restrict a = st + (top - 2) * B;
                    const double* __restrict b = a + B;
                    for (std::size_t k = 0; k < L; ++k) a[k] = a[k] - b[k];
                    --top;
                    break;
                }
                case Op::Multiply: {
                    double* __restrict a = st + (top - 2) * B;
                    const double* __restrict b = a + B;
                    for (std::size_t k = 0; k < L; ++k) a[k] = a[k] * b[k];
                    --top;
                    break;
                }
                case Op::Divide: {
                    double* __restrict a = st + (top - 2) * B;
                    const double* __restrict b = a + B;
                    for (std::size_t k = 0; k < L; ++k) {
                        const double d = std::abs(b[k]) < kEpsilon ? kEpsilon : b[k];
                        a[k] = a[k] / d;
                    }
                    --top;
                    break;
                }
            }
        }
        // Suma SECUENCIAL por fila (mismo orden que B): resultados bit a bit iguales.
        for (std::size_t k = 0; k < L; ++k) {
            const double diff = y[r0 + k] - st[k];
            total += diff * diff;
        }
        if (!std::isfinite(total)) return std::numeric_limits<double>::infinity();
    }
    return total / static_cast<double>(rows);
}

double fitness(const Program& p, const Dataset& data) {
    thread_local EvalWorkspace ws;
    return fitness(p, data, ws);
}

// ---------------------------------------------------------------------------
// Operadores geneticos
// ---------------------------------------------------------------------------
std::pair<Program, Program> crossover(const Program& a, const Program& b,
                                      std::mt19937_64& rng, Pick pick) {
    std::size_t ia, ib, sa, sb;
    if (pick == Pick::Fast) {
        ia = random_integer<std::size_t>(rng, 0, a.size() - 1);
        ib = random_integer<std::size_t>(rng, 0, b.size() - 1);
        sa = subtree_start(a, ia);
        sb = subtree_start(b, ib);
    } else {
        Shape& A = shape(0);
        Shape& Bs = shape(1);
        analyze(a, A, false, true);
        analyze(b, Bs, false, true);
        ia = pick_node(a.size(), rng, pick, &A);
        ib = pick_node(b.size(), rng, pick, &Bs);
        sa = ia + 1 - A.size[ia];
        sb = ib + 1 - Bs.size[ib];
    }
    return {splice(a, sa, ia, b.values() + sb, b.ops() + sb, ib - sb + 1),
            splice(b, sb, ib, a.values() + sa, a.ops() + sa, ia - sa + 1)};
}

Program mutation(const Program& parent, std::size_t variable_count, std::mt19937_64& rng,
                 int new_subtree_max_depth, Pick pick) {
    ensure_depth_argument(new_subtree_max_depth);
    // Mismo orden de consumo del RNG que B: profundidad, subarbol nuevo, punto.
    const int replacement_depth = random_integer<int>(rng, 0, new_subtree_max_depth);
    Scratch& s = scratch();
    s.clear();
    grow_impl(replacement_depth, variable_count, rng, false, s);
    std::size_t idx, start;
    if (pick == Pick::Fast) {
        idx = random_integer<std::size_t>(rng, 0, parent.size() - 1);
        start = subtree_start(parent, idx);
    } else {
        Shape& A = shape(0);
        analyze(parent, A, false, true);
        idx = pick_node(parent.size(), rng, pick, &A);
        start = idx + 1 - A.size[idx];
    }
    return splice(parent, start, idx, s.values.data(), s.ops.data(), s.ops.size());
}

std::pair<Program, Program> depth_protected_crossover(const Program& a, const Program& b,
                                                      int max_depth, std::mt19937_64& rng,
                                                      Pick pick) {
    ensure_depth_argument(max_depth);
    const bool preorder = pick == Pick::Preorder;
    Shape& A = shape(0);
    Shape& Bs = shape(1);
    analyze(a, A, true, preorder);
    analyze(b, Bs, true, preorder);
    if (A.height[a.size() - 1] > max_depth || Bs.height[b.size() - 1] > max_depth) {
        throw std::invalid_argument("Un padre supera max_depth");
    }
    // Con profundidad y altura precalculadas, cada intento es una comprobacion O(1):
    //   hijo1 valido <=> depth_A(i) + height_B(j) <= max
    //   hijo2 valido <=> depth_B(j) + height_A(i) <= max
    for (int attempt = 0; attempt < kProtectedCrossoverAttempts; ++attempt) {
        const std::size_t ia = pick_node(a.size(), rng, pick, &A);
        const std::size_t ib = pick_node(b.size(), rng, pick, &Bs);
        if (A.depth[ia] + Bs.height[ib] <= max_depth && Bs.depth[ib] + A.height[ia] <= max_depth) {
            const std::size_t sa = ia + 1 - A.size[ia];
            const std::size_t sb = ib + 1 - Bs.size[ib];
            return {splice(a, sa, ia, b.values() + sb, b.ops() + sb, ib - sb + 1),
                    splice(b, sb, ib, a.values() + sa, a.ops() + sa, ia - sa + 1)};
        }
    }
    // Igual que B: intercambiar las raices siempre respeta el limite.
    return {Program(b), Program(a)};
}

Program depth_protected_mutation(const Program& parent, std::size_t variable_count,
                                 int max_depth, std::mt19937_64& rng, Pick pick) {
    ensure_depth_argument(max_depth);
    Shape& A = shape(0);
    analyze(parent, A, true, pick == Pick::Preorder);
    if (A.height[parent.size() - 1] > max_depth) {
        throw std::invalid_argument("El padre supera max_depth");
    }
    const std::size_t idx = pick_node(parent.size(), rng, pick, &A);
    const int remaining = max_depth - A.depth[idx];
    const int replacement_depth = random_integer<int>(rng, 0, remaining);
    Scratch& s = scratch();
    s.clear();
    grow_impl(replacement_depth, variable_count, rng, false, s);
    const std::size_t start = idx + 1 - A.size[idx];
    return splice(parent, start, idx, s.values.data(), s.ops.data(), s.ops.size());
}

// ---------------------------------------------------------------------------
// GP completo
// ---------------------------------------------------------------------------
GPResult full_gp(const Dataset& data, const GPOptions& options) {
    if (data.variables() == 0 || data.rows() == 0) {
        throw std::invalid_argument("Se necesita un dataset con variables");
    }
    if (options.population_size < 4 || options.population_size % 4 != 0) {
        throw std::invalid_argument("population_size debe ser multiplo de 4 y >= 4");
    }
    ensure_depth_argument(options.max_depth);
    std::mt19937_64 rng(options.seed);
    const std::size_t N = options.population_size;
    EvalWorkspace ws;
    ws.block = options.block;

    std::vector<Individual> population;
    population.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        Program tree = grow_full(options.max_depth, data.variables(), rng);
        const double score = fitness(tree, data, ws);
        population.push_back({std::move(tree), score});
    }

    GPResult result;
    std::vector<const Program*> parent_pool(N);
    std::vector<Individual> offspring;
    offspring.reserve(N);
    for (std::size_t generation = 0; generation <= options.generations; ++generation) {
        const Individual& best = *std::min_element(population.begin(), population.end(), better);
        result.best_fitness_per_generation.push_back(best.score);
        if (generation == options.generations) break;

        // Torneo binario repetido: pool de N ganadores.
        for (std::size_t i = 0; i < N; ++i) {
            const auto& a = population[random_integer<std::size_t>(rng, 0, N - 1)];
            const auto& b = population[random_integer<std::size_t>(rng, 0, N - 1)];
            parent_pool[i] = a.score <= b.score ? &a.tree : &b.tree;
        }
        const auto pick_parent = [&]() -> const Program& {
            return *parent_pool[random_integer<std::size_t>(rng, 0, N - 1)];
        };

        offspring.clear();
        for (std::size_t i = 0; i < N / 4; ++i) {  // N/2 hijos por crossover
            const Program& p1 = pick_parent();
            const Program& p2 = pick_parent();
            auto [c1, c2] = depth_protected_crossover(p1, p2, options.max_depth, rng, options.pick);
            const double f1 = fitness(c1, data, ws);
            const double f2 = fitness(c2, data, ws);
            offspring.push_back({std::move(c1), f1});
            offspring.push_back({std::move(c2), f2});
        }
        for (std::size_t i = 0; i < N / 2; ++i) {  // N/2 hijos por mutacion
            const Program& p = pick_parent();
            Program m = depth_protected_mutation(p, data.variables(), options.max_depth, rng,
                                                 options.pick);
            const double f = fitness(m, data, ws);
            offspring.push_back({std::move(m), f});
        }

        // Reemplazo 50/50: mejores N/2 actuales + mejores N/2 descendientes.
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
    result.best = best.tree;
    result.best_fitness = best.score;
    return result;
}

// ---------------------------------------------------------------------------
// Benchmark de crossover
// ---------------------------------------------------------------------------
CrossoverTiming benchmark_crossover(std::size_t trees, int depth, std::size_t variables,
                                    std::uint64_t seed, std::size_t batch_pairs, Pick pick) {
    if (trees < 2 || trees % 2 != 0 || variables == 0 || batch_pairs == 0) {
        throw std::invalid_argument("trees debe ser par >=2, variables >=1 y batch >=1");
    }
    ensure_depth_argument(depth);
    std::mt19937_64 rng(seed);
    using clock = std::chrono::steady_clock;
    double measured = 0.0;
    double total_nodes = 0.0;
    const std::size_t pairs = trees / 2;
    std::vector<std::pair<Program, Program>> parents;
    std::vector<std::pair<Program, Program>> children;
    parents.reserve(std::min(batch_pairs, pairs));
    children.reserve(std::min(batch_pairs, pairs));

    for (std::size_t offset = 0; offset < pairs; offset += batch_pairs) {
        const std::size_t count = std::min(batch_pairs, pairs - offset);
        parents.clear();
        children.clear();
        for (std::size_t i = 0; i < count; ++i) {
            Program first = grow_full(depth, variables, rng);
            Program second = grow_full(depth, variables, rng);
            total_nodes += static_cast<double>(first.size() + second.size());
            parents.emplace_back(std::move(first), std::move(second));
        }
        const auto start = clock::now();
        for (const auto& p : parents) children.push_back(crossover(p.first, p.second, rng, pick));
        const auto stop = clock::now();
        measured += std::chrono::duration<double>(stop - start).count();
        // Hijos y padres se destruyen fuera del intervalo cronometrado.
    }
    return {depth, trees, pairs, measured, measured / static_cast<double>(pairs),
            total_nodes / static_cast<double>(trees)};
}

// ---------------------------------------------------------------------------
// Pruebas
// ---------------------------------------------------------------------------
void run_tests() {
    const auto check = [](bool condition, const std::string& description) {
        if (!condition) throw std::runtime_error("Fallo en prueba: " + description);
    };
    const auto approx = [](double a, double b) { return std::abs(a - b) <= 1e-10; };
    const auto same_bits = [](double a, double b) {
        return std::memcmp(&a, &b, sizeof(double)) == 0;
    };
    std::mt19937_64 rng(42);

    // --- Parte 1: representacion, expression, eval, MSE --------------------
    const Op f_ops[] = {Op::Variable, Op::Variable, Op::Constant, Op::Multiply, Op::Add};
    const double f_vals[] = {0, 1, 0.5, 0, 0};
    Program formula = make_program(f_ops, f_vals);  // x1 + (x2 * 0.5)
    check(formula.expression() == "(x1 + (x2 * 0.5))", "expression");
    check(formula.postfix() == "x1 x2 0.5 * +", "postfix");
    const std::vector<double> sample{2.0, 3.0};
    check(approx(eval_row(formula, sample), 3.5), "eval");
    check(node_count(formula) == 5, "node_count");
    check(tree_depth(formula) == 2, "tree_depth raiz 0");
    check(subtree_start(formula, 3) == 1 && subtree_start(formula, 4) == 0, "subtree_start");

    const Dataset data = Dataset::from_rows({{2.0, 3.0}, {1.0, 2.0}}, {3.5, 2.0});
    check(approx(fitness(formula, data), 0.0), "MSE cero");

    Program copy = formula;
    copy.values()[2] = 0.7;
    check(approx(eval_row(formula, sample), 3.5), "clonacion profunda");

    const Op div_ops[] = {Op::Constant, Op::Constant, Op::Divide};
    const double div_vals[] = {1.0, 0.0, 0};
    check(approx(eval_row(make_program(div_ops, div_vals), sample), 1e12), "division protegida");
    const Op sq_ops[] = {Op::Constant, Op::Sqrt};
    const double sq_vals[] = {-5.0, 0};
    check(approx(eval_row(make_program(sq_ops, sq_vals), sample), 0.0), "sqrt protegido");

    const Op big_ops[] = {Op::Variable, Op::Constant, Op::Multiply, Op::Constant, Op::Multiply};
    const double big_vals[] = {0, 1e300, 0, 1e300, 0};
    check(std::isinf(fitness(make_program(big_ops, big_vals), data)), "MSE no finito = +inf");

    // Parser del formato de B (ida y vuelta sin perdida con precision 17).
    check(parse_infix("(x1 + (x2 * 0.5))").postfix() == formula.postfix(), "parse_infix");
    check(parse_infix("sqrt((x3 / -0.25))").expression() == "sqrt((x3 / -0.25))", "parse negativo");

    // --- Parte 2: generacion -----------------------------------------------
    for (int depth = 0; depth <= 8; ++depth) {
        Program full = grow_full(depth, 3, rng);
        check(is_well_formed(full) && tree_depth(full) == depth, "grow_full profundidad exacta");
        Program rnd = grow_random(depth, 3, rng);
        check(is_well_formed(rnd) && tree_depth(rnd) <= depth, "grow_random profundidad maxima");
        const Program back = parse_infix(full.expression(17));
        check(back.size() == full.size()
              && std::memcmp(back.ops(), full.ops(), full.size()) == 0
              && std::memcmp(back.values(), full.values(), full.size() * sizeof(double)) == 0,
              "ida y vuelta expression(17) -> parse_infix");
    }

    // Evaluacion por lotes == evaluacion escalar fila por fila, bit a bit,
    // para varios tamanos de bloque (incluye bloques parciales).
    {
        std::uniform_real_distribution<double> u(-3.0, 3.0);
        std::vector<std::vector<double>> x(150, std::vector<double>(3));
        std::vector<double> y(150);
        for (std::size_t r = 0; r < x.size(); ++r) {
            for (auto& v : x[r]) v = (r % 17 == 0) ? 0.0 : u(rng);
            y[r] = u(rng);
        }
        const Dataset d = Dataset::from_rows(x, y);
        for (int t = 0; t < 300; ++t) {
            const Program p = grow_random(1 + t % 8, 3, rng);
            double ref = 0.0;
            bool finite = true;
            for (std::size_t r = 0; r < x.size() && finite; ++r) {
                const double diff = y[r] - eval_row(p, x[r]);
                ref += diff * diff;
                finite = std::isfinite(ref);
            }
            ref = finite ? ref / static_cast<double>(x.size())
                         : std::numeric_limits<double>::infinity();
            for (std::size_t block : {1u, 7u, 64u, 256u}) {
                EvalWorkspace ws;
                ws.block = block;
                check(same_bits(fitness(p, d, ws), ref), "eval por lotes == escalar");
            }
        }
    }

    // subtree_start coincide con los tamanos calculados por analyze.
    for (int t = 0; t < 50; ++t) {
        const Program p = grow_random(7, 2, rng);
        Shape& s = shape(0);
        analyze(p, s, true, true);
        for (std::size_t i = 0; i < p.size(); ++i) {
            check(subtree_start(p, i) == i + 1 - s.size[i], "subtree_start vs size");
        }
        check(s.pre2post.size() == p.size() && s.pre2post[0] == p.size() - 1, "preorden");
        check(s.height[p.size() - 1] == tree_depth(p), "altura raiz");
    }

    // --- Crossover y mutacion -----------------------------------------------
    Program a = grow_full(4, 2, rng);
    Program b = grow_full(3, 2, rng);
    const std::string old_a = a.expression(17);
    const std::string old_b = b.expression(17);
    for (Pick pick : {Pick::Fast, Pick::Preorder}) {
        for (int i = 0; i < 100; ++i) {
            auto [c, d] = crossover(a, b, rng, pick);
            check(is_well_formed(c) && is_well_formed(d), "crossover produce arboles validos");
            check(c.size() + d.size() == a.size() + b.size(), "crossover conserva nodos");
            Program m = mutation(a, 2, rng, 3, pick);
            check(is_well_formed(m), "mutacion valida");
            auto [x, y] = depth_protected_crossover(a, b, 4, rng, pick);
            check(tree_depth(x) <= 4 && tree_depth(y) <= 4, "crossover protegido");
            Program z = depth_protected_mutation(a, 2, 4, rng, pick);
            check(is_well_formed(z) && tree_depth(z) <= 4, "mutacion protegida");
        }
    }
    check(a.expression(17) == old_a && b.expression(17) == old_b, "operadores no alteran padres");

    // --- GP completo ----------------------------------------------------------
    GPOptions options;
    options.population_size = 24;
    options.max_depth = 3;
    options.generations = 3;
    GPResult run = full_gp(data, options);
    check(!run.best.empty(), "full_gp devuelve arbol");
    check(run.best_fitness_per_generation.size() == 4, "historial de fitness");
    for (std::size_t i = 1; i < run.best_fitness_per_generation.size(); ++i) {
        check(run.best_fitness_per_generation[i] <= run.best_fitness_per_generation[i - 1],
              "elitismo: mejor MSE no aumenta");
    }
    // El tamano de bloque no debe cambiar NADA del resultado.
    GPOptions o1 = options;
    o1.population_size = 40;
    o1.max_depth = 5;
    o1.block = 1;
    GPOptions o2 = o1;
    o2.block = 64;
    const GPResult r1 = full_gp(data, o1);
    const GPResult r2 = full_gp(data, o2);
    check(r1.best_fitness_per_generation == r2.best_fitness_per_generation
          && r1.best.expression(17) == r2.best.expression(17), "bloque no altera el GP");
}

}  // namespace gpc
