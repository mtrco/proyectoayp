// Verificacion cruzada B (nodos + unique_ptr) vs C (arreglo postfijo).
// Enlaza ambas implementaciones en un solo ejecutable y comprueba:
//   1. Generacion: con la misma semilla, grow_full/grow_random producen el MISMO
//      arbol (misma expresion y mismo estado final del generador).
//   2. Conversion: B -> texto -> C y C -> B conservan el arbol exacto.
//   3. Evaluacion y MSE: resultados identicos BIT A BIT en varios datasets.
//   4. Operadores: crossover, mutacion y sus versiones protegidas producen los
//      mismos hijos con la misma semilla (C en modo Preorder).
//   5. GP completo: mismo historial de mejor MSE por generacion y mismo mejor arbol.
//
// Uso: cross_check [ruta/ejemplo.csv]

#include "gp.hpp"   // implementacion B
#include "gpc.hpp"  // implementacion C

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 20) std::cerr << "  FALLO: " << what << '\n';
    }
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// B -> texto sin perdida (mismo formato que expression(), 17 digitos).
std::string b_text(const gp::Node& n) {
    switch (n.kind) {
        case gp::Kind::Constant: {
            std::ostringstream o;
            o << std::setprecision(17) << n.value;
            return o.str();
        }
        case gp::Kind::Variable: return "x" + std::to_string(n.variable_index + 1);
        case gp::Kind::Sqrt: return "sqrt(" + b_text(*n.left) + ")";
        default: break;
    }
    const char* s = n.kind == gp::Kind::Add ? " + " : n.kind == gp::Kind::Subtract ? " - "
                  : n.kind == gp::Kind::Multiply ? " * " : " / ";
    return "(" + b_text(*n.left) + s + b_text(*n.right) + ")";
}

// C -> B construyendo los nodos con una pila.
gp::Tree to_b(const gpc::Program& p) {
    std::vector<gp::Tree> stack;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double v = p.values()[i];
        switch (p.ops()[i]) {
            case gpc::Op::Constant: stack.push_back(gp::Node::constant(v)); break;
            case gpc::Op::Variable: stack.push_back(gp::Node::variable(static_cast<std::size_t>(v))); break;
            case gpc::Op::Sqrt: stack.back() = gp::Node::unary(gp::Kind::Sqrt, std::move(stack.back())); break;
            default: {
                gp::Tree r = std::move(stack.back());
                stack.pop_back();
                const gpc::Op op = p.ops()[i];
                const gp::Kind k = op == gpc::Op::Add ? gp::Kind::Add
                                 : op == gpc::Op::Subtract ? gp::Kind::Subtract
                                 : op == gpc::Op::Multiply ? gp::Kind::Multiply : gp::Kind::Divide;
                stack.back() = gp::Node::binary(k, std::move(stack.back()), std::move(r));
                break;
            }
        }
    }
    return std::move(stack.back());
}

gpc::Program to_c(const gp::Node& n) { return gpc::parse_infix(b_text(n)); }

bool same_tree(const gp::Node& b, const gpc::Program& c) {
    return b_text(b) == c.expression(17) && b.expression() == c.expression(8);
}

struct Pair {
    gp::Dataset b;
    gpc::Dataset c;
    std::string name;
};

Pair make_pair_from_rows(const std::vector<std::vector<double>>& x, const std::vector<double>& y,
                         const std::string& name) {
    return {gp::Dataset{x, y}, gpc::Dataset::from_rows(x, y), name};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string csv = argc > 1 ? argv[1] : "data/ejemplo.csv";

    // ---- Datasets -----------------------------------------------------------
    std::vector<Pair> datasets;
    {
        const gp::Dataset b = gp::Dataset::from_csv(csv);
        datasets.push_back({b, gpc::Dataset::from_csv(csv), csv});
    }
    {
        // Dataset "hostil": ceros (division protegida), negativos (sqrt
        // protegida), magnitudes grandes y pequenas.
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<double> u(-3.0, 3.0);
        std::vector<std::vector<double>> x(203, std::vector<double>(4));
        std::vector<double> y(x.size());
        for (std::size_t r = 0; r < x.size(); ++r) {
            for (std::size_t v = 0; v < 4; ++v) {
                const double base = u(rng);
                x[r][v] = (r % 11 == v) ? 0.0 : (r % 29 == 0 ? base * 1e8 : (r % 31 == 0 ? base * 1e-9 : base));
            }
            y[r] = u(rng) * 10.0;
        }
        datasets.push_back(make_pair_from_rows(x, y, "sintetico_hostil(203x4)"));
    }
    {
        // Valores enormes: fuerza desbordamientos (MSE = +inf) y NaN intermedios.
        std::mt19937_64 rng(11);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::vector<std::vector<double>> x(97, std::vector<double>(2));
        std::vector<double> y(x.size());
        for (std::size_t r = 0; r < x.size(); ++r) {
            x[r][0] = u(rng) * 1e150;
            x[r][1] = (r % 5 == 0) ? 0.0 : u(rng) * 1e-150;
            y[r] = u(rng);
        }
        datasets.push_back(make_pair_from_rows(x, y, "desbordamiento(97x2)"));
    }

    // ---- 1. Generacion -----------------------------------------------------
    std::cout << "[1] Generacion con la misma semilla...\n";
    for (std::uint64_t seed = 1; seed <= 40; ++seed) {
        for (int depth = 0; depth <= 9; ++depth) {
            for (std::size_t vars = 1; vars <= 4; ++vars) {
                std::mt19937_64 rb(seed), rc(seed);
                gp::Tree bf = gp::grow_full(depth, vars, rb);
                gpc::Program cf = gpc::grow_full(depth, vars, rc);
                check(same_tree(*bf, cf), "grow_full igual");
                gp::Tree br = gp::grow_random(depth, vars, rb);
                gpc::Program cr = gpc::grow_random(depth, vars, rc);
                check(same_tree(*br, cr), "grow_random igual");
                check(rb == rc, "mismo estado del RNG tras generar");
                check(gp::tree_depth(*bf) == gpc::tree_depth(cf)
                      && gp::node_count(*bf) == gpc::node_count(cf), "misma profundidad y tamano");
            }
        }
    }

    // ---- 2 y 3. Conversion, eval y MSE ---------------------------------------
    std::cout << "[2-3] Conversion y MSE bit a bit...\n";
    std::mt19937_64 rng(123);
    std::size_t infinite = 0;
    for (const Pair& d : datasets) {
        for (int t = 0; t < 3000; ++t) {
            const std::size_t vars = d.c.variables();
            gp::Tree b = gp::grow_random(1 + t % 10, vars, rng);
            const gpc::Program c = to_c(*b);
            check(same_tree(*b, c), "B -> C conserva el arbol");
            const gp::Tree back = to_b(c);
            check(b_text(*back) == b_text(*b), "C -> B conserva el arbol");
            const double fb = gp::fitness(*b, d.b);
            const double fc = gpc::fitness(c, d.c);
            check(same_bits(fb, fc), "MSE identico en " + d.name);
            if (std::isinf(fb)) ++infinite;
            const std::vector<double> row = d.b.x[t % d.b.x.size()];
            check(same_bits(b->eval(row), gpc::eval_row(c, row)), "eval identico");
        }
    }
    std::cout << "     (" << infinite << " arboles con MSE = +inf, tambien coinciden)\n";

    // ---- 4. Operadores geneticos -------------------------------------------
    std::cout << "[4] Operadores con la misma semilla (C en modo Preorder)...\n";
    for (std::uint64_t seed = 1; seed <= 300; ++seed) {
        std::mt19937_64 g(seed);
        const int max_depth = 2 + static_cast<int>(seed % 7);
        gp::Tree pa = gp::grow_full(max_depth, 3, g);
        gp::Tree pb = gp::grow_random(max_depth, 3, g);
        const gpc::Program ca = to_c(*pa);
        const gpc::Program cb = to_c(*pb);

        std::mt19937_64 rb(seed * 977), rc(seed * 977);
        auto [b1, b2] = gp::crossover(*pa, *pb, rb);
        auto [c1, c2] = gpc::crossover(ca, cb, rc, gpc::Pick::Preorder);
        check(same_tree(*b1, c1) && same_tree(*b2, c2), "crossover igual");

        gp::Tree bm = gp::mutation(*pa, 3, rb);
        gpc::Program cm = gpc::mutation(ca, 3, rc, 3, gpc::Pick::Preorder);
        check(same_tree(*bm, cm), "mutation igual");

        auto [b3, b4] = gp::depth_protected_crossover(*pa, *pb, max_depth, rb);
        auto [c3, c4] = gpc::depth_protected_crossover(ca, cb, max_depth, rc, gpc::Pick::Preorder);
        check(same_tree(*b3, c3) && same_tree(*b4, c4), "crossover protegido igual");

        gp::Tree bp = gp::depth_protected_mutation(*pa, 3, max_depth, rb);
        gpc::Program cp = gpc::depth_protected_mutation(ca, 3, max_depth, rc, gpc::Pick::Preorder);
        check(same_tree(*bp, cp), "mutacion protegida igual");
        check(rb == rc, "mismo estado del RNG tras los operadores");
    }

    // ---- 5. GP completo -------------------------------------------------------
    std::cout << "[5] GP completo con la misma semilla...\n";
    struct Case { std::size_t pop; int depth; std::size_t gens; std::uint64_t seed; };
    const Case cases[] = {{40, 3, 8, 1}, {100, 4, 10, 2}, {200, 5, 10, 3}, {400, 6, 8, 4}, {120, 7, 6, 5}};
    for (const Pair& d : datasets) {
        for (const Case& k : cases) {
            gp::GPOptions ob;
            ob.population_size = k.pop;
            ob.max_depth = k.depth;
            ob.generations = k.gens;
            ob.seed = k.seed;
            gpc::GPOptions oc;
            oc.population_size = k.pop;
            oc.max_depth = k.depth;
            oc.generations = k.gens;
            oc.seed = k.seed;
            oc.pick = gpc::Pick::Preorder;
            const gp::GPResult rb = gp::full_gp(d.b, ob);
            const gpc::GPResult rc = gpc::full_gp(d.c, oc);
            bool same_history = rb.best_fitness_per_generation.size() == rc.best_fitness_per_generation.size();
            for (std::size_t i = 0; same_history && i < rb.best_fitness_per_generation.size(); ++i) {
                same_history = same_bits(rb.best_fitness_per_generation[i], rc.best_fitness_per_generation[i]);
            }
            check(same_history, "historial de MSE identico (" + d.name + ", pop=" + std::to_string(k.pop)
                                + ", depth=" + std::to_string(k.depth) + ")");
            check(same_tree(*rb.best, rc.best), "mismo mejor arbol");
            std::cout << "     " << d.name << " pop=" << k.pop << " depth=" << k.depth
                      << " -> MSE final B=" << std::setprecision(10) << rb.best_fitness
                      << "  C=" << rc.best_fitness << '\n';
        }
    }

    std::cout << "\nComprobaciones: " << checks << ", fallos: " << failures << '\n';
    if (failures == 0) std::cout << "OK: B y C se comportan de forma identica.\n";
    return failures == 0 ? 0 : 1;
}
