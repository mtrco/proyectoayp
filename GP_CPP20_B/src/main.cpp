//Cambios, no usa std:: ya que esta definido directamente ya en gp.hpp, se elimina la libreria string, ya esta incluida en gp.hp
//Se acomoda la impresión de los datos para que sean mas entendibles y se vean acomodados.
#include "gp.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
struct Args {
     string mode = "--help";
     string data = "data/dataset1a.csv";
     string depths;
     string file;
     string history;  // --history=archivo.csv: mejor MSE de cada generacion
     size_t trees = 10000;
     size_t count = 100;
    int depth = 5;
    bool full = false;
    bool fold_constants = true; // --no-fold lo apaga
    bool early_cut = true;      // --no-cut lo apaga
    bool order_by_parent = true; // --no-order lo apaga
    bool skip_duplicates = true; // --no-dup lo apaga
    bool quiet = false;          // --quiet: no mostrar el MSE de cada generacion en vivo
     size_t population = 5000;
     size_t generations = 100; // El manual no lo fija; se acordo usar 100.
     size_t variables = 2;
     size_t batch_pairs = 0; // 0 = los 10000 arboles en memoria a la vez
     uint64_t seed = 0;
    bool seed_given = false; // sin --seed se elige una al azar (resolve_seed)
};

//  stoull acepta "-1" y lo convierte en 18446744073709551615: se exige un
// entero sin signo escrito solo con digitos.
 uint64_t parse_unsigned(const  string& key, const  string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") !=  string::npos) {
        throw  invalid_argument(key + " requiere un entero no negativo: " + value);
    }
    return  stoull(value);
}

Args parse_args(int argc, char** argv) {
    Args opts;
    if (argc > 1) opts.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const  string arg = argv[i];
        if (arg == "--full") { opts.full = true; continue; }
        if (arg == "--no-fold") { opts.fold_constants = false; continue; }
        if (arg == "--no-cut") { opts.early_cut = false; continue; }
        if (arg == "--no-order") { opts.order_by_parent = false; continue; }
        if (arg == "--no-dup") { opts.skip_duplicates = false; continue; }
        if (arg == "--quiet") { opts.quiet = true; continue; }
        const  size_t equal = arg.find('=');
        if (equal ==  string::npos) {
            throw  invalid_argument("Se esperaba --opcion=valor: " + arg);
        }
        const  string key = arg.substr(0, equal);
        const  string value = arg.substr(equal + 1);
        if (key == "--data") opts.data = value;
        else if (key == "--depths") opts.depths = value;
        else if (key == "--file") opts.file = value;
        else if (key == "--history") opts.history = value;
        else if (key == "--count") opts.count = parse_unsigned(key, value);
        else if (key == "--depth") opts.depth =  stoi(value);
        else if (key == "--trees") opts.trees = parse_unsigned(key, value);
        else if (key == "--population") opts.population = parse_unsigned(key, value);
        else if (key == "--generations") opts.generations = parse_unsigned(key, value);
        else if (key == "--variables") opts.variables = parse_unsigned(key, value);
        else if (key == "--seed") { opts.seed = parse_unsigned(key, value); opts.seed_given = true; }
        else if (key == "--batch-pairs") opts.batch_pairs = parse_unsigned(key, value);
        else throw  invalid_argument("Opcion desconocida: " + key);
    }
    return opts;
}

// Sin --seed, una semilla al azar (de 1 a 999999, facil de copiar). Se imprime por stderr
// para no mezclarla con la salida normal y poder repetir la corrida con --seed=S.
// random_device se mezcla con el reloj por si en algun compilador fuera determinista.
void resolve_seed(Args& args) {
    if (args.seed_given) return;
    random_device device;
    const uint64_t noise =
        (static_cast<uint64_t>(device()) << 32) ^ device() ^
        static_cast<uint64_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
    args.seed = 1 + mt19937_64(noise)() % 999999;
    cerr << "Semilla aleatoria: " << args.seed << " (para repetir esta corrida: --seed="
         << args.seed << ")" << endl;
}

 vector<int> parse_depths( string list) {
     vector<int> result;
     stringstream stream(list);
     string item;
    while ( getline(stream, item, ',')) {
        result.push_back( stoi(item));
    }
    if (result.empty()) throw  invalid_argument("Lista de profundidades vacia");
    return result;
}

void print_help() {
     cout << R"(GP B - C++20, implementacion convencional

  gp_b --test
  gp_b --demo [--data=data/ejemplo.csv]
  gp_b --bench-crossover [--trees=10000] [--depths=7,9,10,12,15]
                          [--variables=2] [--seed=S] [--batch-pairs=0]
  gp_b --bench-gp [--data=dataset.csv] [--population=5000]
                  [--depths=7,9,12] [--generations=100] [--seed=S] [--history=archivo.csv]
                  [--no-fold] [--no-cut] [--no-order] [--no-dup] [--quiet]
  gp_b --eval-file --file=arboles.txt [--data=dataset.csv]
       Lee una expresion por linea e imprime MSE (17 digitos), profundidad,
       nodos y la expresion, separados por tabuladores (para comparar con A y C).
  gp_b --export-trees [--count=100] [--depth=5] [--variables=2] [--seed=S] [--full]
       Escribe arboles grow_random (o grow_full) sin perdida, uno por linea.

El CSV debe tener cabecera opcional, x1,x2,...,y, y como ultima columna.
--seed: semilla del generador aleatorio. Sin --seed se elige una al azar y se imprime
(por stderr) para poder repetir la corrida con --seed=S. Los scripts de benchmark siempre
pasan una (5 por defecto).
El manual no fija el numero de generaciones; se acordo usar 100 (valor por defecto).
--history=archivo.csv: escribe el mejor MSE de cada generacion (depth,generation,best_mse,
con 17 digitos), fuera del cronometro. La generacion 0 es la poblacion inicial.
--bench-gp muestra en vivo, por la salida de errores (stderr), el mejor MSE de cada generacion
y el tiempo que lleva; la salida normal (stdout) queda como CSV. El tiempo de imprimir esas
lineas se descuenta del cronometro. --quiet las apaga.
En el benchmark de crossover se mide solo el crossover, no generar padres.
Por defecto los 10000 padres estan en memoria a la vez (protocolo acordado).
--batch-pairs=2500 deja 5000 a la vez si no alcanza la RAM.
--no-fold: no plegar subarboles sin variables. --no-cut: sin corte temprano exacto.
--no-order: evaluar los hijos en orden de creacion (por defecto, primero los de mejores padres).
--no-dup: evaluar tambien los hijos identicos a su padre (por defecto reciben el MSE del padre).
Las cuatro optimizaciones dan el mismo resultado; se pueden apagar si el profesor no las acepta.
)";
}

void run_gp_benchmark(const gp::Dataset& data, const  vector<int>& depths,
                      const Args& args) {
     ofstream history;
    if (!args.history.empty()) {
        history.open(args.history);
        if (!history) throw  runtime_error("No se pudo crear: " + args.history);
        history << "depth \t generation \t best_mse\n" <<  setprecision(17);
    }
    for (int depth : depths) {
        gp::GPOptions opts;
        opts.population_size = args.population;
        opts.max_depth = depth;
        opts.generations = args.generations;
        opts.seed = args.seed;
        opts.fold_constants = args.fold_constants;
        opts.early_cut = args.early_cut;
        opts.order_by_parent = args.order_by_parent;
        opts.skip_duplicates = args.skip_duplicates;
        // Avance en vivo por stderr (stdout queda como CSV para los scripts). Lo que tarda
        // imprimir se acumula en `paused` y se descuenta del tiempo medido.
        using clock =  chrono::steady_clock;
        clock::time_point start;
        clock::duration paused{};
        if (!args.quiet) {
            opts.on_generation = [&]( size_t generation, double best) {
                const auto now = clock::now();
                const double so_far =  chrono::duration<double>(now - start - paused).count();
                char line[160];
                 snprintf(line, sizeof line, " depth %d  gen %3zu/%zu  MSE %.10g  (%.1f s)\n",
                              depth, generation, opts.generations, best, so_far);
                 fputs(line, stderr);
                 fflush(stderr);
                paused += clock::now() - now;
            };
        }
        start = clock::now();
        auto result = gp::full_gp(data, opts);
        const auto stop = clock::now();
        const double seconds =  chrono::duration<double>(stop - start - paused).count();
        if (!args.quiet) {  // cuanto costo imprimir el avance (ya descontado de `seconds`)
             fprintf(stderr, "  prof %d  tiempo de imprimir el avance (descontado): %.3f ms\n",
                         depth,  chrono::duration<double,  milli>(paused).count());
        }
     cout << "depth ||  population  ||  generations  ||  elapsed_seconds  ||  best_mse \n";
         cout << depth << "     ||     " << opts.population_size << "      ||   " << opts.generations << "     ||     "<<  setprecision(9) << seconds << "  ||    " << result.best_fitness <<endl;
cout<<"best_expression\n\""<< result.best->expression() << "\"\n";
         cout.flush();
        // Mejor MSE de cada generacion (0 = poblacion inicial), despues del cronometro.
        if (history) {
            const auto& per_generation = result.best_fitness_per_generation;
            for ( size_t g = 0; g < per_generation.size(); ++g) {
                history << depth << "  ||  " << g << "  ||  " << per_generation[g] << '\n';
            }
            history.flush();
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.mode == "--demo" || args.mode == "--bench-crossover" ||
            args.mode == "--bench-gp" || args.mode == "--export-trees") {
            resolve_seed(args);
        }
        if (args.mode == "--help" || args.mode == "-h") {
            print_help();
        } else if (args.mode == "--test") {
            gp::run_tests();
             cout << "OK: todas las pruebas terminaron correctamente.\n";
        } else if (args.mode == "--demo") {
            const gp::Dataset data = gp::Dataset::from_csv(args.data);
            gp::GPOptions options;
            options.population_size = 40;
            options.max_depth = 4;
            options.generations = 5;
            options.seed = args.seed;
            options.fold_constants = args.fold_constants;
            options.early_cut = args.early_cut;
            options.order_by_parent = args.order_by_parent;
            gp::GPResult result = gp::full_gp(data, options);
             cout << "MSE final: " << result.best_fitness << '\n';
             cout << "Expresion: " << result.best->expression() << '\n';
             cout << "Mejor MSE por generacion:\n";
            for ( size_t i = 0; i < result.best_fitness_per_generation.size(); ++i) {
                 cout << "  Generacion " << i << ": "
                          << result.best_fitness_per_generation[i] << '\n';
            }
        } else if (args.mode == "--bench-crossover") {
            const auto depths = parse_depths(args.depths.empty()
                ? "7,9,10,12,15" : args.depths);
             cout << "depth  ||  trees  ||  resident_trees  ||  crossovers  ||  elapsed_seconds,"
                         "mean_seconds_per_crossover  ||  mean_nodes\n";
            for (int depth : depths) {
                const auto time = gp::benchmark_crossover(args.trees, depth,
                                                            args.variables, args.seed,
                                                            args.batch_pairs);
                 cout << time.depth << "  ||  " << time.trees << "  ||  " << time.resident_trees
                          << "  ||  " << time.crossovers << "  ||  " <<  setprecision(9)
                          << time.seconds << "  ||  " << time.mean_seconds_per_crossover << "  ||  "
                          <<  setprecision(6) << time.mean_nodes_per_tree <<  endl;
            }
        } else if (args.mode == "--eval-file") {
            if (args.file.empty()) throw  invalid_argument("Falta --file=arboles.txt");
            const gp::Dataset data = gp::Dataset::from_csv(args.data);
             ifstream in(args.file);
            if (!in) throw  runtime_error("No se pudo abrir: " + args.file);
             string line;
             cout <<  setprecision(17);
            while ( getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.find_first_not_of(" \t") ==  string::npos) continue;
                const gp::Tree tree = gp::parse_expression(line);
                 cout << gp::fitness(*tree, data) << '\t' << gp::tree_depth(*tree) << '\t'
                          << gp::node_count(*tree) << '\t' << line << '\n';
            }
        } else if (args.mode == "--export-trees") {
             mt19937_64 rng(args.seed);
            for ( size_t i = 0; i < args.count; ++i) {
                const gp::Tree tree = args.full ? gp::grow_full(args.depth, args.variables, rng)
                                                : gp::grow_random(args.depth, args.variables, rng);
                 cout << tree->expression(17) << '\n';
            }
        } else if (args.mode == "--bench-gp") {
            const gp::Dataset data = gp::Dataset::from_csv(args.data);
            const auto depths = parse_depths(args.depths.empty() ? "7,9,12" : args.depths);
            run_gp_benchmark(data, depths, args);
        } else {
            throw  invalid_argument("Modo no reconocido: " + args.mode);
        }
        return 0;
    } catch (const  exception& error) {
         cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}