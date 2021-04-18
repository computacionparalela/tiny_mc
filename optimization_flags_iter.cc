#include <bits/stdc++.h>
#include <unistd.h>
#include "t_test.h"
using namespace std;

#ifdef GCC
vector<string> Oflags = {"-O1", "-O2", "-O3", "-Ofast"};
vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math", "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", "-ffast-math", "-funroll-loops", "-funroll-all-loops",
 "-fpeel-loops", "-flto", "-fprefetch-loop-arrays", "-mfma", "-mrecip", "-msse2"};
string make_command("make clean > /dev/null && make CC=gcc-10 ");
#endif

#ifdef CLANG
vector<string> Oflags = {"-O0", "-O1", "-O2", "-O3"};
vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-freciprocal-math", "-ffinite-math-only",
 "-fno-signed-zeros", "-fno-trapping-math", "-ffast-math", "-funroll-loops", "-flto", "-mfma", "-mrecip", "-msse2"};
string make_command("make clean > /dev/null && make CC=clang-11 ");
#endif

#ifdef INTEL
vector<string> Oflags = {"-O0", "-O1", "-O2", "-O3"};
vector<string> compiler_flags =
{"-parallel", "-m64", "-qopenmp-offload", "-qopenmp", "-qopt-prefetch", "-fimf-precision=simple",
 "-no-prec-sqrt", "-no-prec-div", "-Istd", "-Istdi", "-Lstd", "-fast-transcendentals"};
string make_command("make clean > /dev/null && make CC=icc ");
#endif

vector<float> compiler_with(const vector<string>& flags)
{
    // Create the shell command
    string compile_opt("OPT_FLAGS=\""), command;
    for (string f:flags) {
        compile_opt.append(f);
        compile_opt.push_back(' ');
    }
    // Execute the program and save the output at report.out
    command = make_command + compile_opt + string("\" > /dev/null");
    system(command.c_str());
    command = string("./run.sh > report.out");
    sleep(1);
    system(command.c_str());
    // Clean the output and get the average runtime
    float ms = 0;
    vector<float> result;
    std::fstream fs("report.out", std::fstream::in | std::fstream::out);
    fs >> command;
    while (command != ">") {
        fs >> command;
    }
    while (command == ">") {
        command.clear();
        fs >> ms >> command;
        result.push_back(ms);
    }

    return result;
}

void print_vector(const vector<string>& vs)
{
    if (vs.empty()) {
        cout << "Empty" << endl;
        return;
    }
    for (auto v:vs) {
        cout << v << ' ';
    }
    cout << endl;
}

void seleccion(vector<string>& state)
{
    srand(time(NULL));
    random_shuffle(state.begin(), state.end());
    vector<string> current_state(state.begin(), state.end());
    bool changed;

    cout << "Start with: ";
    print_vector(current_state);

    cout << "Remove: " << flush;
    do {
        changed = false;
        random_shuffle(current_state.begin(), current_state.end());
        vector<float> base = compiler_with(current_state);
        for (string flag:current_state) {
            vector<string> step(current_state.size());
            for (string f:current_state) {
                if (f != flag) {
                    step.push_back(f);
                }
            }
            vector<float> rstep = compiler_with(step);
            if (t_test(base, rstep, false) == REMOVE) {
                cout << flag << ' ' << flush;
                current_state = step;
                changed = true;
                break;
            }
        }
    } while (changed);
    cout << endl;
    vector<float> base = compiler_with(state);
    vector<float> rstep = compiler_with(current_state);
    t_test(base, rstep, true);

    cout << "\n****Best Photons: ";
    print_vector(current_state);
}

int main(void)
{
    // ~ system(". /opt/intel/oneapi/setvars.sh");
    cout << "Compilaciones falsas" << flush;
    for (int i = 0; i < 5; i++) {
        compiler_with({});
        cout << '.' << flush;
    }
    cout << endl;

    cout << "Comienzo seleccion de flags" << endl;
    for (auto Oflag:Oflags) {
        vector<string> v(compiler_flags);
        v.push_back(Oflag);
        seleccion(v);
        cout << endl;
    }

    system("make clean > /dev/null && rm report.out");

    return 0;
}
// g++-10 -o "optimization_flags" "optimization_flags.cc" -lm -std=c++17 -O3 -Wno-unused-result
