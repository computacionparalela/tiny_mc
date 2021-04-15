#include <bits/stdc++.h>
#include <unistd.h>
#include "t_test.h"
using namespace std;

vector<string> compiler_flags = {"-parallel", "-m64", "-qopenmp-offload", "-qopenmp", "-qopt-report", "-qopt-prefetch", "-fimf-precision=simple", "-no-prec-sqrt", "-no-prec-div", "-Istd", "-Istdi", "-Lstd", "-march=native", "-fp-speculation", "-fast-transcendentals"};

vector<float> compiler_with(const vector<string>& flags)
{
    // Create the shell command
    string compile_opt("OPT_FLAGS=\""), command;
    for (string f:flags) {
        compile_opt.append(f);
        compile_opt.push_back(' ');
    }
    // Execute the program and save the output at report.out
    command = string("make clean > /dev/null && make CC=icc") + compile_opt +
              string("\" > /dev/null");
    system(command.c_str());
    command = string("./run_iter.sh > report.out");
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
    unordered_set<string> current_state(state.begin(), state.end());
    bool changed;

    do {
        cout << "Current State: ";
        print_vector(vector<string>(current_state.begin(), current_state.end()));
        changed = false;
        unordered_set<string> next_state(current_state);
        vector<float> base = compiler_with(vector<string>(current_state.begin(), current_state.end()));
        for (string flag:current_state) {
            unordered_set<string> step(current_state);
            step.erase(flag);
            vector<float> rstep = compiler_with(vector<string>(step.begin(), step.end()));
            if (t_test(base, rstep, false) == REMOVE) {
                current_state.erase(flag);
                changed = true;
                break;
            }
        }
    } while (changed);

    vector<float> base = compiler_with(vector<string>(state.begin(), state.end()));
    vector<float> rstep = compiler_with(vector<string>(current_state.begin(), current_state.end()));
    t_test(base, rstep, true);

    cout << "\n****Best Time: ";
    print_vector(vector<string>(current_state.begin(), current_state.end()));
}

int main(void)
{
	system(". /opt/intel/oneapi/setvars.sh");
    cout << "Compilaciones falsas" << flush;
    for (int i = 0; i < 5; i++) {
        compiler_with({});
        cout << '.' << flush;
    }
    cout << endl;

    cout << "Comienzo seleccion de flags" << endl;
    for (auto Oflag:{"-O0", "-O1", "-O2", "-O3"}) {
        vector<string> v(compiler_flags);
        v.push_back(Oflag);
        seleccion(v);
    }

    system("make clean > /dev/null && rm report.out");

    return 0;
}
// g++-10 -o "optimization_flags" "optimization_flags.cc" -lm -std=c++17 -O3 -Wno-unused-result
