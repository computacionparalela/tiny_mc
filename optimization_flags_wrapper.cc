#include <bits/stdc++.h>
#include <unistd.h>
#define MAXN 2147483647
using namespace std;

vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math",
 "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", "-ffast-math",
 "-funroll-loops", "-funroll-all-loops", "-fpeel-loops", "-flto",
 "-fprefetch-loop-arrays", "-mfma", "-mrecip", "-msse2"};

float compiler_with(const vector<string>& flags)
{
    // Create the shell command
    string compile_opt("OPT_FLAGS=\""), command;
    for (string f:flags) {
        compile_opt.append(f);
        compile_opt.push_back(' ');
    }
    // Execute the program and save the output at report.out
    command = string("make clean > /dev/null && make ") + compile_opt +
              string("\" > /dev/null");
    system(command.c_str());
    command = string("./run.sh > report.out");
    sleep(1);
    system(command.c_str());
    // Clean the output and get the average runtime
    float ms = 0;
    std::fstream fs("report.out", std::fstream::in | std::fstream::out);
    while (command != ">>") {
        fs >> command;
    }
    fs >> command >> ms;

    return ms;
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

void seleccion(vector<string>& next_step)
{
    vector<pair<float, vector<string> > > results;

    float minimal = compiler_with(next_step);
    //cout << "Ms: " << minimal << "\tFlags: "; print_vector(next_step);
    results.push_back({minimal, next_step});

    while (!next_step.empty()) {
        minimal = MAXN;
        vector<string> current_state = next_step;
        std::random_shuffle(current_state.begin(), current_state.end());
        for (string cf:current_state) {
            vector<string> new_step;
            float ms = 0;
            for (string f:current_state) {
                if (f != cf) {
                    new_step.push_back(f);
                }
            }
            ms += compiler_with(new_step);
            ms += compiler_with(new_step);
            ms /= 2.0f;
            if (ms < minimal) {
                minimal = ms;
                next_step = new_step;
            }
        }
        //cout << "Ms: " << minimal << "\tFlags: "; print_vector(next_step);
        results.push_back({minimal, next_step});
    }

    next_step.clear();
    minimal = MAXN;
    for (auto p:results) {
        if (p.first <= minimal) {
            minimal = p.first;
            next_step = p.second;
        }
    }
    cout << "\n****Best Time:\tMs: " << minimal << "\tFlags: ";
    print_vector(next_step);
    /*
    cout << "All results:" << endl;
    sort(results.begin(), results.end());
    reverse(results.begin(), results.end());
    for (auto p:results) {
        cout << "Ms: " << p.first << "\tFlags: "; print_vector(p.second);
    }
    */
}

int main(void)
{
    cout << "Compilaciones falsas" << flush;
    for (int i = 0; i < 5; i++) {
        compiler_with({});
        cout << '.' << flush;
    }
    cout << endl;

    cout << "Comienzo seleccion de flags" << endl;
    for (auto Oflag:{"", "-O1", "-O2", "-O3", "-Ofast"}) {
        vector<string> v(compiler_flags);
        v.push_back(Oflag);
        seleccion(v);
    }

    system("make clean > /dev/null && rm report.out");

    return 0;
}
// g++-10 -o "optimization_flags" "optimization_flags.cc" -lm -std=c++17 -Ofast -Wno-unused-result
