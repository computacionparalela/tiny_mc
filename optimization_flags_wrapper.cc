#include <bits/stdc++.h>
#include <unistd.h>
using namespace std;

#ifdef GCC
ofstream myfile("GCC_PH.txt");
vector<string> Oflags = {"-O1", "-O2", "-O3", "-Ofast"};
vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math", "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", "-ffast-math", "-funroll-loops", "-funroll-all-loops",
 "-fpeel-loops", "-flto", "-fprefetch-loop-arrays", "-mfma", "-mrecip", "-msse2"};
string make_command("make clean > /dev/null && make CC=gcc-10 ");
#endif

#ifdef CLANG
ofstream myfile("CLANG_PH.txt");
vector<string> Oflags = {"-O0", "-O1", "-O2", "-O3"};
vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-freciprocal-math", "-ffinite-math-only",
 "-fno-signed-zeros", "-fno-trapping-math", "-ffast-math", "-funroll-loops", "-flto", "-mfma", "-mrecip", "-msse2"};
string make_command("make clean > /dev/null && make CC=clang-11 ");
#endif

#ifdef INTEL
ofstream myfile("ICC_PH.txt");
vector<string> Oflags = {"-O0", "-O1", "-O2", "-O3"};
vector<string> compiler_flags =
{"-parallel", "-m64", "-qopenmp-offload", "-qopenmp", "-qopt-report", "-qopt-prefetch", "-fimf-precision=simple",
 "-no-prec-sqrt", "-no-prec-div", "-Istd", "-Istdi", "-Lstd", "-fp-speculation", "-fast-transcendentals"};
string make_command("make clean > /dev/null && make CC=icc ");
#endif

float compiler_with(const vector<string>& flags)
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
    std::fstream fs("report.out", std::fstream::in | std::fstream::out);
    while (command != ">>>") {
        fs >> command;
    }
    fs >> command >> ms;

    return ms;
}

void print_vector(const vector<string>& vs, ofstream& f)
{
    if (vs.empty()) {
        f << "Empty" << endl;
        return;
    }
    for (auto v:vs) {
        f << v << ' ';
    }
    f << endl;
}

void seleccion(vector<string>& next_step)
{
    vector<pair<float, vector<string> > > results;

    float maximal = compiler_with(next_step);
    // cout << "Ms: " << minimal << "\tFlags: "; print_vector(next_step);
    results.push_back({maximal, next_step});

    while (!next_step.empty()) {
        maximal = -1.0f;
        vector<string> current_state = next_step;
        std::random_shuffle(current_state.begin(), current_state.end());
        for (string cf:current_state) {
            vector<string> new_step;
            for (string f:current_state) {
                if (f != cf) {
                    new_step.push_back(f);
                }
            }
            float ms = compiler_with(new_step);
            cout << "Ms: " << ms << ' ' << maximal << endl;
            if (maximal < ms) {
                maximal = ms;
                next_step = new_step;
            }
        }
        cout << "Photons: " << maximal << endl;// print_vector(next_step);
        results.push_back({maximal, next_step});
    }

    next_step.clear();
    maximal = -1.0f;
    for (auto p:results) {
        if (p.first >= maximal) {
            maximal = p.first;
            next_step = p.second;
        }
    }

    myfile << "\n****Best Photons:\tMs: " << maximal << "\tFlags: ";
    print_vector(next_step, myfile);
    myfile << "All results:" << endl;
    sort(results.begin(), results.end());
    for (auto p:results) {
        myfile << "Photons: " << p.first << "\tFlags: ";
        print_vector(p.second, myfile);
    }
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
    }

    myfile.close();
    system("make clean > /dev/null && rm report.out");

    return 0;
}
// g++-10 -o "optimization_flags" "optimization_flags.cc" -lm -std=c++17 -Ofast -Wno-unused-result
