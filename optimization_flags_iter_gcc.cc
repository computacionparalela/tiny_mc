#include <bits/stdc++.h>
#include <unistd.h>
#include "t_test.h"
using namespace std;

vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math",
 "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", "-ffast-math",
 "-funroll-loops", "-funroll-all-loops", "-fpeel-loops", "-flto", "-fprefetch-loop-arrays", "-mfma", "-mrecip", "-msse2"};

vector<string> compiler_flags_full =
{"-funroll-loops", "-funroll-all-loops", "-fpeel-loops", "-flto", "-fauto-inc-dec", "-fbranch-count-reg",
 "-fcombine-stack-adjustments", "-fcompare-elim", "-fcprop-registers", "-fdce", "-fdefer-pop", "-fdelayed-branch",
 "-fdse", "-fforward-propagate", "-fguess-branch-probability", "-fif-conversion", "-fif-conversion2",
 "-finline-functions-called-once", "-fipa-profile", "-fipa-pure-const", "-fipa-reference",
 "-fipa-reference-addressable", "-fmerge-constants", "-fmove-loop-invariants", "-fomit-frame-pointer",
 "-freorder-blocks", "-fshrink-wrap", "-fshrink-wrap-separate", "-fsplit-wide-types", "-fssa-backprop",
 "-fssa-phiopt", "-ftree-bit-ccp", "-ftree-ccp", "-ftree-ch", "-ftree-coalesce-vars", "-ftree-copy-prop",
 "-ftree-dce", "-ftree-dominator-opts", "-ftree-dse", "-ftree-forwprop", "-ftree-fre", "-ftree-phiprop",
 "-ftree-pta", "-ftree-scev-cprop", "-ftree-sink", "-ftree-slsr", "-ftree-sra", "-ftree-ter", "-funit-at-a-time",
 "-falign-functions", "-falign-jumps", "-falign-labels", "-falign-loops", "-fcaller-saves", "-fcode-hoisting",
 "-fcrossjumping", "-fcse-follow-jumps", "-fcse-skip-blocks", "-fdelete-null-pointer-checks", "-fdevirtualize",
 "-fdevirtualize-speculatively", "-fexpensive-optimizations", "-ffinite-loops", "-fgcse", "-fgcse-lm",
 "-fhoist-adjacent-loads", "-finline-functions", "-finline-small-functions", "-findirect-inlining", "-fipa-bit-cp",
 "-fipa-cp", "-fipa-icf", "-fipa-ra", "-fipa-sra", "-fipa-vrp", "-fisolate-erroneous-paths-dereference",
 "-flra-remat", "-foptimize-sibling-calls", "-foptimize-strlen", "-fpartial-inlining", "-fpeephole2",
 "-freorder-blocks-algorithm=stc", "-freorder-blocks-and-partition", "-freorder-functions", "-frerun-cse-after-loop",
 "-fschedule-insns", "-fschedule-insns2", "-fsched-interblock", "-fsched-spec", "-fstore-merging",
 "-fstrict-aliasing", "-fthread-jumps", "-ftree-builtin-call-dce", "-ftree-pre", "-ftree-switch-conversion",
 "-ftree-tail-merge", "-ftree-vrp", "-fgcse-after-reload", "-fipa-cp-clone", "-floop-interchange",
 "-floop-unroll-and-jam", "-fpeel-loops", "-fpredictive-commoning", "-fsplit-loops", "-fsplit-paths",
 "-ftree-loop-distribution", "-ftree-loop-vectorize", "-ftree-partial-pre", "-ftree-slp-vectorize",
 "-funswitch-loops", "-fvect-cost-model", "-fvect-cost-model=dynamic", "-fversion-loops-for-strides",
 "-falign-functions", "-falign-jumps", "-falign-labels", "-falign-loops", "-fprefetch-loop-arrays",
 "-freorder-blocks-algorithm=stc", "-fallow-store-data-races", "-fno-math-errno",
 "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math", "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", ""};

vector<float> compiler_with(const vector<string>& flags)
{
    // Create the shell command
    string compile_opt("OPT_FLAGS=\""), command;
    for (string f:flags) {
        compile_opt.append(f);
        compile_opt.push_back(' ');
    }
    // Execute the program and save the output at report.out
    command = string("make clean > /dev/null && make CC=gcc-10") + compile_opt +
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
    cout << "Compilaciones falsas" << flush;
    for (int i = 0; i < 5; i++) {
        compiler_with({});
        cout << '.' << flush;
    }
    cout << endl;

    cout << "Comienzo seleccion de flags" << endl;
    for (auto Oflag:{"-O1", "-O2", "-O3", "-Ofast"}) {
        vector<string> v(compiler_flags);
        v.push_back(Oflag);
        seleccion(v);
    }

    system("make clean > /dev/null && rm report.out");

    return 0;
}
// g++-10 -o "optimization_flags" "optimization_flags.cc" -lm -std=c++17 -O3 -Wno-unused-result
