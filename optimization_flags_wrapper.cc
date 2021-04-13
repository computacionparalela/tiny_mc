#include <bits/stdc++.h>
#include <unistd.h>
#define MAXN 2147483647
using namespace std;

vector<string> compiler_flags_test =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans"};

vector<string> compiler_flags =
{"-fno-math-errno", "-funsafe-math-optimizations", "-ffinite-math-only", "-fno-rounding-math", "-fno-signaling-nans",
 "-fcx-limited-range", "-fexcess-precision=fast", "-freciprocal-math",
 "-ffinite-math-only", "-fno-signed-zeros",
 "-fno-trapping-math", "-frounding-math", "-fsignaling-nans", "-ffast-math",
 "-funroll-loops", "-funroll-all-loops", "-fpeel-loops", "-flto", "-ftree-vectorize", "-fno-tree-loop-vectorize",
 "-fprefetch-loop-arrays", "-ftree-parallelize-loops", "-mfmaf", "-mrecip", "-msse2", "-mvis3"};

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
    cout << "Ms: " << minimal << "\tFlags: "; print_vector(next_step);
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
        cout << "Ms: " << minimal << "\tFlags: "; print_vector(next_step);
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
    cout << "All results:" << endl;
    sort(results.begin(), results.end());
    reverse(results.begin(), results.end());
    for (auto p:results) {
        cout << "Ms: " << p.first << "\tFlags: "; print_vector(p.second);
    }
}

int main(void)
{
    cout << "Compilaciones falsas" << flush;
    for (int i = 0; i < 3; i++) {
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
