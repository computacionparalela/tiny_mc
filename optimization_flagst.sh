#!/bin/bash

#Check https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

#Ver:
#    * -fbranch-probabilities, -fprofile-arcs, -fprofile-values, -fprofile-reorder-functions

O_FLAGS="-O0 -O1 -O2 -O3 -Ofast"
MATH_FLAGS="-fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-rounding-math -fno-signaling-nans -fcx-limited-range -fexcess-precision=fast -freciprocal-math -ffinite-math-only -fno-signed-zeros -fno-trapping-math -frounding-math -fsignaling-nans -ffast-math"
LOOP_FLAGS="-funroll-loops -funroll-all-loops -fpeel-loops -flto"

FLAGS="$MATH_FLAGS $LOOP_FLAGS"

for oflag in $O_FLAGS
do
	FLG="$oflag"
	for flag in $FLAGS
	do
		FLG=$(echo "$FLG $flag")
		echo "Testing $FLG"
		make clean > /dev/null
		make OPT_FLAGS="$oflag $flag" > /dev/null
		PERF=$(./run.sh | grep '>>')
		echo "$PERF"
	done
done
