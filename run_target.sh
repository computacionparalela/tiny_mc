#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=1

#90 ejecuciones

echo "Ejecutando run_target.sh"

for target in "sse4-i8x16" "sse4-i16x8" "sse4-i32x4" "sse4-i32x8" "avx2-i8x32" "avx2-i16x16" "avx2-i32x4" "avx2-i32x8" "avx2-i32x16"
do
	make clean
	make ADD_FLAGS=-DPHOTONS=1048576 ISPC_TARGET=$target

	sleep 1
	TOTAL_MS=0
	TOTAL_PH=0
	for it in $(seq 1 $ITERATIONS)
	do
		./tiny_mc_ispc > /dev/null
		./tiny_mc_ispc > "$it-$FILE"
		LOCAL_MS=$(grep '+>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
		TOTAL_MS=$(echo $TOTAL_MS $LOCAL_MS | awk '{printf "%5.3f\n",$1+$2}')
		LOCAL_PH=$(grep '+>>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
		TOTAL_PH=$(echo $TOTAL_PH $LOCAL_PH | awk '{printf "%5.3f\n",$1+$2}')
		rm -f "$it-$FILE"
		echo "> $LOCAL_PH $LOCAL_MS"
	done

	TOTAL_MS=$(echo $TOTAL_MS $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
	TOTAL_PH=$(echo $TOTAL_PH $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
	echo "tiny_mc_ispc $target:"	
	echo "    >>  TOTAL_MS: $TOTAL_MS"
	echo "    >>> TOTAL_PH: $TOTAL_PH"
	make clean
done
