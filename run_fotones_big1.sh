#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=10

echo "Ejecutando run_fotones_big1.sh"

for cnt_photons in "1048576"  "2097152" "4194304" "8388608"
do
	make clean
	make ADD_FLAGS="-DPHOTONS=$cnt_photons -DREDUCTION -DTHREADS=28"

	for version in "tiny_mc_m256" "tiny_mc_m256_omp" "tiny_mc_omp"
	do
		TOTAL_MS=0
		TOTAL_PH=0
                ./$version > /dev/null
		for it in $(seq 1 $ITERATIONS)
		do
			sleep 3
			./$version > /dev/null
			./$version > "$it-$FILE"
			LOCAL_MS=$(grep '+>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
			TOTAL_MS=$(echo $TOTAL_MS $LOCAL_MS | awk '{printf "%5.3f\n",$1+$2}')
			LOCAL_PH=$(grep '+>>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
			TOTAL_PH=$(echo $TOTAL_PH $LOCAL_PH | awk '{printf "%5.3f\n",$1+$2}')
			rm -f "$it-$FILE"
			echo "> $LOCAL_PH $LOCAL_MS"
		done

		TOTAL_MS=$(echo $TOTAL_MS $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
		TOTAL_PH=$(echo $TOTAL_PH $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
		echo "$cnt_photons $version:"	
		echo "    >>  TOTAL_MS: $TOTAL_MS"
		echo "    >>> TOTAL_PH: $TOTAL_PH"
	done
	make clean
done
