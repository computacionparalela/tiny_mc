#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=5

#160 ejecuciones

echo "Ejecutando run_fotones.sh"

for cnt_photons in "16384" "32768" "65536" "131072" "262144" "524288" "1048576" "2097152"
do
	make clean
	make ADD_FLAGS=-DPHOTONS=$cnt_photons

	for version in "tiny_mc" "tiny_mc_ispc" "tiny_mc_m128" "tiny_mc_m256"
	do
		sleep 1
		TOTAL_MS=0
		TOTAL_PH=0
		for it in $(seq 1 $ITERATIONS)
		do
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
