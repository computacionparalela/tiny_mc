#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=2

echo "Ejecutando run_block_size.sh"

for block_size in "16" "32" "64" "128" "256" "512" "1024"
do
	for photon_pthr in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
	do
		make clean
		make ADD_FLAGS="-DBLOCK_SIZE=$block_size -DPHOTONS_PER_THREAD=$photon_pthr"
		for version in "tiny_mc_gpu"
		do
			TOTAL_MS=0
			TOTAL_PH=0
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
			echo "$block_size $photon_pthr $version:"
			echo "    >>  TOTAL_MS: $TOTAL_MS"
			echo "    >>> TOTAL_PH: $TOTAL_PH"
		done
		make clean
	done
done
