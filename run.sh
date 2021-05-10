#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=10

make clean
make -j5

for version in "tiny_mc" "tiny_mc_ispc" "tiny_mc_m256"
do
	sleep 30
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
	echo "$version:"	
	echo "    >>  TOTAL_MS: $TOTAL_MS"
	echo "    >>> TOTAL_PH: $TOTAL_PH"
done
make clean
