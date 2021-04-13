#!/bin/bash

FILE="report.out"
TOTAL_MS=0
TOTAL_PH=0
ITERATIONS=20

echo "El programa debe estar compilado para ejecutar este script"
for it in $(seq 1 $ITERATIONS)
do
	./tiny_mc > /dev/null
	./tiny_mc > "$it-$FILE"
	LOCAL_MS=$(grep '+>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
	TOTAL_MS=$(echo $TOTAL_MS $LOCAL_MS | awk '{printf "%5.3f\n",$1+$2}')
	LOCAL_PH=$(grep '+>>> ' "$it-$FILE" | awk -F ' ' '{print $2}')
	TOTAL_PH=$(echo $TOTAL_PH $LOCAL_PH | awk '{printf "%5.3f\n",$1+$2}')
	rm -f "$it-$FILE"
	echo "> $LOCAL_MS"
done

TOTAL_MS=$(echo $TOTAL_MS $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
TOTAL_PH=$(echo $TOTAL_PH $ITERATIONS | awk '{printf "%5.3f\n",$1/$2}')
echo ">>  TOTAL_MS: $TOTAL_MS"
echo ">>> TOTAL_PH: $TOTAL_PH"
