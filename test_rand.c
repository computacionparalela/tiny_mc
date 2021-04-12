/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500  // M_PI

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN (655350l)
#define MAXM (MAXN)

long long unsigned number[MAXN];

int main(void)
{

        fast_srand(SEED);

        long double mean = 0, sd = 0;

        for(int i=0; i<MAXM; i++) {
                unsigned idx = FAST_RAND()*MAXN;
                number[idx]++;
                mean += idx;
        }

        mean = mean/(long double)MAXM;

        for(int i=0; i<MAXN; i++) {
                for(unsigned long long j=0; j<number[i]; j++) {
                        sd += fabs(i - mean)*fabs(i - mean);
                }
        }
        sd = sqrt(sd / MAXM);
        sd = sd*sd;

        printf("Expected:\tMean: %Lf V:%Lf\n",(long double)(MAXN/2.0),(long double)(1.0/12.0)*(MAXN*MAXN));
        printf("Got:\t\tMean: %Lf V:%Lf\n",mean,sd);

        return 0;
}
