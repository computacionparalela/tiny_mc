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

/***
 * Main matter
 ***/

int main(void)
{
        // configure RNG
        fast_srand(SEED);
        // start timer
        int k=0;
        double start = wtime();
        // simulation
        for (int i = 0; i < 10000000; ++i) {
                k = FAST_RAND();
        }
        // stop timer
        double end = wtime();
        assert(start <= end);
        double elapsed = (end - start)*1000.0;

        printf("MS: %f\n", elapsed);

        return k;
}
