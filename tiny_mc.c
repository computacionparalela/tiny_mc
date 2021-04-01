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

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];


/***
 * Photon
 ***/

static void photon(void)
{
        static const float albedo = MU_S / (MU_S + MU_A);
        static const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

        /* Step 1: Launching a photon packet */
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        float w = 1.0f;
        float weight = 1.0f;

        for (;;) {
                /* Step 2: Step size selection and photon packet movement */
                float t = -logf(rand() / (float)RAND_MAX);
                /* move */
                x += t * u;
                y += t * v;
                z += t * w;

                /* Step 3: Absorption and scattering */
                unsigned int shell = MIN(sqrtf(x * x + y * y + z * z) * shells_per_mfp, SHELLS - 1); /* absorb */
                float _heat = (1.0f - albedo) * weight;
                heat[shell] += _heat;
                heat2[shell] += _heat * _heat; /* add up squares */
                weight *= albedo;

                /* New direction, rejection method */
                float xi1, xi2, tmp;
                do {
                        xi1 = 2.0f * rand() / (float)RAND_MAX - 1.0f;
                        xi2 = 2.0f * rand() / (float)RAND_MAX - 1.0f;
                        t = xi1 * xi1 + xi2 * xi2;
                } while (1.0f < t);
                u = 2.0f * t - 1.0f;
                tmp = sqrtf((1.0f - u * u) / t);
                v = xi1 * tmp;
                w = xi2 * tmp;

                /* Step 4: Photon termination */
                if (weight < 0.001f) { /* roulette */
                        weight /= 0.1f;
                        if (rand() / (float)RAND_MAX > 0.1f) {
                                return;
                        }
                }
        }
}


/***
 * Main matter
 ***/

int main(void)
{
        // heading
        printf("# %s\n# %s\n# %s\n", t1, t2, t3);
        printf("# Scattering = %8.3f/cm\n", MU_S);
        printf("# Absorption = %8.3f/cm\n", MU_A);
        printf("# Photons    = %8d\n#\n", PHOTONS);

        memset(heat,0,sizeof(float)*SHELLS);
        memset(heat2,0,sizeof(float)*SHELLS);

        // configure RNG
        srand(SEED);

        // start timer
        double start = wtime();
        // simulation
        for (unsigned int i = 0; i < PHOTONS; ++i) {
                photon();
        }
        // stop timer
        double end = wtime();
        assert(start <= end);
        double elapsed = (end - start)*1000.0;

        printf("# Radius\tHeat\n");
        printf("# [microns]\t[W/cm^3]\tError\n");
        float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
        for (unsigned int i = 0; i < SHELLS - 1; ++i) {
                printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
                       heat[i] / t / (i * i + i + 1.0 / 3.0),
                       sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
        }
        printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

        printf(">> %lf ms\n", elapsed);
        printf(">>> %lf K photons per second\n", 1e-3 * PHOTONS / (elapsed/1000.0));

        return 0;
}
