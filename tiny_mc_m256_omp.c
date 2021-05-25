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
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>


char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

/***
 * Photon
 ***/

#define ALBEDO         (MU_S / (MU_S + MU_A))
#define SHELLS_PER_MFP (1e4 / MICRONS_PER_SHELL / (MU_A + MU_S))

static inline __m256 _m256_flog(__m256 x)
{
        // union { float f; unsigned int i; } vx = { x };
        union { __m256 f; __m256i i; } vx = { x };
        // float y = vx.i;y *= 8.2629582881927490e-8f;
        __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(vx.i), _mm256_set1_ps(8.2629582881927490e-8f));
        // return y - 87.989971088f;
        return _mm256_sub_ps(y, _mm256_set1_ps(87.989971088f));
}

static inline __m256 _m256_rand(__m256i * _seed)
{
        __m256i x = *_seed;
        // x ^= x << 13;
        x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
        // x ^= x >> 17;
        x = _mm256_xor_si256(x, _mm256_srai_epi32(x, 17));
        // x ^= x << 5;
        x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
        *_seed = x;
        return _mm256_div_ps(_mm256_add_ps(_mm256_cvtepi32_ps(x), _mm256_set1_ps(2147483647.0f)),
                             _mm256_set1_ps(4294967296.0f));
}

static void photon(float (*heat)[2], int photons, float * _seed)
{
        float lheat[8], lheat2[8], is_working[8], shellidx[8];
        /* Step 1: Launching a photon packet */
        // float albedo = MU_S / (MU_S + MU_A);
        const __m256 albedo =
                _mm256_div_ps(_mm256_set1_ps(MU_S), _mm256_add_ps(_mm256_set1_ps(MU_S), _mm256_set1_ps(MU_A)));
        // float shell_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
        const __m256 shells_per_mfp =
                _mm256_div_ps(_mm256_div_ps(_mm256_set1_ps((float)1e4), _mm256_set1_ps((float)MICRONS_PER_SHELL)),
                              _mm256_add_ps(_mm256_set1_ps(MU_A), _mm256_set1_ps(MU_S)));

        // float x = 0.0f;
        __m256 x = _mm256_setzero_ps();
        // float y = 0.0f;
        __m256 y = _mm256_setzero_ps();
        // float z = 0.0f;
        __m256 z = _mm256_setzero_ps();
        // float u = 0.0f;
        __m256 u = _mm256_setzero_ps();
        // float v = 0.0f;
        __m256 v = _mm256_setzero_ps();
        // float w = 1.0f;
        __m256 w = _mm256_set1_ps(1.0f);
        // float weight = 1.0f;
        __m256 weight = _mm256_set1_ps(1.0f);

        __m256i vseed  = _mm256_cvtps_epi32(_mm256_loadu_ps(_seed));

        __m256 working;
        for (int k = 0; k < 8; k++) {
                photons--;
                if (photons < 0) {
                        is_working[k] = 0.0f;
                } else {
                        is_working[k] = -NAN;
                }
        }
        working = _mm256_load_ps(is_working);

        do {
                /* Step 2: Step size selection and photon packet movement */
                // float t = -logf(FAST_RAND());
                __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), _m256_flog(_m256_rand(&vseed))); // FIX
                /* move */
                // x += t * u;
                x = _mm256_add_ps(_mm256_mul_ps(t, u), x);
                // y += t * v;
                y = _mm256_add_ps(_mm256_mul_ps(t, v), y);
                // z += t * w;
                z = _mm256_add_ps(_mm256_mul_ps(t, w), z);

                /* Step 3: Absorption and scattering */
                // unsigned int shell = min(sqrtf(x * x + y * y + z * z) * shell_per_mfp, SHELLS - 1); /* absorb */
                __m256 sroot =
                        _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_add_ps(_mm256_mul_ps(y, y), _mm256_mul_ps(z, z))));
                __m256 shell =
                        _mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(sroot, shells_per_mfp)),
                                      _mm256_sub_ps(_mm256_set1_ps(SHELLS), _mm256_set1_ps(1.0f)));

                // float _heat = (1.0f - albedo) * weight;
                __m256 _heat = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), albedo), weight);

                _mm256_store_ps(shellidx, shell);
                _mm256_store_ps(lheat, _heat);
                _mm256_store_ps(lheat2, _mm256_mul_ps(_heat, _heat));
                char is_alive = _mm256_movemask_ps(working);
                for (int k = 0; k < 8; k++) {
                        if ((is_alive & (1 << k)) != 0) {
                                unsigned int idx = shellidx[k];
                                heat[idx][0] += lheat[k];
                                heat[idx][1] += lheat2[k];
                        }
                }
                // weight *= albedo;
                weight = _mm256_mul_ps(weight, albedo);

                /* New direction, rejection method */
                __m256 q, xi1, xi2, tmp, loop_mask = working;
                char finish_loop;
                do {
                        // xi1 = 2.0f * FAST_RAND() - 1.0f;
                        xi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _m256_rand(&vseed)), _mm256_set1_ps(1.0f));
                        // xi2 = 2.0f * FAST_RAND() - 1.0f;
                        xi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _m256_rand(&vseed)), _mm256_set1_ps(1.0f));
                        // t = xi1 * xi1 + xi2 * xi2;
                        tmp = _mm256_add_ps(_mm256_mul_ps(xi1, xi1), _mm256_mul_ps(xi2, xi2));
                        t = _mm256_blendv_ps(t, tmp, loop_mask);
                        loop_mask = _mm256_and_ps(loop_mask, _mm256_cmp_ps(_mm256_set1_ps(1.0f), t, _CMP_LT_OQ));
                        finish_loop = _mm256_movemask_ps(loop_mask);
                } while (finish_loop != 0); // 1.0f < t

                // u = 2.0f * t - 1.0f;
                u = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t), _mm256_set1_ps(1.0f));
                q = _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(u, u)), t));
                v = _mm256_mul_ps(xi1, q);
                w = _mm256_mul_ps(xi2, q);

                /* Step 4: Photon termination */
                // if (weight < 0.001f) { /* roulette */
                //        weight /= 0.1f;
                //        if (FAST_RAND() > 0.1f) {
                //                return;
                //        }
                // }
                __m256 weight_mask = _mm256_cmp_ps(weight, _mm256_set1_ps(0.001f), _CMP_LT_OQ);
                __m256 return_mask =
                        _mm256_and_ps(weight_mask, _mm256_cmp_ps(_m256_rand(&vseed), _mm256_set1_ps(0.1f), _CMP_GT_OQ));
                weight = _mm256_blendv_ps(weight, _mm256_div_ps(weight, _mm256_set1_ps(0.1f)), weight_mask);

                // Update x,y,z,u,v,w,weight
                x = _mm256_blendv_ps(x, _mm256_setzero_ps(), return_mask);
                y = _mm256_blendv_ps(y, _mm256_setzero_ps(), return_mask);
                z = _mm256_blendv_ps(z, _mm256_setzero_ps(), return_mask);
                u = _mm256_blendv_ps(u, _mm256_setzero_ps(), return_mask);
                v = _mm256_blendv_ps(v, _mm256_setzero_ps(), return_mask);
                w = _mm256_blendv_ps(w, _mm256_set1_ps(1.0f), return_mask);
                weight = _mm256_blendv_ps(weight, _mm256_set1_ps(1.0f), return_mask);

                char has_returned = _mm256_movemask_ps(_mm256_and_ps(working, return_mask));
                _mm256_store_ps(is_working, working);
                // Check photons
                for (int k = 0; k < 8; k++) {
                        if ((has_returned & (1 << k)) != 0) {
                                photons -= 1;
                                if (photons < 0) {
                                        is_working[k] = 0.0f;
                                }
                        }
                }
                working = _mm256_load_ps(is_working);
        } while (_mm256_movemask_ps(working) != 0);
}
/***
 * Main matter
 ***/

int main(void)
{

        static float heat[SHELLS][2];
        static float seed[8];// Avoid buffer overflow
        if (verbose) { // heading
                printf("# %s\n# %s\n# %s\n", t1, t2, t3);
                printf("# Scattering = %8.3f/cm\n", MU_S);
                printf("# Absorption = %8.3f/cm\n", MU_A);
                printf("# Photons    = %8d\n#\n", PHOTONS);
        }

        // configure RNG
        srand(SEED);
        for (int i = 0; i < 8; i++) {
                seed[i] = rand();
        }
    #pragma omp parallel for
        for (int i = 0; i < 8; ++i) {
                photon(heat, 256, seed);
        }
        // first run
        memset(heat, 0, 2 * sizeof(float) * SHELLS);
        // start timer
        double start = wtime();
        // simulation
    #pragma omp parallel for firstprivate(seed) num_threads(THREADS) schedule(SCHEDULE) reduction(+:heat[:SHELLS][:2]) default(none)
        for (int i = 0; i < CHUNKS; ++i) {
                for(int j=0; j<8; j++) {
                        seed[j] *= i+1;
                }
                photon(heat, PHOTONS / CHUNKS, seed);
        }
        // stop timer
        double end = wtime();
        assert(start <= end);
        double elapsed = (end - start) * 1000.0;
        if (verbose) {
                printf("# Radius\tHeat\n");
                printf("# [microns]\t[W/cm^3]\tError\n");
                float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
                for (unsigned int i = 0; i < SHELLS - 1; ++i) {
                        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
                               heat[i][0] / t / (i * i + i + 1.0 / 3.0),
                               sqrt(heat[i][1] - heat[i][0] * heat[i][0] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
                }
                printf("# extra\t%12.5f\n", heat[SHELLS - 1][0] / PHOTONS);
        }
        printf("+>> %lf ms\n", elapsed);
        printf("+>>> %lf K photons per second\n", 1e-3 * PHOTONS / (elapsed / 1000.0));

        return 0;
}
