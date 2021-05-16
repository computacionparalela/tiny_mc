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
#include <stdint.h>
#include <immintrin.h>


char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];
static float seed[32];// Avoid buffer overflow


/***
 * Photon
 ***/

#define ALBEDO         (MU_S / (MU_S + MU_A))
#define SHELLS_PER_MFP (1e4 / MICRONS_PER_SHELL / (MU_A + MU_S))

static inline __m128 _m128_flog(__m128 x)
{
    // union { float f; unsigned int i; } vx = { x };
    union { __m128 f; __m128i i; } vx = { x };
    // float y = vx.i;y *= 8.2629582881927490e-8f;
    __m128 y = _mm_mul_ps(_mm_cvtepi32_ps(vx.i), _mm_set1_ps(8.2629582881927490e-8f));
    // return y - 87.989971088f;
    return _mm_sub_ps(y, _mm_set1_ps(87.989971088f));
}

static inline __m128 _m128_rand(__m128i * _seed)
{
    __m128i x = *_seed;
    // x ^= x << 13;
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
    // x ^= x >> 17;
    x = _mm_xor_si128(x, _mm_srai_epi32(x, 17));
    // x ^= x << 5;
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
    *_seed = x;
    return _mm_div_ps(_mm_add_ps(_mm_cvtepi32_ps(x), _mm_set1_ps(2147483647.0f)), _mm_set1_ps(4294967296.0f));
}

static void photon(int photons, float * _seed)
{
    float lheat[8], lheat2[8], is_working[8], shellidx[8];
    /* Step 1: Launching a photon packet */
    // float albedo = MU_S / (MU_S + MU_A);
    const __m128 albedo =  _mm_div_ps(_mm_set1_ps(MU_S), _mm_add_ps(_mm_set1_ps(MU_S), _mm_set1_ps(MU_A)));
    // float shell_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
    const __m128 shells_per_mfp = _mm_div_ps(_mm_div_ps(_mm_set1_ps((float)1e4), _mm_set1_ps((float)MICRONS_PER_SHELL)),
                                             _mm_add_ps(_mm_set1_ps(MU_A), _mm_set1_ps(MU_S)));

    // float x = 0.0f;
    __m128 x = _mm_setzero_ps();
    // float y = 0.0f;
    __m128 y = _mm_setzero_ps();
    // float z = 0.0f;
    __m128 z = _mm_setzero_ps();
    // float u = 0.0f;
    __m128 u = _mm_setzero_ps();
    // float v = 0.0f;
    __m128 v = _mm_setzero_ps();
    // float w = 1.0f;
    __m128 w = _mm_set1_ps(1.0f);
    // float weight = 1.0f;
    __m128 weight = _mm_set1_ps(1.0f);

    __m128i vseed  = _mm_cvtps_epi32(_mm_loadu_ps(_seed));

    __m128 working;
    for (int k = 0; k < 4; k++) {
        photons--;
        if (photons < 0) {
            is_working[k] = 0.0f;
        } else {
            is_working[k] = -NAN;
        }
    }
    working = _mm_load_ps(is_working);

    do {
        /* Step 2: Step size selection and photon packet movement */
        // float t = -logf(FAST_RAND());
        __m128 t = _mm_sub_ps(_mm_setzero_ps(), _m128_flog(_m128_rand(&vseed)));        // FIX
        /* move */
        // x += t * u;
        x = _mm_add_ps(_mm_mul_ps(t, u), x);
        // y += t * v;
        y = _mm_add_ps(_mm_mul_ps(t, v), y);
        // z += t * w;
        z = _mm_add_ps(_mm_mul_ps(t, w), z);

        /* Step 3: Absorption and scattering */
        // unsigned int shell = min(sqrtf(x * x + y * y + z * z) * shell_per_mfp, SHELLS - 1); /* absorb */
        __m128 sroot = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_add_ps(_mm_mul_ps(y, y), _mm_mul_ps(z, z))));
        __m128 shell =
            _mm_add_ps(_mm_min_ps(_mm_floor_ps(_mm_mul_ps(sroot, shells_per_mfp)),
                                  _mm_sub_ps(_mm_set1_ps(SHELLS), _mm_set1_ps(1.0f))), _mm_set1_ps(0.1f));

        // float _heat = (1.0f - albedo) * weight;
        __m128 _heat = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), albedo), weight);

        _mm_store_ps(shellidx, shell);
        _mm_store_ps(lheat, _heat);
        _mm_store_ps(lheat2, _mm_mul_ps(_heat, _heat));
        char is_alive = _mm_movemask_ps(working);
        for (int k = 0; k < 4; k++) {
            if ((is_alive & (1 << k)) != 0) {
                heat[(int)shellidx[k]]  += lheat[k];
                heat2[(int)shellidx[k]] += lheat2[k];
            }
        }
        // weight *= albedo;
        weight = _mm_mul_ps(weight, albedo);

        /* New direction, rejection method */
        __m128 q, xi1, xi2, tmp, loop_mask = working;
        char finish_loop;
        do {
            // xi1 = 2.0f * FAST_RAND() - 1.0f;
            xi1 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _m128_rand(&vseed)), _mm_set1_ps(1.0f));
            // xi2 = 2.0f * FAST_RAND() - 1.0f;
            xi2 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _m128_rand(&vseed)), _mm_set1_ps(1.0f));
            // t = xi1 * xi1 + xi2 * xi2;
            tmp = _mm_add_ps(_mm_mul_ps(xi1, xi1), _mm_mul_ps(xi2, xi2));
            t = _mm_blendv_ps(t, tmp, loop_mask);
            loop_mask = _mm_and_ps(loop_mask, _mm_cmp_ps(_mm_set1_ps(1.0f), t, _CMP_LT_OQ));
            finish_loop = _mm_movemask_ps(loop_mask);
        } while (finish_loop != 0);        // 1.0f < t

        // u = 2.0f * t - 1.0f;
        u = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), t), _mm_set1_ps(1.0f));
        q = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(u, u)), t));
        v = _mm_mul_ps(xi1, q);
        w = _mm_mul_ps(xi2, q);

        /* Step 4: Photon termination */
        // if (weight < 0.001f) { /* roulette */
        //        weight /= 0.1f;
        //        if (FAST_RAND() > 0.1f) {
        //                return;
        //        }
        // }
        __m128 weight_mask = _mm_cmp_ps(weight, _mm_set1_ps(0.001f), _CMP_LT_OQ);
        __m128 return_mask = _mm_and_ps(weight_mask, _mm_cmp_ps(_m128_rand(&vseed), _mm_set1_ps(0.1f), _CMP_GT_OQ));
        weight = _mm_blendv_ps(weight, _mm_div_ps(weight, _mm_set1_ps(0.1f)), weight_mask);

        // Update x,y,z,u,v,w,weight
        x = _mm_blendv_ps(x, _mm_setzero_ps(), return_mask);
        y = _mm_blendv_ps(y, _mm_setzero_ps(), return_mask);
        z = _mm_blendv_ps(z, _mm_setzero_ps(), return_mask);
        u = _mm_blendv_ps(u, _mm_setzero_ps(), return_mask);
        v = _mm_blendv_ps(v, _mm_setzero_ps(), return_mask);
        w = _mm_blendv_ps(w, _mm_set1_ps(1.0f), return_mask);
        weight = _mm_blendv_ps(weight, _mm_set1_ps(1.0f), return_mask);

        char has_returned = _mm_movemask_ps(_mm_and_ps(working, return_mask));
        _mm_store_ps(is_working, working);
        // Check photons
        for (int k = 0; k < 4; k++) {
            if ((has_returned & (1 << k)) != 0) {
                photons -= 1;
                if (photons < 0) {
                    is_working[k] = 0.0f;
                }
            }
        }
        working = _mm_load_ps(is_working);
    } while (_mm_movemask_ps(working) != 0);
}
/***
 * Main matter
 ***/

int main(void)
{
    if (verbose) {   // heading
        printf("# %s\n# %s\n# %s\n", t1, t2, t3);
        printf("# Scattering = %8.3f/cm\n", MU_S);
        printf("# Absorption = %8.3f/cm\n", MU_A);
        printf("# Photons    = %8d\n#\n", PHOTONS);
    }

    // configure RNG
    srand(SEED);
    for (int i = 0; i < 32; i++) {
        seed[i] = rand();
    }
    photon(256, seed);
    // first run
    memset(heat, 0, sizeof(float) * SHELLS);
    memset(heat2, 0, sizeof(float) * SHELLS);

    // start timer
    double start = wtime();
    // simulation
    // for (int i = 0; i < PHOTONS; ++i) {
    photon(PHOTONS, seed);
    // }
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
                   heat[i] / t / (i * i + i + 1.0 / 3.0),
                   sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
        }
        printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
    }
    printf("+>> %lf ms\n", elapsed);
    printf("+>>> %lf K photons per second\n", 1e-3 * PHOTONS / (elapsed / 1000.0));

    return 0;
}
