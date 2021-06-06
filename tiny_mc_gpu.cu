#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>

#include "params.h"

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return 1e-9 * ts.tv_nsec + (double)ts.tv_sec;
}

__global__ void photon(float ** heat, int photons)
{
    /* Step 0: Inicializo el PRNG */
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init((7 + idx) * 9967, (idx + 1), (7 + idx) * 197, &state);

    float4 rand = curand_uniform4(&state);

    for (int cnt = 0; cnt < photons; cnt++) {
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
            float t = -logf(rand.w);
            /* move */
            x += t * u;
            y += t * v;
            z += t * w;

            /* Step 3: Absorption and scattering */
            unsigned int shell = min((int)(sqrtf(x * x + y * y + z * z) * SHELLS_PER_MFP), SHELLS - 1);             /* absorb */
            float _heat = (1.0f - ALBEDO) * weight;
            weight *= ALBEDO;
            // Barrera de sincronizacion
            heat[idx][2 * shell + 0] += _heat;
            heat[idx][2 * shell + 1] += _heat * _heat;/* add up squares */
            ///////////////////////////
            /* New direction, rejection method */
            float x1, x2;
            do {
                x1 = 2.0f * 0.5f - 1.0f;
                x2 = 2.0f * 0.5f - 1.0f;
                t = x1 * x1 + x2 * x2;
                rand = curand_uniform4(&state);
            } while (1.0f < t);
            u = 2.0f * t - 1.0f;
            t = sqrtf((1.0f - u * u) / t);
            v = x1 * t;
            w = x2 * t;
            /* Step 4: Photon termination */
            if (weight < 0.001f) {/* roulette */
                weight /= 0.1f;
                if (rand.z > 0.1f) {
                    break;
                }
            }
        }
    }

    return;
}

unsigned run_kernel(float (*_heat)[2],float ** heat, int photons)
{
    dim3 grid(1);
    dim3 block(CUDA_CORES);
    photon << < grid, block >> > (heat, photons / CUDA_CORES);
    cudaDeviceSynchronize();

    #pragma omp parallel for shared(heat) schedule(SCHEDULE) reduction(+:_heat[:SHELLS][:2]) default(none)
    for (int i = 0; i < CUDA_CORES; i++) {
        for (int j = 0; j < SHELLS; j++) {
            _heat[j][0] += heat[i][2 * j + 0];
            _heat[j][1] += heat[i][2 * j + 1];
        }
    }

    return 0;
}

int main()
{

    if (verbose) { // heading
        printf("# %s\n# %s\n# %s\n", t1, t2, t3);
        printf("# Scattering = %8.3f/cm\n", MU_S);
        printf("# Absorption = %8.3f/cm\n", MU_A);
        printf("# Photons    = %8d\n#\n", PHOTONS);
    }


    float _heat[SHELLS][2];
    float ** heat;
    cudaMallocManaged(&heat, CUDA_CORES * sizeof(float *));
    for (int i = 0; i < CUDA_CORES; i++) {
        cudaMallocManaged(&heat[i], 2 * SHELLS * sizeof(float));
        for (int j = 0; j < 2 * SHELLS; j++) {
            heat[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < SHELLS; i++) {
        _heat[i][0] = _heat[i][1] = 0.0f;
    }

    double start = wtime();
    run_kernel(_heat, heat, PHOTONS);
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


    cudaFree(heat[0]);
    cudaFree(heat[1]);
    cudaFree(heat);

    return 0;
}
