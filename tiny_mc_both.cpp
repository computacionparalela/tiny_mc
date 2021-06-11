#include "params.h"
#include "tiny_mc_cpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define checkCudaCall(val) _checkCudaReturnValue((val), #val, __FILE__, __LINE__)

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


inline void _checkCudaReturnValue(cudaError_t result, const char* const func, const char* const file, const int line)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(static_cast<int>(result));
    }
}

unsigned run_gpu_tiny_mc(float ** heat, const int photons, const bool sync);

double wtime(void);

void run_both_tiny_mc(float (*heat)[2], float ** heat_gpu, const unsigned photons)
{
    // GPU 7 veces mas rapida que CPU
    const unsigned frac = 8;
    const unsigned photons_cpu = photons - run_gpu_tiny_mc(heat_gpu, photons - (photons / frac), false);
    run_cpu_tiny_mc(heat, photons_cpu);
    checkCudaCall(cudaDeviceSynchronize());
    #pragma \
    omp parallel for firstprivate(heat_gpu)  num_threads(THREADS) schedule(SCHEDULE) reduction(+:heat[:SHELLS][:2]) default(none)
    for (int i = 0; i < SHELLS; i++) {
        heat[i][0] += heat_gpu[i][0];
        heat[i][1] += heat_gpu[i][1];
    }
}

int main()
{
    if (verbose) { // heading
        printf("# %s\n# %s\n# %s\n", t1, t2, t3);
        printf("# Scattering = %8.3f/cm\n", MU_S);
        printf("# Absorption = %8.3f/cm\n", MU_A);
        printf("# Photons    = %8d\n#\n", PHOTONS);
    }
    static float heat[SHELLS][2];
    float ** heat_gpu;
    cudaMallocManaged(&heat_gpu, SHELLS * sizeof(float *));
    for (int i = 0; i < SHELLS; i++) {
        cudaMallocManaged(&heat_gpu[i], 2 * sizeof(float));
        heat_gpu[i][0] = heat_gpu[i][1] = heat[i][0] = heat[i][1] = 0.0f;
    }

    run_both_tiny_mc(heat, heat_gpu, PHOTONS);

    for (int i = 0; i < SHELLS; i++) {
        heat_gpu[i][0] = heat_gpu[i][1] = heat[i][0] = heat[i][1] = 0.0f;
    }


    // start timer
    double start = wtime();
    // simulation
    run_both_tiny_mc(heat, heat_gpu, PHOTONS);
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
    printf("+>>> %lf photons per millisecond\n", 1e-3 * PHOTONS / (elapsed / 1000.0));

    return 0;
}
