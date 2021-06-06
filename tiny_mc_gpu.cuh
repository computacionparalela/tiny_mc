#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>

#include "params.h"

#define checkCudaCall(val) _checkCudaReturnValue((val), #val, __FILE__, __LINE__)

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

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return 1e-9 * ts.tv_nsec + (double)ts.tv_sec;
}

static __global__ void photon(float ** global_heat, int photons)
{
    /* Step 0: Inicializo el PRNG */
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init((7 + idx) * 9967, (idx + 1), (7 + idx) * 197, &state);


    float heat[SHELLS][2];
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
            heat[shell][0] += _heat;
            heat[shell][1] += _heat * _heat;/* add up squares */
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

    __shared__ float shared_heat[SHELLS][2];
    for (int i = 0; i < SHELLS; i++) {
        atomicAdd(&shared_heat[i][0], heat[i][0]);
        atomicAdd(&shared_heat[i][1], heat[i][1]);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < SHELLS; i++) {
            atomicAdd(&global_heat[i][0], shared_heat[i][0]);
            atomicAdd(&global_heat[i][1], shared_heat[i][1]);
        }
    }

    return;
}

unsigned run_gpu_tiny_mc(float ** heat, const int photons, const bool sync = true)
{
    static const unsigned photons_per_thread = 1 << 12;
    dim3 block(BLOCK_SIZE);
    dim3 grid((photons / photons_per_thread) / BLOCK_SIZE);
    photon << < grid, block >> > (heat, photons_per_thread);
    checkCudaCall(cudaGetLastError());
    if (sync) {
        checkCudaCall(cudaDeviceSynchronize());
    }

    return (1 << 12) * block.x * grid.x;
}
