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

static __global__ void photon(float ** global_heat)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Step 0: Inicializo el PRNG */
    curandStatePhilox4_32_10_t state;
    curand_init((7 + idx) * 9967, (idx + 1), (7 + idx) * 197, &state);

    // Defino matrices auxiliares
    float local_heat[SHELLS][2];
    for (int i = 0; i < SHELLS; i++) {
        local_heat[i][0] = local_heat[i][1] = 0.0f;
    }

    __shared__ float shared_heat[REDUCE_SIZE][SHELLS][2];
    for (int i = threadIdx.x; i < 2 * SHELLS; i += blockDim.x) {
        int x = i / 2, y = i % 2;// La division y el modulo son operaciones costosas, tratar de evitarlas
        for (int j = 0; j < REDUCE_SIZE; j++) {
            shared_heat[j][x][y] = 0.0f;
        }
    }
    __syncthreads();

    /* Step 1: Launching a photon packet */
    float x, y, z, u, v, w, weight;
    for (unsigned cnt = 0; cnt < PHOTONS_PER_THREAD; cnt++) {
        x = y = z = u = v = 0.0f;
        w = weight = 1.0f;
        for (;;) {
            float4 rand = curand_uniform4(&state);
            /* Step 2: Step size selection and photon packet movement */
            float t = -logf(rand.x);
            /* move */
            x += t * u;
            y += t * v;
            z += t * w;

            /* Step 3: Absorption and scattering */
            unsigned int shell = min((int)(sqrtf(x * x + y * y + z * z) * SHELLS_PER_MFP), SHELLS - 1); /* absorb */
            float _heat = (1.0f - ALBEDO) * weight;
            local_heat[shell][0] += _heat;
            local_heat[shell][1] += _heat * _heat;    /* add up squares */
            weight *= ALBEDO;

            /* New direction, rejection method */
            float r = sqrt(rand.y), theta = rand.z * 2.0f * M_PI;
            float x1 = 2.0f * r * sin(theta) - 1.0f;
            float x2 = 2.0f * r * cos(theta) - 1.0f;
            t = x1 * x1 + x2 * x2;
            u = 2.0f * t - 1.0f;
            t = sqrtf((1.0f - u * u) / t);
            v = x1 * t;
            w = x2 * t;
            /* Step 4: Photon termination */
            if (weight < 0.001f) {    /* roulette */
                if (rand.w > 0.1f) {
                    break;
                }
                weight /= 0.1f;
            }
        }
    }

    const unsigned designed_shared_matrix = threadIdx.x % REDUCE_SIZE;
    for (int i = 0; i < SHELLS; i++) {
        atomicAdd(&shared_heat[designed_shared_matrix][i][0], local_heat[i][0]);
        atomicAdd(&shared_heat[designed_shared_matrix][i][1], local_heat[i][1]);
    }
    __syncthreads();

    for (unsigned size = REDUCE_SIZE / 2; 0 < size; size /= 2) {
        unsigned group_size = blockDim.x / size, group_pos = threadIdx.x / size, group_id = threadIdx.x % size;
        for (unsigned i = group_pos; i < 2 * SHELLS; i += group_size) {
            shared_heat[group_id][i / 2][i % 2] += shared_heat[group_id + size][i / 2][i % 2];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < 2 * SHELLS; i += blockDim.x) {
        atomicAdd(&global_heat[i / 2][i % 2], shared_heat[0][i / 2][i % 2]);
    }

    return;
}

unsigned run_gpu_tiny_mc(float ** heat, const int photons, const bool sync = true)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((photons / PHOTONS_PER_THREAD) / BLOCK_SIZE);
    photon << < grid, block >> > (heat);
    checkCudaCall(cudaGetLastError());
    if (sync) {
        checkCudaCall(cudaDeviceSynchronize());
    }
    return PHOTONS_PER_THREAD * BLOCK_SIZE * grid.x;
}
