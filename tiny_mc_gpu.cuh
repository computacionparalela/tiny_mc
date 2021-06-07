#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>

#include "params.h"

#define MAX_RAND_MATRIX 170

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

inline __device__ void init_rand_matrix(float rand_matrix[MAX_RAND_MATRIX][2], curandStatePhilox4_32_10_t * state)
{
    for (int i = 0; i < MAX_RAND_MATRIX; i += 2) {
        float4 rand = curand_uniform4(state);
        float r1, r2, theta1, theta2;
        r1 = sqrt(rand.x);
        r2 = sqrt(rand.y);
        theta1 = rand.z * 2.0f * M_PI;
        theta2 = rand.w * 2.0f * M_PI;
        rand_matrix[i][0] = 2.0f * r1 * cos(theta1) - 1.0f;
        rand_matrix[i][1] = 2.0f * r1 * sin(theta1) - 1.0f;
        rand_matrix[i + 1][0] = 2.0f * r2 * cos(theta2) - 1.0f;
        rand_matrix[i + 1][1] = 2.0f * r2 * sin(theta2) - 1.0f;
    }
}

static __global__ void photon(float ** global_heat)
{
    /* Step 0: Inicializo el PRNG */
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init((7 + idx) * 9967, (idx + 1), (7 + idx) * 197, &state);

    float4 rand;
    float rand_matrix[MAX_RAND_MATRIX][2];
    init_rand_matrix(rand_matrix, &state);

    // Defino matrices locales
    float local_heat[SHELLS][2];
    for (int i = 0; i < SHELLS; i++) {
        local_heat[i][0] = local_heat[i][1] = 0.0f;
    }

    __shared__ float shared_heat[SHELLS][2];
    for (int i = threadIdx.x; i < 2 * SHELLS; i += blockDim.x) {
        shared_heat[i / 2][i % 2] = 0.0f;
    }
    __syncthreads();

    /* Step 1: Launching a photon packet */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;
    unsigned rand_idx = 0;
    for (int k = 0; k < 170; k++) {
        float r1, r2;
        if (k % 2 == 0) {
            rand = curand_uniform4(&state);
            r1 = rand.x;
            r2 = rand.y;
        } else {
            r1 = rand.w;
            r2 = rand.z;
        }
        /* Step 2: Step size selection and photon packet movement */
        float t = -logf(r1);
        /* move */
        x += t * u;
        y += t * v;
        z += t * w;

        /* Step 3: Absorption and scattering */
        unsigned int shell = min((int)(sqrtf(x * x + y * y + z * z) * SHELLS_PER_MFP), SHELLS - 1);                 /* absorb */
        float _heat = (1.0f - ALBEDO) * weight;
        weight *= ALBEDO;
        // Barrera de sincronizacion
        local_heat[shell][0] += _heat;
        local_heat[shell][1] += _heat * _heat;    /* add up squares */
        ///////////////////////////

        /* New direction, rejection method */
        float x1 = rand_matrix[rand_idx][0];
        float x2 = rand_matrix[rand_idx++][1];
        t = x1 * x1 + x2 * x2;
        u = 2.0f * t - 1.0f;
        t = sqrtf((1.0f - u * u) / t);
        v = x1 * t;
        w = x2 * t;
        /* Step 4: Photon termination */
        if (weight < 0.001f) {    /* roulette */
            if (r2 > 0.1f) {
                break;
            }
            weight /= 0.1f;
        }
    }

    for (int i = 0; i < SHELLS; i++) {
        atomicAdd(&shared_heat[i][0], local_heat[i][0]);
        atomicAdd(&shared_heat[i][1], local_heat[i][1]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * SHELLS; i += blockDim.x) {
        atomicAdd(&global_heat[i / 2][i % 2], shared_heat[i / 2][i % 2]);
    }

    return;
}

unsigned run_gpu_tiny_mc(float ** heat, const int photons, const bool sync = true)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(photons / BLOCK_SIZE);
    photon << < grid, block >> > (heat);
    checkCudaCall(cudaGetLastError());
    if (sync) {
        checkCudaCall(cudaDeviceSynchronize());
    }
    return block.x * grid.x;
}
