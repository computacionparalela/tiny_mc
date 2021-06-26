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

float4 __device__ __forceinline__ fast_rand(int4 * seed)
{
    float4 rnd;
    seed->x ^= seed->x << 13;
    seed->y ^= seed->y << 13;
    seed->z ^= seed->z << 13;
    seed->w ^= seed->w << 13;
    seed->x ^= seed->x >> 17;
    seed->y ^= seed->y >> 17;
    seed->z ^= seed->z >> 17;
    seed->w ^= seed->w >> 17;
    seed->x ^= seed->x << 5;
    seed->y ^= seed->y << 5;
    seed->z ^= seed->z << 5;
    seed->w ^= seed->w << 5;
    rnd.x = seed->x;
    rnd.y = seed->y;
    rnd.z = seed->z;
    rnd.w = seed->w;
    rnd.x = (rnd.x + 2147483647.0f) / 4294967296.0f;
    rnd.y = (rnd.y + 2147483647.0f) / 4294967296.0f;
    rnd.z = (rnd.z + 2147483647.0f) / 4294967296.0f;
    rnd.w = (rnd.w + 2147483647.0f) / 4294967296.0f;
    return rnd;
}

float __device__ __forceinline__ warp_reducef(float val)
{
    #pragma unroll
    for (int offset = CUDA_HALF_WARP_SIZE; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

static __global__ void photon(float ** global_heat)
{
    const unsigned lane = threadIdx.x & CUDA_WARP_MASK;
    const unsigned warp = threadIdx.x / CUDA_WARP_SIZE;
    const unsigned tid  = threadIdx.x;
    const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned bdim = blockDim.x;

    // Inicializo el PRNG
    int4 state = make_int4((7 + gtid) * 9967, (gtid + 1), (7 + gtid) * 197, (1 + gtid) * 9997);

    // Defino matrices auxiliares
    float local_heat[SHELLS][2];
    __shared__ float shared_heat[REDUCE_SIZE][SHELLS][2];
    for (int i = tid; i < 2 * SHELLS; i += bdim) {
        int x = i / 2, y = i % 2;// La division y el modulo son operaciones caras
        for (int j = 0; j < REDUCE_SIZE; j++) {
            shared_heat[j][x][y] = 0.0f;
        }
    }
    #pragma unroll
    for (int i = 0; i < SHELLS; i++) {
        local_heat[i][0] = local_heat[i][1] = 0.0f;
    }
    __syncthreads();

    /* Step 1: Launching a photon packet */
    float x, y, z, u, v, w, weight;
    unsigned photons = PHOTONS_PER_THREAD;
    x = y = z = u = v = 0.0f;
    w = weight = 1.0f;
    while (photons > 0) {
        float4 st = fast_rand(&state);
        /* Step 2: Step size selection and photon packet movement */
        float t = -logf(st.x);
        /* move */
        x += t * u;
        y += t * v;
        z += t * w;
        /* Step 3: Absorption and scattering */
        unsigned int shell = min((unsigned)(sqrtf(x * x + y * y + z * z) * SHELLS_PER_MFP), SHELLS - 1); /* absorb */
        float _heat = (1.0f - ALBEDO) * weight;
        local_heat[shell][0] += _heat;
        local_heat[shell][1] += _heat * _heat;
        weight *= ALBEDO;
        /* New direction */
        float r = sqrtf(st.y), _sin, _cos;
        sincospi(2.0f * st.z, &_sin, &_cos);
        float x1 = 2.0f * r * _sin - 1.0f;
        float x2 = 2.0f * r * _cos - 1.0f;
        t = x1 * x1 + x2 * x2;
        u = 2.0f * t - 1.0f;
        t = sqrtf((1.0f - u * u) / t);
        v = x1 * t;
        w = x2 * t;
        /* Step 4: Photon termination */
        if (weight < 0.001f) {    /* roulette */
            weight /= 0.1f;
            if (st.w > 0.1f) {
                photons--;
                x = y = z = u = v = 0.0f;
                w = weight = 1.0f;
            }
        }
    }
    __syncwarp();

    /* Step 5: Reduce */
    #pragma unroll
    for (unsigned i = 0; i < SHELLS + (SHELLS % CUDA_WARP_SIZE); i++) {
        float k0 = i < SHELLS ? local_heat[i][0] : 0.0f;
        float k1 = i < SHELLS ? local_heat[i][1] : 0.0f;
        k0 = warp_reducef(k0);
        k1 = warp_reducef(k1);
        if (lane == 0) {
            atomicAdd(&shared_heat[warp % REDUCE_SIZE][i][0], k0);
            atomicAdd(&shared_heat[warp % REDUCE_SIZE][i][1], k1);
        }
        __syncwarp();
    }
    __syncthreads();

    #pragma unroll
    for (unsigned size = REDUCE_SIZE / 2; 0 < size; size /= 2) {
        unsigned group_size = bdim / size, group_pos = tid % group_size, group_id = tid / group_size;
        for (unsigned i = group_pos; i < 2 * SHELLS; i += group_size) {
            shared_heat[group_id][i / 2][i % 2] += shared_heat[group_id + size][i / 2][i % 2];
        }
        __syncthreads();
    }

    for (int i = tid; i < 2 * SHELLS; i += bdim) {
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
