#pragma once

#include <time.h> // time

#ifndef SHELLS
#define SHELLS 101 // discretization level
#endif

#ifndef PHOTONS
#define PHOTONS 32768 // 32K photons
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED (time(NULL)) // random seed
#endif

#ifndef VERBOSE
static const unsigned verbose = 0;
#else
static const unsigned verbose = 1;
#endif

#ifdef RAND0
#include <stdlib.h>

void fast_srand(int seed) {
        srand(seed);
}

float FAST_RAND(void) {
        return rand() / (float)RAND_MAX;
}

#endif

#ifdef RAND1
#define MAXRAND 2147483646.0f

int __rand_x = 0;

void fast_srand(int seed) {
        __rand_x = seed;
}

float FAST_RAND(void) {
        __rand_x = __rand_x * 48271 % 2147483647;
        return __rand_x/MAXRAND;
}

#endif

#ifdef RAND2
#include <stdint.h>

uint64_t s[2] = { 0x41, 0x29837592 };

void fast_srand(int x) {
        (void)x;
}

uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
}

uint64_t next(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        s[1] = rotl(s1, 36); // c

        return result;
}

double FAST_RAND() {
        return next()*(1.0/18446744073709551616.0);
}
#endif

#ifdef RAND3
#include <stdint.h>
//Ver:
// * https://yurichev.com/blog/LCG/
// * https://stackoverflow.com/questions/44960864/analysis-of-linear-congruential-generator-is-wrong
#define MAXRAND 32767.0f
static uint_fast32_t g_seed = 0;

void fast_srand(int seed) {
        g_seed = seed;
}

float FAST_RAND(void) {
        g_seed = 214013*g_seed+2531011;
        return ((g_seed>>16)&0x7FFF)/MAXRAND;
}

#endif

#ifdef RAND4
#include <stdint.h>
#define MAXRAND 32767.0f
#define MAXM 10000000

static uint_fast32_t g_seed0 = 0;
static uint_fast32_t g_seed1 = 0;
static uint_fast32_t g_seed2 = 0;
static uint_fast32_t g_seed3 = 0;
static uint_fast32_t rand_id = 0;
static float mr[MAXM];

float FAST_RAND(void) {
        return mr[rand_id++];
}

float FAST_RAND_INTERNAL0(void) {
        g_seed0 = 214013*g_seed0+2531011;
        return ((g_seed0>>16)&0x7FFF)/MAXRAND;
}

float FAST_RAND_INTERNAL1(void) {
        g_seed1 = 214013*g_seed1+2531011;
        return ((g_seed1>>16)&0x7FFF)/MAXRAND;
}

float FAST_RAND_INTERNAL2(void) {
        g_seed2 = 214013*g_seed2+2531011;
        return ((g_seed2>>16)&0x7FFF)/MAXRAND;
}

float FAST_RAND_INTERNAL3(void) {
        g_seed3 = 214013*g_seed3+2531011;
        return ((g_seed3>>16)&0x7FFF)/MAXRAND;
}

void fast_srand(int seed) {
        g_seed0 = seed+1;
        g_seed1 = seed+2;
        g_seed2 = seed+3;
        g_seed3 = seed+4;
        for(int i=0; i<MAXM; i+=4) {
                mr[i+0] = FAST_RAND_INTERNAL0();
                mr[i+1] = FAST_RAND_INTERNAL1();
                mr[i+2] = FAST_RAND_INTERNAL2();
                mr[i+3] = FAST_RAND_INTERNAL3();
        }
}
#endif

#ifdef RAND5

#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397 /* changes to STATE_VECTOR_LENGTH also require changes to this */
#define UPPER_MASK      0x80000000
#define LOWER_MASK      0x7fffffff
#define TEMPERING_MASK_B    0x9d2c5680
#define TEMPERING_MASK_C    0xefc60000

typedef struct tagMTRand {
        unsigned long mt[STATE_VECTOR_LENGTH];
        int index;
} MTRand;

static MTRand internal_r;

static void m_seedRand(MTRand* rand, unsigned long seed) {
        /* set initial seeds to mt[STATE_VECTOR_LENGTH] using the generator
         * from Line 25 of Table 1 in: Donald Knuth, "The Art of Computer
         * Programming," Vol. 2 (2nd Ed.) pp.102.
         */
        rand->mt[0] = seed & 0xffffffff;
        for(rand->index=1; rand->index<STATE_VECTOR_LENGTH; rand->index++) {
                rand->mt[rand->index] = (6069 * rand->mt[rand->index-1]) & 0xffffffff;
        }
}

/**
 * Creates a new random number generator from a given seed.
 */
MTRand seedRand(unsigned long seed) {
        MTRand rand;
        m_seedRand(&rand, seed);
        return rand;
}

/**
 * Generates a pseudo-randomly generated long.
 */
unsigned long genRandLong(MTRand* rand) {

        unsigned long y;
        static unsigned long mag[2] = {0x0, 0x9908b0df}; /* mag[x] = x * 0x9908b0df for x = 0,1 */
        if(rand->index >= STATE_VECTOR_LENGTH || rand->index < 0) {
                /* generate STATE_VECTOR_LENGTH words at a time */
                int kk;
                if(rand->index >= STATE_VECTOR_LENGTH+1 || rand->index < 0) {
                        m_seedRand(rand, 4357);
                }
                for(kk=0; kk<STATE_VECTOR_LENGTH-STATE_VECTOR_M; kk++) {
                        y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
                        rand->mt[kk] = rand->mt[kk+STATE_VECTOR_M] ^ (y >> 1) ^ mag[y & 0x1];
                }
                for(; kk<STATE_VECTOR_LENGTH-1; kk++) {
                        y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
                        rand->mt[kk] = rand->mt[kk+(STATE_VECTOR_M-STATE_VECTOR_LENGTH)] ^ (y >> 1) ^ mag[y & 0x1];
                }
                y = (rand->mt[STATE_VECTOR_LENGTH-1] & UPPER_MASK) | (rand->mt[0] & LOWER_MASK);
                rand->mt[STATE_VECTOR_LENGTH-1] = rand->mt[STATE_VECTOR_M-1] ^ (y >> 1) ^ mag[y & 0x1];
                rand->index = 0;
        }
        y = rand->mt[rand->index++];
        y ^= (y >> 11);
        y ^= (y << 7) & TEMPERING_MASK_B;
        y ^= (y << 15) & TEMPERING_MASK_C;
        y ^= (y >> 18);
        return y;
}

/**
 * Generates a pseudo-randomly generated double in the range [0..1].
 */
double genRand(MTRand* rand) {
        return((double)genRandLong(rand) / (unsigned long)0xffffffff);
}

void fast_srand(int seed) {
        internal_r = seedRand(seed);
}

float FAST_RAND(void) {
        return genRand(&internal_r);
}
#endif
