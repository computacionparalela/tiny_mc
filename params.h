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
