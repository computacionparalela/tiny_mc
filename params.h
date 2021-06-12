#ifndef PARAMS_H
#define PARAMS_H

#include <time.h> // time
#include <omp.h>
#include <sys/sysinfo.h>
#include <limits.h>

#ifndef THREADS
#define THREADS           28
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE        128
#endif

#ifndef PHOTONS_PER_THREAD
#define PHOTONS_PER_THREAD 32
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE       16
#endif

#ifndef CHUNKS
#define CHUNKS            1000
#endif

#ifndef SCHEDULE
#define SCHEDULE          dynamic
#endif

#ifndef SHELLS
#define SHELLS            101 // discretization level
#endif

#ifndef PHOTONS
#define PHOTONS           (1 << 25)
#endif

#ifndef MU_A
#define MU_A              2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S              20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED              (time(NULL)) // random seed
#endif

#ifndef M_PI
#define M_PI              3.14159265358979323846
#endif


#ifndef VERBOSE
static const unsigned verbose = 0;
#else
static const unsigned verbose = 1;
#endif

#define ALBEDO         (MU_S / (MU_S + MU_A))
#define SHELLS_PER_MFP (1e4 / MICRONS_PER_SHELL / (MU_A + MU_S))

#endif //PARAMS_H
