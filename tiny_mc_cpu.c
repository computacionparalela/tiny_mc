#include "tiny_mc_cpu.h"
#include "wtime.h"

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

int main(void)
{

        static float heat[SHELLS][2];
        static float seed[8];// Avoid buffer overflow
        if (verbose) { // heading
                printf("# %s\n# %s\n# %s\n", t1, t2, t3);
                printf("# Scattering = %8.3f/cm\n", MU_S);
                printf("# Absorption = %8.3f/cm\n", MU_A);
                printf("# Photons    = %8d\n#\n", PHOTONS);
        }

        // configure RNG
        srand(SEED);
        for (int i = 0; i < 8; i++) {
                seed[i] = rand();
        }
    #pragma omp parallel for
        for (int i = 0; i < 8; ++i) {
                photon(heat, 256, seed);
        }
        // first run
        memset(heat, 0, 2 * sizeof(float) * SHELLS);
        // start timer
        double start = wtime();
        // simulation
        run_cpu_tiny_mc(heat, PHOTONS);
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
