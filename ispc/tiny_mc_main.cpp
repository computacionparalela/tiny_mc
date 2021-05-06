/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _XOPEN_SOURCE 500  // M_PI

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include the header file that the ispc compiler generates
#include "tiny_mc_ispc.h"
using namespace ispc;

static float heat[SHELLS];
static float heat2[SHELLS];
static unsigned int seed[64];

int main() {

	srand(SEED);
	for(int i=0;i<64;i++)seed[i] = rand();
	memset(heat,0,sizeof(float)*SHELLS);
	memset(heat2,0,sizeof(float)*SHELLS);

	// start timer
	double start = wtime();
	// simulation
	photonV(heat, heat2, seed, PHOTONS);
	// stop timer
	double end = wtime();
	assert(start <= end);
	double elapsed = (end - start)*1000.0;
	if(verbose) {
			printf("# Radius\tHeat\n");
			printf("# [microns]\t[W/cm^3]\tError\n");
			float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
			for (unsigned int i = 0; i < SHELLS - 1; ++i) {
					printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
						   heat[i] / t / (i * i + i + 1.0 / 3.0),
						   sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
			}
			printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
	}
	printf("+>> %lf ms\n", elapsed);
	printf("+>>> %lf K photons per second\n", 1e-3 * PHOTONS / (elapsed/1000.0));


    return 0;
}
