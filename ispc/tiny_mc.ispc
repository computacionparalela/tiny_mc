#define MAXRAND 4294967296.0f
#define MU_A 2.0f
#define MU_S 20.0f
#define ALBEDO (MU_S / (MU_S + MU_A))
#define SHELLS 101
#define MICRONS_PER_SHELL 50
#define SHELLS_PER_MFP (1e4 / MICRONS_PER_SHELL / (MU_A + MU_S))

static inline float rnd(unsigned int &number) {
	unsigned int x = number;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	number = x;
	return number / MAXRAND;
}

static void photonT2(uniform float heat[], uniform float heat2[], uniform unsigned int seed[], uniform int gang_counter)
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
	float w = 1.0f;
	float old_weight;
	float weight = 1.0f;

	foreach (index = 0 ... gang_counter) {
	
		unsigned int rseed = seed[index];
	
		do{
			float t = -log(rnd(rseed));
			
			x += t * u;
			y += t * v;
			z += t * w;

			float nshell = sqrt(x * x + y * y + z * z) * SHELLS_PER_MFP;
			unsigned int shell = min((int)nshell, SHELLS - 1);
			float _heat = (1.0f - ALBEDO) * weight;
			atomic_add_local(&heat[shell],_heat);
			atomic_add_local(&heat2[shell],_heat*_heat);
			weight = weight * ALBEDO;
			
			float xi1, xi2, tmp;
			do {
				xi1 = 2.0f * rnd(rseed) - 1.0f;
				xi2 = 2.0f * rnd(rseed) - 1.0f;
				t = xi1 * xi1 + xi2 * xi2;
			} while (1.0f < t);
			u = 2.0f * t - 1.0f;
			tmp = sqrt((1.0f - u * u) / t);
			v = xi1 * tmp;
			w = xi2 * tmp;

			old_weight = weight;
			if (weight < 0.001f) {
				weight /= 0.1f;
			}
			seed[index] = rseed + 223;
		} while(!(old_weight < 0.001f && rnd(rseed) > 0.1f));
	}
}

static void photonT(uniform float heat[], uniform float heat2[], uniform unsigned int seed[], uniform int gang_counter, uniform int photons)
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
	float w = 1.0f;
	float weight = 1.0f;

	foreach (index = 0 ... gang_counter) {
	
		unsigned int rseed = seed[index];
	
		do{
			float t = -log(rnd(rseed));
			
			x += t * u;
			y += t * v;
			z += t * w;

			float _heat = (1.0f - ALBEDO) * weight;
			float nshell = sqrt(x * x + y * y + z * z) * SHELLS_PER_MFP;
			unsigned int shell = min((int)nshell, SHELLS - 1);
			weight = weight * ALBEDO;
			
			atomic_add_local(&heat[shell],_heat);
			atomic_add_local(&heat2[shell],_heat*_heat);
			
			float xi1, xi2, tmp;
			do {
				xi1 = 2.0f * rnd(rseed) - 1.0f;
				xi2 = 2.0f * rnd(rseed) - 1.0f;
				t = xi1 * xi1 + xi2 * xi2;
			} while (1.0f < t);
			u = 2.0f * t - 1.0f;
			tmp = sqrt((1.0f - u * u) / t);
			v = xi1 * tmp;
			w = xi2 * tmp;
			float old_weight = weight;

			if (weight < 0.001f) {
				weight /= 0.1f;
				if (rnd(rseed) > 0.1f){
					x = y = z = u = v = 0.0f;
					w = weight = 1.0f;
					atomic_subtract_local(&photons, 1);
				}
			}
			seed[index] = rseed + 223;

		} while(photons > 0);
	}
}

export void photonV(uniform float heat[], uniform float heat2[], uniform unsigned int seed[], uniform int photons)
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
	float w = 1.0f;
	float weight = 1.0f;

	unsigned int rseed = seed[programIndex];
	//Saco los primeros N fotones
	atomic_subtract_local(&photons, 1);
	do{
		float t = -log(rnd(rseed));
		x += t * u;
		y += t * v;
		z += t * w;

		float _heat = (1.0f - ALBEDO) * weight;
		float nshell = sqrt(x * x + y * y + z * z) * SHELLS_PER_MFP;
		unsigned int shell = min((int)nshell, SHELLS - 1);
		
		atomic_add_local(&heat[shell],_heat);
		atomic_add_local(&heat2[shell],_heat*_heat);
		
		float xi1, xi2, tmp;
		do {
			xi1 = 2.0f * rnd(rseed) - 1.0f;
			xi2 = 2.0f * rnd(rseed) - 1.0f;
			t = xi1 * xi1 + xi2 * xi2;
		} while (1.0f < t);
		u = 2.0f * t - 1.0f;
		tmp = sqrt((1.0f - u * u) / t);
		v = xi1 * tmp;
		w = xi2 * tmp;

		weight = weight * ALBEDO;
		if (weight < 0.001f) {
			weight /= 0.1f;
			if (rnd(rseed) > 0.1f){
				//Preparar para uno nuevo?
				if(atomic_subtract_local(&photons, 1) > 0){
					x = y = z = u = v = 0.0f;
					w = weight = 1.0f;
				}else{
					break;
				}
			}
		}
	} while(1);
}
