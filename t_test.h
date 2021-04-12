#ifndef __T_TEST_H__
#define __T_TEST_H__

#include <bits/stdc++.h>

enum Answer {KEEP, REMOVE};

//Dado un vector de errores realiza un t-test sobre los mismo
Answer t_test(const std::vector<float>& base,const std::vector<float>& step, bool verbosity = true){

        assert(base.size()==step.size());

        //Valor correspondiente a la tabla 5.6 para 95% de confianza y 9 grados de libertad.
        //Libro de Mitchell, sección 5.6, p.145.
        //float table_t_value = 2.101;

        float mean = 0.0, sd = 0.0, t_value = 0.0;
        float mean_1 = 0.0, mean_2 = 0.0, sd_1 = 0.0, sd_2 = 0.0;
        float size = base.size();
        std::vector<float> d;
        for(int i=0; i<size; i++) {
                float df = base[i]-step[i];
                d.push_back(df);
                mean+=df;
                mean_1 += base[i];
                mean_2 += step[i];
        }
        mean /= size;
        mean_1 /= size;
        mean_2 /= size;

        for(int i=0; i<size; i++) {
                sd += (d[i]-mean)*(d[i]-mean);
                sd_1 += (base[i]-mean_1)*(base[i]-mean_1);
                sd_2 += (step[i]-mean_2)*(step[i]-mean_2);
        }
        sd = sqrtf(sd/(size-1.0));
        sd_1 = sqrtf(sd_1/(size-1.0));
        sd_2 = sqrtf(sd_2/(size-1.0));

        if(sd==0) {
                t_value = 0.0;
        }else{
                t_value = mean/(sd/sqrtf(size));
        }

        if(verbosity) {
                std::cout << "Resultados:" << std::endl;
                std::cout << "Media de ejecucion del conjunto 1: " << mean_1 << std::endl;
                std::cout << "Media de ejecucion del conjunto 2: " << mean_2 << std::endl;
                std::cout << "Desviacion de la muestra del conjunto 1: " << sd_1 << std::endl;
                std::cout << "Desviacion de la muestra del conjunto 2: " << sd_2 << std::endl;
                std::cout << "Valor T: " << t_value << std::endl;
        }

        if(fabs(t_value)<0.8f) {//SAME
                return REMOVE;
        }else if(t_value>3.0f) {//Better to remove
                return REMOVE;
        }
        //Not sure
        return KEEP;
}

#endif // __T_TEST_H__
