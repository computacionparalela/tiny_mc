# Tiny Monte Carlo

El objetivo de este trabajo práctico es realizar optimizaciones sobre un programa que pretende mostrar de forma representativa el método Monte Carlo.
El código realiza simulaciones de propagación de luz, a partir de una fuente puntual, sobre un medio infinito con dispersión isotrópica.

- [Página en Wikipedia sobre el problema](https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport)
- [Código original](https://omlc.org/software/mc/) de [Scott Prahl](https://omlc.org/~prahl/)

# Laboratorio 4: CUDA

En este cuarto laboratorio utilizamos CUDA para poder ejecutar nuestro programa dentro de la GPU tratando de alcanzar un rendimiento mucho mayor.

Las ejecuciones se realizaron sobre el servidor Jupiterace que cuenta con un CPU `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz` y una GPU `NVIDIA RTX 2080 TI`.
Como métrica decidimos utilizar la cantidad de fotones procesados por milisegundo, que fue la utilizada para la entrega anterior.

## Kernel

Esta sección va a estar dedicada a describir el funcionamiento del kernel generado para ejecutar las simulaciones del problema `tiny_mc` en la GPU. El código completo puede encontrarse en el archivo [tiny_mc_kernel.cu](https://github.com/barufa/tiny_mc/blob/lab4/tiny_mc_kernel.cu).

### Generador de numeros aleatorios

Como generador de números aleatorios se decidió utilizar una implementación de Xorshift implementada en CUDA para generar 4 números aleatorios por llamada. Sin embargo, tambien se obtuvieron muy buenos resultados con el generador de numeros [Philox](https://developer.nvidia.com/curand) que nos brinda la libreria [curand](https://docs.nvidia.com/cuda/archive/10.1/curand/host-api-overview.html) implementada por Nvidia.

```
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

```

### Calculo de nueva direccion del foton

En las entregas anteriores, para calcular la nueva dirección del fotón utilizabamos el método del rechazo, que consiste en generar dos puntos aleatorios y verificar que cumplan con ciertas condiciones:

```
        float x1, x2;
        do {
                x1 = 2.0f * FAST_RAND() - 1.0f;
                x2 = 2.0f * FAST_RAND() - 1.0f;
        } while (1.0f < x1 * x1 + x2 * x2);
```

Sin embargo, se obtuvieron mejores resultados al utilizar un método alternativo que utiliza coordenadas polares.

```
        float r = sqrtf(st.y), _sin, _cos;
        sincospi(2.0f * st.z, &_sin, &_cos);
        float x1 = 2.0f * r * _sin - 1.0f;
        float x2 = 2.0f * r * _cos - 1.0f;
```

Considero que este método se adapta mejor al modelo de ejecución de CUDA ya que todos los hilos ejecutan las mismas instrucciones sin divergencias. En el caso anterior, la ejecución no hubiera podido continuar hasta que los 32 hilos del warp obtuvieran los valores adecuados para x1 y x2, lo que se traduce en un retraso innecesario para la mayoría de los hilos.

En esta implementacion se decidio utilizar la funcion [sincospi](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gfc99d7acfc1b14dcb6f6db56147d2560) ya que parece brindar un mejor rendimiento que calcular el seno y coseno por separado.

### Agrupacion de las simulaciones

De forma similar a lo hecho en el laboratorio 3, se decidió partir el problema entre los distintos hilos para que cada uno resuelva una pequeña parte del problema y almacenar los resultados parciales en los arreglos `local_heat` para luego sumarlos y obtener el resultado final. Para ello, se definieron 3 jerarquías de arreglos hasta llegar al resultado final: `local_heat`, `shared_heat` y `global_heat`.

En la primer jerarquía se encuentra el arreglo `local_heat`. Cada hilo cuenta con su versión de este arreglo (oculta para los demás hilos). De esta forma, cada hilo puede realizar las simulaciones y escribir dentro de estos arreglos sin tener que utilizar funciones de sincronización (como `atomicAdd`).

Al finalizar las simulaciones, cada hilo tiene un resultado parcial de nuestro problema alocado en su `local_heat`. El siguiente paso se basa en combinar estos resultados parciales y almacenar los resultados en los arreglos `shared_heat`. Para ello implementamos una reducción a nivel de warp que nos permite sumar los distintos arreglos `local_heat` y almacenarlos en el lane 0 de cada warp, para luego escribir esos resultados en memoria compartida para que puedan ser accedidos por todos los hilos del bloque.

Para simplificar el código se implementó la función `warp_reducef` que suma todos los distintos valores de la misma variable dentro de un warp y los retorna al lane 0 del warp.

```
float __device__ __forceinline__ warp_reducef(float val)
{
    #pragma unroll
    for (int offset = CUDA_HALF_WARP_SIZE; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}
```

El siguiente código muestra como podemos reducir los distintos arreglos `local_heat` dentro de un mismo warp y almacenarlos en el arreglo `shared_heat` correspondiente.

```
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
```

En este punto, los resultados parciales se encuentran acumulados dentro de los arreglos `shared_heat`. El paso final es combinar estos arreglos y almacenar el resultado final(del bloque) dentro del arreglo `global_heat`, cuya dirección de memoria nos fue brindada como argumento del kernel.

Para realizar esta tarea lo que hacemos es almacenar el resultado final dentro de uno de los arreglos `shared_heat`(que se encuentra dentro de la memoria compartida) y luego copiarlo a `global_heat`(que se encuentra en la memoria RAM de la GPU). De esta forma, minimizamos la cantidad de accesos que realizamos a la memoria RAM.

En el siguiente fragmento de código podemos ver una reducción de los arreglos `shared_heat`. Lo que se busca es reducir a los arreglos en forma de árbol para poder aprovechar al máximo todos los hilos disponible del bloque.

```
    #pragma unroll
    for (unsigned size = REDUCE_SIZE / 2; 0 < size; size /= 2) {
        unsigned group_size = bdim / size, group_pos = tid % group_size, group_id = tid / group_size;
        for (unsigned i = group_pos; i < 2 * SHELLS; i += group_size) {
            shared_heat[group_id][i / 2][i % 2] += shared_heat[group_id + size][i / 2][i % 2];
        }
        __syncthreads();
    }
```

Supongamos que tenemos un 8 arreglos `shared_heat`, numerados del 0 al 7. En la primer iteración del código anterior lo que haremos será sumar el arreglo `i`(con i en [0 ,3]) con el arreglo `i + size`(donde size = 4). Entonces, tendríamos que el arreglo `shared_heat[0]` almacena la suma del `shared_heat[0] + shared_heat[0 + 4]`, el arreglo `shared_heat[1]` almacenaría `shared_heat[1] + shared_heat[1 + 4]`, etc. En la segunda iteración estaríamos sumando el arreglo `i`(con i en [0, 1]) con el arreglo `i + size`(con size = 2). Por ende, en el arreglo `shared_heat[0]` almacenaremos `shared_heat[0] + shared_heat[0 + 2]` y en `shared_heat[1]` los valores de `shared_heat[1] + shared_heat[1 + 2]`. Finalmente haríamos una última combinación donde sumamos los arreglos `shared_heat[0]` y `shared_heat[1]` para obtener el resultado final.

```
//En resumen, en la primer iteracion:
shared_heat[0] = shared_heat[0] + shared_heat[4]
shared_heat[1] = shared_heat[1] + shared_heat[5]
shared_heat[2] = shared_heat[2] + shared_heat[6]
shared_heat[3] = shared_heat[3] + shared_heat[7]
//En resumen, en la segunda iteracion:
shared_heat[0] = shared_heat[0] + shared_heat[2]//que equivaldria a (shared_heat[0] + shared_heat[4]) + (shared_heat[2] + shared_heat[6])
shared_heat[1] = shared_heat[1] + shared_heat[3]//que equivaldria a (shared_heat[1] + shared_heat[5]) + (shared_heat[3] + shared_heat[7])
//Ultima iteracion
shared_heat[0] = shared_heat[0] + shared_heat[1]
//que equivaldria a ((shared_heat[0] + shared_heat[4]) + (shared_heat[2] + shared_heat[6])) + ((shared_heat[1] + shared_heat[5]) + (shared_heat[3] + shared_heat[7]))
```

Para hacer un mejor uso de los recursos, asignamos cada hilo a un "grupo" que se encargará de sumar todos los valores del arreglo `shared_heat[i + size]` al arreglo `shared_heat[i]`. En cada iteración los hilos se irán reagrupando para distribuirse esta tarea de una forma más uniforme.

Finalmente, copiamos el resultado del cómputo que se encuentra alojado en el primer arreglo `shared_heat` al arreglo `global_heat` para comunicar los resultados y  terminar la ejecución del kernel.

```
    for (int i = tid; i < 2 * SHELLS; i += bdim) {
        atomicAdd(&global_heat[i / 2][i % 2], shared_heat[0][i / 2][i % 2]);
    }
```

### Distrubucion de los fotones

Para hacer un mejor uso de los recursos se decidió simular más de un fotón por hilo. De esta forma, se realizan menos combinaciones de resultados parciales por fotón y se obtiene un mejor rendimiento en general (ya que los arreglos `local_heat` almacenan más de una simulación). Para determinar la cantidad adecuada de fotones que van a simular cada hilo se realizó un barrio sobre la variable `PHOTONS_PER_THREAD`.

## Resultados

Se realizó un barrido entre los distintos valores de tamaño de bloque y cantidad de fotones simulados por hilo. Esto para determinar cuál es la configuración más adecuada de esta variables para maximizar la cantidad de fotones procesados por milisegundos.

![Comparacion de las distintas configuraciones](https://raw.githubusercontent.com/barufa/tiny_mc/lab4/data/bloque.png "Comparacion de las distintas configuraciones")

Podemos ver que el tamaño del bloque no parece afectar tanto al problema, siempre y cuando este sea mayor al tamaño del warp. Sin embargo, la cantidad de fotones que procesa cada hilo parece tener una mayor influencia en el rendimiento del programa. Se observa que los valores máximos se alcanzan cuando cada hilo simula entre 64 y 512 fotones, alcanzando un valor máximo con un tamaño de bloque de 128 y simulando 256 fotones por cada hilo.

| Fotones por hilo |      16     |      32     |      64     |     128     |     256     |     512     |
| ---------------- |:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|                1 |  156178,500 |  287080,109 |  328289,641 |  328049,169 |  308161,535 |  291981,410 |
|                2 |  233319,763 |  444726,507 |  541992,309 |  532925,321 |  486880,761 |  477154,026 |
|                4 |  295193,054 |  566949,929 |  624255,721 |  623743,438 |  613003,355 |  611340,584 |
|                8 |  326749,115 |  631933,837 |  657502,086 |  657240,829 |  656746,370 |  650555,647 |
|               16 |  340976,968 |  664433,539 |  679742,191 |  679442,948 |  679012,890 |  675486,035 |
|               32 |  348672,264 |  683140,541 |  694569,122 |  693420,736 |  693667,780 |  692035,723 |
|               64 |  353293,764 |  694176,272 |  704214,607 |  705172,062 |  703713,579 |  703168,482 |
|              128 |  356031,948 |  699540,595 |  710365,823 |  710458,785 |  710422,478 |  704768,334 |
|              256 |  357431,318 |  699926,448 |  712848,339 |**714021,849**|  708882,127 |  698477,769 |
|              512 |  357332,423 |  691702,088 |  710188,201 |  709933,406 |  700032,000 |  679796,493 |
|             1024 |  352732,762 |  682014,409 |  697668,309 |  697254,071 |  677625,005 |  681195,688 |
|             2048 |  349416,660 |  667001,304 |  670304,418 |  668654,132 |  676443,686 |  680594,604 |

También se llevaron a cabo simulaciones para ver cómo afecta el tamaño del problema al rendimiento general del sistema.

![Comparacion cambiando el tamaño del problema](https://raw.githubusercontent.com/barufa/tiny_mc/lab4/data/fotones.png "Comparacion cambiando el tamaño del problema")

Podemos observar que para simular una cantidad relativamente pequeña de fotones nos conviene seguir utilizando la CPU. Sin embargo, podemos ver una drástica mejora en la eficiencia de las versión que utilizan GPU a medida que aumentamos la cantidad de fotones. En su punto máximo, nuestra versión GPU del problema `tiny_mc` logra alcanzar los 717000 fotones procesados por milisegundo.

| Cantidad de fotones |     CPU    |     GPU    |    Ambos   |
| ------------------- |:----------:| ----------:| ----------:|
|               16384 |  16562,047 |   2104,102 |          - |
|               32768 |  15900,915 |   4030,608 |          - |
|               65536 |  20645,588 |   8037,926 |          - |
|              131072 |  24215,856 |  15922,411 |          - |
|              262144 |  26517,053 |  31858,501 |          - |
|              524288 |  26509,410 |  63627,700 |          - |
|             1048576 |  31932,893 | 126425,331 |          - |
|             2097152 |  39137,289 | 251869,611 |          - |
|             4194304 |  37326,687 | 448321,688 |          - |
|             8388608 |  41644,795 | 618491,519 |          - |
|            16777216 |  45506,039 | 616672,277 |          - |
|            33554432 |  45084,391 | 647633,023 |          - |
|            67108864 |  45928,736 | 687211,411 |          - |
|           134217728 |  45822,735 | 705969,478 |          - |
|           268435456 |  45574,366 | 712733,964 | 754680,672 |
|           536870912 |  45521,027 | 715365,600 | 762535,549 |
|          1073741824 |  45604,697 | 716367,745 | 764723,474 |
|          2147483647 |  45728,695 |**717974,175**|**765449,832**|

También se decidió tratar de combinar el poder de cómputo de la CPU con la GPU para seguir incrementando la cantidad de fotones procesados por milisegundos. Para ello simplemente se le pasó una parte de las simulaciones a la CPU.

```
void run_both_tiny_mc(float (*heat)[2], float ** heat_gpu, const unsigned photons)
{
    const unsigned frac = 16;
    const unsigned photons_gpu = photons - (photons / frac);
    const unsigned photons_cpu = photons - run_gpu_tiny_mc(heat_gpu, photons_gpu);
    run_cpu_tiny_mc(heat, photons_cpu);
    checkCudaCall(cudaDeviceSynchronize());
    #pragma omp parallel for firstprivate(heat_gpu)  num_threads(THREADS) schedule(SCHEDULE) reduction(+:heat[:SHELLS][:2]) default(none)
    for (int i = 0; i < SHELLS; i++) {
        heat[i][0] += heat_gpu[i][0];
        heat[i][1] += heat_gpu[i][1];
    }
}
```

Esta nueva versión logró superar los 765000 fotones procesados por milisegundo, a pesar de que la división del trabajo es demasiado básica.

## Conclusiones

Gracias a la naturaleza del problema fue posible aprovechar al máximo los recursos de la GPU y lograr mejoras en el rendimiento muy buenas. Mediante la programación en CUDA se logró multiplicar la cantidad de fotones procesados por milisegundos por 15.7 respecto a la mejor versión obtenida en la entrega anterior, pasando de 45000 alcanzados con OpenMP a 717000 fotones y logrando un máximo de 765449,832 al combinar ambas tecnologías (16.73 veces más).

Como trabajo a futuro se pueden seguir realizando pequeñas optimización al kernel presentado en esta entrega, en particular se podría hacer una mejor uso de las primitivas de cuda a nivel de warp para agrupar los distintos resultados almacenados en el arreglo `shared_heat`. Una forma de hacerlo sería cargar en la primera mitad del warp 16 valores de `shared_heat[x]` y otros 16 valores de `shared_heat[y]` para luego realizar una suma sobre las posiciones correspondientes mediante una unica llamada a `__shfl_down_sync`.
