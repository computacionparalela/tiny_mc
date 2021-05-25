# Tiny Monte Carlo

El objetivo de este trabajo práctico es realizar optimizaciones sobre un programa que pretende mostrar de forma representativa el método Monte Carlo.
El código realiza simulaciones de propagación de luz, a partir de una fuente puntual, sobre un medio infinito con dispersión isotrópica.

- [Página en Wikipedia sobre el problema](https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport)
- [Código original](https://omlc.org/software/mc/) de [Scott Prahl](https://omlc.org/~prahl/)

# Laboratorio 3: Multithreading

En este tercer laboratorio adaptamos el código de nuestro programa para que pueda ejecutarse en varios cores a la vez para hacer un mejor uso de los recursos.

Las ejecuciones se realizaron sobre el servidor Jupiterace que cuenta con un `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz`.
Como métrica decidimos utilizar la cantidad de fotones procesados por milisegundo, ya que es más fácil de manejar debido al tamaño de los números.

## Multithreading

Para paralelizar el código hacemos uso de la herramienta OpenMP, que nos permite lanzar varios hilos mediante una interfaz que requiere realizar modificaciones mínimas al código secuencial.

Nuestro código secuencial se encontraba casi en su totalidad dentro de la función `photon` que tenía la siguiente firma :

```
void photon(float ** heat, int photons, float * _seed);
```

Algo también a destacar es que se pueden combinar los resultados de dos llamadas a la función `photon` para que sea equivalente a un llamada con la suma de las cantidades simuladas, es decir

```
photon(heat, fotones1, seed1) + photon(heat, fotones2, seed2) = photon(heat, fotones1 + fotones2, seed3)
```

donde el operador `+` hace referencia a realizar la sumas de la matrices `heat` obtenidas en cada ejecución.

Por todo esto, paralelizar el código puede reducirse a ejecutar una (o más) instancias de la función `photon` en cada core. Luego, los resultados pueden combinarse durante las simulaciones utilizando locks (cada instancia escribe en los mismos arreglos heat y heat2) o al finalizar todas las simulaciones (combinando los resultados parciales).

En resultados preliminares notamos que (como vimos en las clases teóricas) utilizar locks requiere una gran comunicación entre los hilos, lo que es mucho menos eficiente en comparación con combinar los resultados parciales. Por lo tanto, decidimos optar por combinar los resultados haciendo uso del operador `reduction` que nos brinda OpenMP.

En resumen, OpenMP nos permite paralelizar nuestro código mediante el agregado de una única línea antes del llamado a nuestra función `photon`:

```
 #pragma omp parallel private(seed) reduction(+:heat[:SHELLS][:2]) num_threads(THREADS)
{
    //Inicializo seed de alguna forma(que sea única para cada thread)
    for(int i=0; i<8; i++) {
        seed[i] = (i+1)*223*(10000*omp_get_thread_num()+1);
    }
    //Llamó a la función photon para simular una cantidad de fotones determinada en
    //base a la cantidad de threads.
    photon(heat, PHOTONS / THREADS, seed);
}
```

De esta forma, lanzamos `THREADS` hilos (idealmente 1 por core) para que ejecuten distintas instancias de la función `photon` y combinen los resultados parciales.

Es importante notar que cada instancia de la función `photon` contará con una semilla distinta. Si esto no fuese así, estaríamos simulando los mismos fotones más de una vez lo cual alteraría nuestros resultados.

Sin embargo, paralelizar de esta forma, donde cada core realiza una cantidad de trabajo similar, puede traer problemas. Si uno de los cores se retrasa, esto afectará los tiempos de ejecución del programa en general. Una forma de contrarrestar esto es dividir el trabajo en más segmentos y repartirlos entre los distintos hilos a medida que estos se van desocupando.

```
    #pragma omp parallel for firstprivate(seed) shared(heat) num_threads(THREADS)
    for (int i = 0; i < CHUNKS; ++i) {
        for(int j=0; j<8; j++) {
            seed[j] *= i+1;
        }
        photon(PHOTONS / CHUNKS, seed[i]);
    }
```

## Resultados

Una de las primeras mediciones que realizamos fue modificar el tamaño del problema, para ver como la cantidad de fotones afecta a la eficiencia general del programa.


![Comparacion en funcion de la cantidad de fotones simulados](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/fotones.png "Comparacion en funcion de la cantidad de fotones simulados")


En este gráfico podemos ver el rendimiento de nuestro programa `secuencial` que se corresponde con la mejor versión obtenida para el laboratorio 2. La versión `Estática` hace referencia a la primera forma de paralelismo presentada en este informe; y la versión `Fraccionado` a la división del trabajo en distintos segmentos.

Podemos apreciar que ambas versiones parecen funcionar de manera muy similar para distintas cantidades de fotones, incrementando el rendimiento a medida que crece el problema hasta estancarse alrededor de 60 millones de fotones simulados. La versión fraccionada parece ser la más eficiente, alcanzando un rendimiento cercano a los 43000 fotones procesados por milisegundo.

Por todo esto, podemos afirmar que nuestro problema es `weak scaling` ya que el rendimiento aumenta (hasta cierto punto) a medida que crece el tamaño del problema.

| Cantidad de fotones | Secuencial | Fraccionado | Estatico  |
| ------------------- |:----------:| -----------:| ---------:|
| 16384               | 1207.429   | 17869.705   | 17839.473 |
| 32768               | 1205.894   | 20136.526   | 18224.442 |
| 65536               | 1361.314   | 18738.365   | 19923.566 |
| 131072              | 1353.791   | 19230.836   | 16977.111 |
| 262144              | 1539.867   | 24280.308   | 21611.241 |
| 524288              | 1638.288   | 30379.821   | 27303.171 |
| 1048576             | 1655.496   | 35791.689   | 33957.702 |
| 2097152             | 1714.668   | 36974.701   | 32423.821 |
| 4194304             | 1706.935   | 37193.205   | 31143.943 |
| 8388608             | 1743.716   | 38909.313   | 36469.921 |
| 67108864            | 1724.661   | 43215.317   | 39336.608 |
| 134217728           | 1716.628   | 43241.849   | 40080.642 |
| 268435456           | 1706.094   | 43197.346   | 38910.273 |

Un experimento similar se realizó modificando la cantidad de hilos utilizado por nuestro programa, para simular 60 millones de fotones. 

![Comparacion en funcion de la cantidad de hilos](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/Cantidad_hilos.png "Comparacion en funcion de la cantidad de hilos")

En este caso, agregamos un rendimiento “teórico” que se corresponde con una mejora lineal con respecto al programa secuencial en función de la cantidad de cores (a N cores le corresponde una mejora de N veces el rendimiento del programa secuencuencial).

Podemos ver que ambas versiones paralelizadas se comportan de forma muy similar mientras la cantidad de hilos no supere a la cantidad de cores. Sin embargo, se aprecia una fuerte caída en el rendimiento de la versión estática al superar este límite. Al lanzar tantos hilos, al menos 2 se van a ejecutar sobre el mismo core (teorema del palomar) y por ende estos se van a retrasar con respecto a los demás. Esto no ocurre de una forma tan pronunciada con la versión fraccionada, ya que el trabajo de distribuye en función de los hilos que vayan terminando, lo que evita que un thread retrase a todo el programa.

| Cantidad de hilos |  Estatico | Fraccionado |
| ----------------- |:---------:| -----------:|
| 2                 | 3370.145  | 3558.017    |
| 4                 | 6200.341  | 6554.006    |
| 6                 | 8824.453  | 9342.395    |
| 8                 | 11760.708 | 12422.951   |
| 10                | 14678.632 | 15541.019   |
| 12                | 17646.347 | 18631.469   |
| 14                | 20408.276 | 21708.007   |
| 16                | 23314.020 | 24926.190   |
| 18                | 26043.183 | 27976.720   |
| 20                | 29385.516 | 30985.961   |
| 22                | 32124.560 | 33938.779   |
| 24                | 35276.739 | 36837.365   |
| 26                | 38168.980 | 39793.275   |
| 28                | 39098.399 | 43215.317   |
| 30                | 23411.808 | 40203.769   |
| 32                | 23951.047 | 40409.491   |

También se probaron distintos tipos de scheduling sobre nuestra versión fraccionada para ver si efectivamente se produce alguna mejora.

![Comparacion en funcion del scheduler con 28 hilos](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/schedule_1.png "Comparacion en funcion del scheduler con 28 hilos")

En esta gráfica se muestran los resultados para 28 hilos (igual que el número de cores). En principio parece que todas las formas de scheduling dan resultados similares, aunque parece haber un leve decremento del rendimiento con `static`.

![Comparacion en funcion del scheduler con 30 hilos](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/schedule_2.png "Comparacion en funcion del scheduler con 30 hilos")

Ahora bien, si aumentamos el número de hilos a 30 (por encima del número de cores), podemos ver que el scheduler `dynamic` brindó los mejores resultados. Por lo tanto considero que sería adecuado utilizar este scheduler para tratar de minimizar el ruido que introducen los demás procesos que se ejecutan dentro de la misma máquina.

| Scheduler |  28 hilos | 30 hilos  |
| --------- |:---------:| ---------:|
| guided    | 40529.116 | 33620.834 |
| static    | 39034.809 | 23372.563 |
| dynamic   | 40848.486 | 38519.951 |

Finalmente decidimos analizar el speedup y la eficiencia de la versión fraccionada, que fue la que brindó los mejores resultados.

![Speed Up](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/speed_up.png "Speed Up")

En esta gráfica se muestra la función identidad (que podemos considerar como un límite teórico) y el speed up obtenido mediante las modificaciones anteriormente discutidas. 

![Eficiencia](https://raw.githubusercontent.com/barufa/tiny_mc/lab3/imagenes/eficiencia.png "Eficiencia")

Podemos ver con la ayuda de ambos gráficos que la eficiencia obtenida es cercana al 90%.

## Conclusiones

Si bien hay lugar para mejoras, considero que se obtuvieron muy buenos resultados mediante la inserción de unas pocas líneas de código de OpenMP. 

Mediante las mejoras descritas en este informe, se incrementó la cantidad de fotones procesados por un factor de 24 utilizando los 28 cores disponibles en Jupiterace. En términos de la unidad de medida seleccionada, partiendo de 1724 fotones procesados se llegó a 43241 fotones procesados por milisegundo.

