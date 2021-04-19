# Tiny Monte Carlo

El objetivo de este trabajo práctico es realizar optimizaciones sobre un programa que pretende mostrar de forma representativa el método Monte Carlo.
El código realiza simulaciones de propagación de luz, a partir de una fuente puntual, sobre un medio infinito con dispersión isotrópica.

- [Página en Wikipedia sobre el problema](https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport)
- [Código original](https://omlc.org/software/mc/) de [Scott Prahl](https://omlc.org/~prahl/)


## Características de la PC

Las ejecuciones se realizaron sobre una notebook Asus vivobook S14 con las siguiente prestaciones:


```

	Ubuntu 18.04.5 LTS

	Kernel 5.4.0-72-generic

	Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz x 8

	16 GB de RAM DDR4 2400 MT/s 64 bits band-width

	Compiladores: GCC 10.1.0 | CLANG 11.1.0 | ICC 2021.2.0

```

Sobre este equipo se ejecutó el programa Empirical Roofline Toolkit obteniendo los siguientes resultados:

![Empirical Roofline Toolkit](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image9.jpg "Grafico ERT con varios hilo")

Y al ejecutarlo para un solo hilo nos queda:

![Empirical Roofline Toolkit](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image7.jpg "Grafico ERT con un solo hilo")


## Métrica utilizada

Como métrica del problema se decidió utilizar la cantidad de fotones procesados por segundo, mayoritariamente representada de a miles. Esto se debe a que, en nuestra opinión, es una magnitud más fácil de comprender y visualizar.Siempre que se habla de mejoras en el programa, se estará buscando un incremento en la métrica elegida.


## Optimizaciones realizadas sobre el código

Luego de una primera evaluación del programa se decidió realizar pruebas sobre el
generador de números aleatorios, notando que la misma ocupaba gran parte del tiempo de
ejecución.

A partir de esto se procedió a evaluar una amplia cantidad de generadores de números
aleatorios y luego de diversas evaluaciones se concluyo que lo mas optimo era utilizar el
método de Lehmer. Dicho método fue elegido ya que no solo presenta uno de los dos
mejores rendimientos sino que también proviene de una fuente más confiable (Lehmer fue
quien introdujo los generadores congruenciales lineales que se usan hoy en día).

Tras realizar dichos cambios se observó un incremento del 50% en la cantidad de fotones
procesados por segundo, junto con una disminución al 30% en los tiempos de ejecución en
la porción de código correspondiente al mismo.

Por último se buscó optimizar el cálculo de la nueva dirección del fotón, cambiando el
método de rechazo por coordenadas polares. Sin embargo, este cambio requería utilizar
funciones matemáticas que ralentizaban el código produciendo una reducción en la cantidad
de fotones. Por ende no se siguió explorando esta posible optimización del código.


## Metodología de elección de banderas

Para explorar el espacio de banderas de optimizaciones decidimos utilizar un método
Wrapper de selección de variables, el cual es usado en machine learning y fue adaptado a
este problema.

En machine learning el método wrapper evalúa subconjuntos de variables para determinar
una combinación adecuada para el problema que se quiere resolver.

![Metodo de seleccion de variables Wrapper](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image3.png "Wrapper")

Dentro de nuestro problema, las variables son representadas por las distintas banderas del
compilador y el rendimiento está dado por el archivo binario. Para facilitar la comprensión
del mismo se proporciona el siguiente ejemplo:

![Espacio de busqueda](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image1.png "Espacio de busqueda")

En este caso se trabaja con un espacio de 4 banderas de compilación. El punto inicial es un
estado que cuenta con todas las banderas habilitadas y en cada paso se trata de quitar una
ellas, probando exhaustivamente cuál es el estado de mejor rendimiento. Este proceso se
repite hasta llegar al estado vacío.

Concretamente, en este gráfico se parte de un estado inicial con las banderas {A,B,C,D},
avanzando al estado {A,C,D}, luego al {A,C}, posteriormente {A} y finalmente se llega al
estado vacío. Cabe aclarar que no es posible ir del estado {A,C,D} al {B,D} o {B,C,D} puesto
que la bandera B ya ha sido descartada.

Si se quisiera explorar todo el espacio de banderas y sus posibles combinaciones sería
necesario realizar 2^n compilaciones, puesto que este problema crece exponencialmente a
medida aumenta el número de banderas. Por otro lado, el orden del método de selección
elegido es cuadrático respecto a la cantidad de banderas presentes, reduciendo
ampliamente los tiempos pertinentes a la exploración.


## Resultados en los distintos compiladores y cual es mejor

En el siguiente gráfico podemos ver los resultados obtenidos para las distintas
combinaciones de banderas de optimización a través del método presentado.

![Comparacion de compiladores](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image6.png "Compiladores")

| Compilador | NONE    | O0+     | O1+     | O2+     | O3+     | Ofast+  |
| ---------- |:-------:| -------:| -------:| -------:| -------:| -------:|
| GCC        | 299.525 | 312.613 | 541.420 | 783.162 | 805.886 | 812.322 |
| CLANG      | 262.130 | 265.107 | 368.600 | 798.277 | 810.487 | -       |
| ICC        | 492.891 | 200.372 | 632.288 | 494.507 | 497.35  | -       |

Las banderas obtenidas en cada caso fueron:

1. GCC:
* O0+: -ffast-math -funroll-loops -fpeel-loops -march=native
* O1+: -ffast-math -funroll-loops -fprefetch-loop-arrays -fpeel-loops -march=native
* O2+: -funroll-loops  -ffast-math -fprefetch-loop-arrays -fpeel-loops -flto -march=native
* O3+: -ffast-math -flto -funroll-loops -fpeel-loops -march=native
* Ofast+: -funroll-all-loops -flto -march=native

2. CLANG:
* O0+: -funroll-loops -flto -ffast-math -march=native
* O1+: -funroll-loops -flto -funsafe-math-optimizations -ffinite-math-only -freciprocal-math -fno-math-errno -march=native
* O2+:  -ffast-math -funroll-loops -flto - march=native
* O3+: -funroll-loops -flto -ffast-math -march=native

3. ICC:
* O0+: -qopenmp-offload -march=native -fast-transcendentals -parallel -fimf-precision=simple -qopt-prefetch -no-prec-div -no-prec-sqrt -qopenmp -fp-speculation
* O1+: -qopenmp-offload -march=native -fast-transcendentals -parallel -fimf-precision=simple -qopt-prefetch -no-prec-div -no-prec-sqrt -qopenmp
* O2+: -qopenmp-offload -march=native -fast-transcendentals
* O3+: -qopenmp-offload -march=native -parallel -fimf-precision=simple -qopt-prefetch -no-prec-div -no-prec-sqrt -fp-speculation

En todos los casos, None hace referencia a la compilación sin el agregado de ninguna bandera.

Podemos ver claramente que los mejores resultados fueron obtenidos con GCC y Clang, alcanzando los 800.000 fotones por segundo procesados, mientras que el compilador de Intel (ICC) solamente logra unos 630.000.

También se puede apreciar que al utilizar el compilador ICC con la bandera -O0 se produce una drástica reducción en el rendimiento, debido a que este compilador tiene habilitada la bandera -O2 por defecto. 

Por último, es posible notar que el mejor rendimiento usando ICC se alcanzó con la bandera -O1. Este resultado genera desconcierto y todavía no se ha conseguido esclarecer la causa del mismo. Una hipótesis vigente es la falta de conocimientos pertinentes a la hora de realizar las pruebas en dicho procesador. La otra es que la bandera -O2 de este compilador pone en funcionamiento ciertas optimizaciones que, quizás, en el procesador de trabajo actual  producen una caída en el rendimiento.


## Relación lineal entre fotones y tiempo

![Grafico tiempo y cantidad de fotones](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image5.png "Grafico tiempo y cantidad de fotones")

En este gráfico se puede ver que el orden del problema es lineal respecto al número de fotones. Esto es así ya que la memoria requerida por el programa permanece constante a lo largo del mismo y al aumentar la cantidad de fotones solo incrementa el procesamiento.


## Resultados de Perf

Se ejecutó con la versión original y la versión optimizada del programa con Perf para realizar una comparación entre ellas.

![Imagen Perf Original](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image4.png "Imagen Perf Original")

![Imagen Perf Optimizado](https://raw.githubusercontent.com/barufa/tiny_mc/main/Imagenes/image2.png "Imagen Perf Optimizado")

Podemos ver que la mayor diferencia entre las ejecuciones radica en la cantidad de instrucciones ejecutadas, la cual se reduce un 71,66%. Esta reducción se traduce en un decremento en los tiempos de ejecución en un 72,94% y un incremento cercano al 305,13% en el número de fotones procesados por segundo.


## Conclusiones

* De los experimentos realizados podemos concluir que los mejores resultados se obtuvieron con los compiladores GCC y CLANG.
* Una posible mejora al método de exploración sería utilizar evaluaciones estadísticas, como la prueba t de Student, para determinar a qué estado es óptimo avanzar, en lugar de simplemente realizar comparaciones entre los promedios de ejecución.
* Se probaron las optimizaciones guiadas por ejecución en GCC (-fprofile-generate y -fprofile-use), pero solo se encontraron mejoras en la cantidad de fotones procesada cercanas al 1%.
