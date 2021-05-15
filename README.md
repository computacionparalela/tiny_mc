# Tiny Monte Carlo

El objetivo de este trabajo práctico es realizar optimizaciones sobre un programa que pretende mostrar de forma representativa el método Monte Carlo.
El código realiza simulaciones de propagación de luz, a partir de una fuente puntual, sobre un medio infinito con dispersión isotrópica.

- [Página en Wikipedia sobre el problema](https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport)
- [Código original](https://omlc.org/software/mc/) de [Scott Prahl](https://omlc.org/~prahl/)

# Laboratorio 2: Vectorización

En este segundo laboratorio nos enfocamos en la vectorización del código correspondiente al problema tiny_mc utilizando las herramientas ispc e intrinsics.

Las ejecuciones se realizaron sobre el servidor Jupiterace y sobre una notebook Asus vivobook S14(utilizada para la entrega del primer laboratorio).

Como métrica decidimos utilizar la cantidad de fotones procesados por segundo, que fue la utilizada durante la primera entrega.

## Modificaciones

Se han realizado dos modificaciones relevantes al código de la primer entrega:
* Decidimos utilizar el generador de números aleatorios xoshiro, ya que nos permite mantener un rendimiento similar al obtenido mediante el generador
  Lehmery posee un código que encontramos más simple para vectorizar mediante intrinsics.
* Utilizamos una aproximación de la función log para mejorar la performance general del programa.


## Vectorización mediante el compilador

Una primera aproximación al problema fue tratar de realizar modificaciones al código para ayudar a que el compilador lo vectoricé. Sin embargo, debido a 
la naturaleza del algoritmo esta tarea nos pareció compleja por lo que decidimos pasar directamente a la primera herramienta(ispc). 

Algunas banderas que encontramos útiles durante esta etapa son `-fopt-info-vec` y `-fopt-info-vec-missed` para obtener mayor información acerca de los 
problemas que tenía el compilador para vectorizar nuestro código.


## ISPC

La primera herramienta utilizada fue ispc. Fue relativamente simple reescribir la función `photon` aprovechando el modelo de paralelización presentado por 
este compilador. 

Esta herramienta nos presenta un modelo de paralelismo compuesto por lanes o “carriles”. Dentro de estos carriles podemos ejecutar distintas instancias de nuestra 
función photon al mismo tiempo, lo que nos permite simular varios fotones a la vez. La idea principal fue mantener todos estos carriles ocupados durante la mayor 
cantidad de tiempo posible. Para ello, cambiamos a la firma de la función photon para que tome una variable que indique la cantidad de fotones a simular. Al 
comenzar con la con la ejecución de la función cargamos a todos los carriles con la simulación de un fotón y, cuando alguna de las simulaciones termina simplemente 
iniciamos una nueva simulación dentro de ese carril. De esta forma, maximizamos el uso de los vectores para mejorar la eficiencia del programa.

Fue necesario la utilización de la función `atomic_add_local` de la librería estándar de ispc para evitar data races dentro del código.

## Intrinsics

En el caso de intrinsics la adaptación de la función photon fue más compleja. De forma similar a ispc, se trató de mantener a todos los carriles ocupados durante 
la mayor cantidad de tiempo posible siguiendo un esquema de ejecución similar, en el cual si una simulación termina se vuelva a iniciar la próxima dentro del mismo 
carril para evitar que este se desaproveche.

En este caso la escritura sobre los arreglos heat y heat2 se realizó de forma secuencial, copiando el valor de los vectores a dos arreglos de punto flotante. De esta 
forma, se eliminó la posibilidad de data races dentro del código.

## Benchmark

Una de las primeras mediciones que realizamos fue modificar el tamaño del problema, para ver como la cantidad de fotones afecta a la eficiencia general del programa.

![Comparacion de cantidad de fotones en maquina local](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/local_cantidad_fotones.png "Cantidad de fotones")

| Cantidad de fotones | Original    | ISPC     | Intrinsics 128 | Intrinsics 256 |
| ------------------- |:-----------:| --------:| --------------:| --------------:|
| 16384               | 815,742     | 1065,263 | 1199,765       | 1463,758       |
| 32768               | 979,514     | 1328,133 | 1416,63        | 1600,647       |
| 65536               | 1026,407    | 1540,078 | 1584,695       | 1942,96        |
| 131072              | 1032,93     | 1587,244 | 1628,891       | 2005,029       |
| 262144              | 1035,281    | 1631,488 | 1630,939       | 2022,555       |
| 524288              | 1041,045    | 1648,66  | 1639,804       | 2036,699       |
| 1048576             | 1061,585    | 1642,284 | 1634,971       | 2033,041       |
| 2097152             | 852,189     | 1343,108 | 1459,442       | 1957,137       |

![Comparacion de cantidad de fotones en Jupiterace](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/server_cantidad_fotones.png "Cantidad de fotones")

| Cantidad de fotones | Original    | ISPC     | Intrinsics 128 | Intrinsics 256 |
| ------------------- |:-----------:| --------:| --------------:| --------------:|
| 16384               | 605,528     | 782,766  | 825,963        | 1207,429       |
| 32768               | 611,367     | 847,596  | 880,697        | 1205,894       |
| 65536               | 641,139     | 901,991  | 851,948        | 1361,314       |
| 131072              | 730,764     | 1015,439 | 926,863        | 1353,790       |
| 262144              | 800,319     | 1168,219 | 1043,524       | 1539,867       |
| 524288              | 823,566     | 1243,558 | 1113,042       | 1638,288       |
| 1048576             | 837,790     | 1279,144 | 1147,236       | 1655,496       |
| 2097152             | 846,529     | 1284,745 | 1156,730       | 1714,668       |

Podemos ver que la cantidad de fotones procesados tiene una cierta influencia en la eficiencia del programa, llegando a un valor máximo cuando se superan los 
cien mil fotones. Dado que este fenómeno afecta a todas las versiones de la misma manera, de ahora en adelante utilizaremos el valor 131072 como la cantidad de fotones 
a procesar.

En la siguiente gráfica hacemos una comparación de las distintas versiones del programa `tiny_mc`. Además, para cada versión mostramos resultados utilizando distintos 
compiladores para determinar si hay alguna diferencia real en el ejecutable generado.

![Comparacion de las distintas versiones en maquina local](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/local_compiladores.png "Comparacion de las distintas versiones")

| Compiladores | Original    | ISPC     | Intrinsics 128 | Intrinsics 256 |
| ------------ |:-----------:| --------:| --------------:| --------------:|
| GCC          | 1132,407    | 1599,902 | 1629,829       | 2049,539       |
| CLANG        | 1083,929    | 1597,140 | 1739,319       | 2153,411       |
| ICC          | 708,901     | 1612,188 | 1486,677       | 1773,948       |

![Comparacion de las distintas versiones en maquina Jupiterace](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/server_compiladores.png "Comparacion de las distintas versiones")

| Compiladores | Original    | ISPC     | Intrinsics 128 | Intrinsics 256 |
| ------------ |:-----------:| --------:| --------------:| --------------:|
| GCC          | 751,016     | 1020,218 | 974,951        | 1294,357       |
| CLANG        | 716,246     | 978,152  | 923,689        | 1272,414       |
| ICC          | 524,083     | 1005,528 | 915,663        | 1058,426       |

Como última comparación decidimos realizar mediciones de la versión adaptada a ispc con distintos targets. Decidimos comparar vectores de sse4 y avx2 con sus distintas
configuraciones. Esto se consigue simplemente modificando el valor de la opción `--target=` del compilador ispc.

![Comparacion de distintos targets en ispc en maquina local](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/local_target.png "Comparacion de distintos targets en ispc")

| Target      | KPPS     |
| ----------- |:--------:|
| sse4-i16x8  | 1017,83  |
| sse4-i32x4  | 858,035  |
| sse4-i32x8  | 1045,75  |
| avx2-i16x16 | 1536,187 |
| avx2-i32x4  | 1222,55  |
| avx2-i32x8  | 1443,946 |
| avx2-i32x16 | 1674,938 |

![Comparacion de distintos targets en ispc en maquina Jupiterace](https://raw.githubusercontent.com/barufa/tiny_mc/lab2/data/server_target.png "Comparacion de distintos targets en ispc")

| Target      | KPPS     |
| ----------- |:--------:|
| sse4-i16x8  | 903,312  |
| sse4-i32x4  | 726,404  |
| sse4-i32x8  | 939,453  |
| avx2-i16x16 | 1133,242 |
| avx2-i32x4  | 862,087  |
| avx2-i32x8  | 1171,010 |
| avx2-i32x16 | 1277,371 |

Se observa que los mejores resultados se obtuvieron con `avx2`, y en particular con los valores `avx2-i32x16` y `avx2-i16x16`.

## Conclusiones

* ISPC es una herramienta que nos permite vectorizar código de una manera simple y rápida, aunque no suele brindar los mejores resultados.
* Intrinsics alcanzó los mejores rendimientos, pero adaptar algoritmos a esta herramienta puede ser complejo requerir de mucho trabajo.
* No se vio una diferencia muy marcada entre GCC y CLANG, ya que ambos compiladores brindaron resultados similares para todas las versiones.
