.SILENT:

# Compilers
CC = gcc-10 #clang-11 gcc-10 icc
CXX = g++-10


ADD_FLAGS = 

#Banderas para vectorizacion:
##-fopt-info-vec: para ver si pudo vectorizar.
##-fopt-info-vec-missed: para ver que NO pudo vectorizar.
##-ftree-vectorize

# Flags
OPT_FLAGS = -O3 -ffast-math -march=native -fopenmp -lpthread#-floop-nest-optimize
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -Wno-unknown-pragmas -DRAND7 -DVERBOSE -DSEED=223 $(OPT_FLAGS) $(ADD_FLAGS)
LDFLAGS = -lm

# Binary file
TARGET = tiny_mc

# Files
C_OBJS_256 =  wtime.h wtime.c params.h tiny_mc_m256.c
C_OBJS_OMP =  wtime.h wtime.c params.h tiny_mc_omp.c
C_OBJS_256_OMP =  wtime.h wtime.c params.h tiny_mc_m256_omp.c

# Rules
all: m256 omp m256_omp

m256:
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_256) -DM256
	@mv a.out tiny_mc_m256

m256_omp:
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_256_OMP) -DM256
	@mv a.out tiny_mc_m256_omp

omp: 
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_OMP) -DM256
	@mv a.out tiny_mc_omp

clean:
	rm -f $(TARGET) *.o "ispc/$(TARGET)" tiny_mc_m256_omp tiny_mc_m256 tiny_mc_omp tiny_mc_m128 tiny_mc_ispc *.gch ispc/*.o
