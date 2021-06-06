.SILENT:

# Compilers
CC = gcc-10 #clang-11 gcc-10 icc
CXX = g++-10
NVCC = nvcc

ADD_FLAGS = 

#Banderas para vectorizacion:
##-fopt-info-vec: para ver si pudo vectorizar.
##-fopt-info-vec-missed: para ver que NO pudo vectorizar.
##-ftree-vectorize

# Flags
OPT_FLAGS = -O3 -ffast-math -march=native -fopenmp -lpthread -floop-nest-optimize -flto
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -Wno-unknown-pragmas -DRAND7 -DVERBOSE -DSEED=223 $(OPT_FLAGS) $(ADD_FLAGS)
LDFLAGS = -lm

CU_FLAGS_XCOMP = -Xcompiler=-I/usr/lib/gcc/x86_64-linux-gnu/5.5.0/include/ -Xcompiler=-fopenmp=libiomp5 -Xcompiler=-lm -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-Wno-unused-command-line-argument -Xcompiler=-DVERBOSE
CU_FLAGS = -ccbin clang-4.0 --use_fast_math -O3 $(CU_FLAGS_XCOMP)

# Binary file
TARGET = tiny_mc

# Files
C_OBJS_CPU =  wtime.h wtime.c params.h tiny_mc_cpu.c

# Rules
all: cpu gpu

cpu:
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_CPU) -DM256
	@mv a.out tiny_mc_cpu

gpu:
	$(NVCC) $(CU_FLAGS) tiny_mc_gpu.cu -o tiny_mc_gpu

clean:
	rm -f $(TARGET) *.o "ispc/$(TARGET)" tiny_mc_cpu tiny_mc_gpu *.gch ispc/*.o
