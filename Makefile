.SILENT:

# Compilers
CC = gcc-10
CXX = g++-10
NVCC = nvcc

# Flags
OPT_FLAGS = -O3 -ffast-math -march=native -fopenmp -lpthread -floop-nest-optimize
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -Wno-unknown-pragmas -flto -DRAND7 -DVERBOSE $(OPT_FLAGS)
LDFLAGS = -lm

CU_FLAGS_XCOMP = $(LDFLAGS) $(OPT_FLAGS) `pkg-config --libs --cflags cuda` -DVERBOSE
CU_FLAGS = -O3 --use_fast_math -arch=sm_70 --compiler-options "$(CU_FLAGS_XCOMP)"

# Files
C_OBJS_CPU = wtime.h wtime.c params.h tiny_mc_cpu.c

# Rules
all: cpu gpu both

cpu:
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_CPU) -DM256 -o tiny_mc_cpu

gpu:
	$(NVCC) $(CU_FLAGS) tiny_mc_gpu.cu -o tiny_mc_gpu

both:
	$(CXX) $(CU_FLAGS_XCOMP) -c tiny_mc_both.cpp -o tiny_mc_both.o
	nvcc $(CU_FLAGS) tiny_mc_kernel.cu tiny_mc_both.o -o tiny_mc_both

clean:
	rm -f *.o tiny_mc_cpu tiny_mc_gpu tiny_mc_both *.gch
