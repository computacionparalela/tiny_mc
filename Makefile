# Compilers
CC = clang-11 #gcc-10
CPP = g++-10

#Banderas para vectorizacion:
##-fopt-info-vec: para ver si pudo vectorizar.
##-fopt-info-vec-missed: para ver que NO pudo vectorizar.
##-ftree-vectorize

# Flags
OPT_FLAGS = -Ofast -funroll-loops -march=native #-floop-nest-optimize
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -DRAND7 -DVERBOSE -DSEED=223 $(OPT_FLAGS)
LDFLAGS = -lm

# Binary file
TARGET = tiny_mc

# Files
C_SOURCES = tiny_mc.c wtime.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))
C_OBJS_256 =  wtime.h wtime.c params.h tiny_mc_m256.c
C_OBJS_128 =  wtime.h wtime.c params.h tiny_mc_m128.c

# Rules
all: $(TARGET) m128 m256

$(TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@./ispc/ispc -woff --opt=fast-math --opt=fast-masked-vload --opt=force-aligned-memory --target-os=linux -O3 --math-lib=fast --addressing=64 --target=avx2-i32x16 --pic ispc/tiny_mc.ispc -o ispc/tiny_mc_ispc.o
	$(CPP) ispc/tiny_mc_ispc.o ispc/tiny_mc_main.cpp ispc/tiny_mc_ispc.h wtime.c wtime.h params.h $(OPT_FLAGS) -DVERBOSE -o "ispc/$(TARGET)"
	@mv ispc/tiny_mc tiny_mc_ispc

m256: $(C_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_256) -DM256
	@mv a.out tiny_mc_m256

m128: $(C_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(C_OBJS_128) -DM256
	@mv a.out tiny_mc_m128

clean_gcda:
	rm -f *.gcda

clean:
	rm -f $(TARGET) *.o "ispc/$(TARGET)" tiny_mc_m256 tiny_mc_m128 tiny_mc_ispc *.gch ispc/*.o
