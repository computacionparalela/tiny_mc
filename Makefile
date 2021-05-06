# Compilers
CC = gcc-11

#Banderas para vectorizacion:
##-fopt-info-vec: para ver si pudo vectorizar.
##-fopt-info-vec-missed: para ver que NO pudo vectorizar.
##-ftree-vectorize

# Flags
OPT_FLAGS = -Ofast -funroll-loops -fpeel-loops -floop-nest-optimize -march=native -fopt-info-vec -fopt-info-vec-missed
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -DRAND7 -DVERBOSE $(OPT_FLAGS)
LDFLAGS = -lm

# Binary file
TARGET = tiny_mc

# Files
C_SOURCES = tiny_mc.c wtime.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

# Rules
all: $(TARGET)

$(TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./ispc/ispc --opt=fast-math --opt=fast-masked-vload --opt=force-aligned-memory --target-os=linux -O3 --math-lib=fast --addressing=64 --target=avx2-i32x16 --pic ispc/tiny_mc.ispc -o ispc/tiny_mc_ispc.o
	g++-11 ispc/tiny_mc_ispc.o ispc/tiny_mc_main.cpp ispc/tiny_mc_ispc.h wtime.c wtime.h params.h $(OPT_FLAGS) -DVERBOSE -o "ispc/$(TARGET)"

clean_gcda:
	rm -f *.gcda

clean:
	rm -f $(TARGET) *.o "ispc/$(TARGET)"
