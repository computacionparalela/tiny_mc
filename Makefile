# Compilers
CC = gcc-10

# Flags
OPT_FLAGS = -O0 -ffast-math -fno-signaling-nans -funroll-loops -fno-signed-zeros -fpeel-loops -freciprocal-math
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -DRAND7 -DVERBOSE -march=native $(OPT_FLAGS)
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

clean_gcda:
	rm -f *.gcda

clean:
	rm -f $(TARGET) *.o

