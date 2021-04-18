# Compilers
CC = gcc-10

# Flags
OPT_FLAGS = -Ofast -ftree-vectorize -ftree-loop-optimize -ftree-loop-vectorize -faligned-new -msse -msse2 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -flto
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -DRAND7 -DVERBOSE -march=native $(OPT_FLAGS) $(PHOTONS)
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

