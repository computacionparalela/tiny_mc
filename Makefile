# Compilers
CC = gcc-10

# Flags
OPT_FLAGS = -O2 -ffast-math -fexcess-precision=fast -funroll-all-loops
CFLAGS = -std=c11 -Wall -Wextra -Wshadow -DRAND3 -march=native $(OPT_FLAGS)
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

clean:
	rm -f $(TARGET) *.o

