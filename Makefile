# Compilers
CC = gcc

# Flags
CFLAGS = -std=c11 -Wall -Wextra
TINY_LDFLAGS = -lm
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = tiny_mc cg_mc

# Files
C_SOURCES = wtime.c photon.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

tiny_mc: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TINY_LDFLAGS)

cg_mc: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

