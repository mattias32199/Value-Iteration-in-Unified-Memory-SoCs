# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Grid world library objects
LIB_OBJS = gridworld.o verify.o

# Targets
all: test_gridworld

test_gridworld: test_gridworld.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Dependencies
test_gridworld.o: test_gridworld.c gridworld.h
gridworld.o: gridworld.c gridworld.h
verify.o: verify.c verify.h

clean:
	rm -f *.o test_gridworld

.PHONY: all clean
