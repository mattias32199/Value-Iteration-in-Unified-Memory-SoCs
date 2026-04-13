CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build

OBJECTS = $(BUILD_DIR)/gridworld.o

all: test_gridworld cpu_vi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

test_gridworld: $(BUILD_DIR)/test_gridworld.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

cpu_vi: $(BUILD_DIR)/cpu_vi.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) test_gridworld cpu_vi *.bin

.PHONY: all clean
