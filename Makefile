# Makefile
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -std=c99
NVCCFLAGS = -O2
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build

OBJECTS = $(BUILD_DIR)/gridworld.o

# Build CPU targets by default, CUDA targets explicitly
all: test_gridworld cpu_vi

cuda: discrete_vi unified_vi unified_vi_prefetch

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# C compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# CPU targets
test_gridworld: $(BUILD_DIR)/test_gridworld.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

cpu_vi: $(BUILD_DIR)/cpu_vi.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CUDA targets
# nvcc compiles the .cu file and links against the C gridworld object
discrete_vi: $(SRC_DIR)/discrete_vi.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

unified_vi: $(SRC_DIR)/unified_vi.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

unified_vi_prefetch: $(SRC_DIR)/unified_vi.cu $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -DUSE_PREFETCH -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) test_gridworld cpu_vi discrete_vi unified_vi unified_vi_prefetch *.bin

.PHONY: all cuda clean
