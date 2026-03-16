CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data

TARGET = $(BUILD_DIR)/speck
SRC = $(SRC_DIR)/speck.c

N_TRAIN = 10000
N_TEST = 2500
POOL = 64
SEED = 42
ROUNDS = 4 5 6 7 8

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

run-tests: $(TARGET) | $(DATA_DIR)
	$(TARGET) test
	$(TARGET) avalanche 5
	$(TARGET) avalanche 8
	$(TARGET) diffstat 5

generate-data: $(TARGET) | $(DATA_DIR)
	@for r in $(ROUNDS); do \
		echo ">>> Generating nr=$$r ..."; \
		$(TARGET) split $(N_TRAIN) $(N_TEST) $$r \
			$(DATA_DIR)/train_nr_$$r.csv \
			$(DATA_DIR)/test_nr_$$r.csv \
			--pool-size $(POOL) --seed $(SEED); \
	done

test: run-tests
datafiles: generate-data

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(DATA_DIR)/*.csv
	rm -f data_nb*.csv

rebuild: clean all

.PHONY: all test run-tests datafiles generate-data clean rebuild