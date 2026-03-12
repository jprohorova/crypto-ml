CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic
TARGET = speck

N_TRAIN = 10000
N_TEST = 2500
POOL = 64
SEED = 42
ROUNDS = 4 5 6 7 8

$(TARGET): speck.c
	$(CC) $(CFLAGS) -o $@ $< -lm

test: $(TARGET)
	./$(TARGET) test
	./$(TARGET) avalanche 5
	./$(TARGET) avalanche 8
	./$(TARGET) diffstat 5

data: $(TARGET)
	@for r in $(ROUNDS); do \
		echo ">>> Generating nr=$$r ..."; \
		./$(TARGET) split $(N_TRAIN) $(N_TEST) $$r \
			train_nr_$$r.csv test_nr_$$r.csv \
			--pool-size $(POOL) --seed $(SEED); \
	done

all: $(TARGET) test data

clean:
	rm -f $(TARGET) *.csv data_nr*.csv

.PHONY: test data all clean