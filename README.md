
# SPECK32


This project combines a reduced-round **SPECK32/64** implementation in C with a Python-based neural distinguisher. The C program generates labeled ciphertext-pair datasets, and the Python script trains a simple CNN to distinguish pairs produced from a chosen input differential from random pairs.

## Prerequisites

- GCC or Clang with C99 support
- Python 3.10+
- `pip`

## Setup and run

Run the steps in this order:

```bash
make
make test
make data
pip install -r requirements.txt
python src/distinguisher.py --rounds 4
```

## Build commands

### Compile the binary

```bash
make
```

This compiles `src/speck.c` into:

```text
build/speck
```

### Run basic tests

```bash
make test
```

This runs:

- the SPECK32/64 self-test
- avalanche checks for selected round counts
- a small differential-distribution experiment

### Generate datasets

```bash
make data
```

This creates CSV files inside `data/`:

- `data/train_nr_4.csv`, `data/test_nr_4.csv`
- `data/train_nr_5.csv`, `data/test_nr_5.csv`
- `data/train_nr_6.csv`, `data/test_nr_6.csv`
- `data/train_nr_7.csv`, `data/test_nr_7.csv`
- `data/train_nr_8.csv`, `data/test_nr_8.csv`

## Python distinguisher

The script `src/distinguisher.py` is now parameterized with `--rounds` and automatically loads:

```text
data/train_nr_<rounds>.csv
data/test_nr_<rounds>.csv
```

Available options:

```bash
python src/distinguisher.py --help
```

Main arguments:

- `--rounds`: required, selects the dataset pair
- `--data-dir`: dataset directory, default is `data`
- `--epochs`: training epochs, default is `10`
- `--batch-size`: training batch size, default is `256`

## Dataset format

Each CSV file starts with one metadata line followed by a header. The Python loader skips the metadata line with `comment="#"`.

Columns:

- `c0l`, `c0r`, `c1l`, `c1r`: two ciphertext blocks
- `delta_left`, `delta_right`: ciphertext XOR differences
- `v0`, `v1`: simple derived values
- `label`: `1` for differential pairs, `0` for random pairs

## Key policies

The C generator supports several key-selection policies:

- `per_sample_key`: a fresh random master key is generated for each pair
- `fixed_key`: one key is reused for the whole dataset
- `key_pool`: keys are sampled from a predefined pool
- `split_keyset`: train and test datasets are generated from disjoint key pools

The `split` command uses `split_keyset`, which is the safest default for ML experiments because it reduces train/test leakage through shared keys.

## Example generator commands

Generate one dataset:

```bash
./build/speck generate 10000 4 data/custom_nr_4.csv --policy pool --pool-size 64 --seed 42
```

Generate train/test datasets with separate key pools:

```bash
./build/speck split 10000 2500 4 data/train_nr_4.csv data/test_nr_4.csv --pool-size 64 --seed 42
```

## Notes

- The C code implements up to `22` SPECK32/64 rounds.
- The default differential is defined in `src/speck.c` through `DIFF_IN_L` and `DIFF_IN_R`.
- The current neural model is a compact 1D CNN over the bit representation of ciphertext pairs.

