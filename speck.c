#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define ROUNDS_FULL 22
#define ALPHA 7
#define BETA 2
#define MASK16 0xFFFF
#define ROR16(x, r) (((x) >> (r)) | ((x) << (16 - (r)))) // cyclic rotation right
#define ROL16(x, r) (((x) << (r)) | ((x) >> (16 - (r)))) // cyclic rotation left
#define DIFF_IN_L 0x0040 // XOR difference left
#define DIFF_IN_R 0x0000 // XOR difference right
#define MAX_POOL_SIZE 1024 // maximum number of pre-generated master keys in pool
#define MAX_TRACK 1024 // maximum number of distinct output differences tracked

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef enum { 
    KP_PER_SAMPLE = 0, // generate a fresh random key for each pair
    KP_FIXED = 1, // use one fixed key for the entire dataset
    KP_POOL = 2, // draw keys randomly from a predefined key pool
    KP_SPLIT = 3 // separate key pools for training and testing
} KeyPolicy;

typedef struct { 
    KeyPolicy policy; 
    int pool_size; 
    u16 fixed_key[4]; 
    u16 pool[MAX_POOL_SIZE][4]; 
} KeyInfo;

typedef struct { 
    u16 c0l, c0r, // first ciphertext produced from plaintext P0
    c1l, c1r; // second ciphertext produced from related plaintext P1
    int label; // 1 = pair follows chosen differential, 0 = random pair
} CipherPair;

void speck_key_schedule(const u16 key[4], u16 round_keys[ROUNDS_FULL]) {
    u16 ks = key[3]; 
    u16 l[3]={key[2],key[1],key[0]}; 
    round_keys[0]= ks;
    for (int i=0; i < ROUNDS_FULL - 1; i++) { 
        int index = i%3; 
        l[index] = (u16)((ROR16(l[index], ALPHA) + ks)^(u16)i); 
        ks = (u16)(ROL16(ks, BETA)^l[index]); 
        round_keys[i+1] = ks; 
    }
}

void speck_encrypt_nb(u16 *x, u16 *y, const u16 round_keys[ROUNDS_FULL], int nb) {
    for (int i=0; i < nb; i++) { // Speck ARX round
        *x = (u16)((ROR16(*x, ALPHA) + *y)^round_keys[i]); 
        *y = (u16)(ROL16(*y, BETA)^*x); 
    }
}

void speck_decrypt_nb(u16 *x, u16 *y, const u16 round_keys[ROUNDS_FULL], int nb) {
    for (int i = nb-1; i >= 0; i--) { 
        *y = (u16)(ROR16(*y^*x, BETA)); 
        *x = (u16)(ROL16((u16)((*x^round_keys[i]) - *y), ALPHA)); 
    }
}

static inline u16 rand16(void) { return (u16)(rand()&MASK16); } // generate a random 16-bit value

// assign a random 16-bit value to the current key word
void rand_key(u16 key[4]) {
    for (int i=0; i<4; i++) key[i] = rand16(); 
}

static void fill_pool(u16 pool[][4], int size, unsigned seed) {
    unsigned s = seed;
    for (int i=0; i<size; i++) { // generate each key in the pool
        for (int j=0; j<4; j++) { 
            s = s * 1664525u + 1013904223u; // // LCG update: s = a*s + c
            pool[i][j] = (u16)(s >> 16); // use high 16 bits as key word
        }
    }
}

/* In the ML-based literature, the construction of labeled ciphertext pairs and the choice of keys used to generate 
them are central experimental choices, because the distinguisher is trained on ciphertext pairs labeled “real” or “random,” 
and performance can depend on whether keys are fixed, varied, or separated across training and testing. */

KeyInfo create_key(KeyPolicy policy, int pool_size, unsigned seed_train, unsigned seed_test, const u16 *fixed_key) {
    KeyInfo box; // local container
    (void)seed_test; // unused in current implementation
    memset(&box, 0, sizeof(box)); 
    box.policy = policy; 
    box.pool_size = (pool_size > 0 && pool_size <= MAX_POOL_SIZE) ? pool_size : 64;
    switch (policy) {
        case KP_FIXED:
            if (fixed_key) memcpy(box.fixed_key, fixed_key, 4 * sizeof(u16)); else rand_key(box.fixed_key); // if the caller provides a key, it is copied into the structure. Otherwise a random 64-bit key is generated
            break;
        case KP_POOL:
            fill_pool(box.pool, box.pool_size, seed_train);
            break;
        case KP_SPLIT: // prevents training and testing from using the same key pool
            fill_pool(box.pool, box.pool_size, seed_train);
            break;
        default: // prevents training and testing from using the same key pool, so no stored key needed
            break;
    }
    return box;
}

static unsigned lcg_next(unsigned *state) {
    *state = *state * 1664525u + 1013904223u; // generating deterministic sequences
    return *state;
}

static void select_key(const KeyInfo *box, unsigned *lcg_state, u16 round_keys[ROUNDS_FULL]) {
    u16 key[4]; // buffer for the selected master key
    switch (box -> policy) {
        case KP_FIXED:
            memcpy(key, box -> fixed_key, sizeof(key)); // reuse, memcpy used for arrays
            break;
        case KP_POOL:
        case KP_SPLIT: {
            int index = (int)(lcg_next(lcg_state) % (unsigned) box -> pool_size); // pick one key
            memcpy(key, box -> pool[index], sizeof(key));
            break;
        }
        default:
            rand_key(key);
            break;
    }
    speck_key_schedule(key,round_keys); // master key -> round subkeys
}

CipherPair make_real_pair(int nb, const KeyInfo *box, unsigned *lcg_state) {
    u16 round_keys[ROUNDS_FULL]; 
    select_key(box, lcg_state, round_keys);
    u16 p0l = rand16(), p0r = rand16(); // generate a random plaintext
    u16 c0l = p0l, c0r = p0r;
    u16 c1l = p0l^DIFF_IN_L, c1r = p0r^DIFF_IN_R; // apply xor difference
    speck_encrypt_nb(&c0l, &c0r, round_keys, nb); 
    speck_encrypt_nb(&c1l, &c1r, round_keys, nb);
    CipherPair cp = {c0l, c0r, c1l, c1r, 1}; // 1 = real pair
    return cp;
}

CipherPair make_random_pair(int nb, const KeyInfo *box, unsigned *lcg_state) { // produces pairs that do not enforce any relation between the plaintexts
    u16 round_keys[ROUNDS_FULL]; select_key(box,lcg_state,round_keys);
    u16 c0l = rand16(), c0r = rand16(), c1l = rand16(), c1r = rand16(); // independent random plaintext blocks
    speck_encrypt_nb(&c0l, &c0r, round_keys, nb); speck_encrypt_nb(&c1l, &c1r, round_keys, nb);
    CipherPair cp = {c0l, c0r, c1l, c1r, 0}; // 0 = random pair
    return cp;
}

int generate_dataset(const char *filename, long n_pairs, int nb, const KeyInfo *box, unsigned seed) {
    FILE *f = fopen(filename, "w"); 
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return -1; }
    srand(seed);
    unsigned lcg_state = seed; 
    const char *policy_names[] = {"per_sample_key", "fixed_key", "key_pool", "split_keyset"};
    // make a csv file layout
    fprintf(f, "# key_policy=%s rounds=%d n_pairs=%ld seed=%u\n", policy_names[box -> policy], nb, n_pairs, seed);
    fprintf(f, "c0l,c0r,c1l,c1r,delta_left,delta_right,v0,v1,label\n");
    for (long i=0; i<n_pairs; i++) {
        CipherPair cp = (i%2 == 0) ? make_real_pair(nb, box, &lcg_state) : make_random_pair(nb, box, &lcg_state);
        u16 delta_l = cp.c0l^cp.c1l; 
        u16 delta_r = cp.c0r^cp.c1r; 
        u16 v0 = cp.c0l^cp.c0r; 
        u16 v1 = cp.c1l^cp.c1r;
        fprintf(f, "%u,%u,%u,%u,%u,%u,%u,%u,%d\n", cp.c0l, cp.c0r, cp.c1l, cp.c1r, delta_l, delta_r, v0, v1, cp.label);
    }
    fclose(f); printf("Generated %ld pairs (nb = %d, policy = %s, seed = %u) -> %s\n", n_pairs, nb, policy_names[box -> policy], seed, filename);
    return 0;
}

// avalanche effect: a very small modification in the plaintext should produce a large, hard-to-predict modification in the ciphertext
// if I flip exactly one bit of the plaintext, how many ciphertext bits change on average after nb rounds?
void avalanche_test(int nb, int n_samples) {
    printf("\nAvalanche Test (%d rounds, %d samples)\n", nb, n_samples);
    double total_change = 0.0; // sum of all observed bit changes
    u16 key[4];
    u16 round_keys[ROUNDS_FULL];
    for (int s=0; s < n_samples; s++) {
        rand_key(key); 
        speck_key_schedule(key, round_keys);
        u16 pl = rand16(), 
        pr = rand16(); 
        u16 cl = pl,
        cr = pr;
        speck_encrypt_nb(&cl, &cr, round_keys, nb);
        for (int bit = 0; bit < 32; bit++) {
            u16 pl2 = pl,
            pr2 = pr; 
            if (bit < 16) pl2 ^= (1 << bit); else pr2 ^= (1 << (bit - 16)); // flip by one with XOR
            u16 cl2 = pl2,
            cr2 = pr2; 
            speck_encrypt_nb(&cl2, &cr2, round_keys, nb);
            u32 diff = ((u32)(cl^cl2) << 16) | (cr ^ cr2); 
            total_change += __builtin_popcount(diff); // Hamming weight of the output difference, counts 1's in code and adds them 
        }
    }
    double avg = total_change / (n_samples * 32);
    printf("Average bits changed per 1-bit flip: %.2f / 32 (%.1f%%)\n", avg, avg / 32.0 * 100.0);
    printf("Ideal: 16.0 bits (50.0%%)\n");
}

// if i impose one fixed input xor-difference, which output xor-differences appear after nb rounds, and how often?
// it fixes one input xor-difference, encrypts many related plaintext pairs, and estimates which output xor-differences the reduced-round cipher prefers
void compute_diff_distribution(int nb, int n_samples) {
    printf("\nDifferential Distribution (%d rounds)\n", nb);
    u16 key[4],
    round_keys[ROUNDS_FULL]; 
    int n_unique = 0; // how many different output differences
    u32 seen_diff[MAX_TRACK]; // arrays that store observed differences and counts
    int seen_cnt[MAX_TRACK];
    memset(seen_diff, 0xFF, sizeof(seen_diff)); 
    memset(seen_cnt, 0 , sizeof(seen_cnt));
    for (int s=0; s < n_samples; s++) {
        rand_key(key); 
        speck_key_schedule(key, round_keys);
        u16 pl = rand16(), pr = rand16();
        u16 p1l = pl^DIFF_IN_L, p1r = pr^DIFF_IN_R; 
        u16 cl = pl, cr = pr, c1l = p1l, c1r=p1r;
        speck_encrypt_nb(&cl, &cr, round_keys, nb);
        speck_encrypt_nb(&c1l, &c1r, round_keys, nb);
        u32 out_diff = ((u32)(cl^c1l) << 16) | (cr^c1r); // observed output difference
        for (int i=0; i < MAX_TRACK; i++) { // searches already seen differences, if more than MAX_TRACK unique differences appear, the code has no room to store new one
            if (seen_diff[i] == out_diff) { seen_cnt[i]++; break; }
            if (seen_diff[i] == 0xFFFFFFFF) { 
                seen_diff[i] = out_diff; // new output difference discovered
                seen_cnt[i] = 1; 
                n_unique++; break;
            }
        }
    }
    printf("Top 5 output differences:\n"); 
    printf("%-12s %-8s %-8s\n", "Difference", "Count", "Prob");
    for (int rank = 0; rank < 5; rank++) {
        int best_i = 0;
        for (int i=1; i < MAX_TRACK; i++) if (seen_cnt[i]>seen_cnt[best_i]) best_i = i;
        if (seen_cnt[best_i] == 0) break;
        printf("0x%08X %-8d 2^%.1f\n", seen_diff[best_i], seen_cnt[best_i], log2((double)seen_cnt[best_i] / n_samples)); // proba
        seen_cnt[best_i] = 0;
    }
    printf("Unique differences observed: %d\n", n_unique);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s generate|split|test|avalanche|diffstat ...\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "generate") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s generate <n_pairs> <n_rounds> <output.csv> [--policy ...] [--pool-size K] [--seed N]\n", argv[0]);
            return 1;
        }

        long n = atol(argv[2]);
        long nb_l = atol(argv[3]);
        int nb = (int)nb_l;

        if (n <= 0) {
            fprintf(stderr, "n_pairs must be > 0\n");
            return 1;
        }

        if (nb < 1 || nb > ROUNDS_FULL) {
            fprintf(stderr, "Rounds must be in 1..%d\n", ROUNDS_FULL);
            return 1;
        }

        KeyPolicy policy = KP_PER_SAMPLE;
        int pool_sz = 64;
        unsigned seed = 12345;

        for (int a = 5; a < argc; a++) {
            if (strcmp(argv[a], "--policy") == 0) {
                if (a + 1 >= argc) {
                    fprintf(stderr, "Missing value for --policy\n");
                    return 1;
                }
                a++;

                if (strcmp(argv[a], "per_sample") == 0) policy = KP_PER_SAMPLE;
                else if (strcmp(argv[a], "fixed") == 0) policy = KP_FIXED;
                else if (strcmp(argv[a], "pool") == 0) policy = KP_POOL;
                else if (strcmp(argv[a], "split") == 0) policy = KP_SPLIT;
                else {
                    fprintf(stderr, "Unknown policy: %s\n", argv[a]);
                    return 1;
                }

            } else if (strcmp(argv[a], "--pool-size") == 0) {
                if (a + 1 >= argc) {
                    fprintf(stderr, "Missing value for --pool-size\n");
                    return 1;
                }
                pool_sz = atoi(argv[++a]);

            } else if (strcmp(argv[a], "--seed") == 0) {
                if (a + 1 >= argc) {
                    fprintf(stderr, "Missing value for --seed\n");
                    return 1;
                }
                seed = (unsigned)atol(argv[++a]);
            }
        }

        KeyInfo box = create_key(policy, pool_sz, seed, seed + 1, NULL);
        return generate_dataset(argv[4], n, nb, &box, seed);
    }

    if (strcmp(argv[1], "split") == 0) {
        if (argc < 7) {
            fprintf(stderr, "Usage: %s split <n_train> <n_test> <n_rounds> <train.csv> <test.csv> [--pool-size K] [--seed N]\n", argv[0]);
            return 1;
        }

        long n_train = atol(argv[2]);
        long n_test = atol(argv[3]);
        long nb_l = atol(argv[4]);
        int nb = (int)nb_l;

        if (n_train <= 0 || n_test <= 0) {
            fprintf(stderr, "n_train and n_test must be > 0\n");
            return 1;
        }

        if (nb < 1 || nb > ROUNDS_FULL) {
            fprintf(stderr, "Rounds must be in 1..%d\n", ROUNDS_FULL);
            return 1;
        }

        int pool_sz = 64;
        unsigned seed = 12345;

        for (int a = 7; a < argc; a++) {
            if (strcmp(argv[a], "--pool-size") == 0) {
                if (a + 1 >= argc) {
                    fprintf(stderr, "Missing value for --pool-size\n");
                    return 1;
                }
                pool_sz = atoi(argv[++a]);

            } else if (strcmp(argv[a], "--seed") == 0) {
                if (a + 1 >= argc) {
                    fprintf(stderr, "Missing value for --seed\n");
                    return 1;
                }
                seed = (unsigned)atol(argv[++a]);
            }
        }

        printf("Generating TRAIN dataset (seed = %u)...\n", seed);
        KeyInfo box_train = create_key(KP_SPLIT, pool_sz, seed, seed + 1, NULL);
        if (generate_dataset(argv[5], n_train, nb, &box_train, seed) != 0) return 1;

        printf("Generating TEST dataset (seed = %u)...\n", seed + 1);
        KeyInfo box_test = create_key(KP_SPLIT, pool_sz, seed + 1, seed + 2, NULL);
        if (generate_dataset(argv[6], n_test, nb, &box_test, seed + 1) != 0) return 1;

        printf("Done. Train and test use disjoint key pools.\n");
        return 0;
    }

    if (strcmp(argv[1], "test") == 0) {
        printf("SPECK32/64 Self-Test\n");

        u16 key[4] = {0x1918, 0x1110, 0x0908, 0x0100};
        u16 round_keys[ROUNDS_FULL];
        speck_key_schedule(key, round_keys);

        u16 pt_l = 0x6574, pt_r = 0x694c;
        u16 ct_l = pt_l, ct_r = pt_r;
        speck_encrypt_nb(&ct_l, &ct_r, round_keys, ROUNDS_FULL);

        printf("PT: 0x%04X 0x%04X\n", pt_l, pt_r);
        printf("CT: 0x%04X 0x%04X\n", ct_l, ct_r);
        printf("Expected: 0xA868 0x42F2\n");

        u16 dec_l = ct_l, dec_r = ct_r;
        speck_decrypt_nb(&dec_l, &dec_r, round_keys, ROUNDS_FULL);

        printf("Dec: 0x%04X 0x%04X %s\n",
               dec_l, dec_r,
               (dec_l == pt_l && dec_r == pt_r) ? "[OK]" : "[FAIL]");

        KeyInfo def_box = create_key(KP_PER_SAMPLE, 0, 0, 0, NULL);
        printf("\nGenerating mini-datasets (10000 pairs each)...\n");

        for (int nb = 5; nb <= 8; nb++) {
            char fname[64];
            snprintf(fname, sizeof(fname), "./data_nb%d.csv", nb);
            generate_dataset(fname, 10000, nb, &def_box, 99999u + (unsigned)nb);
        }

        return 0;
    }

    if (strcmp(argv[1], "avalanche") == 0) {
        int nb = (argc >= 3) ? atoi(argv[2]) : 8;

        if (nb < 1 || nb > ROUNDS_FULL) {
            fprintf(stderr, "Rounds must be in 1..%d\n", ROUNDS_FULL);
            return 1;
        }

        avalanche_test(nb, 1000);
        return 0;
    }

    if (strcmp(argv[1], "diffstat") == 0) {
        int nb = (argc >= 3) ? atoi(argv[2]) : 5;

        if (nb < 1 || nb > ROUNDS_FULL) {
            fprintf(stderr, "Rounds must be in 1..%d\n", ROUNDS_FULL);
            return 1;
        }

        compute_diff_distribution(nb, 5000);
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
