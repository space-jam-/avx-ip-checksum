#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "util.h"
#include "checksum_scalar.h"

/* Benchmark the performance of AVX2 IP checksum calculation. */
int main(int argc, char* argv[])
{
    int i, j;
    uint16_t check;
    uint16_t data_16[2048];
    char* data;

    if (argc < 2){
        fprintf(stderr, "Usage:\n %s sample_count\n", argv[0]);
        return 1;
    }

    int num_samples = atoi(argv[1]);
    int data_size = 1024;

    init_data(&data, num_samples * data_size);
    if (!data){
        fprintf(stderr, "ERROR: failed to allocate data: %s\n", strerror(errno));
        return 1;
    }

    timing_t *stats;
    stats = malloc(sizeof(timing_t) * num_samples);

    uint16_t *results;
    results = malloc(sizeof(uint16_t) * num_samples);

    for (i = 0; i < num_samples; i++)
    {
        timing_t start, end;
        rdtscll(start);
        results[i] = csum((uint8_t*)data + data_size * i, data_size);
        rdtscll(end);
        stats[i] = end - start;
    }

    timing_print(stats, num_samples, 0);
    return 0;
}
