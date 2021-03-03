CFLAGS=-O3 -Wall
AVX2_CFLAGS=$(CFLAGS) -mavx -mavx2
AVX512_CFLAGS=$(CFLAGS) -mavx -mavx2 -mavx512bw

all: bin/bench_scalar bin/bench_exa bin/bench_avx2 bin/bench_avx512

bin/bench_scalar: bench_scalar.c checksum_scalar.h util.c
	mkdir -p bin
	gcc $(CFLAGS) bench_scalar.c util.c -o $@

bin/bench_exa: bench_exa.c checksum_exa.h util.c
	mkdir -p bin
	gcc $(CFLAGS) bench_exa.c util.c -o $@

bin/bench_avx2: bench_avx2.c checksum_avx2.h util.c
	mkdir -p bin
	gcc $(AVX2_CFLAGS) bench_avx2.c util.c -o $@

bin/bench_avx512: bench_avx512.c checksum_avx512.h util.c
	mkdir -p bin
	gcc $(AVX512_CFLAGS) bench_avx512.c util.c -o $@

clean:
	rm -rf bin/*
