#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

void dump_m256i(__m256i vec)
{
    printf("m256: ");
    printf("%d ",_mm256_extract_epi32(vec, 0));
    printf("%d ",_mm256_extract_epi32(vec, 1));
    printf("%d ",_mm256_extract_epi32(vec, 2));
    printf("%d ",_mm256_extract_epi32(vec, 3));
    printf("%d ",_mm256_extract_epi32(vec, 4));
    printf("%d ",_mm256_extract_epi32(vec, 5));
    printf("%d ",_mm256_extract_epi32(vec, 6));
    printf("%d ",_mm256_extract_epi32(vec, 7));
    printf("\n");
}

void dump_hex_m256i(__m256i vec)
{
    printf("m256: ");
    printf("%08x ",_mm256_extract_epi32(vec, 0));
    printf("%08x ",_mm256_extract_epi32(vec, 1));
    printf("%08x ",_mm256_extract_epi32(vec, 2));
    printf("%08x ",_mm256_extract_epi32(vec, 3));
    printf("%08x ",_mm256_extract_epi32(vec, 4));
    printf("%08x ",_mm256_extract_epi32(vec, 5));
    printf("%08x ",_mm256_extract_epi32(vec, 6));
    printf("%08x ",_mm256_extract_epi32(vec, 7));
    printf("\n");
}

void dump_m256i_16(__m256i vec)
{
    printf("m256: ");
    printf("%d ",_mm256_extract_epi16(vec, 0));
    printf("%d ",_mm256_extract_epi16(vec, 1));
    printf("%d ",_mm256_extract_epi16(vec, 2));
    printf("%d ",_mm256_extract_epi16(vec, 3));
    printf("%d ",_mm256_extract_epi16(vec, 4));
    printf("%d ",_mm256_extract_epi16(vec, 5));
    printf("%d ",_mm256_extract_epi16(vec, 6));
    printf("%d ",_mm256_extract_epi16(vec, 7));
    printf("%d ",_mm256_extract_epi16(vec, 8));
    printf("%d ",_mm256_extract_epi16(vec, 9));
    printf("%d ",_mm256_extract_epi16(vec, 10));
    printf("%d ",_mm256_extract_epi16(vec, 11));
    printf("%d ",_mm256_extract_epi16(vec, 12));
    printf("%d ",_mm256_extract_epi16(vec, 13));
    printf("%d ",_mm256_extract_epi16(vec, 14));
    printf("%d ",_mm256_extract_epi16(vec, 15));
    printf("\n");
}

void dump_hex_m256i_16(__m256i vec)
{
    printf("m256: ");
    printf("%04x ",_mm256_extract_epi16(vec, 0));
    printf("%04x ",_mm256_extract_epi16(vec, 1));
    printf("%04x ",_mm256_extract_epi16(vec, 2));
    printf("%04x ",_mm256_extract_epi16(vec, 3));
    printf("%04x ",_mm256_extract_epi16(vec, 4));
    printf("%04x ",_mm256_extract_epi16(vec, 5));
    printf("%04x ",_mm256_extract_epi16(vec, 6));
    printf("%04x ",_mm256_extract_epi16(vec, 7));
    printf("%04x ",_mm256_extract_epi16(vec, 8));
    printf("%04x ",_mm256_extract_epi16(vec, 9));
    printf("%04x ",_mm256_extract_epi16(vec, 10));
    printf("%04x ",_mm256_extract_epi16(vec, 11));
    printf("%04x ",_mm256_extract_epi16(vec, 12));
    printf("%04x ",_mm256_extract_epi16(vec, 13));
    printf("%04x ",_mm256_extract_epi16(vec, 14));
    printf("%04x ",_mm256_extract_epi16(vec, 15));
    printf("\n");
}

/* Horizontally add all 16b elements in 256b vector */
inline uint16_t _mm256i_flatten_epi16(__m256i vec)
{
    __m256i zero = _mm256_setzero_si256();
    __m256i lo_256 = _mm256_unpacklo_epi16(vec, zero);
    __m256i hi_256 = _mm256_unpackhi_epi16(vec, zero);

    /* 16x16b is spread across two 256b vectors. */
    /* Each element contains 4 leading zeros then 16b of actual data. */
    __m256i sum_256 = _mm256_add_epi32(lo_256, hi_256);

    /* Add the high bits in sum256 to the low bits in sum256. */
    __m128i hi_128 = _mm256_extracti128_si256(sum_256, 1);

    /* 64b of valid data spread across 128b as before. */
    __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(sum_256), hi_128);

    /* Shift the upper 64b of the first 128b lane down. */
    __m128i hi_64 = _mm_srli_si128(sum_128, 8);
    sum_128 = _mm_add_epi32(sum_128, hi_64);

    /* 32b of valid data in the lower 64b of 128b vector. */
    __m128i hi_32 = _mm_srli_si128(sum_128, 4);
    sum_128 = _mm_add_epi32(sum_128, hi_32);

    /* Scalar time! */
    uint32_t result = _mm_extract_epi32(sum_128, 0);

    while(result >> 16){
        result = (result & 0xFFFF) + (result >> 16);
    }
    
    return result;
}

uint16_t csum_avx2_test(uint8_t* data, size_t len)
{
    size_t i = 0;
    uint32_t carry_scalar = 0;
    uint8_t* p = data;
    __m256i sum = _mm256_setzero_si256();
    __m256i carry = _mm256_setzero_si256();
    __m256i carry_256 = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi16(1);

    /* Checksum 256b of data if possible. */
    while(len >= 32){
        /* Load next 256b chunk of data */
        __m256i chunk_256 = _mm256_load_si256((__m256i*)p);
        //__m256i chunk_256 = _mm256_lddqu_si256((const __m256i*)p);

        /* Add to running sum */
        __m256i _sum = _mm256_add_epi16(sum, chunk_256);

        /* Add with saturation. Any words which overflowed as a result of the previous addition are set to 0xffff */
        carry = _mm256_adds_epu16(sum, chunk_256);

        /* Any words which are detected as overflow are set to != 0. */
        carry = _mm256_xor_si256(carry, _sum);
       
        /* Set all of the words which overflowed to 1. */
        carry = _mm256_min_epu16(carry, one);

        /* Add to the total carry vector */
        carry_256 = _mm256_add_epi16(carry_256, carry);

        sum = _sum;
        p += 32; 
        len -= 32;
    }

    /* Use scalar implementation for last < 32B */    
    carry_scalar = _mm256i_flatten_epi16(carry_256);
    uint32_t csum = _mm256i_flatten_epi16(sum);
    csum += carry_scalar;

    while(len >= 2){
        csum += *(uint16_t*)p;
        p += 2;
        len -= 2;
    }

    if(len == 1){
        csum += *p;
    }

    while(csum >> 16){
        csum = (csum & 0xffff) + (csum >> 16);
    }
    
    return (uint16_t)~csum;
}


/*

 512:
load 512

add
cmplt - produces mmask
_mm512_mask_adds_epu16 have a 'ones' reg and a 'zero' reg. add from 'ones' reg according to the mask.

add mask to carry?


xor (old, new)


*/
