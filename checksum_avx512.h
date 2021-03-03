#include <immintrin.h>
#include <stdint.h>

/* Horizontally reduce all 32b elements in 512b vector */
inline uint16_t _mm512i_flatten_epu16(__m512i vec)
{
    __m512i zero = _mm512_setzero_si512();
    __m512i lo_512 = _mm512_unpacklo_epi16(vec, zero);
    __m512i hi_512 = _mm512_unpackhi_epi16(vec, zero);
    uint32_t result = 0;

    /* TODO: Check whether the older style is faster than the mystery sequence this generates. */
    result = _mm512_reduce_add_epi32(lo_512) + _mm512_reduce_add_epi32(hi_512);
    
    while(result >> 16){
        result = (result & 0xFFFF) + (result >> 16);
    }
    
    return result;
}


inline uint16_t csum_avx512(uint8_t* data, size_t len)
{
    uint8_t* p = data;
    __m512i sum = _mm512_setzero_si512();
    __m512i carry = _mm512_setzero_si512();
    __m512i zero = _mm512_setzero_si512();
    __m512i one = _mm512_set1_epi16(1);

    /* Checksum 512b of data if possible */
    while(len >= 64){
        /* Load next 512b chunk of data */
        __m512i chunk_512 = _mm512_load_si512((__m512i*)p);

        /* Add to running sum */
        sum = _mm512_add_epi16(sum, chunk_512);

        /* Check for rollover */
        __mmask32 roll_mask = _mm512_cmplt_epu16_mask(sum, chunk_512);
        
        /* Add to carry vector */
        carry = _mm512_mask_adds_epu16(carry, roll_mask, zero, one);

        p += 64;
        len -= 64;
    }

    /* Use scalar implementation for last < 64B */
    uint32_t carry_scalar = _mm512i_flatten_epu16(carry);
    uint32_t csum = _mm512i_flatten_epu16(sum);
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
