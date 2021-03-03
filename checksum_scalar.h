#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

uint16_t csum(uint8_t *data, size_t len)
{
    uint32_t sum = 0;
    uint16_t *data_u16 = (uint16_t*)data;

    while (len >= sizeof(uint16_t)){
        sum += *data_u16;
        len -= sizeof(uint16_t);
        data_u16++;
    }
    
    if (len == 1){
        sum += *(uint8_t*)data_u16;
    }

    while(sum >> 16){
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    return (uint16_t)~sum;
}
