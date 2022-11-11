#pragma once

#include "uint128.h"

__device__ __forceinline__ void singleBarrett(uint128_t& a, unsigned long long& q, unsigned long long& mu, unsigned long long& qbit)
{
    uint128_t rx;

    rx = a >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(a, rx);

    
    if (a.low >= q)
        a.low -= q;
    
    /*
    while (a.low >= q) {
        a.low -= q;
    }
    */
}

__device__ __forceinline__ void singleBarrett_64(unsigned long long& a, unsigned& q, unsigned& mu, unsigned& qbit)
{
    unsigned long long rx;

    rx = a >> (qbit - 2);

    rx = rx * mu;

    rx = rx >> (qbit + 2);

    rx = rx * q;

    a = a - rx;

    
    if (a >= q)
        a -= q;
    
    /*
    while (a >= q) {
        a -= q;
    }
    */
}