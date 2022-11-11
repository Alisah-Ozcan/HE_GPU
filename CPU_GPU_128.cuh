#pragma once


#include "uint128.h"

#include <iostream>
#include <tuple>

int bitreverse(int index, int n_power) {

    int res_1 = 0;
    for (int i = 0; i < n_power; i++)
    {
        res_1 <<= 1;
        res_1 = (index & 1) | res_1;
        index >>= 1;
    }
    return res_1;
}


void multiply_128(uint64_t operand1, uint64_t operand2, unsigned long long* result128)
{

    auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
    auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
    operand1 >>= 32;
    operand2 >>= 32;

    auto middle1 = operand1 * operand2_coeff_right;
    auto middle2 = operand2 * operand1_coeff_right;

    auto left = operand1 * operand2;
    auto right = operand1_coeff_right * operand2_coeff_right;

    //carry operations
    auto temp1 = right >> 32;
    auto temp2 = middle2 & 0x00000000FFFFFFFF;

    middle1 = middle1 + temp1 + temp2;

    temp1 = middle1 >> 32;
    temp2 = middle2 >> 32;

    left = left + temp1 + temp2;

    result128[1] = static_cast<unsigned long long>(left);
    result128[0] = static_cast<unsigned long long>((middle1 << 32) | (right & 0x00000000FFFFFFFF));

}

void sub_128(unsigned long long* input1, unsigned long long* input2)
{
    unsigned long long in1_low = input1[0];
    unsigned long long in1_high = input1[1];

    unsigned long long in2_low = input2[0];
    unsigned long long in2_high = input2[1];

    if (in1_low < in2_low) {

        in2_low = 0xFFFFFFFFFFFFFFFF - in2_low;
        //in1_low = in1_low + in2_low;
        in1_low = in1_low + in2_low + 1;

        in1_high = in1_high - (in2_high + 1);

    }
    else {
        in1_low = in1_low - in2_low;
        in1_high = in1_high - in2_high;
    }

    input1[0] = in1_low;
    input1[1] = in1_high;

}

unsigned long long shiftr_128(unsigned long long* x, const unsigned shift)
{
    unsigned long long xlow = x[0];
    unsigned long long xhigh = x[1];

    xlow = xlow >> shift;
    xlow = (xhigh << (64 - shift)) | xlow;

    return xlow;

}

unsigned long long Barrett_128_ali(unsigned long long* a, unsigned long long q, unsigned long long mu, unsigned long long qbit)
{
    unsigned long long temp[2];
    temp[0] = a[0];
    temp[1] = a[1];

    unsigned long long rx = shiftr_128(temp, qbit - 2);

    multiply_128(rx, mu, temp);

    rx = shiftr_128(temp, qbit + 2);

    multiply_128(rx, q, temp);

    sub_128(a, temp);

    if (a[0] >= q)
        a[0] -= q;


    //return a[0];

    unsigned long long rslt = a[0];
    return rslt;
}



std::tuple<signed long long, signed long long, signed long long> Egcd(signed long long a, signed long long b)
{
    if (a == 0) {
        return std::make_tuple(b, 0, 1);
    }

    signed long long gcd, x, y;

    std::tie(gcd, x, y) = Egcd(b % a, a);

    return std::make_tuple(gcd, (y - (b / a) * x), x);
}

unsigned long long Mod_Inverse(signed long long a, signed long long b)
{
    signed long long gcd, x, y;
    std::tie(gcd, x, y) = Egcd(a, b);

    unsigned long long result = x + b;
    return result % b;
}

unsigned long long Mu_calculator(unsigned long long q, int q_bit) {


    uint128_t num;
    num.high = pow(2, (q_bit * 2 - 64));
    uint128_t result = num / q;

    return result.low;


}








__device__ unsigned long long modpow128(unsigned long long a, unsigned long long b, unsigned long long mod)
{
    unsigned long long res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        uint128_t t128 = host64x2(a, a);
        a = (t128 % mod).low;
        if (b & 1)
        {
            uint128_t r128 = host64x2(res, a);
            res = (r128 % mod).low;
        }

    }
    return res;
}

__device__ unsigned long long modinv128(unsigned long long a, unsigned long long q)
{
    unsigned long long ainv = modpow128(a, q - 2, q);
    return ainv;
}


__global__ void NTT_Table(unsigned long long* q_device, unsigned long long* Psi_root_device, unsigned long long* ForwardPsi_device, unsigned long long* InversePsi_device, unsigned long long* Double_InversePsi_device, int N, int size, long double n_power_device) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_host = q_device[int(idx / N)];
    unsigned long long psi_root_host = Psi_root_device[int(idx / N)];

    int counter_2 = idx % N;


    unsigned long long res = modpow128(psi_root_host, idx % N, q_host);

    unsigned long long res_1 = 0;

    for (int i = 0; i < n_power_device; i++)
    {
        res_1 <<= 1;
        res_1 = (counter_2 & 1) | res_1;
        counter_2 >>= 1;
    }

    ForwardPsi_device[res_1 + N * (int(idx / N))] = res;

    res = modinv128(res, q_host);

    InversePsi_device[res_1 + N * (int(idx / N))] = res;

    for (int lps = 0; lps < 2; lps++) {

        Double_InversePsi_device[(lps * N) + res_1 + 2 * N * (int(idx / N))] = res;
    }

}


__global__ void NTT_Table_plain(unsigned long long* plain_device, unsigned long long* plain_psi_device, unsigned long long* ForwardPsi_device, unsigned long long* InversePsi_device, int N, long double n_power_device) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_host = plain_device[0];
    unsigned long long psi_root_host = plain_psi_device[0];

    int counter_2 = idx % N;

    unsigned long long res = modpow128(psi_root_host, idx % N, q_host);

    unsigned long long res_1 = 0;

    for (int i = 0; i < n_power_device; i++)
    {
        res_1 <<= 1;
        res_1 = (counter_2 & 1) | res_1;
        counter_2 >>= 1;
    }

    ForwardPsi_device[res_1] = res;

    res = modinv128(res, q_host);

    InversePsi_device[res_1] = res;

}

__global__ void NTT_Table_BSK(unsigned long long* q_device, unsigned long long* Psi_root_device, unsigned long long* ForwardPsi_device, unsigned long long* InversePsi_device, int N, int size, long double n_power_device) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_host = q_device[int(idx / N)];
    unsigned long long psi_root_host = Psi_root_device[int(idx / N)];

    int counter_2 = idx % N;


    unsigned long long res = modpow128(psi_root_host, idx % N, q_host);

    unsigned long long res_1 = 0;

    for (int i = 0; i < n_power_device; i++)
    {
        res_1 <<= 1;
        res_1 = (counter_2 & 1) | res_1;
        counter_2 >>= 1;
    }

    ForwardPsi_device[res_1 + N * (int(idx / N))] = res;

    res = modinv128(res, q_host);

    InversePsi_device[res_1 + N * (int(idx / N))] = res;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

