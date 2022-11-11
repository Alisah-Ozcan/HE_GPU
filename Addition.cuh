#pragma once
// --------------------- //
// Author: Alisah Ozcan
// --------------------- //


#include "NTT.cuh"
#include "GPU_Context.cuh"

//Addition
__global__ void Addition(unsigned long long* Enc1, unsigned long long* Enc2, unsigned long long* results, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
    int N, int Q_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_ = q_device[int(idx / N) % Q_count];
    unsigned long long mu_ = mu_device[int(idx / N) % Q_count];
    unsigned long long qbit = q_bit_device[int(idx / N) % Q_count];
    unsigned long long term1 = Enc1[idx];
    unsigned long long term2 = Enc2[idx];
    uint128_t sum = (term1 + term2);
    //singleBarrett(sum, q_, mu_, qbit);
    sum -= q_ * (sum >= q_);
    results[idx] = sum.low;
    //results_device[idx] = sum % q_;

}

__global__ void Addition_Inplace(unsigned long long* Enc1, unsigned long long* Enc2, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
    int N, int Q_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_ = q_device[int(idx / N) % Q_count];
    unsigned long long mu_ = mu_device[int(idx / N) % Q_count];
    unsigned long long qbit = q_bit_device[int(idx / N) % Q_count];
    unsigned long long term1 = Enc1[idx];
    unsigned long long term2 = Enc2[idx];
    uint128_t sum = (term1 + term2);
    //singleBarrett(sum, q_, mu_, qbit);
    sum -= q_ * (sum >= q_);
    Enc1[idx] = sum.low;
    //results_device[idx] = sum % q_;

}

//Subtraction
__global__ void Subtraction(unsigned long long* Enc1, unsigned long long* Enc2, unsigned long long* results, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
    int N, int Q_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_ = q_device[int(idx / N) % Q_count];
    unsigned long long mu_ = mu_device[int(idx / N) % Q_count];
    unsigned long long qbit = q_bit_device[int(idx / N) % Q_count];
    unsigned long long term1 = Enc1[idx];
    unsigned long long term2 = Enc2[idx];
    uint128_t sum = (q_ + term1);
    sum = sum - term2;
    //singleBarrett(sum, q_, mu_, qbit);
    sum -= q_ * (sum >= q_);
    results[idx] = sum.low;
    //results_device[idx] = sum % q_;

}

__global__ void Subtraction_Inplace(unsigned long long* Enc1, unsigned long long* Enc2, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
    int N, int Q_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long q_ = q_device[int(idx / N) % Q_count];
    unsigned long long mu_ = mu_device[int(idx / N) % Q_count];
    unsigned long long qbit = q_bit_device[int(idx / N) % Q_count];
    unsigned long long term1 = Enc1[idx];
    unsigned long long term2 = Enc2[idx];
    uint128_t sum = (q_ + term1);
    sum = sum - term2;
    //singleBarrett(sum, q_, mu_, qbit);
    sum -= q_ * (sum >= q_);
    Enc1[idx] = sum.low;
    //results_device[idx] = sum % q_;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////// ------------------------------------------------------------------------------------------ /////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// KERNEL FOR 3x
__global__ void Addition_3x(unsigned long long* Enc1_1, unsigned long long* Enc1_2, unsigned long long* Enc2_1, unsigned long long* Enc2_2, unsigned long long* results_1, unsigned long long* results_2,
    unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
    int N, int Q_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = int(idx / N) % Q_count;

    unsigned long long q_ = q_device[loc1];
    unsigned long long mu_ = mu_device[loc1];
    unsigned long long qbit = q_bit_device[loc1];

    if (idx < 2 * N * Q_count) {

        unsigned long long term1 = Enc1_1[idx];
        unsigned long long term2 = Enc2_1[idx];
        uint128_t sum = (term1 + term2);

        sum -= q_ * (sum >= q_);
        results_1[idx] = sum.low;

    }
    else {
        int addr = idx - 2 * N * Q_count;
        unsigned long long term1 = Enc1_2[addr];
        unsigned long long term2 = Enc2_2[addr];
        uint128_t sum = (term1 + term2);

        sum -= q_ * (sum >= q_);
        results_2[addr] = sum.low;

    }

}

__host__ void GPU_Addition(GPU_Ciphertext input1, GPU_Ciphertext input2, GPU_Ciphertext output, Lib_Parameters context)
{
    //Class Features
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Addition << < 2 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, output.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}

__host__ void GPU_Addition_x3(GPU_Ciphertext input1, GPU_Ciphertext input2, GPU_Ciphertext output, Lib_Parameters context)
{
    //Class Features
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    //Addition << < 3 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, output.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
    Addition_3x << < 3 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input1.GPU_Location2, input2.GPU_Location, input2.GPU_Location2, output.GPU_Location, output.GPU_Location2, q_device, mu_device, q_bit_device, n, (qcount - 1));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void GPU_Addition_Inplace(GPU_Ciphertext input1, GPU_Ciphertext input2, Lib_Parameters context)
{
    //Class Features
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Addition_Inplace << < 2 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}

__host__ void GPU_Addition_Inplace_x3(GPU_Ciphertext input1, GPU_Ciphertext input2, Lib_Parameters context)
{
    //Class Features
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Addition_Inplace << < 3 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void GPU_Subtraction(GPU_Ciphertext input1, GPU_Ciphertext input2, GPU_Ciphertext output, Lib_Parameters context)
{
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Subtraction << < 2 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, output.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}


__host__ void GPU_Subtraction_x3(GPU_Ciphertext input1, GPU_Ciphertext input2, GPU_Ciphertext output, Lib_Parameters context)
{
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Subtraction << < 3 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, output.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void GPU_Subtraction_Inplace(GPU_Ciphertext input1, GPU_Ciphertext input2, Lib_Parameters context)
{
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Subtraction_Inplace << < 2 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}

__host__ void GPU_Subtraction_Inplace_x3(GPU_Ciphertext input1, GPU_Ciphertext input2, Lib_Parameters context)
{
    int n = input1.ring_size;
    unsigned qcount = input1.coeff_mod_count + 1;

    //Extract Context Parameters
    unsigned long long* q_device = context.Context_GPU;
    unsigned long long* mu_device = q_device + qcount;
    unsigned long long* q_bit_device = mu_device + qcount;

    Subtraction_Inplace << < 3 * (qcount - 1) * (n / 1024), 1024 >> > (input1.GPU_Location, input2.GPU_Location, q_device, mu_device, q_bit_device, n, (qcount - 1));
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////