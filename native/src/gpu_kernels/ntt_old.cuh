// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://ieeexplore.ieee.org/document/10097488

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "../common.cuh"
#include "cuda_runtime.h"
#include "nttparameters.cuh"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_FFT_CORE_H
#define NTT_FFT_CORE_H

typedef unsigned location_t;

enum type
{
    FORWARD,
    INVERSE
};

struct ntt_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse* mod_inverse;
    cudaStream_t stream;
};

__device__ void CooleyTukeyUnit(Data& U, Data& V, Root& root, Modulus& modulus);

__device__ void GentlemanSandeUnit(Data& U, Data& V, Root& root,
                                   Modulus& modulus);

__global__ void FORWARD_NTT_IEEE_SHARED(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus* modulus, int logm, int N_power,
                            bool reduction_poly_check, int mod_count);

__global__ void INVERSE_NTT_IEEE_SHARED(Data* polynomial_in, Data* polynomial_out, Root* inverse_root_of_unity_table,
                            Modulus* modulus, int logm, int k, int N_power,
                            bool reduction_poly_check, int mod_count);

__global__ void FORWARD_NTT_IEEE_REG(Data* Inputs, Data* Outputs, Root* root_of_unity_table, Modulus* modulus, int m_, int t_2_, int k_, int outer_loop, int inner_loop, int N_power, int mod_count);

__global__ void INVERSE_NTT_IEEE_REG(Data* Inputs, Data* Outputs, Root* root_of_unity_table, Modulus* modulus, int m_, int t_2_, int k_, int outer_loop, int inner_loop, int N_power, Ninverse* n_inverse, int mod_count);

__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table,
                      Modulus* modulus, ntt_configuration cfg, int batch_size, int mod_count);

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table,
                      Modulus* modulus, ntt_configuration cfg, int batch_size, int mod_count);

__global__ void GPU_ACTIVITY(unsigned long long* output,
                             unsigned long long fix_num);
__host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                unsigned long long fix_num);

__global__ void GPU_ACTIVITY2(unsigned long long* input1, unsigned long long* input2);
__host__ void GPU_ACTIVITY2_HOST(unsigned long long* input1, unsigned long long* input2, unsigned size);


#endif  // NTT_FFT_CORE_H
