// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "common.cuh"
#include "cuda_runtime.h"
#include "gpu_kernels/ntt.cuh"
#include "context.cuh"

#ifndef KEYGEN_H
#define KEYGEN_H


__global__ void sk_kernel(Data* secret_key, Modulus* modulus, int n_power, int rns_mod_count, int seed);

__host__ void HESecretkeygen(Secretkey &sk, Parameters context);




__global__ void error_kernel(Data* a_e, Modulus* modulus, int n_power, int rns_mod_count, int seed);

__global__ void pk_kernel(Data* public_key, Data* secret_key, Modulus* modulus, int n_power, int rns_mod_count);

__host__ void HEPublickeygen(Publickey &pk, Secretkey &sk, Parameters context);




__global__ void relinkey_kernel(Data* relin_key, Data* secret_key, Data* e_a, Modulus* modulus, Data* factor, int n_power, int rns_mod_count);

__host__ void HERelinkeygen(Relinkey &rk, Secretkey &sk, Parameters context);




int steps_to_galois_elt(int steps, int coeff_count);

__device__ int bitreverse_gpu(int index, int n_power);

__device__ int permutation(int index, int galois_elt, int coeff_count, int n_power);

__global__ void galoiskey_kernel(Data* galois_key, Data* secret_key, Data* e_a, Modulus* modulus, Data* factor, int galois_elt, int n_power, int rns_mod_count);

__host__ void HEGaloiskeygen(Galoiskey &gk, Secretkey &sk, Parameters context);

#endif // KEYGEN_H