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

#include "../common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "../context.cuh"

#ifndef SWITCHKEY_H
#define SWITCHKEY_H



__global__ void CipherBroadcast(Data* input, Data* output, int n_power, int iteration);
__global__ void CipherBroadcast2(Data* input, Data* output, Modulus* modulus, int n_power, int rns_mod_count);

__global__ void MultiplyAcc(Data* input, Data* relinkey, Data* output, Modulus* modulus, int n_power, int decomp_mod_count);

__global__ void DivideRoundLastq(Data* input, Data* ct, Data* output, Modulus* modulus, Data half, Data* half_mod, Data* last_q_modinv, int n_power, int decomp_mod_count);

__host__ void HERelinearization(Ciphertext &input1, Relinkey key, Parameters context);


__global__ void apply_galois(Data* cipher, Data* out0, Data* out1, Modulus* modulus, int galois_elt, int n_power, int decomp_mod_count);

__host__ void HERotation(Ciphertext &input, Ciphertext &output, int shift, Galoiskey key, Parameters context);

#endif // SWITCHKEY_H