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
#include "../context.cuh"

#ifndef DECRYPTION_H
#define DECRYPTION_H

__global__ void sk_multiplication(Data* ct1, Data* sk, Modulus* modulus, int n_power, int decomp_mod_count);

__global__ void decryption_kernel(Data* ct0, Data* ct1, Data* plain, Modulus* modulus, Modulus plain_mod, Modulus gamma,
    Data* Qi_t, Data* Qi_gamma, Data* Qi_inverse, Data mulq_inv_t, Data mulq_inv_gamma, Data inv_gamma,
    int n_power, int decomp_mod_count);


//__host__ void HEDecryption(Plaintext plain, Ciphertext cipher, Secretkey sk, Parameters context);



#endif // DECRYPTION_H