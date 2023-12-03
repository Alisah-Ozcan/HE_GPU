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

#ifndef ENCRYPTION_H
#define ENCRYPTION_H


__global__ void enc_error_kernel(Data* u_e, Modulus* modulus, int n_power, int rns_mod_count, int seed);

__global__ void pk_u_kernel(Data* pk, Data* u, Data* pk_u, Modulus* modulus, int n_power, int rns_mod_count);

__global__ void EncDivideRoundLastq(Data* pk, Data* e, Data* plain, Data* ct, Modulus* modulus, Data half, Data* half_mod, Data* last_q_modinv,
 Modulus plain_mod, Data Q_mod_t, Data upper_threshold, Data* coeffdiv_plain, int n_power, int decomp_mod_count);


//__host__ void HEEncryption(Ciphertext cipher, Plaintext plain, Publickey pk, Parameters context);



#endif // ENCRYPTION_H