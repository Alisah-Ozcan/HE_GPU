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
#include "gpu_kernels/decryption.cuh"
#include "gpu_kernels/ntt.cuh"
#include "context.cuh"

#ifndef DECRYPTOR_H
#define DECRYPTOR_H

__global__ void sk_multiplication(Data* ct1, Data* sk, Modulus* modulus, int n_power, int decomp_mod_count);

__global__ void decryption_kernel(Data* ct0, Data* ct1, Data* plain, Modulus* modulus, Modulus plain_mod, Modulus gamma,
    Data* Qi_t, Data* Qi_gamma, Data* Qi_inverse, Data mulq_inv_t, Data mulq_inv_gamma, Data inv_gamma,
    int n_power, int decomp_mod_count);


//__host__ void HEDecryption(Plaintext plain, Ciphertext cipher, Secretkey sk, Parameters context);



class HEDecryptor
{
public:

    __host__ HEDecryptor(const Parameters &context, const Secretkey &secret_key);

    __host__ void decrypt(Plaintext &plaintext, const Ciphertext &ciphertext);

    __host__ void decrypt(Plaintext &plaintext, const Ciphertext &ciphertext, HEStream stream);

private:

    Data* secret_key_;

    int n;

    int n_power;

    int rns_mod_count_;

    Modulus* modulus_;

    Modulus plain_modulus_;

    Modulus gamma_;

    Data* Qi_t_;
    
    Data* Qi_gamma_;
    
    Data* Qi_inverse_;

    Data mulq_inv_t_;
    
    Data mulq_inv_gamma_;
    
    Data inv_gamma_;

    Root* ntt_table_;
    Root* intt_table_; 

    Ninverse* n_inverse_;

};


#endif // DECRYPTOR_H

