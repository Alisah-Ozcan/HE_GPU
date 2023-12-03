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
#include "gpu_kernels/encryption.cuh"
#include "gpu_kernels/ntt.cuh"
#include "context.cuh"

#ifndef ENCRYPTOR_H
#define ENCRYPTOR_H

class HEEncryptor
{
public:

    __host__ HEEncryptor(const Parameters &context, const Publickey &public_key);

    __host__ void encrypt(Ciphertext &ciphertext, const Plaintext &plaintext);

    __host__ void encrypt(Ciphertext &ciphertext, const Plaintext &plaintext, HEStream stream);

    __host__ void kill();

private:

    Data* public_key_;

    int n;

    int n_power;

    int rns_mod_count_;

    Modulus* modulus_;

    Modulus plain_modulus_;

    Data* last_q_modinv_;

    Root* ntt_table_;
    Root* intt_table_; 

    Ninverse* n_inverse_;

    Data half_;

    Data* half_mod_;

    Data Q_mod_t_;

    Data upper_threshold_;

    Data* coeeff_div_plainmod_;

    Data *temp1_enc;
    Data *temp2_enc;

};


#endif // ENCRYPTOR_H

