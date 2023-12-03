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
#include "gpu_kernels/encoding.cuh"
#include "gpu_kernels/ntt.cuh"

#include "context.cuh"

#ifndef ENCODER_H
#define ENCODER_H


class HEEncoder
{
public:

    __host__ HEEncoder(Parameters &context);

    __host__ void encode(Plaintext &plain, const Message message);

    __host__ void encode(Plaintext &plain, const Message message, HEStream stream);

    __host__ void decode(Message &message, const Plaintext plain);

    __host__ void decode(Message &message, const Plaintext plain, HEStream stream);

    __host__ void kill();

private:

    int n;

    int n_power;

    Data* encoding_location_;

    Modulus* plain_modulus_;

    Root* plain_ntt_tables_;
    Root* plain_intt_tables_;

    Ninverse* n_plain_inverse_;

};

#endif // ENCODER_H