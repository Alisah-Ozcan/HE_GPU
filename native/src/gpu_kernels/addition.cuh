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

#ifndef HE_ADDITION_H
#define HE_ADDITION_H

// Homomorphic Addition
__global__ void Addition(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power);

__host__ void HEAdditionInplace(Ciphertext input1, Ciphertext input2, Parameters context);

__host__ void HEAddition(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context);

__host__ void HEAddition_x3(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context);

// Homomorphic Substraction
__global__ void Substraction(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power);

__host__ void HESubstractionInplace(Ciphertext input1, Ciphertext input2, Parameters context);

__host__ void HESubstraction(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context);

#endif // HE_ADDITION_H