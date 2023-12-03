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

#ifndef ENCODING_H
#define ENCODING_H

__global__ void encode_kernel(Data* messagge_encoded, Data* messagge, Data* location_info);

__global__ void decode_kernel(Data* messagge, Data* messagge_encoded, Data* location_info);

#endif // ENCODING_H