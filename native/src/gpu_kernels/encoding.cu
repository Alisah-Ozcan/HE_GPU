// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "encoding.cuh"

__global__ void encode_kernel(Data* messagge_encoded, Data* messagge, Data* location_info) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int location = location_info[idx];
    messagge_encoded[location] = messagge[idx];

}

__global__ void decode_kernel(Data* messagge, Data* messagge_encoded, Data* location_info) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int location = location_info[idx];
    messagge[idx] = messagge_encoded[location];

}