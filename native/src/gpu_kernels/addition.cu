// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "addition.cuh"


__global__ void Addition(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // rns count
    int idz = blockIdx.z; // cipher count

    int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

    out[location] = VALUE_GPU::add(in1[location], in2[location], modulus[idy]);

}

__host__ void HEAdditionInplace(Ciphertext input1, Ciphertext input2, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;

    Addition << < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> > (input1.location, input2.location, input1.location, context.modulus_, context.n_power);


}

__host__ void HEAddition(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;

    Addition << < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> > (input1.location, input2.location, output.location, context.modulus_, context.n_power);
}

__host__ void HEAddition_x3(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;

    Addition << < dim3((context.n >> 8), decomp_mod_count, 3), 256 >> > (input1.location, input2.location, output.location, context.modulus_, context.n_power);
}



__global__ void Substraction(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // rns count
    int idz = blockIdx.z; // cipher count

    int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

    out[location] = VALUE_GPU::sub(in1[location], in2[location], modulus[idy]);

}

__host__ void HESubstractionInplace(Ciphertext input1, Ciphertext input2, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;

    Substraction << < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> > (input1.location, input2.location, input1.location, context.modulus_, context.n_power);
}

__host__ void HESubstraction(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;

    Substraction << < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> > (input1.location, input2.location, output.location, context.modulus_, context.n_power);
}
