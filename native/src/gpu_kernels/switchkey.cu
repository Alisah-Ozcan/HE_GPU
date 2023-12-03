// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "switchkey.cuh"


/*
__global__ void CipherBroadcast(Data* input, Data* output, int n_power, int rns_mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count

    Data input_ = input[idx + (block_y << n_power)];
#pragma unroll
    for (int i = 0; i < rns_mod_count; i++) {

        output[idx + ((i * rns_mod_count) << n_power)] = input_;

    }

}
*/

__global__ void CipherBroadcast(Data* input, Data* output, int n_power, int rns_mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count

    int location = (rns_mod_count * block_y) << n_power;
    Data input_ = input[idx + (block_y << n_power)];
#pragma unroll
    for (int i = 0; i < rns_mod_count; i++) {

        output[idx + (i << n_power) + location] = input_;

    }

}

__global__ void CipherBroadcast2(Data* input, Data* output, Modulus* modulus, int n_power, int rns_mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count

    int location = (rns_mod_count * block_y) << n_power;
    Data input_ = input[idx + (block_y << n_power)];
    Data one_ = 1;
#pragma unroll
    for (int i = 0; i < rns_mod_count; i++) {

        output[idx + (i << n_power) + location] = VALUE_GPU::mult(one_, input_, modulus[i]);

    }

}

__global__ void MultiplyAcc(Data* input, Data* relinkey, Data* output, Modulus* modulus, int n_power, int decomp_mod_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // RNS Modulus Count

    int key_offset1 = (decomp_mod_count + 1) << n_power;
    int key_offset2 = (decomp_mod_count + 1) << (n_power + 1);

    Data ct_0_sum = 0;
    Data ct_1_sum = 0;
#pragma unroll
    for (int i = 0; i < decomp_mod_count; i++) {

        Data in_piece = input[idx + (block_y << n_power) + ((i * (decomp_mod_count + 1)) << n_power)];
/*
        if((idx == 0) && (block_y == 0)){
            printf("==> %llu : %llu \n", i, in_piece);
        }
*/
        Data rk0 = relinkey[idx + (block_y << n_power) + (key_offset2 * i)];
        Data rk1 = relinkey[idx + (block_y << n_power) + (key_offset2 * i) + key_offset1];

        Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus[block_y]);
        Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus[block_y]);


        ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus[block_y]);
        ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus[block_y]);

    }

    output[idx + (block_y << n_power)] = ct_0_sum;
    output[idx + (block_y << n_power) + key_offset1] = ct_1_sum;

}


__global__ void DivideRoundLastq(Data* input, Data* ct, Data* output, Modulus* modulus, Data half, Data* half_mod, Data* last_q_modinv, int n_power, int decomp_mod_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count
    int block_z = blockIdx.z; // Cipher Size (2)


    Data last_ct = input[idx + (decomp_mod_count << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];

/*
    if((idx == 0) && (block_y == 0) && (block_z == 0)){
        printf("==> %llu \n", last_ct);
    }
    if((idx == 0) && (block_y == 0) && (block_z == 1)){
        printf("==> %llu \n", last_ct);
    }
*/

    last_ct = VALUE_GPU::add(last_ct, half, modulus[decomp_mod_count]);

    Data zero_ = 0;
    last_ct = VALUE_GPU::add(last_ct, zero_, modulus[block_y]);

    last_ct = VALUE_GPU::sub(last_ct, half_mod[block_y], modulus[block_y]);
/*
    if((idx == 0) && (block_z == 1)){
        printf("===> %llu - %llu \n", block_y, last_ct);
    }
*/

    Data input_ = input[idx + (block_y << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];

/*
    if((idx == 0) && (block_z == 1)){
        printf("==> %llu - %llu \n", block_y, input_);
    }
*/
    input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

    input_ = VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);


    Data ct_in = ct[idx + (block_y << n_power) + (((decomp_mod_count) << n_power) * block_z)];
/*
    if((idx == 0) && (block_z == 0)){
        printf("==> %llu - %llu \n", block_y, ct_in);
    }
*/
    ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);


    output[idx + (block_y << n_power) + (((decomp_mod_count) << n_power) * block_z)] = ct_in;


}



__host__ void HERelinearization(Ciphertext &input1, Relinkey key, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned decomp_mod_count = rns_mod_count - 1;
    unsigned n_power = context.n_power; 

    //CipherBroadcast<< < dim3((context.n >> 8), decomp_mod_count, 1), 256 >> >(input1.location + (decomp_mod_count << (n_power + 1)), temp_1, n_power, rns_mod_count);

    CipherBroadcast2<< < dim3((context.n >> 8), decomp_mod_count, 1), 256 >> >(input1.location + (decomp_mod_count << (n_power + 1)), context.temp1_relin, context.modulus_, n_power, rns_mod_count);


    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(context.temp1_relin, context.ntt_table_ , context.modulus_, cfg_ntt, decomp_mod_count * rns_mod_count, rns_mod_count);

    MultiplyAcc<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(context.temp1_relin, key.location, context.temp2_relin, context.modulus_, n_power, decomp_mod_count);

    ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(context.temp2_relin, context.intt_table_, context.modulus_, cfg_intt, 2 * rns_mod_count, rns_mod_count);

    DivideRoundLastq<< < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> >(context.temp2_relin, input1.location, input1.location, context.modulus_, context.half_, context.half_mod_, context.last_q_modinv_, n_power, decomp_mod_count);


}


////////////////////////////////////////////////////////////////////////////


__global__ void apply_galois(Data* cipher, Data* out0, Data* out1, Modulus* modulus, int galois_elt, int n_power, int decomp_mod_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count
    int block_z = blockIdx.z; // Cipher Size (2)

    int rns_mod_count = (decomp_mod_count + 1);

    int coeff_count_minus_one = (1 << n_power) - 1;


    int index_raw = idx * galois_elt;
    int index = index_raw & coeff_count_minus_one;
    Data result_value = cipher[idx + (block_y << n_power) + ((decomp_mod_count << n_power) * block_z)];

    if ((index_raw >> n_power) & 1) {
        result_value = (modulus[block_y].value - result_value);
    }

    if(block_z == 0){

        out0[index + (block_y << n_power) + ((decomp_mod_count << n_power) * block_z)] = result_value;

    }
    else{
        
        int location = (rns_mod_count * block_y) << n_power;

        for (int i = 0; i < rns_mod_count; i++) {

            out1[index + (i << n_power) + location] = result_value; 

        }

    }
}



__host__ void HERotation(Ciphertext &input, Ciphertext &output, int shift, Galoiskey key, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned decomp_mod_count = rns_mod_count - 1;
    unsigned n_power = context.n_power; 

    Data** GaloisKey;
    int* galoiselt;
    // positive or negative
    if (shift > 0) {
        GaloisKey = key.positive_location;
        galoiselt = key.galois_elt_pos;
    }
    else {
        GaloisKey = key.negative_location;
        galoiselt = key.galois_elt_neg;
    }

    int shift_num = abs(shift);
    while (shift_num != 0) {

        int power = int(log2(shift_num));
        int power_2 = pow(2, power);
        shift_num = shift_num - power_2;

        apply_galois<< < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> >(input.location, context.temp0_rotation, context.temp1_rotation, context.modulus_, galoiselt[power], n_power, decomp_mod_count);

        ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

        GPU_NTT_Inplace(context.temp1_rotation, context.ntt_table_ , context.modulus_, cfg_ntt, decomp_mod_count * rns_mod_count, rns_mod_count);

        MultiplyAcc<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(context.temp1_rotation, GaloisKey[power], context.temp2_rotation, context.modulus_, n_power, decomp_mod_count);

        ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.n_inverse_,
        .stream = 0};

        GPU_NTT_Inplace(context.temp2_rotation, context.intt_table_, context.modulus_, cfg_intt, 2 * rns_mod_count, rns_mod_count);

        DivideRoundLastq<< < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> >(context.temp2_rotation, context.temp0_rotation, output.location, context.modulus_, context.half_, context.half_mod_, context.last_q_modinv_, n_power, decomp_mod_count);

    }


    

}

/*

CipherBroadcast2<< < dim3((context.n >> 8), decomp_mod_count, 1), 256 >> >(input1.location + (decomp_mod_count << (n_power + 1)), temp_1, context.modulus_, n_power, rns_mod_count);


    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(temp_1, context.ntt_table_ , context.modulus_, cfg_ntt, decomp_mod_count * rns_mod_count, rns_mod_count);

    MultiplyAcc<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(temp_1, key.location, temp_2, context.modulus_, n_power, decomp_mod_count);

    ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(temp_2, context.intt_table_, context.modulus_, cfg_intt, 2 * rns_mod_count, rns_mod_count);

    DivideRoundLastq<< < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> >(temp_2, input1.location, input1.location, context.modulus_, context.half_, context.half_mod_, context.last_q_modinv_, n_power, decomp_mod_count);


*/