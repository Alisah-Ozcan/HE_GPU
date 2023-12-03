// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "encryption.cuh"


__global__ void enc_error_kernel(Data* u_e, Modulus* modulus, int n_power, int rns_mod_count, int seed){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // u, e1, e2

    if(block_y == 0){ // u
        
        curandState_t state;
        curand_init((idx * seed) + (seed), 0, 0, &state);
        Data random_number = curand(&state) % 3; // 0,1,2

#pragma unroll
    for (int i = 0; i < rns_mod_count; i++) {

        int location = i << n_power;

        Data result = random_number;

        if (result == 2)
            result = modulus[i].value - 1; // -1,0,1

        u_e[idx + location] = result;

    }

    }
    else if(block_y == 1){ // e1

        curandState_t state;

        curand_init((idx * seed) + (seed * seed), 0, 0, &state);

        float rn = curand_normal(&state);
        rn = rn * 3.2; // SIGMA
        //rn = rn * 1;

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++) {
            
            signed long long rn_ = rn;
            
            if (rn_ < 0)
                rn_ = rn_ + modulus[i].value;

            Data rn_ULL = rn_;

            int location = i << n_power;

            u_e[idx + location + ((rns_mod_count) << n_power)] = rn_ULL % modulus[i].value;
        }

    }
    else{ // e2

        curandState_t state;

        curand_init((idx * seed) + (seed * seed *seed), 0, 0, &state);

        float rn = curand_normal(&state);
        rn = rn * 3.2; // SIGMA
        //rn = rn * 1;

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++) {
            
            signed long long rn_ = rn;
            
            if (rn_ < 0)
                rn_ = rn_ + modulus[i].value;

            Data rn_ULL = rn_;

            int location = i << n_power;

            u_e[idx + location + ((rns_mod_count) << (n_power + 1))] = rn_ULL % modulus[i].value;
        }

    }

}

__global__ void pk_u_kernel(Data* pk, Data* u, Data* pk_u, Modulus* modulus, int n_power, int rns_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // rns_mod_count
    int block_z = blockIdx.z; // 2

    Data pk_ = pk[idx + (block_y << n_power) + ((rns_mod_count << n_power) * block_z)];
    Data u_ = u[idx + (block_y << n_power)];

    Data pk_u_ = VALUE_GPU::mult(pk_, u_, modulus[block_y]);

    pk_u[idx + (block_y << n_power) + ((rns_mod_count << n_power) * block_z)] = pk_u_;

}


__global__ void EncDivideRoundLastq(Data* pk, Data* e, Data* plain, Data* ct, Modulus* modulus, Data half, Data* half_mod, Data* last_q_modinv,
 Modulus plain_mod, Data Q_mod_t, Data upper_threshold, Data* coeffdiv_plain, int n_power, int decomp_mod_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // Decomposition Modulus Count
    int block_z = blockIdx.z; // Cipher Size (2)


    Data last_pk = pk[idx + (decomp_mod_count << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];
    Data last_e = e[idx + (decomp_mod_count << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];

    last_pk = VALUE_GPU::add(last_pk, last_e, modulus[decomp_mod_count]);

    last_pk = VALUE_GPU::add(last_pk, half, modulus[decomp_mod_count]);

    Data zero_ = 0;
    last_pk = VALUE_GPU::add(last_pk, zero_, modulus[block_y]);

    last_pk = VALUE_GPU::sub(last_pk, half_mod[block_y], modulus[block_y]);

    Data input_ = pk[idx + (block_y << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];

    //
    Data e_ = e[idx + (block_y << n_power) + (((decomp_mod_count + 1) << n_power) * block_z)];
    input_ = VALUE_GPU::add(input_, e_, modulus[block_y]);
    //


    input_ = VALUE_GPU::sub(input_, last_pk, modulus[block_y]);

    input_ = VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

    if(block_z == 0){

        Data message = plain[idx];
        Data fix = message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);

        Data ct_0 = VALUE_GPU::mult(message, coeffdiv_plain[block_y], modulus[block_y]);
        ct_0 = VALUE_GPU::add(ct_0, fix, modulus[block_y]);

        input_ = VALUE_GPU::add(input_, ct_0, modulus[block_y]);

        ct[idx + (block_y << n_power) + (((decomp_mod_count) << n_power) * block_z)] = input_;
    
    }
    else{
        ct[idx + (block_y << n_power) + (((decomp_mod_count) << n_power) * block_z)] = input_;
    }
   
}

/*
__host__ void HEEncryption(Ciphertext cipher, Plaintext plain, Publickey pk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned n_power = context.n_power;
        
    enc_error_kernel<< < dim3((context.n >> 8), 3, 1), 256 >> >(context.temp1_enc, context.modulus_, n_power, rns_mod_count, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(context.temp1_enc, context.ntt_table_ , context.modulus_, cfg_ntt, rns_mod_count, rns_mod_count);

    pk_u_kernel<< < dim3((context.n >> 8), rns_mod_count, 2), 256 >> >(pk.location, context.temp1_enc, context.temp2_enc, context.modulus_, n_power, rns_mod_count);

    ntt_configuration cfg_intt = {
    .n_power = context.n_power,
    .ntt_type = INVERSE,
    .reduction_poly = ReductionPolynomial::X_N_plus,
    .zero_padding = false,
    .mod_inverse = context.n_inverse_,
    .stream = 0};

    GPU_NTT_Inplace(context.temp2_enc, context.intt_table_ , context.modulus_, cfg_intt, 2 * rns_mod_count, rns_mod_count);

    EncDivideRoundLastq<< < dim3((context.n >> 8), (rns_mod_count - 1), 2), 256 >> >(context.temp2_enc, context.temp1_enc + (rns_mod_count << n_power), plain.location, cipher.location, context.modulus_,
    context.half_, context.half_mod_, context.last_q_modinv_, context.plain_modulus_, context.Q_mod_t_, context.upper_threshold_, context.coeeff_div_plainmod_, n_power, (rns_mod_count - 1)
    );


}
*/