// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "decryption.cuh"


__global__ void sk_multiplication(Data* ct1, Data* sk, Modulus* modulus, int n_power, int decomp_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
    int block_y = blockIdx.y; // decomp_mod_count

    int index = idx + (block_y << n_power);

    Data ct_1 = ct1[index];
    Data sk_ = sk[index];

    ct1[index] = VALUE_GPU::mult(ct_1, sk_, modulus[block_y]);

}

/*
std::vector<Data> Qi_t();
std::vector<Data> Qi_gamma();
std::vector<Data> Qi_inverse();
Data mulq_inv_t();
Data mulq_inv_gamma();
Data inv_gamma();
*/




__global__ void decryption_kernel(Data* ct0, Data* ct1, Data* plain, Modulus* modulus, Modulus plain_mod, Modulus gamma,
    Data* Qi_t, Data* Qi_gamma, Data* Qi_inverse, Data mulq_inv_t, Data mulq_inv_gamma, Data inv_gamma,
    int n_power, int decomp_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

    Data sum_t = 0;
    Data sum_gamma = 0;

    Data one = 1;

#pragma unroll
    for (int i = 0; i < decomp_mod_count; i++) {

        int location = idx + (i << n_power);

        Data mt = VALUE_GPU::add(ct0[location], ct1[location], modulus[i]);

        Data gamma_ = VALUE_GPU::mult(one, gamma.value, modulus[i]);

        mt = VALUE_GPU::mult(mt, plain_mod.value, modulus[i]);

        mt = VALUE_GPU::mult(mt, gamma_, modulus[i]);

        mt = VALUE_GPU::mult(mt, Qi_inverse[i], modulus[i]);


        Data mt_in_t = VALUE_GPU::mult(one, mt, plain_mod);
        Data mt_in_gamma = VALUE_GPU::mult(one, mt, gamma);

        mt_in_t = VALUE_GPU::mult(mt_in_t, Qi_t[i], plain_mod);
        mt_in_gamma = VALUE_GPU::mult(mt_in_gamma, Qi_gamma[i], gamma);

        sum_t = VALUE_GPU::add(sum_t, mt_in_t, plain_mod);
        sum_gamma = VALUE_GPU::add(sum_gamma, mt_in_gamma, gamma);

    }

    sum_t = VALUE_GPU::mult(sum_t, mulq_inv_t, plain_mod);
    sum_gamma = VALUE_GPU::mult(sum_gamma, mulq_inv_gamma, gamma);

    Data gamma_2 = gamma.value >> 1;

    if (sum_gamma > gamma_2) {
        
        Data gamma_ = VALUE_GPU::mult(one, gamma.value, plain_mod);
        Data sum_gamma_ = VALUE_GPU::mult(one, sum_gamma, plain_mod);

        Data result = VALUE_GPU::sub(gamma_, sum_gamma_, plain_mod);
        result = VALUE_GPU::add(sum_t, result, plain_mod);
        result = VALUE_GPU::mult(result, inv_gamma, plain_mod);
  
        plain[idx] = result;

    }
    else {
        
        Data sum_t_ = VALUE_GPU::mult(one, sum_t, plain_mod);
        Data sum_gamma_ = VALUE_GPU::mult(one, sum_gamma, plain_mod);

        Data result = VALUE_GPU::sub(sum_t_, sum_gamma_, plain_mod);
        result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

        plain[idx] = result;
    }

}

/*
__host__ void HEDecryption(Plaintext plain, Ciphertext cipher, Secretkey sk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned decomp_mod_count = rns_mod_count - 1;
    unsigned n_power = context.n_power;

    Data* ct0 = cipher.location;
    Data* ct1 = cipher.location + (decomp_mod_count << n_power);

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(ct1, context.ntt_table_ , context.modulus_, cfg_ntt, decomp_mod_count, decomp_mod_count);

    sk_multiplication<< < dim3((context.n >> 8), decomp_mod_count, 1), 256 >> >(ct1, sk.location, context.modulus_, n_power, decomp_mod_count);

    ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(ct1, context.intt_table_ , context.modulus_, cfg_intt, decomp_mod_count, decomp_mod_count);

    decryption_kernel<< < dim3((context.n >> 8), 1, 1), 256 >> >(ct0, ct1, plain.location, context.modulus_, context.plain_modulus_, context.gamma_,
    context.Qi_t_, context.Qi_gamma_, context.Qi_inverse_, context.mulq_inv_t_, context.mulq_inv_gamma_, context.inv_gamma_,
    n_power, decomp_mod_count);

}
*/
