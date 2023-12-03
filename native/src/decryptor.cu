// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "decryptor.cuh"


//////////////////////////////////////////////////////////////////////////////////

__host__ HEDecryptor::HEDecryptor(const Parameters &context, const Secretkey &secret_key){

    secret_key_ = secret_key.location;

    n = context.n;
    n_power = context.n_power;

    rns_mod_count_ = context.coeff_modulus;

    modulus_ = context.modulus_;

    plain_modulus_ = context.plain_modulus_;

    gamma_= context.gamma_;

    Qi_t_ = context.Qi_t_;

    Qi_gamma_ = context.Qi_gamma_;

    Qi_inverse_ = context.Qi_inverse_;

    mulq_inv_t_ = context.mulq_inv_t_;

    mulq_inv_gamma_ = context.mulq_inv_gamma_;

    inv_gamma_ = context.inv_gamma_;

    ntt_table_ = context.ntt_table_;
    intt_table_ = context.intt_table_;

    n_inverse_ = context.n_inverse_;

}


__host__ void HEDecryptor::decrypt(Plaintext &plaintext, const Ciphertext &ciphertext){


    int decomp_mod_count = rns_mod_count_ - 1;

    Data* ct0 = ciphertext.location;
    Data* ct1 = ciphertext.location + (decomp_mod_count << n_power);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(ct1, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count, decomp_mod_count);

    sk_multiplication<< < dim3((n >> 8), decomp_mod_count, 1), 256 >> >(ct1, secret_key_, modulus_, n_power, decomp_mod_count);

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(ct1, intt_table_ , modulus_, cfg_intt, decomp_mod_count, decomp_mod_count);

    decryption_kernel<< < dim3((n >> 8), 1, 1), 256 >> >(ct0, ct1, plaintext.location, modulus_, plain_modulus_, gamma_,
    Qi_t_, Qi_gamma_, Qi_inverse_, mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
    n_power, decomp_mod_count);

}

__host__ void HEDecryptor::decrypt(Plaintext &plaintext, const Ciphertext &ciphertext, HEStream stream){


    int decomp_mod_count = rns_mod_count_ - 1;

    Data* ct0 = ciphertext.location;
    Data* ct1 = ciphertext.location + (decomp_mod_count << n_power);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    GPU_NTT_Inplace(ct1, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count, decomp_mod_count);

    sk_multiplication<< < dim3((n >> 8), decomp_mod_count, 1), 256, 0, stream.stream >> >(ct1, secret_key_, modulus_, n_power, decomp_mod_count);

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = stream.stream};

    GPU_NTT_Inplace(ct1, intt_table_ , modulus_, cfg_intt, decomp_mod_count, decomp_mod_count);

    decryption_kernel<< < dim3((n >> 8), 1, 1), 256, 0, stream.stream >> >(ct0, ct1, plaintext.location, modulus_, plain_modulus_, gamma_,
    Qi_t_, Qi_gamma_, Qi_inverse_, mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
    n_power, decomp_mod_count);

}