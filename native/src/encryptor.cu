// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "encryptor.cuh"



//////////////////////////////////////////////////////////////////////////////////

__host__ HEEncryptor::HEEncryptor(const Parameters &context, const Publickey &public_key){

    public_key_ = public_key.location;

    n = context.n;
    n_power = context.n_power;

    rns_mod_count_ = context.coeff_modulus;

    modulus_ = context.modulus_;

    plain_modulus_ = context.plain_modulus_;

    last_q_modinv_ = context.last_q_modinv_;

    ntt_table_ = context.ntt_table_;
    intt_table_ = context.intt_table_;

    n_inverse_ = context.n_inverse_;

    half_ = context.half_;

    half_mod_ = context.half_mod_;

    Q_mod_t_ = context.Q_mod_t_;

    upper_threshold_ = context.upper_threshold_;

    coeeff_div_plainmod_ = context.coeeff_div_plainmod_;


    cudaMalloc(&temp1_enc, 3 * n * rns_mod_count_ * sizeof(Data));
    cudaMalloc(&temp2_enc, 2 * n * rns_mod_count_ * sizeof(Data));

}

__host__ void HEEncryptor::kill(){

    cudaFree(temp1_enc);
    cudaFree(temp2_enc);

}

__host__ void HEEncryptor::encrypt(Ciphertext &ciphertext, const Plaintext &plaintext){

    enc_error_kernel<< < dim3((n >> 8), 3, 1), 256 >> >(temp1_enc, modulus_, n_power, rns_mod_count_, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(temp1_enc, ntt_table_ , modulus_, cfg_ntt, rns_mod_count_, rns_mod_count_);

    pk_u_kernel<< < dim3((n >> 8), rns_mod_count_, 2), 256 >> >(public_key_, temp1_enc, temp2_enc, modulus_, n_power, rns_mod_count_);

    ntt_configuration cfg_intt = {
    .n_power = n_power,
    .ntt_type = INVERSE,
    .reduction_poly = ReductionPolynomial::X_N_plus,
    .zero_padding = false,
    .mod_inverse = n_inverse_,
    .stream = 0};

    GPU_NTT_Inplace(temp2_enc, intt_table_ , modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

    EncDivideRoundLastq<< < dim3((n >> 8), (rns_mod_count_ - 1), 2), 256 >> >(temp2_enc, temp1_enc + (rns_mod_count_ << n_power), plaintext.location, ciphertext.location, modulus_,
    half_, half_mod_, last_q_modinv_, plain_modulus_, Q_mod_t_, upper_threshold_, coeeff_div_plainmod_, n_power, (rns_mod_count_ - 1)
    );

}

__host__ void HEEncryptor::encrypt(Ciphertext &ciphertext, const Plaintext &plaintext, HEStream stream){

    enc_error_kernel<< < dim3((n >> 8), 3, 1), 256, 0, stream.stream >> >(stream.temp1_enc, modulus_, n_power, rns_mod_count_, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp1_enc, ntt_table_ , modulus_, cfg_ntt, rns_mod_count_, rns_mod_count_);

    pk_u_kernel<< < dim3((n >> 8), rns_mod_count_, 2), 256, 0, stream.stream >> >(public_key_, stream.temp1_enc, stream.temp2_enc, modulus_, n_power, rns_mod_count_);

    ntt_configuration cfg_intt = {
    .n_power = n_power,
    .ntt_type = INVERSE,
    .reduction_poly = ReductionPolynomial::X_N_plus,
    .zero_padding = false,
    .mod_inverse = n_inverse_,
    .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp2_enc, intt_table_ , modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

    EncDivideRoundLastq<< < dim3((n >> 8), (rns_mod_count_ - 1), 2), 256, 0, stream.stream >> >(stream.temp2_enc, stream.temp1_enc + (rns_mod_count_ << n_power), plaintext.location, ciphertext.location, modulus_,
    half_, half_mod_, last_q_modinv_, plain_modulus_, Q_mod_t_, upper_threshold_, coeeff_div_plainmod_, n_power, (rns_mod_count_ - 1)
    );

}
