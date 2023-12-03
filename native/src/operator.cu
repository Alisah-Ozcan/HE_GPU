// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "operator.cuh"


__host__ HEOperator::HEOperator(const Parameters &context){

    n = context.n;

    n_power = context.n_power;

    rns_mod_count_ = context.coeff_modulus;

    decomp_mod_count_ = rns_mod_count_ - 1;

    bsk_mod_count_ = context.bsk_modulus;

    modulus_ = context.modulus_;

    ntt_table_ = context.ntt_table_;

    intt_table_ = context.intt_table_;

    n_inverse_ = context.n_inverse_;

    last_q_modinv_ = context.last_q_modinv_;

    base_Bsk_ = context.base_Bsk_;

    bsk_ntt_tables_ = context.bsk_ntt_tables_;

    bsk_intt_tables_ = context.bsk_intt_tables_;

    bsk_n_inverse_ = context.bsk_n_inverse_;

    m_tilde_ = context.m_tilde_;

    base_change_matrix_Bsk_ = context.base_change_matrix_Bsk_;

    inv_punctured_prod_mod_base_array_ = context.inv_punctured_prod_mod_base_array_;

    base_change_matrix_m_tilde_ = context.base_change_matrix_m_tilde_;

    inv_prod_q_mod_m_tilde_ = context.inv_prod_q_mod_m_tilde_;

    inv_m_tilde_mod_Bsk_ = context.inv_m_tilde_mod_Bsk_;

    prod_q_mod_Bsk_ = context.prod_q_mod_Bsk_;

    inv_prod_q_mod_Bsk_ = context.inv_prod_q_mod_Bsk_;

    plain_modulus_ = context.plain_modulus_;

    base_change_matrix_q_ = context.base_change_matrix_q_;

    base_change_matrix_msk_ = context.base_change_matrix_msk_;

    inv_punctured_prod_mod_B_array_ = context.inv_punctured_prod_mod_B_array_;

    inv_prod_B_mod_m_sk_ = context.inv_prod_B_mod_m_sk_;

    prod_B_mod_q_ = context.prod_B_mod_q_;

    q_Bsk_merge_modulus_ = context.q_Bsk_merge_modulus_;

    q_Bsk_merge_ntt_tables_ = context.q_Bsk_merge_ntt_tables_;

    q_Bsk_merge_intt_tables_ = context.q_Bsk_merge_intt_tables_;

    q_Bsk_n_inverse_ = context.q_Bsk_n_inverse_;

    half_ = context.half_;
    
    half_mod_ = context.half_mod_;

    upper_threshold_ = context.upper_threshold_;

    upper_halfincrement_ = context.upper_halfincrement_;

    // Temp
    cudaMalloc(&temp1_mul, 4 * n * (bsk_mod_count_+decomp_mod_count_) * sizeof(Data)); 
    cudaMalloc(&temp2_mul, 3 * n * (bsk_mod_count_+decomp_mod_count_) * sizeof(Data)); 

    cudaMalloc(&temp1_relin, n * decomp_mod_count_ * rns_mod_count_ * sizeof(Data));
    cudaMalloc(&temp2_relin, 2 * n * rns_mod_count_ * sizeof(Data));


    cudaMalloc(&temp0_rotation, 2 * n * decomp_mod_count_ * sizeof(Data));
    cudaMalloc(&temp1_rotation, n * decomp_mod_count_ * rns_mod_count_ * sizeof(Data));
    cudaMalloc(&temp2_rotation, 2 * n * rns_mod_count_ * sizeof(Data));


    cudaMalloc(&temp1_plain_mul, n * decomp_mod_count_ * sizeof(Data));

}

void HEOperator::kill(){

    cudaFree(temp1_mul);
    cudaFree(temp2_mul);

    cudaFree(temp1_relin);
    cudaFree(temp2_relin);

    cudaFree(temp0_rotation);
    cudaFree(temp1_rotation);
    cudaFree(temp2_rotation);

    cudaFree(temp1_plain_mul);

}

__host__ void HEOperator::add(Ciphertext &input1, Ciphertext &input2, Ciphertext &output){

    Addition << < dim3((n >> 8), decomp_mod_count_, 2), 256 >> > (input1.location, input2.location, output.location, modulus_, n_power);

}

__host__ void HEOperator::add(Ciphertext &input1, Ciphertext &input2, Ciphertext &output, HEStream stream){

    Addition << < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> > (input1.location, input2.location, output.location, modulus_, n_power);

}

__host__ void HEOperator::add_inplace(Ciphertext &input1, Ciphertext &input2){

    add(input1, input2, input1);

}

__host__ void HEOperator::add_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream){

    add(input1, input2, input1, stream);

}

__host__ void HEOperator::sub(Ciphertext &input1, Ciphertext &input2, Ciphertext &output){

    Substraction << < dim3((n >> 8), decomp_mod_count_, 2), 256 >> > (input1.location, input2.location, output.location, modulus_, n_power);

}

__host__ void HEOperator::sub(Ciphertext &input1, Ciphertext &input2, Ciphertext &output,  HEStream stream){

    Substraction << < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> > (input1.location, input2.location, output.location, modulus_, n_power);

}

__host__ void HEOperator::sub_inplace(Ciphertext &input1, Ciphertext &input2){

    sub(input1, input2, input1);

}

__host__ void HEOperator::sub_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream){

    sub(input1, input2, input1, stream);

}

__host__ void HEOperator::multiply(Ciphertext &input1, Ciphertext &input2, Ciphertext &output){

    FastConvertion << < dim3((n >> 8), 4, 1), 256 >> > (input1.location, input2.location, temp1_mul, modulus_, base_Bsk_,
    m_tilde_, inv_prod_q_mod_m_tilde_, inv_m_tilde_mod_Bsk_, prod_q_mod_Bsk_, base_change_matrix_Bsk_, base_change_matrix_m_tilde_, inv_punctured_prod_mod_base_array_, n_power, decomp_mod_count_, bsk_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = q_Bsk_n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(temp1_mul, q_Bsk_merge_ntt_tables_ , q_Bsk_merge_modulus_,
            cfg_ntt, ((bsk_mod_count_ + decomp_mod_count_)*4), (bsk_mod_count_ + decomp_mod_count_));

    CrossMultiplication<< < dim3((n >> 8), (bsk_mod_count_ + decomp_mod_count_), 1), 256 >> > (temp1_mul, temp1_mul + (((bsk_mod_count_ + decomp_mod_count_)*2) * n), temp2_mul, q_Bsk_merge_modulus_, n_power, (bsk_mod_count_ + decomp_mod_count_));

    GPU_NTT_Inplace(temp2_mul, q_Bsk_merge_intt_tables_, q_Bsk_merge_modulus_,
            cfg_intt, (3 * (bsk_mod_count_ + decomp_mod_count_)), (bsk_mod_count_ + decomp_mod_count_));


    FastFloor2 << < dim3((n >> 8), 3, 1), 256 >> >(temp2_mul, output.location, modulus_, base_Bsk_, plain_modulus_, inv_punctured_prod_mod_base_array_, 
        base_change_matrix_Bsk_, inv_prod_q_mod_Bsk_, inv_punctured_prod_mod_B_array_, base_change_matrix_q_, base_change_matrix_msk_, inv_prod_B_mod_m_sk_, prod_B_mod_q_, n_power, decomp_mod_count_, bsk_mod_count_);


}

__host__ void HEOperator::multiply(Ciphertext &input1, Ciphertext &input2, Ciphertext &output, HEStream stream){

    FastConvertion << < dim3((n >> 8), 4, 1), 256, 0, stream.stream >> > (input1.location, input2.location, stream.temp1_mul, modulus_, base_Bsk_,
    m_tilde_, inv_prod_q_mod_m_tilde_, inv_m_tilde_mod_Bsk_, prod_q_mod_Bsk_, base_change_matrix_Bsk_, base_change_matrix_m_tilde_, inv_punctured_prod_mod_base_array_, n_power, decomp_mod_count_, bsk_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = q_Bsk_n_inverse_,
        .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp1_mul, q_Bsk_merge_ntt_tables_ , q_Bsk_merge_modulus_,
            cfg_ntt, ((bsk_mod_count_ + decomp_mod_count_)*4), (bsk_mod_count_ + decomp_mod_count_));

    CrossMultiplication<< < dim3((n >> 8), (bsk_mod_count_ + decomp_mod_count_), 1), 256, 0, stream.stream >> > (stream.temp1_mul, stream.temp1_mul + (((bsk_mod_count_ + decomp_mod_count_)*2) * n), stream.temp2_mul, q_Bsk_merge_modulus_, n_power, (bsk_mod_count_ + decomp_mod_count_));

    GPU_NTT_Inplace(stream.temp2_mul, q_Bsk_merge_intt_tables_, q_Bsk_merge_modulus_,
            cfg_intt, (3 * (bsk_mod_count_ + decomp_mod_count_)), (bsk_mod_count_ + decomp_mod_count_));


    FastFloor2 << < dim3((n >> 8), 3, 1), 256, 0, stream.stream >> >(stream.temp2_mul, output.location, modulus_, base_Bsk_, plain_modulus_, inv_punctured_prod_mod_base_array_, 
        base_change_matrix_Bsk_, inv_prod_q_mod_Bsk_, inv_punctured_prod_mod_B_array_, base_change_matrix_q_, base_change_matrix_msk_, inv_prod_B_mod_m_sk_, prod_B_mod_q_, n_power, decomp_mod_count_, bsk_mod_count_);

}


__host__ void HEOperator::multiply_inplace(Ciphertext &input1, Ciphertext &input2){

    multiply(input1, input2, input1);

}
    
__host__ void HEOperator::multiply_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream){

    multiply(input1, input2, input1, stream);

}


__host__ void HEOperator::multiply_plain(Ciphertext &input1, Plaintext &input2, Ciphertext &output){

    Threshold_Kernel << < dim3((n >> 8), decomp_mod_count_, 1), 256 >> > (input2.location, temp1_plain_mul, modulus_, upper_halfincrement_, upper_threshold_, n_power, decomp_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(temp1_plain_mul, ntt_table_ , modulus_,
            cfg_ntt, decomp_mod_count_, decomp_mod_count_);

    GPU_NTT(input1.location, output.location, ntt_table_ , modulus_,
            cfg_ntt, 2 * decomp_mod_count_, decomp_mod_count_);

    CipherPlain_Kernel << < dim3((n >> 8), decomp_mod_count_, 2), 256 >> > (output.location, temp1_plain_mul, output.location, modulus_, n_power, decomp_mod_count_);

    GPU_NTT_Inplace(output.location, intt_table_ , modulus_,
            cfg_intt, 2 * decomp_mod_count_, decomp_mod_count_);

}

__host__ void HEOperator::multiply_plain(Ciphertext &input1, Plaintext &input2, Ciphertext &output, HEStream stream){

    Threshold_Kernel << < dim3((n >> 8), decomp_mod_count_, 1), 256, 0, stream.stream >> > (input2.location, stream.temp1_plain_mul, modulus_, upper_halfincrement_, upper_threshold_, n_power, decomp_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp1_plain_mul, ntt_table_ , modulus_,
            cfg_ntt, decomp_mod_count_, decomp_mod_count_);

    GPU_NTT(input1.location, output.location, ntt_table_ , modulus_,
            cfg_ntt, 2 * decomp_mod_count_, decomp_mod_count_);

    CipherPlain_Kernel << < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> > (output.location, stream.temp1_plain_mul, output.location, modulus_, n_power, decomp_mod_count_);

    GPU_NTT_Inplace(output.location, intt_table_ , modulus_,
            cfg_intt, 2 * decomp_mod_count_, decomp_mod_count_);

}

__host__ void HEOperator::multiply_plain_inplace(Ciphertext &input1, Plaintext &input2){

    multiply_plain(input1, input2, input1);

}

__host__ void HEOperator::multiply_plain_inplace(Ciphertext &input1, Plaintext &input2, HEStream stream){

    multiply_plain(input1, input2, input1, stream);
    
}

__host__ void HEOperator::relinearize_inplace(Ciphertext &input1, const Relinkey &relin_key){

    CipherBroadcast2<< < dim3((n >> 8), decomp_mod_count_, 1), 256 >> >(input1.location + (decomp_mod_count_ << (n_power + 1)), temp1_relin, modulus_, n_power, rns_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(temp1_relin, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count_ * rns_mod_count_, rns_mod_count_);

    MultiplyAcc<< < dim3((n >> 8), rns_mod_count_, 1), 256 >> >(temp1_relin, relin_key.location, temp2_relin, modulus_, n_power, decomp_mod_count_);

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(temp2_relin, intt_table_, modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

    DivideRoundLastq<< < dim3((n >> 8), decomp_mod_count_, 2), 256 >> >(temp2_relin, input1.location, input1.location, modulus_, half_, half_mod_, last_q_modinv_, n_power, decomp_mod_count_);

}

__host__ void HEOperator::relinearize_inplace(Ciphertext &input1, const Relinkey &relin_key, HEStream stream){

    CipherBroadcast2<< < dim3((n >> 8), decomp_mod_count_, 1), 256, 0, stream.stream >> >(input1.location + (decomp_mod_count_ << (n_power + 1)), stream.temp1_relin, modulus_, n_power, rns_mod_count_);

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp1_relin, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count_ * rns_mod_count_, rns_mod_count_);

    MultiplyAcc<< < dim3((n >> 8), rns_mod_count_, 1), 256, 0, stream.stream >> >(stream.temp1_relin, relin_key.location, stream.temp2_relin, modulus_, n_power, decomp_mod_count_);

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = stream.stream};

    GPU_NTT_Inplace(stream.temp2_relin, intt_table_, modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

    DivideRoundLastq<< < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> >(stream.temp2_relin, input1.location, input1.location, modulus_, half_, half_mod_, last_q_modinv_, n_power, decomp_mod_count_);

}


__host__ void HEOperator::rotate(Ciphertext &input1, Ciphertext &output, Galoiskey &galois_key, int shift){

    Data** GaloisKey;
    int* galoiselt;
    // positive or negative
    if (shift > 0) {
        GaloisKey = galois_key.positive_location;
        galoiselt = galois_key.galois_elt_pos;
    }
    else {
        GaloisKey = galois_key.negative_location;
        galoiselt = galois_key.galois_elt_neg;
    }

    int shift_num = abs(shift);
    while (shift_num != 0) {

        int power = int(log2(shift_num));
        int power_2 = pow(2, power);
        shift_num = shift_num - power_2;

        apply_galois<< < dim3((n >> 8), decomp_mod_count_, 2), 256 >> >(input1.location, temp0_rotation, temp1_rotation, modulus_, galoiselt[power], n_power, decomp_mod_count_);

        ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

        GPU_NTT_Inplace(temp1_rotation, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count_ * rns_mod_count_, rns_mod_count_);

        MultiplyAcc<< < dim3((n >> 8), rns_mod_count_, 1), 256 >> >(temp1_rotation, GaloisKey[power], temp2_rotation, modulus_, n_power, decomp_mod_count_);

        ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, intt_table_, modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

        DivideRoundLastq<< < dim3((n >> 8), decomp_mod_count_, 2), 256 >> >(temp2_rotation, temp0_rotation, output.location, modulus_, half_, half_mod_, last_q_modinv_, n_power, decomp_mod_count_);

    }

}
    

__host__ void HEOperator::rotate(Ciphertext &input1, Ciphertext &output, Galoiskey &galois_key, int shift, HEStream stream){

        Data** GaloisKey;
    int* galoiselt;
    // positive or negative
    if (shift > 0) {
        GaloisKey = galois_key.positive_location;
        galoiselt = galois_key.galois_elt_pos;
    }
    else {
        GaloisKey = galois_key.negative_location;
        galoiselt = galois_key.galois_elt_neg;
    }

    int shift_num = abs(shift);
    while (shift_num != 0) {

        int power = int(log2(shift_num));
        int power_2 = pow(2, power);
        shift_num = shift_num - power_2;

        apply_galois<< < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> >(input1.location, stream.temp0_rotation, stream.temp1_rotation, modulus_, galoiselt[power], n_power, decomp_mod_count_);

        ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_ , modulus_, cfg_ntt, decomp_mod_count_ * rns_mod_count_, rns_mod_count_);

        MultiplyAcc<< < dim3((n >> 8), rns_mod_count_, 1), 256, 0, stream.stream >> >(stream.temp1_rotation, GaloisKey[power], stream.temp2_rotation, modulus_, n_power, decomp_mod_count_);

        ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_inverse_,
        .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, intt_table_, modulus_, cfg_intt, 2 * rns_mod_count_, rns_mod_count_);

        DivideRoundLastq<< < dim3((n >> 8), decomp_mod_count_, 2), 256, 0, stream.stream >> >(stream.temp2_rotation, stream.temp0_rotation, output.location, modulus_, half_, half_mod_, last_q_modinv_, n_power, decomp_mod_count_);

    }


}