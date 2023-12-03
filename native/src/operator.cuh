// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "gpu_kernels/addition.cuh"
#include "gpu_kernels/multiplication.cuh"
#include "gpu_kernels/switchkey.cuh"

#ifndef OPERATOR_H
#define OPERATOR_H

class HEOperator
{
public:

    __host__ HEOperator(const Parameters &context);

    __host__ void add(Ciphertext &input1, Ciphertext &input2, Ciphertext &output);
    __host__ void add(Ciphertext &input1, Ciphertext &input2, Ciphertext &output, HEStream stream);

    __host__ void add_inplace(Ciphertext &input1, Ciphertext &input2);
    __host__ void add_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream);

    __host__ void sub(Ciphertext &input1, Ciphertext &input2, Ciphertext &output);
    __host__ void sub(Ciphertext &input1, Ciphertext &input2, Ciphertext &output, HEStream stream);

    __host__ void sub_inplace(Ciphertext &input1, Ciphertext &input2);
    __host__ void sub_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream);

    __host__ void multiply(Ciphertext &input1, Ciphertext &input2, Ciphertext &output);
    __host__ void multiply(Ciphertext &input1, Ciphertext &input2, Ciphertext &output, HEStream stream);

    __host__ void multiply_inplace(Ciphertext &input1, Ciphertext &input2);
    __host__ void multiply_inplace(Ciphertext &input1, Ciphertext &input2, HEStream stream);

    __host__ void multiply_plain(Ciphertext &input1, Plaintext &input2, Ciphertext &output);
    __host__ void multiply_plain(Ciphertext &input1, Plaintext &input2, Ciphertext &output, HEStream stream);

    __host__ void multiply_plain_inplace(Ciphertext &input1, Plaintext &input2);
    __host__ void multiply_plain_inplace(Ciphertext &input1, Plaintext &input2, HEStream stream);

    __host__ void relinearize_inplace(Ciphertext &input1, const Relinkey &relin_key);
    __host__ void relinearize_inplace(Ciphertext &input1, const Relinkey &relin_key, HEStream stream);

    __host__ void rotate(Ciphertext &input1, Ciphertext &output, Galoiskey &galois_key, int shift);
    __host__ void rotate(Ciphertext &input1, Ciphertext &output, Galoiskey &galois_key, int shift, HEStream stream);

    void kill();

private:

    int n;

    int n_power;

    int rns_mod_count_;

    int decomp_mod_count_;

    int bsk_mod_count_;

    Modulus* modulus_;
    Root* ntt_table_;
    Root* intt_table_;
    Ninverse* n_inverse_;
    Data* last_q_modinv_;

    Modulus* base_Bsk_;
    Root* bsk_ntt_tables_;
    Root* bsk_intt_tables_;
    Ninverse* bsk_n_inverse_;

    Modulus m_tilde_;
    Data* base_change_matrix_Bsk_;
    Data* inv_punctured_prod_mod_base_array_;
    Data* base_change_matrix_m_tilde_;

    Data inv_prod_q_mod_m_tilde_;
    Data* inv_m_tilde_mod_Bsk_;
    Data* prod_q_mod_Bsk_;

    Data* inv_prod_q_mod_Bsk_;

    Modulus plain_modulus_;

    Data* base_change_matrix_q_;
    Data* base_change_matrix_msk_;

    Data* inv_punctured_prod_mod_B_array_;
    Data inv_prod_B_mod_m_sk_;
    Data* prod_B_mod_q_;

    Modulus* q_Bsk_merge_modulus_;
    Root* q_Bsk_merge_ntt_tables_;
    Root* q_Bsk_merge_intt_tables_;
    Ninverse* q_Bsk_n_inverse_;

    Data half_;
    Data* half_mod_;

    Data upper_threshold_;
    Data* upper_halfincrement_;

    // Temp
    Data* temp1_mul;
    Data* temp2_mul;

    Data* temp1_relin;
    Data* temp2_relin;

    Data* temp0_rotation;
    Data* temp1_rotation;
    Data* temp2_rotation;

    Data* temp1_plain_mul;

};

#endif // OPERATOR_H
