// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "common.cuh"
#include "contextpool.cuh"
#include "gpu_kernels/nttparameters.cuh"
#include <string>
#include <iostream>
#include <memory>
#include <vector>

#ifndef CONTEXT_H
#define CONTEXT_H


//////////////////////////////////////////////////////////////////////////////////

class Parameters {

public:

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

    Data* factor_;

    

    Modulus* plain_modulus2_;
    Ninverse* n_plain_inverse_;
    Root* plain_ntt_tables_;
    Root* plain_intt_tables_;

    Modulus gamma_;
    Data* coeeff_div_plainmod_;
    Data Q_mod_t_;

    Data upper_threshold_;
    Data* upper_halfincrement_;

    ///
    Data* Qi_t_;
    Data* Qi_gamma_;
    Data* Qi_inverse_;

    Data mulq_inv_t_;
    Data mulq_inv_gamma_;
    Data inv_gamma_;

    int n;
    int total_bits;
    int coeff_modulus;
    int bsk_modulus;
    int n_power;
    std::string scheme;
    PrimePool::security_level sec;


    //////////////////////////////////////////////////////////////////
    Data* encoding_location_;
    
    // for default stream
    Data *temp1_enc;
    Data *temp2_enc;
    
    Data* temp1_mul;
    Data* temp2_mul;
    Data* temp3_mul;

    Data* temp1_relin;
    Data* temp2_relin;

    Data* temp0_rotation;
    Data* temp1_rotation;
    Data* temp2_rotation;

    Data* temp1_plain_mul;



    //////////////////////////////////////////////////////////////////

    __host__ Parameters(std::string scheme_type, int poly_degree, PrimePool::security_level sec_level = PrimePool::security_level::HES_128);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class HEStream
{
public:

    cudaStream_t stream;

    int ring_size;
    int coeff_modulus_count;
    int bsk_modulus_count;

    Data *temp1_enc;
    Data *temp2_enc;
    
    Data* temp1_mul;
    Data* temp2_mul;
    Data* temp3_mul;

    Data* temp1_relin;
    Data* temp2_relin;

    Data* temp0_rotation;
    Data* temp1_rotation;
    Data* temp2_rotation;

    Data* temp1_plain_mul;
    
    __host__ HEStream(Parameters context);

    //__host__ void HEStreamSynchronize();
    //__host__ void HEStreamWait();

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Ciphertext
{
public:

    Data* location;

    int ring_size;
    int coeff_modulus_count;
    int cipher_size;
    

    __host__ Ciphertext();

    __host__ Ciphertext(Parameters context);

    __host__ Ciphertext(Parameters context, HEStream steram);

    __host__ Ciphertext(Data* cipher, Parameters context);

    __host__ Ciphertext(Data* cipher, Parameters context, HEStream stream);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Message
{
public:

    Data* location;

    int ring_size;

    __host__ Message();

    __host__ Message(Parameters context);

    __host__ Message(Parameters context, HEStream stream);

    __host__ Message(Data* message, Parameters context);

    __host__ Message(const std::vector<uint64_t> &message, Parameters context);
    
    __host__ Message(Data* message, Parameters context, HEStream stream);

    __host__ Message(Data* message, int size, Parameters context);

    __host__ Message(const std::vector<uint64_t> &message, int size, Parameters context);

    __host__ Message(Data* message, int size, Parameters context, HEStream stream);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Plaintext
{
public:

    Data* location;

    int ring_size;

    __host__ Plaintext();

    __host__ Plaintext(Parameters context);

    __host__ Plaintext(Parameters context, HEStream stream);

    __host__ Plaintext(Message message, Parameters context);

    __host__ Plaintext(Message message, Parameters context, HEStream stream);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Relinkey
{
public:

    Data* location;
    Data* e_a;

    int ring_size;
    int coeff_modulus_count;

    __host__ Relinkey();

    __host__ Relinkey(Parameters context);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

#define MAX_SHIFT 8

class Galoiskey
{
public:

    int* galois_elt_pos;
    int* galois_elt_neg;

    Data* positive_location[MAX_SHIFT];
    Data* negative_location[MAX_SHIFT];
    Data* e_a;

    int ring_size;
    int coeff_modulus_count;

    __host__ Galoiskey();

    __host__ Galoiskey(Parameters context);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Secretkey
{
public:

    Data* location;

    int ring_size;
    int coeff_modulus_count;

    __host__ Secretkey();

    __host__ Secretkey(Parameters context);

    __host__ void kill();

};

//////////////////////////////////////////////////////////////////////////////////

class Publickey
{
public:

    Data* location;

    int ring_size;
    int coeff_modulus_count;

    __host__ Publickey();

    __host__ Publickey(Parameters context);

    __host__ void kill();

};

#endif // CONTEXT_H



