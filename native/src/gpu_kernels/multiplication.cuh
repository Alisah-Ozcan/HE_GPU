// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "../common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "../context.cuh"

#ifndef HE_MULTIPLICATION_H
#define HE_MULTIPLICATION_H

//__global__ void FastConvertion(Data* in1, Data* in2, Data* out1, Data* out2, Modulus* ibase, Modulus* obase, Modulus m_tilde, Data* base_change_matrix_Bsk, Data* base_change_matrix_m_tilde, Data* inv_punctured_prod_mod_base_array, int n_power, int ibase_size, int obase_size);


__host__ void HEMultiplication(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context);

__global__ void CrossMultiplication(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power, int decomp_size);

__global__ void FastConvertion(Data* in1, Data* in2, Data* out1, Modulus* ibase, Modulus* obase, Modulus m_tilde, Data inv_prod_q_mod_m_tilde, Data* inv_m_tilde_mod_Bsk, Data* prod_q_mod_Bsk,
 Data* base_change_matrix_Bsk, Data* base_change_matrix_m_tilde, Data* inv_punctured_prod_mod_base_array, int n_power, int ibase_size, int obase_size);

__global__ void FastFloor(Data* in_baseq, Data* in_baseBsk, Data* out1, Modulus* ibase, Modulus* obase, Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array, 
Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk,  int n_power, int ibase_size, int obase_size);

__global__ void FastFloor2(Data* in_baseq_Bsk, Data* out1, Modulus* ibase, Modulus* obase, Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array, 
Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk, Data* inv_punctured_prod_mod_B_array, Data* base_change_matrix_q, Data* base_change_matrix_msk, Data inv_prod_B_mod_m_sk, Data* prod_B_mod_q,
  int n_power, int ibase_size, int obase_size);


__global__ void Threshold_Kernel(Data* plain_in, Data* output, Modulus* modulus, Data* plain_upper_half_increment,
	Data plain_upper_half_threshold, int n_power, int decomp_size);
  
__global__ void CipherPlain_Kernel(Data* cipher, Data* plain_in, Data* output, Modulus* modulus, int n_power, int decomp_size);

__host__ void HEPlainMultiplication(Ciphertext input1, Plaintext input2, Ciphertext output, Parameters context);

#endif // HE_MULTIPLICATION_H