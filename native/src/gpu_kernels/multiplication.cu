// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "multiplication.cuh"

//__global__ void FastConvertion(Data* in1, Data* in2, Data* out1, Data* out2, Modulus* ibase, Modulus* obase, Modulus m_tilde, Data* base_change_matrix_Bsk, Data* base_change_matrix_m_tilde, Data* inv_punctured_prod_mod_base_array, int n_power, int ibase_size, int obase_size) {
__global__ void FastConvertion(Data* in1, Data* in2, Data* out1, Modulus* ibase, Modulus* obase, Modulus m_tilde, Data inv_prod_q_mod_m_tilde, Data* inv_m_tilde_mod_Bsk, Data* prod_q_mod_Bsk,
 Data* base_change_matrix_Bsk, Data* base_change_matrix_m_tilde, Data* inv_punctured_prod_mod_base_array, int n_power, int ibase_size, int obase_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // cipher count * 2 // for input
     
    int location = idx + (((idy % 2) * ibase_size) << n_power); // ibase_size = decomp_modulus_count
    Data* input = ((idy >> 1) == 0) ? in1 : in2;

    Data temp[20];
    Data temp_[20];

    // taking input from global and mult with m_tilde
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        temp_[i] = input[location + (i << n_power)];
        temp[i] = VALUE_GPU::mult(temp_[i], m_tilde.value, ibase[i]);
        temp[i] = VALUE_GPU::mult(temp[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
    }
    
    // for Bsk
    Data temp2[20];
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        temp2[i] = 0;
#pragma unroll
        for(int j = 0; j < ibase_size; j++){
            Data mult = VALUE_GPU::mult(temp[j], base_change_matrix_Bsk[j + (i * ibase_size)], obase[i]);
            temp2[i] = VALUE_GPU::add(temp2[i], mult, obase[i]);
        }
    }

    // for m_tilde
    temp2[obase_size] = 0;
#pragma unroll
    for(int j = 0; j < ibase_size; j++){
        Data mult = VALUE_GPU::mult(temp[j], base_change_matrix_m_tilde[j], m_tilde);
        temp2[obase_size] = VALUE_GPU::add(temp2[obase_size], mult, m_tilde);
    }


    // sm_mrq
    Data m_tilde_div_2 = m_tilde.value >> 1;
    Data r_m_tilde = VALUE_GPU::mult(temp2[obase_size], inv_prod_q_mod_m_tilde, m_tilde);
    r_m_tilde = m_tilde.value - r_m_tilde;

#pragma unroll
    for(int i = 0; i < obase_size; i++){
        
        Data temp3 = r_m_tilde; 
        if(temp3 >= m_tilde_div_2){
            temp3 = obase[i].value - m_tilde.value;
            temp3 = VALUE_GPU::add(temp3, r_m_tilde, obase[i]);
        }
       
        temp3 = VALUE_GPU::mult(temp3, prod_q_mod_Bsk[i], obase[i]);
        temp3 = VALUE_GPU::add(temp2[i], temp3, obase[i]);
        temp2[i] = VALUE_GPU::mult(temp3, inv_m_tilde_mod_Bsk[i], obase[i]);

    }

    int location2 = idx + ((idy * (obase_size + ibase_size)) << n_power);
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        out1[location2 + (i << n_power)] = temp_[i];
    }
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        out1[location2 + ((i + ibase_size) << n_power)] = temp2[i];
    }

}

__global__ void CrossMultiplication(Data* in1, Data* in2, Data* out, Modulus* modulus, int n_power, int decomp_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // decomp size + bsk size

    int location = idx + (idy << n_power); 

    Data ct0_0 = in1[location];
    Data ct0_1 = in1[location + (decomp_size << n_power)];

    Data ct1_0 = in2[location];
    Data ct1_1 = in2[location + (decomp_size << n_power)];


    Data out_0 = VALUE_GPU::mult(ct0_0, ct1_0, modulus[idy]);
    Data out_1_0 = VALUE_GPU::mult(ct0_0, ct1_1, modulus[idy]);
    Data out_1_1 = VALUE_GPU::mult(ct0_1, ct1_0, modulus[idy]);
    Data out_2 = VALUE_GPU::mult(ct0_1, ct1_1, modulus[idy]);
    Data out_1 = VALUE_GPU::add(out_1_0, out_1_1, modulus[idy]);


    out[location] = out_0;
    out[location + (decomp_size << n_power)] = out_1;
    out[location + (decomp_size << (n_power + 1))] = out_2;


}

//__global__ void FastFloor(Data* in1, Data* in2, Data* out1, Modulus* ibase, Modulus* obase, Modulus m_tilde, Data inv_prod_q_mod_m_tilde, Data* inv_m_tilde_mod_Bsk, Data* prod_q_mod_Bsk,
// Data* base_change_matrix_Bsk, Data* base_change_matrix_m_tilde, Data* inv_punctured_prod_mod_base_array, int n_power, int ibase_size, int obase_size)
__global__ void FastFloor(Data* in_baseq, Data* in_baseBsk, Data* out1, Modulus* ibase, Modulus* obase, Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array, 
Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk,  int n_power, int ibase_size, int obase_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // 3

    int location_q = idx + ((idy * ibase_size) << n_power); // ibase_size = decomp_modulus_count
    int location_Bsk = idx + ((idy * obase_size) << n_power); // ibase_size = decomp_modulus_count


    Data reg_q[20];
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        reg_q[i] = VALUE_GPU::mult(in_baseq[location_q + (i << n_power)], plain_modulus.value, ibase[i]);
        reg_q[i] = VALUE_GPU::mult(reg_q[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
    }

    Data reg_Bsk[20];
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        reg_Bsk[i] = VALUE_GPU::mult(in_baseBsk[location_Bsk + (i << n_power)], plain_modulus.value, obase[i]);
    }

    // for Bsk
    Data temp[20];
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        temp[i] = 0;
        for(int j = 0; j < ibase_size; j++){
            Data mult = VALUE_GPU::mult(reg_q[j], base_change_matrix_Bsk[j + (i * ibase_size)], obase[i]);
            temp[i] = VALUE_GPU::add(temp[i], mult, obase[i]);
        }
    }

#pragma unroll
    for(int i = 0; i < obase_size; i++){
        Data temp2 =  VALUE_GPU::sub(obase[i].value, temp[i], obase[i]);
        temp2 = VALUE_GPU::add(temp2, reg_Bsk[i], obase[i]);
        reg_Bsk[i] = VALUE_GPU::mult(temp2, inv_prod_q_mod_Bsk[i], obase[i]);
    }
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        out1[location_Bsk + (i << n_power)] = reg_Bsk[i];
    }
    

}

__global__ void FastFloor2(Data* in_baseq_Bsk, Data* out1, Modulus* ibase, Modulus* obase, Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array, 
Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk, Data* inv_punctured_prod_mod_B_array, Data* base_change_matrix_q,  Data* base_change_matrix_msk, Data inv_prod_B_mod_m_sk, Data* prod_B_mod_q,
  int n_power, int ibase_size, int obase_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
    int idy = blockIdx.y; // 3

    int location_q = idx + ((idy * (ibase_size + obase_size)) << n_power); // ibase_size = decomp_modulus_count
    int location_Bsk = idx + ((idy * (ibase_size + obase_size)) << n_power) + (ibase_size << n_power); // ibase_size = decomp_modulus_count


    Data reg_q[20];
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        reg_q[i] = VALUE_GPU::mult(in_baseq_Bsk[location_q + (i << n_power)], plain_modulus.value, ibase[i]);
        reg_q[i] = VALUE_GPU::mult(reg_q[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
    }

    Data reg_Bsk[20];
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        reg_Bsk[i] = VALUE_GPU::mult(in_baseq_Bsk[location_Bsk + (i << n_power)], plain_modulus.value, obase[i]);
    }

    // for Bsk
    Data temp[20];
#pragma unroll
    for(int i = 0; i < obase_size; i++){
        temp[i] = 0;
        for(int j = 0; j < ibase_size; j++){
            Data mult = VALUE_GPU::mult(reg_q[j], base_change_matrix_Bsk[j + (i * ibase_size)], obase[i]);
            temp[i] = VALUE_GPU::add(temp[i], mult, obase[i]);
        }
    }

#pragma unroll
    for(int i = 0; i < obase_size; i++){
        Data temp2 =  VALUE_GPU::sub(obase[i].value, temp[i], obase[i]);
        temp2 = VALUE_GPU::add(temp2, reg_Bsk[i], obase[i]);
        reg_Bsk[i] = VALUE_GPU::mult(temp2, inv_prod_q_mod_Bsk[i], obase[i]);
    }


    /*
    for(int i = 0; i < obase_size; i++){
        out1[location_Bsk + (i << n_power)] = reg_Bsk[i];
    }
    */
    // //////////////////////////////////////////////////////////////////////////////////////
    Data temp3[20];
#pragma unroll
    for(int i = 0; i < obase_size - 1; i++){ // only B bases
        temp3[i] = VALUE_GPU::mult(reg_Bsk[i], inv_punctured_prod_mod_B_array[i], obase[i]);
    }

    Data temp4[20];
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        temp4[i] = 0;
#pragma unroll
        for(int j = 0; j < obase_size - 1; j++){
            Data mult = VALUE_GPU::mult(temp3[j], base_change_matrix_q[j + (i * (obase_size - 1))], ibase[i]);
            temp4[i] = VALUE_GPU::add(temp4[i], mult, ibase[i]);
        }
    }

    // for m_sk
    temp4[ibase_size] = 0;
#pragma unroll
    for(int j = 0; j < obase_size - 1; j++){
        Data mult = VALUE_GPU::mult(temp3[j], base_change_matrix_msk[j], obase[obase_size - 1]);
        temp4[ibase_size] = VALUE_GPU::add(temp4[ibase_size], mult, obase[obase_size - 1]);
    }
    
    Data alpha_sk = VALUE_GPU::sub(obase[obase_size - 1].value, reg_Bsk[obase_size - 1], obase[obase_size - 1]);
    alpha_sk = VALUE_GPU::add(alpha_sk, temp4[ibase_size], obase[obase_size - 1]);
    alpha_sk = VALUE_GPU::mult(alpha_sk, inv_prod_B_mod_m_sk, obase[obase_size - 1]);
    

    Data m_sk_div_2 = obase[obase_size - 1].value >> 1;

   
    /*
    if((idx == 0) && (idy == 0)){
        printf("==> %llu \n", temp4[0]);
        printf("==> %llu \n", temp4[1]);
        printf("==> %llu \n", temp4[2]);
        printf("==> %llu \n", temp4[3]);
    }
    */

#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        /*
        if(alpha_sk > m_sk_div_2){
            Data inner = VALUE_GPU::sub(obase[obase_size - 1].value, alpha_sk, ibase[i]);
            inner = VALUE_GPU::mult(inner, prod_B_mod_q[i], ibase[i]);
            temp4[i] = VALUE_GPU::add(temp4[i], inner, ibase[i]);
        }
        else{
            Data inner = VALUE_GPU::sub(ibase[i].value, prod_B_mod_q[i], ibase[i]);
            inner = VALUE_GPU::mult(inner, alpha_sk, ibase[i]);
            temp4[i] = VALUE_GPU::add(temp4[i], inner, ibase[i]);
        }
        */

        Data one = 1;
        Data obase_ = VALUE_GPU::mult(obase[obase_size - 1].value, one, ibase[i]);
        Data temp4_ = VALUE_GPU::mult(temp4[i], one, ibase[i]);
        if(alpha_sk > m_sk_div_2){
            Data inner = VALUE_GPU::sub(obase_, alpha_sk, ibase[i]);
            inner = VALUE_GPU::mult(inner, prod_B_mod_q[i], ibase[i]);
            temp4[i] = VALUE_GPU::add(temp4_, inner, ibase[i]);
        }
        else{
            Data inner = VALUE_GPU::sub(ibase[i].value, prod_B_mod_q[i], ibase[i]);
            inner = VALUE_GPU::mult(inner, alpha_sk, ibase[i]);
            temp4[i] = VALUE_GPU::add(temp4_, inner, ibase[i]);
        }
       
    }
    
    int location_out = idx + ((idy * ibase_size) << n_power); // ibase_size = decomp_modulus_count
#pragma unroll
    for(int i = 0; i < ibase_size; i++){
        out1[location_out + (i << n_power)] = temp4[i];
    }

}


__host__ void HEMultiplication(Ciphertext input1, Ciphertext input2, Ciphertext output, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;
    unsigned bsk_mod_count = context.bsk_modulus; 

    FastConvertion << < dim3((context.n >> 8), 4, 1), 256 >> > (input1.location, input2.location, context.temp1_mul, context.modulus_, context.base_Bsk_,
    context.m_tilde_, context.inv_prod_q_mod_m_tilde_, context.inv_m_tilde_mod_Bsk_, context.prod_q_mod_Bsk_, context.base_change_matrix_Bsk_, context.base_change_matrix_m_tilde_, context.inv_punctured_prod_mod_base_array_, context.n_power, decomp_mod_count, bsk_mod_count);

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.q_Bsk_n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(context.temp1_mul, context.q_Bsk_merge_ntt_tables_ , context.q_Bsk_merge_modulus_,
            cfg_ntt, ((bsk_mod_count + decomp_mod_count)*4), (bsk_mod_count + decomp_mod_count));

    CrossMultiplication<< < dim3((context.n >> 8), (bsk_mod_count + decomp_mod_count), 1), 256 >> > (context.temp1_mul, context.temp1_mul + (((bsk_mod_count + decomp_mod_count)*2) * context.n), context.temp2_mul, context.q_Bsk_merge_modulus_, context.n_power, (bsk_mod_count + decomp_mod_count));

    GPU_NTT_Inplace(context.temp2_mul, context.q_Bsk_merge_intt_tables_, context.q_Bsk_merge_modulus_,
            cfg_intt, (3 * (bsk_mod_count + decomp_mod_count)), (bsk_mod_count + decomp_mod_count));


    FastFloor2 << < dim3((context.n >> 8), 3, 1), 256 >> >(context.temp2_mul, output.location, context.modulus_, context.base_Bsk_, context.plain_modulus_, context.inv_punctured_prod_mod_base_array_, 
        context.base_change_matrix_Bsk_, context.inv_prod_q_mod_Bsk_, context.inv_punctured_prod_mod_B_array_, context.base_change_matrix_q_, context.base_change_matrix_msk_, context.inv_prod_B_mod_m_sk_, context.prod_B_mod_q_, context.n_power, decomp_mod_count, bsk_mod_count);

}





__global__ void Threshold_Kernel(Data* plain_in, Data* output, Modulus* modulus, Data* plain_upper_half_increment,
	Data plain_upper_half_threshold, int n_power, int decomp_size)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
	int block_y = blockIdx.y; // decomp_mod

	Data plain_reg = plain_in[idx];

	if (plain_reg >= plain_upper_half_threshold) {
		output[idx + (block_y << n_power)] = VALUE_GPU::add(plain_reg, plain_upper_half_increment[block_y], modulus[block_y]);//plain_reg + plain_upper_half_increment[block_y];
	}
	else {
		output[idx + (block_y << n_power)] = plain_reg;
	}

}

__global__ void CipherPlain_Kernel(Data* cipher, Data* plain_in, Data* output, Modulus* modulus, int n_power, int decomp_size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
	int block_y = blockIdx.y; // decomp_mod
    int block_z = blockIdx.z; // cipher size

    int index1 = idx + (block_y << n_power);
    int index2 = index1 + ((decomp_size << n_power) * block_z);


	output[index2] = VALUE_GPU::mult(cipher[index2], plain_in[index1], modulus[block_y]);

}

__host__ void HEPlainMultiplication(Ciphertext input1, Plaintext input2, Ciphertext output, Parameters context){

    unsigned decomp_mod_count = context.coeff_modulus - 1;
    unsigned bsk_mod_count = context.bsk_modulus; 

    Threshold_Kernel << < dim3((context.n >> 8), decomp_mod_count, 1), 256 >> > (input2.location, context.temp1_plain_mul, context.modulus_, context.upper_halfincrement_, context.upper_threshold_, context.n_power, decomp_mod_count);

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    ntt_configuration cfg_intt = {
        .n_power = context.n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = context.n_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(context.temp1_plain_mul, context.ntt_table_ , context.modulus_,
            cfg_ntt, decomp_mod_count, decomp_mod_count);

    GPU_NTT(input1.location, output.location, context.ntt_table_ , context.modulus_,
            cfg_ntt, 2 * decomp_mod_count, decomp_mod_count);

    CipherPlain_Kernel << < dim3((context.n >> 8), decomp_mod_count, 2), 256 >> > (output.location, context.temp1_plain_mul, output.location, context.modulus_, context.n_power, decomp_mod_count);

    GPU_NTT_Inplace(output.location, context.intt_table_ , context.modulus_,
            cfg_intt, 2 * decomp_mod_count, decomp_mod_count);

}