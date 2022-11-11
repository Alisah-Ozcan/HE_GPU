#pragma once

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "NTT.cuh"
#include "NTTx.cuh"
#include "GPU_Context.cuh"



__global__ void Op1(unsigned long long* plaintext, unsigned long long* adjust_poly, unsigned long long* plain_upper_half_increment,
	unsigned long long* plain_upper_half_threshold, int coeff_count, int q_count) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int loc1 = idx % coeff_count;
	int loc2 = int(idx / coeff_count);

	unsigned long long plaintext_reg = plaintext[loc1];

	if (plaintext_reg >= plain_upper_half_threshold[0]) {
		adjust_poly[idx] = plaintext_reg + plain_upper_half_increment[loc2];
	}
	else {
		adjust_poly[idx] = plaintext_reg;
	}

}

__global__ void Op2(unsigned long long* ciphertext, unsigned long long* adjust_poly, unsigned long long* result, unsigned long long* q,
	unsigned long long* mu, unsigned long long* q_bitlength, int coeff_count, int q_count) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int loc2 = idx % (coeff_count * (q_count - 1));
	int loc1 = int(loc2 / coeff_count);

	uint128_t cipher_reg = ciphertext[idx];

	mul64(cipher_reg.low, adjust_poly[loc2], cipher_reg);
	singleBarrett(cipher_reg, q[loc1], mu[loc1], q_bitlength[loc1]);
	result[idx] = cipher_reg.low;

}


// CLASSES VERSION
__host__ void GPU_Multiply_Plain(GPU_Ciphertext input_cipher, GPU_Plaintext input_plain, GPU_Ciphertext output_cipher, Lib_Parameters context)
{
	// Class Features
	unsigned long long* ciphertext = input_cipher.GPU_Location;
	unsigned long long* plaintext = input_plain.GPU_Location;
	unsigned long long* result = output_cipher.GPU_Location;
	unsigned long long* Temporary_Pool = context.adjust_poly;
	int n = input_cipher.ring_size;
	int coeff_modulus = input_cipher.coeff_mod_count + 1;

	// Parameters
	unsigned long long* q_device = context.Context_GPU; // Tık
	unsigned long long* mu_device = q_device + coeff_modulus; // Tık
	unsigned long long* q_bit_device = mu_device + coeff_modulus; // Tık
	unsigned long long* q_INTT_device = q_bit_device + coeff_modulus; // Tık
	unsigned long long* q_mu_INTT_device = q_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* q_bit_INTT_device = q_mu_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* ForwardPsi_device = q_bit_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* InversePsi_device = ForwardPsi_device + coeff_modulus * n; // Tık
	unsigned long long* DoubleInversePsi_device = InversePsi_device + coeff_modulus * n; // Tık
	unsigned long long* lastq_modinv_device = DoubleInversePsi_device + coeff_modulus * 2 * n; // Tık
	unsigned long long* half_device = lastq_modinv_device + (coeff_modulus - 1); // Tık
	unsigned long long* half_mod_device = half_device + 1; // Tık
	unsigned long long* INTT_inv_q = half_mod_device + (coeff_modulus - 1); // Tık
	unsigned long long* INTT_inv_double_q = INTT_inv_q + coeff_modulus; // Tık

	unsigned long long* plainmod_device = INTT_inv_double_q + coeff_modulus * 2;
	unsigned long long* plainpsi_device = plainmod_device + 1;
	unsigned long long* plain_ninverse = plainpsi_device + 1;
	unsigned long long* ForwardPlainPsi_device = plain_ninverse + 1;
	unsigned long long* InversePlainPsi_device = ForwardPlainPsi_device + n;
	unsigned long long* plain_upper_half_increment_device = InversePlainPsi_device + n;
	unsigned long long* plain_upper_half_threshold_device = plain_upper_half_increment_device + (coeff_modulus - 1);


	Op1 << < ((coeff_modulus - 1) * (n / 1024)), 1024 >> > (plaintext, Temporary_Pool, plain_upper_half_increment_device, plain_upper_half_threshold_device, n, coeff_modulus);

	Forward_NTT_Inplace(Temporary_Pool, q_device, mu_device, q_bit_device, n, ForwardPsi_device, (coeff_modulus - 1), (coeff_modulus - 1));

	Forward_NTTx(ciphertext, result, q_device, mu_device, q_bit_device, n, ForwardPsi_device, 2 * (coeff_modulus - 1), (coeff_modulus - 1));
	
	Op2 << < (2 * (coeff_modulus - 1) * (n / 1024)), 1024 >> > (result, Temporary_Pool, result, q_device, mu_device, q_bit_device, n, coeff_modulus);

	Inverse_NTT_Inplace(result, q_device, mu_device, q_bit_device, n, InversePsi_device, 2 * (coeff_modulus - 1), (coeff_modulus - 1), INTT_inv_q);

	
}



__host__ void GPU_Multiply_Plain_inplace(GPU_Ciphertext input_cipher, GPU_Plaintext input_plain, Lib_Parameters context)
{
	// Class Features
	unsigned long long* ciphertext = input_cipher.GPU_Location;
	unsigned long long* plaintext = input_plain.GPU_Location;
	unsigned long long* Temporary_Pool = context.adjust_poly;
	int n = input_cipher.ring_size;
	int coeff_modulus = input_cipher.coeff_mod_count + 1;

	// Parameters
	unsigned long long* q_device = context.Context_GPU; // Tık
	unsigned long long* mu_device = q_device + coeff_modulus; // Tık
	unsigned long long* q_bit_device = mu_device + coeff_modulus; // Tık
	unsigned long long* q_INTT_device = q_bit_device + coeff_modulus; // Tık
	unsigned long long* q_mu_INTT_device = q_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* q_bit_INTT_device = q_mu_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* ForwardPsi_device = q_bit_INTT_device + coeff_modulus * 2; // Tık
	unsigned long long* InversePsi_device = ForwardPsi_device + coeff_modulus * n; // Tık
	unsigned long long* DoubleInversePsi_device = InversePsi_device + coeff_modulus * n; // Tık
	unsigned long long* lastq_modinv_device = DoubleInversePsi_device + coeff_modulus * 2 * n; // Tık
	unsigned long long* half_device = lastq_modinv_device + (coeff_modulus - 1); // Tık
	unsigned long long* half_mod_device = half_device + 1; // Tık
	unsigned long long* INTT_inv_q = half_mod_device + (coeff_modulus - 1); // Tık
	unsigned long long* INTT_inv_double_q = INTT_inv_q + coeff_modulus; // Tık

	unsigned long long* plainmod_device = INTT_inv_double_q + coeff_modulus * 2;
	unsigned long long* plainpsi_device = plainmod_device + 1;
	unsigned long long* plain_ninverse = plainpsi_device + 1;
	unsigned long long* ForwardPlainPsi_device = plain_ninverse + 1;
	unsigned long long* InversePlainPsi_device = ForwardPlainPsi_device + n;
	unsigned long long* plain_upper_half_increment_device = InversePlainPsi_device + n;
	unsigned long long* plain_upper_half_threshold_device = plain_upper_half_increment_device + (coeff_modulus - 1);


	Op1 << < ((coeff_modulus - 1) * (n / 1024)), 1024 >> > (plaintext, Temporary_Pool, plain_upper_half_increment_device, plain_upper_half_threshold_device, n, coeff_modulus);

	Forward_NTT_Inplace(Temporary_Pool, q_device, mu_device, q_bit_device, n, ForwardPsi_device, (coeff_modulus - 1), (coeff_modulus - 1));

	Forward_NTT_Inplace(ciphertext, q_device, mu_device, q_bit_device, n, ForwardPsi_device, 2 * (coeff_modulus - 1), (coeff_modulus - 1));

	Op2 << < (2 * (coeff_modulus - 1) * (n / 1024)), 1024 >> > (ciphertext, Temporary_Pool, ciphertext, q_device, mu_device, q_bit_device, n, coeff_modulus);

	Inverse_NTT_Inplace(ciphertext, q_device, mu_device, q_bit_device, n, InversePsi_device, 2 * (coeff_modulus - 1), (coeff_modulus - 1), INTT_inv_q);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// New Multiplication:(Dimentional Optimal)

__device__ void first_OP1(unsigned long long& input1, unsigned long long& output1, unsigned long long& q, unsigned long long& mu, unsigned long long& bitlength,
	unsigned long long& aux_m_tilde_device, unsigned long long& p)
{
	uint128_t mult1;
	mul64(aux_m_tilde_device, input1, mult1);
	singleBarrett(mult1, q, mu, bitlength);

	mul64(mult1.low, p, mult1);
	singleBarrett(mult1, q, mu, bitlength);
	output1 = mult1.low;
}

__device__ void first_OP2(unsigned long long* input1, unsigned long long* output1, unsigned long long& aux, unsigned long long aux_mu, unsigned long long aux_bit,
	unsigned long long& inv_prod_q_mod_m_tilde, unsigned long long* base_change_matrix_device2, int SIZE, int N, int idx, int tid, int loc)
{

	uint128_t sum = 0;

	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t mult;
		mul64(base_change_matrix_device2[lp], input1[(lp * N) + idx + loc], mult);
		//mul64(base_change_matrix_device1[lp], input1[lp], mult); // Read from register
		sum = mult + sum;
		singleBarrett(sum, aux, aux_mu, aux_bit);
	}
	mul64(sum.low, inv_prod_q_mod_m_tilde, sum);
	singleBarrett(sum, aux, aux_mu, aux_bit);

	output1[tid + idx] = aux - sum.low;

}

__device__ void first_OP2reg(unsigned long long* input1, unsigned long long& output1, unsigned long long& aux, unsigned long long aux_mu, unsigned long long aux_bit,
	unsigned long long& inv_prod_q_mod_m_tilde, unsigned long long* base_change_matrix_device2, int SIZE, int N, int idx, int tid, int loc)
{

	uint128_t sum = 0;

	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t mult;
		//mul64(base_change_matrix_device2[lp], input1[(lp * N) + idx + loc], mult);
		mul64(base_change_matrix_device2[lp], input1[lp], mult); // Read from register
		sum = mult + sum;
		singleBarrett(sum, aux, aux_mu, aux_bit);
	}
	mul64(sum.low, inv_prod_q_mod_m_tilde, sum);
	singleBarrett(sum, aux, aux_mu, aux_bit);

	output1 = aux - sum.low;

}

__device__ void first_OP3(unsigned long long* input1, unsigned long long* input2, unsigned long long* output1, unsigned long long* aux, unsigned long long* aux_mu, unsigned long long* aux_bit,
	unsigned long long& aux_m, unsigned long long* base_change_matrix_device1, unsigned long long* prod_q_mod_Bsk, unsigned long long* inv_m_tilde_mod_Bsk,
	int SIZE, int N, int idx, int tid, int tid2, int tid3)
{

	for (int lp1 = 0; lp1 < (SIZE + 1); lp1++) {

		uint128_t sum = 0;
		uint128_t mult;

		for (int lp2 = 0; lp2 < SIZE; lp2++) {

			mul64(base_change_matrix_device1[lp2 * (SIZE + 1) + lp1], input1[lp2 * N + tid + idx], mult);
			sum = mult + sum;
			singleBarrett(sum, aux[lp1], aux_mu[lp1], aux_bit[lp1]);

		}

		unsigned long long rmtilda = input2[tid2 + idx];
		rmtilda += (rmtilda >= (aux_m >> 1)) * (aux[lp1] - aux_m);

		mul64(rmtilda, prod_q_mod_Bsk[lp1], mult);
		mult = mult + sum.low;
		singleBarrett(mult, aux[lp1], aux_mu[lp1], aux_bit[lp1]);

		mul64(mult.low, inv_m_tilde_mod_Bsk[lp1], mult);
		singleBarrett(mult, aux[lp1], aux_mu[lp1], aux_bit[lp1]);
		output1[(lp1 * N) + tid3 + idx] = mult.low;

	}

}

__device__ void first_OP3reg(unsigned long long* input1, unsigned long long& input2, unsigned long long* output1, unsigned long long* aux, unsigned long long* aux_mu, unsigned long long* aux_bit,
	unsigned long long& aux_m, unsigned long long* base_change_matrix_device1, unsigned long long* prod_q_mod_Bsk, unsigned long long* inv_m_tilde_mod_Bsk,
	int SIZE, int N, int idx, int tid, int tid2, int tid3)
{

	for (int lp1 = 0; lp1 < (SIZE + 1); lp1++) {

		uint128_t sum = 0;
		uint128_t mult;

		for (int lp2 = 0; lp2 < SIZE; lp2++) {

			mul64(base_change_matrix_device1[lp2 * (SIZE + 1) + lp1], input1[lp2], mult);
			sum = mult + sum;
			singleBarrett(sum, aux[lp1], aux_mu[lp1], aux_bit[lp1]);

		}

		unsigned long long rmtilda = input2;
		rmtilda += (rmtilda >= (aux_m >> 1)) * (aux[lp1] - aux_m);

		mul64(rmtilda, prod_q_mod_Bsk[lp1], mult);
		mult = mult + sum.low;
		singleBarrett(mult, aux[lp1], aux_mu[lp1], aux_bit[lp1]);

		mul64(mult.low, inv_m_tilde_mod_Bsk[lp1], mult);
		singleBarrett(mult, aux[lp1], aux_mu[lp1], aux_bit[lp1]);
		output1[(lp1 * N) + tid3 + idx] = mult.low;

	}

}


__global__ void fastconvb_NEW(unsigned long long* a_device, unsigned long long* b_device, unsigned long long a_device_temp[], unsigned long long b_device_temp[], unsigned long long q_device[], unsigned long long mu_device[], unsigned long long aux_m_tilde_device[], unsigned long long p[], int N, const int SIZE, unsigned long long bit_length[],
	unsigned long long* base_change_matrix2_device, unsigned long long* inv_prod_q_mod_m_tilde_device,
	unsigned long long* base_change_matrix1_device, unsigned long long* aux_B_m_sk_device, unsigned long long* base_Bsk_elt_mu_device, unsigned long long* base_Bsk_bitlength_device,// 3
	unsigned long long* inv_m_tilde_mod_Bsk_device, unsigned long long* prod_q_mod_Bsk_device, unsigned long long* dest11_device, unsigned long long* dest12_device//3
) {

	//__shared__ unsigned long long result1_device_shr[3][8192];
	register unsigned long long result1_device_reg[20];
	register unsigned long long r_m_tilde_11_reg;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	if (j < 2) {

		unsigned long long* input;

		if (k == 0)
			input = a_device;
		else
			input = b_device;

		for (int lp = 0; lp < SIZE; lp++) {

			first_OP1(input[(lp * N) + (j * N * SIZE) + idx], result1_device_reg[lp], q_device[lp], mu_device[lp], bit_length[lp], aux_m_tilde_device[0], p[lp]);

		}

		// --- Operation 1 END ---

		first_OP2reg(result1_device_reg, r_m_tilde_11_reg, aux_m_tilde_device[0], 17179869184, 33,
			inv_prod_q_mod_m_tilde_device[0], base_change_matrix2_device, SIZE, N, idx, (k * N * 2) + (j * N), (k * N * SIZE * 2) + (j * N * SIZE));

		// --- Operation 2 END ---

		unsigned long long* output;

		if (k == 0)
			output = dest11_device;
		else
			output = dest12_device;

		first_OP3reg(result1_device_reg, r_m_tilde_11_reg, output, aux_B_m_sk_device, base_Bsk_elt_mu_device, base_Bsk_bitlength_device,
			aux_m_tilde_device[0], base_change_matrix1_device, prod_q_mod_Bsk_device, inv_m_tilde_mod_Bsk_device,
			SIZE, N, idx, (k * N * SIZE * 2) + (j * N * SIZE), (k * N * 2) + (j * N), (j * N * (SIZE + 1)));

		// --- Operation 3 END ---

	}//------------------------------------------------------------------------------------------------------------------------------------------------
	else {

		int j_ = j - 2;

		unsigned long long* input;
		unsigned long long* output;

		if (k == 0) {
			input = a_device;
			output = a_device_temp;
		}
		else {
			input = b_device;
			output = b_device_temp;
		}

		output[(j_ * N) + idx] = input[(j_ * N) + idx];

	}

}

//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void MULTS(unsigned long long* input1q, unsigned long long* input2q, unsigned long long qa_device[], unsigned long long mu_device[], unsigned long long* outputq, int N, int SIZE, unsigned long long bit_length[],
	unsigned long long input1bsk[], unsigned long long input2bsk[], unsigned long long base_Bsk_elt_device[], unsigned long long base_Bsk_elt_mu_device[], unsigned long long* outputbsk, unsigned long long base_Bsk_bitlength_device[]) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	if (j < SIZE) {

		unsigned long long q = qa_device[j];
		unsigned long long mu = mu_device[j];
		unsigned long long bitlength = bit_length[j];

		unsigned long long input1_0 = input1q[j * N + idx];
		unsigned long long input1_1 = input1q[(N * SIZE) + j * N + idx];
		unsigned long long input2_0 = input2q[j * N + idx];
		unsigned long long input2_1 = input2q[(N * SIZE) + j * N + idx];

		uint128_t mult;
		mul64(input1_0, input2_0, mult);
		singleBarrett(mult, q, mu, bitlength);
		outputq[j * N + idx] = mult.low;

		//-------------
		uint128_t mult_0;
		uint128_t mult_1;
		mul64(input1_0, input2_1, mult_0);
		mul64(input1_1, input2_0, mult_1);
		mult_0 = mult_0 + mult_1;
		singleBarrett(mult_0, q, mu, bitlength);
		outputq[(N * SIZE) + (j * N) + idx] = mult_0.low;

		//-------------
		uint128_t mult_;
		mul64(input1_1, input2_1, mult_);
		singleBarrett(mult_, q, mu, bitlength);
		outputq[(N * SIZE * 2) + (j * N) + idx] = mult_.low;

	}
	else {
		int j_ = j - SIZE;

		unsigned long long q = base_Bsk_elt_device[j_];
		unsigned long long mu = base_Bsk_elt_mu_device[j_];
		unsigned long long bitlength = base_Bsk_bitlength_device[j_];

		unsigned long long input1_0 = input1bsk[(j_ * N) + idx];
		unsigned long long input1_1 = input1bsk[(N * (SIZE + 1)) + (j_ * N) + idx];
		unsigned long long input2_0 = input2bsk[(j_ * N) + idx];
		unsigned long long input2_1 = input2bsk[(N * (SIZE + 1)) + (j_ * N) + idx];

		uint128_t mult;
		mul64(input1_0, input2_0, mult);
		singleBarrett(mult, q, mu, bitlength);
		outputbsk[j_ * N + idx] = mult.low;

		//-------------
		uint128_t mult_0;
		uint128_t mult_1;
		mul64(input1_0, input2_1, mult_0);
		mul64(input1_1, input2_0, mult_1);
		mult_0 = mult_0 + mult_1;
		singleBarrett(mult_0, q, mu, bitlength);
		outputbsk[(N * (SIZE + 1)) + (j_ * N) + idx] = mult_0.low;

		//-------------
		uint128_t mult_;
		mul64(input1_1, input2_1, mult_);
		singleBarrett(mult_, q, mu, bitlength);
		outputbsk[(N * (SIZE + 1) * 2) + (j_ * N) + idx] = mult_.low;

	}

}

//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------
__device__ void second_OP1(unsigned long long* input1, unsigned long long* output1, unsigned long long* q, unsigned long long* mu, unsigned long long* bit,
	unsigned long long& t, int SIZE, int N, int idx, int tid)
{
	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t ct0 = input1[(lp * N) + tid + idx];
		mul64(ct0.low, t, ct0);
		singleBarrett(ct0, q[lp], mu[lp], bit[lp]);
		output1[lp] = ct0.low;

	}


}

__device__ void second_OP2(unsigned long long* input1, unsigned long long* output1, unsigned long long* q, unsigned long long* mu, unsigned long long* bit,
	unsigned long long* p, int SIZE)
{
	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t ct0 = input1[lp];
		mul64(ct0.low, p[lp], ct0);
		singleBarrett(ct0, q[lp], mu[lp], bit[lp]);
		output1[lp] = ct0.low;

	}


}

__device__ void second_OP3(unsigned long long* input1, unsigned long long* input2, unsigned long long* output1, unsigned long long* aux, unsigned long long* auxmu, unsigned long long* auxbit,
	unsigned long long* base_change_matrix, unsigned long long* inv_prod_q_mod_Bsk_device, int SIZE)
{
	for (int lp1 = 0; lp1 < (SIZE + 1); lp1++) {

		unsigned long long sum = 0;

		for (int lp2 = 0; lp2 < SIZE; lp2++) {

			uint128_t ct0 = input1[lp2];
			mul64(ct0.low, base_change_matrix[(lp2 * (SIZE + 1)) + lp1], ct0);
			ct0 = ct0 + sum;
			singleBarrett(ct0, aux[lp1], auxmu[lp1], auxbit[lp1]);
			sum = ct0.low;

		}

		unsigned long long temp = (input2[lp1] + (aux[lp1] - sum));
		temp -= (temp >= aux[lp1]) * aux[lp1];

		uint128_t result;
		mul64(temp, inv_prod_q_mod_Bsk_device[lp1], result);
		singleBarrett(result, aux[lp1], auxmu[lp1], auxbit[lp1]);
		output1[lp1] = result.low;

	}

}

__device__ void second_OP4(unsigned long long* input1, unsigned long long* output1, unsigned long long* aux, unsigned long long* auxmu, unsigned long long* auxbit,
	unsigned long long* p, int SIZE)
{
	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t ct0 = input1[lp];
		mul64(ct0.low, p[lp], ct0);
		singleBarrett(ct0, aux[lp], auxmu[lp], auxbit[lp]);
		output1[lp] = ct0.low;

	}

}

__device__ void second_OP5(unsigned long long* input1, unsigned long long* input2, unsigned long long* output1,
	unsigned long long* q, unsigned long long* qmu, unsigned long long* qbit,
	unsigned long long* aux, unsigned long long* auxmu, unsigned long long* auxbit,
	unsigned long long* p, unsigned long long* base_change_matrix3, unsigned long long* base_change_matrix4, unsigned long long& msk, unsigned long long& _inv_prod_B_mod_m_sk,
	unsigned long long* prod_B_mod_q,
	int N, int SIZE, int idx, int tid)
{
	unsigned long long _m_sk_div_2 = msk >> 1;

	for (int lp1 = 0; lp1 < (SIZE); lp1++) {

		unsigned long long sum1 = 0;
		unsigned long long sum2 = 0;

		unsigned long long q_reg = q[lp1];
		unsigned long long mu_reg = qmu[lp1];
		unsigned long long bit_reg = qbit[lp1];

		for (int lp2 = 0; lp2 < SIZE; lp2++) {

			uint128_t nid = input1[lp2]; // ok
			mul64(nid.low, base_change_matrix3[(lp2 * SIZE) + lp1], nid);// ok
			nid = nid + sum1;
			singleBarrett(nid, q_reg, mu_reg, bit_reg);
			sum1 = nid.low;

			uint128_t c_1 = input2[lp2];// ok
			mul64(c_1.low, p[lp2], c_1);// ok
			singleBarrett(c_1, aux[lp2], auxmu[lp2], auxbit[lp2]);

			mul64(c_1.low, base_change_matrix4[lp2], nid);
			nid = nid + sum2;
			singleBarrett(nid, msk, auxmu[SIZE], auxbit[SIZE]);
			sum2 = nid.low;

		}

		uint128_t temp = (sum2 + (msk - input2[SIZE]));
		mul64(temp.low, _inv_prod_B_mod_m_sk, temp);
		singleBarrett(temp, msk, auxmu[SIZE], auxbit[SIZE]);

		if (temp.low > _m_sk_div_2) {

			unsigned long long sub = msk - temp.low;
			uint128_t prod_B = prod_B_mod_q[lp1];
			mul64(prod_B.low, sub, prod_B);
			prod_B = prod_B + sum1;
			singleBarrett(prod_B, q_reg, mu_reg, bit_reg);
			output1[(lp1 * N) + tid + idx] = prod_B.low;

		}
		else {

			uint128_t sub = q_reg - prod_B_mod_q[lp1];
			mul64(sub.low, temp.low, sub);
			sub = sub + sum1;
			singleBarrett(sub, q_reg, mu_reg, bit_reg);
			output1[(lp1 * N) + tid + idx] = sub.low;

		}

	}

}


__device__ void second_OP4_globall(unsigned long long* input1, unsigned long long* output1, unsigned long long* aux, unsigned long long* auxmu, unsigned long long* auxbit,
	unsigned long long* p, int SIZE)
{
	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t ct0 = input1[lp];
		mul64(ct0.low, p[lp], ct0);
		singleBarrett(ct0, aux[lp], auxmu[lp], auxbit[lp]);
		output1[lp] = ct0.low;

	}

}

__global__ void FastFloor_NEW(unsigned long long* inputq, unsigned long long* inputbsk,
	unsigned long long BSK[], unsigned long long BSKmu[], unsigned long long BSKbit[],
	unsigned long long q[], unsigned long long qmu[], unsigned long long qbit[],
	unsigned long long t[], int N, int SIZE, int aux_b_len,// mult_t
	unsigned long long* p_device, // first_fast_floor
	unsigned long long* base_change_matrix1, unsigned long long* inv_prod_q_mod_Bsk_device,// --
	unsigned long long* p1_device,
	unsigned long long* base_change_matrix3, unsigned long long* base_change_matrix4,
	unsigned long long* msk_device, unsigned long long* _inv_prod_B_mod_m_sk,
	unsigned long long* prod_B_mod_q, unsigned long long* out1, unsigned long long* out2
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;

	register unsigned long long ct_bsk_pol_t_0_reg1[20];// MAX 20 q
	register unsigned long long ct_bsk_pol_t_0_reg2[20];// MAX 20 q

	second_OP1(inputq, ct_bsk_pol_t_0_reg1, q, qmu, qbit,
		t[0], SIZE, N, idx, (j * N * SIZE));

	second_OP1(inputbsk, ct_bsk_pol_t_0_reg2, BSK, BSKmu, BSKbit,
		t[0], aux_b_len, N, idx, (j * N * aux_b_len));

	// mult_t ------- END -------

	second_OP2(ct_bsk_pol_t_0_reg1, ct_bsk_pol_t_0_reg1, q, qmu, qbit,
		p_device, SIZE);

	// first_fast_floor ------- END -------


	second_OP3(ct_bsk_pol_t_0_reg1, ct_bsk_pol_t_0_reg2, ct_bsk_pol_t_0_reg2, BSK, BSKmu, BSKbit,
		base_change_matrix1, inv_prod_q_mod_Bsk_device, SIZE);

	// fast_floor2 ------- END -------


	//////////////////////////////////////////////////////////////////
	second_OP4_globall(ct_bsk_pol_t_0_reg2, ct_bsk_pol_t_0_reg1, BSK, BSKmu, BSKbit,
		p1_device, SIZE);

	// fourth_fast_floor ------- END -------


	unsigned long long* output;
	int location;

	if (j < 2) {
		output = out1;
		location = (j * N * SIZE);
	}
	else {
		output = out2;
		location = 0;
	}

	second_OP5(ct_bsk_pol_t_0_reg1, ct_bsk_pol_t_0_reg2, output,
		q, qmu, qbit,
		BSK, BSKmu, BSKbit,
		p1_device, base_change_matrix3, base_change_matrix4, msk_device[0], _inv_prod_B_mod_m_sk[0],
		prod_B_mod_q, N, SIZE, idx, location);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void second_OP4_global(unsigned long long* input1, unsigned long long* output1, unsigned long long* aux, unsigned long long* auxmu, unsigned long long* auxbit,
	unsigned long long* p, int SIZE)
{
	for (int lp = 0; lp < SIZE; lp++) {

		uint128_t ct0 = input1[lp];
		mul64(ct0.low, p[lp], ct0);
		singleBarrett(ct0, aux[lp], auxmu[lp], auxbit[lp]);
		output1[lp] = ct0.low;

	}

}


__host__ void GPU_Multiplication(GPU_Ciphertext input1, GPU_Ciphertext input2, GPU_Ciphertext output, Lib_Parameters context)
{
	// CLASS FEATURES
	int n = input1.ring_size;
	int q_length = input1.coeff_mod_count + 1;

	//output.pre_mult(); // necessary operation before multiplication (increase cipher size from 2 to 3)

	unsigned long long* Multiplication_Pool = context.Multiplication_Pool;

	//Extract Context Parameters
	unsigned long long* q_device = context.Context_GPU;
	unsigned long long* q_mudevice = q_device + q_length;
	unsigned long long* bitlength_device = q_mudevice + q_length;
	unsigned long long* q_INTT_device = bitlength_device + q_length;
	unsigned long long* q_mu_INTT_device = q_INTT_device + q_length * 2;
	unsigned long long* q_bit_INTT_device = q_mu_INTT_device + q_length * 2;
	unsigned long long* key_device = q_bit_INTT_device + q_length * 2;
	unsigned long long* key_inverse_device = key_device + q_length * n;
	unsigned long long* DoubleInversePsi_device = key_inverse_device + q_length * n;
	unsigned long long* lastq_modinv_device = DoubleInversePsi_device + q_length * 2 * n;
	unsigned long long* half_device = lastq_modinv_device + (q_length - 1);
	unsigned long long* half_mod_device = half_device + 1;
	unsigned long long* INTT_inv_q = half_mod_device + (q_length - 1); // Tık
	unsigned long long* INTT_inv_double_q = INTT_inv_q + q_length; // Tık

	unsigned long long* plainmod_device = INTT_inv_double_q + q_length * 2;
	unsigned long long* plainpsi_device = plainmod_device + 1;
	unsigned long long* plain_ninverse = plainpsi_device + 1;
	unsigned long long* ForwardPlainPsi_device = plain_ninverse + 1;
	unsigned long long* InversePlainPsi_device = ForwardPlainPsi_device + n;
	unsigned long long* plain_upper_half_increment_device = InversePlainPsi_device + n;
	unsigned long long* plain_upper_half_threshold_device = plain_upper_half_increment_device + (q_length - 1);

	unsigned long long* plainmu_device = plain_upper_half_threshold_device + 1;
	unsigned long long* plain_bit_device = plainmu_device + 1;


	//for multiply
	unsigned long long* aux_m_tilde_device = plain_bit_device + 1;
	unsigned long long* p_device = aux_m_tilde_device + 1;
	unsigned long long* aux_B_m_sk_device = p_device + (q_length - 1);
	unsigned long long* base_change_matrix1_device = aux_B_m_sk_device + q_length;
	unsigned long long* base_change_matrix2_device = base_change_matrix1_device + ((q_length - 1) * q_length);
	unsigned long long* inv_prod_q_mod_m_tilde_device = base_change_matrix2_device + (q_length - 1);
	unsigned long long* inv_m_tilde_mod_Bsk_device = inv_prod_q_mod_m_tilde_device + 1;
	unsigned long long* prod_q_mod_Bsk_device = inv_m_tilde_mod_Bsk_device + q_length;
	unsigned long long* base_Bsk_elt_mu_device = prod_q_mod_Bsk_device + q_length;
	unsigned long long* base_Bsk_bitlength_device = base_Bsk_elt_mu_device + q_length;
	unsigned long long* t_device = base_Bsk_bitlength_device + q_length;
	unsigned long long* aux_B1_device = t_device + 1;
	unsigned long long* inv_prod_q_mod_Bsk_device = aux_B1_device + q_length;
	unsigned long long* p1_device = inv_prod_q_mod_Bsk_device + q_length;
	unsigned long long* base_change_matrix3_device = p1_device + q_length;
	unsigned long long* base_change_matrix4_device = base_change_matrix3_device + (q_length - 1) * (q_length - 1);
	unsigned long long* m_sk_device = base_change_matrix4_device + (q_length - 1);
	unsigned long long* inv_prod_B_mod_m_sk_device = m_sk_device + 1;
	unsigned long long* prod_B_mod_q_device = inv_prod_B_mod_m_sk_device + 1;
	unsigned long long* ForwardPsi_device_BSK = prod_B_mod_q_device + (q_length - 1);
	unsigned long long* InversePsi_device_BSK = ForwardPsi_device_BSK + q_length * n;
	unsigned long long* INTT_inv_bsk = InversePsi_device_BSK + q_length * n;

	//Extract Temporary Parameters
	unsigned long long* result1_device = Multiplication_Pool;
	unsigned long long* temp2_11_device = result1_device + (4 * (q_length - 1) * n);
	unsigned long long* r_m_tilde_11_device = temp2_11_device + (4 * (q_length)*n);
	unsigned long long* dest11_device = r_m_tilde_11_device + (4 * n);
	unsigned long long* dest12_device = dest11_device + (2 * q_length * n);
	unsigned long long* ct_0_device = dest12_device + (2 * q_length * n);
	unsigned long long* ct_bsk_pol_t_0 = ct_0_device + (3 * (q_length - 1) * n);
	unsigned long long* nid1 = ct_bsk_pol_t_0 + (3 * ((q_length - 1) + q_length) * n);
	unsigned long long* a_device_temp = nid1 + (3 * (q_length - 1) * n);
	unsigned long long* b_device_temp = a_device_temp + (2 * (q_length - 1) * n);


	dim3 numBlocks1((n / 512), 2 + (2 * (q_length - 1)), 2);
	fastconvb_NEW << <numBlocks1, 512 >> > (input1.GPU_Location, input2.GPU_Location, a_device_temp, b_device_temp, q_device, q_mudevice, aux_m_tilde_device, p_device, n, (q_length - 1), bitlength_device, base_change_matrix2_device, inv_prod_q_mod_m_tilde_device,
		base_change_matrix1_device, aux_B_m_sk_device, base_Bsk_elt_mu_device, base_Bsk_bitlength_device, inv_m_tilde_mod_Bsk_device, prod_q_mod_Bsk_device, dest11_device, dest12_device);

	Forward_NTT_Inplace(a_device_temp, q_device, q_mudevice, bitlength_device, n, key_device, (4 * (q_length - 1)), (q_length - 1)); // normal ciphertext'lerin ntt sonuçları

	Forward_NTT_Inplace(dest11_device, aux_B_m_sk_device, base_Bsk_elt_mu_device, base_Bsk_bitlength_device, n, ForwardPsi_device_BSK, 4 * q_length, q_length); //BSK ciphertext'lerinin ntt sonuçları

	dim3 numBlocks2((n / 512), (q_length - 1) + q_length, 1);
	MULTS << <numBlocks2, 512 >> > (a_device_temp, b_device_temp, q_device, q_mudevice, ct_0_device, n, (q_length - 1), bitlength_device,
		dest11_device, dest12_device, aux_B_m_sk_device, base_Bsk_elt_mu_device, temp2_11_device, base_Bsk_bitlength_device);

	Inverse_NTT_Inplace(ct_0_device, q_device, q_mudevice, bitlength_device, n, key_inverse_device, 3 * (q_length - 1), (q_length - 1), INTT_inv_q); // normal ciphertext'lerin inverse ntt sonuçları

	Inverse_NTT_Inplace(temp2_11_device, aux_B_m_sk_device, base_Bsk_elt_mu_device, base_Bsk_bitlength_device, n, InversePsi_device_BSK, 3 * q_length, q_length, INTT_inv_bsk); // BSK ciphertext'lerin inverse ntt sonuçları


	dim3 numBlocks3((n / 512), 3, 1);
	FastFloor_NEW << < numBlocks3, 512 >> > (ct_0_device, temp2_11_device,
		aux_B_m_sk_device, base_Bsk_elt_mu_device, base_Bsk_bitlength_device,
		q_device, q_mudevice, bitlength_device,
		t_device, n, (q_length - 1), q_length,// mult_t
		p_device, // first_fast_floor
		base_change_matrix1_device, inv_prod_q_mod_Bsk_device, // fast_floor2 // fourth_fast_floor(zaten var)
		p1_device,
		base_change_matrix3_device, base_change_matrix4_device, m_sk_device, inv_prod_B_mod_m_sk_device, prod_B_mod_q_device, output.GPU_Location, output.GPU_Location2); // fast_floor3_class 



}



