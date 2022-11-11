#pragma once

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "Barret.cuh"
//#include "GPU_Context.cuh"

__device__ void Butterfly(unsigned long long& input1, unsigned long long& input2,
	unsigned long long& output1, unsigned long long& output2, unsigned long long& PsiTable, unsigned long long& q, unsigned long long& mu, unsigned long long& qbit)
{

	uint128_t V_prime;

	mul64(input2, PsiTable, V_prime);
	singleBarrett(V_prime, q, mu, qbit);

	unsigned long long U_prime = input1 + V_prime.low;
	U_prime -= (U_prime >= q) * q;

	input1 = input1 + q;
	V_prime.low = input1 - V_prime.low;
	V_prime.low -= (V_prime >= q) * q;

	output1 = U_prime;
	output2 = V_prime.low;

}

__device__ void Butterfly_inplace(unsigned long long& input1, unsigned long long& input2,
	unsigned long long& PsiTable, unsigned long long& q, unsigned long long& mu, unsigned long long& qbit)
{
	unsigned long long U_prime;
	uint128_t V_prime;

	mul64(input2, PsiTable, V_prime);
	singleBarrett(V_prime, q, mu, qbit);

	U_prime = input1 + V_prime.low;
	U_prime -= (U_prime >= q) * q;

	input1 = input1 + q;
	V_prime.low = input1 - V_prime.low;
	V_prime.low -= (V_prime >= q) * q;

	input1 = U_prime;
	input2 = V_prime.low;


}

__device__ void Butterfly_GS(unsigned long long& input1, unsigned long long& input2,
	unsigned long long& output1, unsigned long long& output2, unsigned long long& PsiTable, unsigned long long& q, unsigned long long& mu, unsigned long long& qbit)
{

	unsigned long long U_prime;
	uint128_t V_prime;

	U_prime = input1 + input2;
	U_prime -= q * (U_prime >= q);

	V_prime.low = input1 + q * (input1 < input2);
	V_prime.low = V_prime.low - input2;

	mul64(V_prime.low, PsiTable, V_prime);
	singleBarrett(V_prime, q, mu, qbit);

	output1 = U_prime;
	output2 = V_prime.low;
}

/////////////////////////////////////////////////////SHARED////////////////////////////////////////////////////*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*
__global__ void FORWARD_NTT_DIM_NEW_Shared_ALL(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];

	int n_power = N_power;
	int t_2 = 10;//n_power - 2; /// buraları parametrik yapp // her zaman 10 yani 1024
	int t = 1 << t_2;
	int n = N;
	int m = M;
	int n_2 = n >> 1; // 4096 / 2 = 2048

	int dividx = (((gridDim.y * k) + j) << n_power);
	int idx_psi = j << n_power; // bu kalacak

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int address = dividx + ((idx >> t_2) << t_2) + idx;

	int shrd_dixidx_t = (local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;

	Butterfly(Inputs[address], Inputs[address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;

	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);


	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);


	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;

	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;

	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;

	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);


	t = t >> 1;
	m = m << 1;
	t_2 -= 1;

	address = dividx + ((idx >> t_2) << t_2) + idx;

	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	Butterfly(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], Inputs[address], Inputs[address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);


}


__global__ void INVERSE_NTT_DIM_NEW_Shared(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];

	int n_power = N_power;
	int t_2 = 0;
	int t = 1 << t_2;
	int n = N;
	int m = n >> 1;
	int n_2 = n >> 1;

	int dividx = (((gridDim.y * k) + j) << n_power);
	int idx_psi = j << n_power;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int address = dividx + ((idx >> t_2) << t_2) + idx;

	int shrd_dixidx_t = (local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	Butterfly_GS(Inputs[address], Inputs[address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);

	t = t << 1;
	m = m >> 1;
	t_2 += 1;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	address = dividx + ((idx >> t_2) << t_2) + idx;

	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	Butterfly_GS(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t], Outputs[address], Outputs[address + t], PsiTable[m + (idx >> t_2) + idx_psi], q_thread, q_mu_thread, q_bit_thread);


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*




/////////////////////////////////////////////////////START/////////////////////////////////////////////////////*
////////////////////////////////////////////- - - - - 4096 - - - - -///////////////////////////////////////////*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*
//FORWARD
__global__ void FORWARD_NTT_DIM_LOCAL_4096(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	register unsigned long long localmemorysx_0, localmemorysx_1;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];


	//Input 
	/////////////////////////////////////////
	int shrd_read = i + (((gridDim.y * k) + j) << n_power); // 11 ==> 2048  ////// (modidx_N << n_power) = (((gridDim.y * blockIdx.z) + blockIdx.y) * n)
	localmemorysx_0 = Inputs[shrd_read];
	localmemorysx_1 = Inputs[shrd_read + 2048];
	/////////////////////////////////////////

	int address_psi = (i) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	Butterfly_inplace(localmemorysx_0, localmemorysx_1, PsiTable[m + address_psi + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// OUTPUT
	/////////////////////////////////
	shrd_read = i + (((gridDim.y * k) + j) << n_power); // 11 ==> 2048
	Outputs[shrd_read] = localmemorysx_0;
	Outputs[shrd_read + 2048] = localmemorysx_1;
	/////////////////////////////////

}

//INVERSE
__global__ void INVERSE_NTT_DIM_NEW_Local_4096(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int M, int T, int q_count, unsigned long long* modinv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];
	unsigned long long invv = modinv[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long localmemorysx_0, localmemorysx_1;
	//Input 
	/////////////////////////////////////////

	localmemorysx_0 = Inputs[addresss];
	localmemorysx_1 = Inputs[addresss + 2048];
	/////////////////////////////////////////

	Butterfly_GS(localmemorysx_0, localmemorysx_1, localmemorysx_0, localmemorysx_1, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	uint128_t U_prime;
	mul64(localmemorysx_0, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	uint128_t V_prime;
	mul64(localmemorysx_1, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = V_prime.low;



}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////- - - - - 4096 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////




/////////////////////////////////////////////////////START/////////////////////////////////////////////////////
////////////////////////////////////////////- - - - - 8192 - - - - -///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//FORWARD
__global__ void FORWARD_NTT_DIM_NEW_LOCAL_8192(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long local0, local1, local2, local3;


	/// ilk for başlar

	//int address_psi = i >> t_2;
	Butterfly(Inputs[addresss], Inputs[addresss + 4096], local0, local2, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// second part

	//address_psi = (2048 + i) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //
	Butterfly(Inputs[addresss + 2048], Inputs[addresss + 6144], local1, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m << 1;
	t_2 = t_2 - 1;

	/// ilk for biter

	/////////////////////////////////

	/// ikinci for başlar

	//address_psi = i >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //
	Butterfly(local0, local1, Outputs[addresss], Outputs[addresss + 2048], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// second part

	//address_psi = (2048 + i) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //
	Butterfly(local2, local3, Outputs[addresss + 4096], Outputs[addresss + 6144], PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	/// ikinci for biter

}

//INVERSE
__global__ void INVERSE_NTT_DIM_NEW_Local_8192(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int M, int T, int q_count, unsigned long long* modinv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];
	unsigned long long invv = modinv[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	/// ilk for başlar


	Butterfly_GS(local0, local1, local0, local1, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// second part

	Butterfly_GS(local2, local3, local2, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	/////////////////////////////////

	/// ikinci for başlar

	Butterfly_GS(local0, local2, local0, local2, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	uint128_t U_prime;
	uint128_t V_prime;

	mul64(local0, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(local2, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = V_prime.low;

	// second part

	Butterfly_GS(local1, local3, local1, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	mul64(local1, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(local3, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = V_prime.low;

	/// ikinci for biter

}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////- - - - - 8192 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////


/////////////////////////////////////////////////////START/////////////////////////////////////////////////////
///////////////////////////////////////////- - - - - 16384 - - - - -///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//FORWARD
__global__ void FORWARD_NTT_DIM_NEW_LOCAL_16384(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long local0, local1, local2, local3, local4, local5, local6, local7;


	/// ilk for başlar
	Butterfly(Inputs[addresss], Inputs[addresss + 8192], local0, local4, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 2048], Inputs[addresss + 10240], local1, local5, PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 4096], Inputs[addresss + 12288], local2, local6, PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 6144], Inputs[addresss + 14336], local3, local7, PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	m = m << 1;
	t_2 = t_2 - 1;

	/// ilk for biter

	/// ikinci for başlar
	Butterfly(local0, local2, local0, local2, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local1, local3, local1, local3, PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local4, local6, local4, local6, PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local5, local7, local5, local7, PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	m = m << 1;
	t_2 = t_2 - 1;

	/// ikinci for biter

	/// üçüncü for başlar
	Butterfly(local0, local1, Outputs[addresss], Outputs[addresss + 2048], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local2, local3, Outputs[addresss + 4096], Outputs[addresss + 6144], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local4, local5, Outputs[addresss + 8192], Outputs[addresss + 10240], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local6, local7, Outputs[addresss + 12288], Outputs[addresss + 14336], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	/// üçüncü for biter


}


//INVERSE
__global__ void INVERSE_NTT_DIM_NEW_Local_16384(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int M, int T, int q_count, unsigned long long* modinv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];
	unsigned long long invv = modinv[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	unsigned long long local4 = Inputs[addresss + 8192];
	unsigned long long local5 = Inputs[addresss + 10240];

	unsigned long long local6 = Inputs[addresss + 12288];
	unsigned long long local7 = Inputs[addresss + 14336];

	/// ilk for başlar
	Butterfly_GS(local0, local1, local0, local1, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// second part
	Butterfly_GS(local2, local3, local2, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// third part
	Butterfly_GS(local4, local5, local4, local5, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// forth part
	Butterfly_GS(local6, local7, local6, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// ikinci for başlar
	Butterfly_GS(local0, local2, local0, local2, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// second part
	Butterfly_GS(local1, local3, local1, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// third part
	Butterfly_GS(local4, local6, local4, local6, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// forth part
	Butterfly_GS(local5, local7, local5, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m >> 1;
	t_2 = t_2 + 1;


	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// üçüncü for başlar
	uint128_t U_prime;
	uint128_t V_prime;

	Butterfly_GS(local0, local4, local0, local4, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local0, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(local4, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8192] = V_prime.low;


	// second part

	Butterfly_GS(local1, local5, local1, local5, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local1, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(local5, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10240] = V_prime.low;


	// third part

	Butterfly_GS(local2, local6, local2, local6, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local2, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = U_prime.low;

	mul64(local6, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = V_prime.low;



	// third part

	Butterfly_GS(local3, local7, local3, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local3, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = U_prime.low;

	mul64(local7, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14336] = V_prime.low;



}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////- - - - - 16384 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////




/////////////////////////////////////////////////////START/////////////////////////////////////////////////////*
/////////////////////////////////////////////- - - - 32768 - - - - -///////////////////////////////////////////*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*
//FORWARD
__global__ void FORWARD_NTT_DIM_NEW_LOCAL_32768_1(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	register unsigned long long local[16];

	int n_2 = 16384;
	int p11 = 11;
	int max = 8;


	// Load
	for (int x = 0; x < max; x++) {

		local[x] = Inputs[addresss + (x << p11)];
		local[x + max] = Inputs[addresss + (x << p11) + n_2];

	}

	/// ilk for başlar
	for (int x = 0; x < max; x++) {

		Butterfly_inplace(local[x], local[x + max], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	}

	m = m << 1;
	t_2 = t_2 - 1;

	/// ilk for biter

	/// ikinci for başlar
	for (int x = 0; x < max; x++) {

		if (x < 4) {
			Butterfly_inplace(local[x], local[x + 4], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}
		else
		{
			Butterfly_inplace(local[x + 4], local[x + 8], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}

	}

	// Store
	for (int x = 0; x < max; x++) {

		Outputs[addresss + (x << p11)] = local[x];
		Outputs[addresss + (x << p11) + n_2] = local[x + max];

	}


}

__global__ void FORWARD_NTT_DIM_NEW_LOCAL_32768_2(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	//int m = 1;
	//int t_2 = n_power - 1;
	int m = 4;
	int t_2 = n_power - 3;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	register unsigned long long local[16];

	//int counter = 0;
	int n_2 = 16384;
	int p11 = 11;
	//int p12 = 12;
	int max = 8;


	// Load
	for (int x = 0; x < max; x++) {

		local[x] = Inputs[addresss + (x << p11)];
		local[x + max] = Inputs[addresss + (x << p11) + n_2];

	}

	/// üçüncü for başlar
	for (int x = 0; x < max; x++) {

		if (x < 2)
		{
			Butterfly_inplace(local[x], local[x + 2], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}
		else if ((x >= 2) && (x < 4))
		{
			Butterfly_inplace(local[x + 2], local[x + 4], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}
		else if ((x >= 4) && (x < 6))
		{
			Butterfly_inplace(local[x + 4], local[x + 6], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}
		else
		{
			Butterfly_inplace(local[x + 6], local[x + 8], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		}

	}
	/// üçüncü for biter

	m = m << 1;
	t_2 = t_2 - 1;


	/// üçüncü for biter

	/// dördüncü for başlar

	for (int x = 0; x < max; x++) {

		//Butterfly(local[x * 2], local[(x * 2) + 1], local[x * 2], local[(x * 2) + 1], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
		Butterfly_inplace(local[x * 2], local[(x * 2) + 1], PsiTable[m + ((i + (x << p11)) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	}

	/// dördüncü for biter

	// Store
	for (int x = 0; x < max; x++) {

		Inputs[addresss + (x << p11)] = local[x];
		Inputs[addresss + (x << p11) + n_2] = local[x + max];

	}
}


// NEW Forward
__global__ void FORWARD_NTT_DIM_NEW_LOCAL_32768NEW(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x + (1024 * blockIdx.x);
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	register unsigned long long local[32];


	/// ilk for başlar
	Butterfly(Inputs[addresss], Inputs[addresss + 16384], local[0], local[16], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 512], Inputs[addresss + 16896], local[1], local[17], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 2048], Inputs[addresss + 18432], local[2], local[18], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 2560], Inputs[addresss + 18944], local[3], local[19], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 4096], Inputs[addresss + 20480], local[4], local[20], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 4608], Inputs[addresss + 20992], local[5], local[21], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 6144], Inputs[addresss + 22528], local[6], local[22], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 6656], Inputs[addresss + 23040], local[7], local[23], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 8192], Inputs[addresss + 24576], local[8], local[24], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 8704], Inputs[addresss + 25088], local[9], local[25], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 10240], Inputs[addresss + 26624], local[10], local[26], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 10752], Inputs[addresss + 27136], local[11], local[27], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 12288], Inputs[addresss + 28672], local[12], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 12800], Inputs[addresss + 29184], local[13], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(Inputs[addresss + 14336], Inputs[addresss + 30720], local[14], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(Inputs[addresss + 14848], Inputs[addresss + 31232], local[15], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	m = m << 1;
	t_2 = t_2 - 1;

	/// ikinci for başlar

	Butterfly(local[0], local[8], local[0], local[8], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[1], local[9], local[1], local[9], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[2], local[10], local[2], local[10], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[3], local[11], local[3], local[11], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[4], local[12], local[4], local[12], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[5], local[13], local[5], local[13], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[6], local[14], local[6], local[14], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[7], local[15], local[7], local[15], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[16], local[24], local[16], local[24], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[17], local[25], local[17], local[25], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[18], local[26], local[18], local[26], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[19], local[27], local[19], local[27], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[20], local[28], local[20], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[21], local[29], local[21], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[22], local[30], local[22], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[23], local[31], local[23], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);



	m = m << 1;
	t_2 = t_2 - 1;

	/// ikinci for biter

	/// üçüncü for başlar
	Butterfly(local[0], local[4], local[0], local[4], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[1], local[5], local[1], local[5], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[2], local[6], local[2], local[6], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[3], local[7], local[3], local[7], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[8], local[12], local[8], local[12], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[9], local[13], local[9], local[13], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[10], local[14], local[10], local[14], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[11], local[15], local[11], local[15], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[16], local[20], local[16], local[20], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[17], local[21], local[17], local[21], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[18], local[22], local[18], local[22], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[19], local[23], local[19], local[23], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[24], local[28], local[24], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[25], local[29], local[25], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[26], local[30], local[26], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[27], local[31], local[27], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	/// üçüncü for biter



	m = m << 1;
	t_2 = t_2 - 1;

	/// dördüncü for başlar
	Butterfly(local[0], local[2], Inputs[addresss], Inputs[addresss + 2048], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[1], local[3], Inputs[addresss + 512], Inputs[addresss + 2560], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[4], local[6], Inputs[addresss + 4096], Inputs[addresss + 6144], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[5], local[7], Inputs[addresss + 4608], Inputs[addresss + 6656], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[8], local[10], Inputs[addresss + 8192], Inputs[addresss + 10240], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[9], local[11], Inputs[addresss + 8704], Inputs[addresss + 10752], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[12], local[14], Inputs[addresss + 12288], Inputs[addresss + 14336], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[13], local[15], Inputs[addresss + 12800], Inputs[addresss + 14848], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[16], local[18], Inputs[addresss + 16384], Inputs[addresss + 18432], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[17], local[19], Inputs[addresss + 16896], Inputs[addresss + 18944], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[20], local[22], Inputs[addresss + 20480], Inputs[addresss + 22528], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[21], local[23], Inputs[addresss + 20992], Inputs[addresss + 23040], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[24], local[26], Inputs[addresss + 24576], Inputs[addresss + 26624], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[25], local[27], Inputs[addresss + 25088], Inputs[addresss + 27136], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly(local[28], local[30], Inputs[addresss + 28672], Inputs[addresss + 30720], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly(local[29], local[31], Inputs[addresss + 29184], Inputs[addresss + 31232], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	/// dördüncü for biter

}


//INVERSE
__global__ void INVERSE_NTT_DIM_NEW_Local_32768_1(unsigned long long* Inputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
	int N, int N_power, int total_array, int M, int T, int q_count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	unsigned long long local4 = Inputs[addresss + 8192];
	unsigned long long local5 = Inputs[addresss + 10240];

	unsigned long long local6 = Inputs[addresss + 12288];
	unsigned long long local7 = Inputs[addresss + 14336];

	unsigned long long local8 = Inputs[addresss + 16384];
	unsigned long long local9 = Inputs[addresss + 18432];

	unsigned long long local10 = Inputs[addresss + 20480];
	unsigned long long local11 = Inputs[addresss + 22528];

	unsigned long long local12 = Inputs[addresss + 24576];
	unsigned long long local13 = Inputs[addresss + 26624];

	unsigned long long local14 = Inputs[addresss + 28672];
	unsigned long long local15 = Inputs[addresss + 30720];

	/// ilk for başlar
	// first part
	Butterfly_GS(local0, local1, local0, local1, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// second part
	Butterfly_GS(local2, local3, local2, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// third part
	Butterfly_GS(local4, local5, local4, local5, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// forth part
	Butterfly_GS(local6, local7, local6, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// fifth part
	Butterfly_GS(local8, local9, local8, local9, PsiTable[m + ((8192 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// sixth part
	Butterfly_GS(local10, local11, local10, local11, PsiTable[m + ((10240 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// seventh part
	Butterfly_GS(local12, local13, local12, local13, PsiTable[m + ((12288 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// eight part
	Butterfly_GS(local14, local15, local14, local15, PsiTable[m + ((14336 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// ikinci for başlar
	Butterfly_GS(local0, local2, local0, local2, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = local2;

	// second part
	Butterfly_GS(local1, local3, local1, local3, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = local3;

	// third part
	Butterfly_GS(local4, local6, local4, local6, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = local6;

	// forth part
	Butterfly_GS(local5, local7, local5, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14336] = local7;

	// fifth part
	Butterfly_GS(local8, local10, local8, local10, PsiTable[m + ((8192 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 16384] = local8;
	Inputs[addresss + 20480] = local10;


	// sixth part
	Butterfly_GS(local9, local11, local9, local11, PsiTable[m + ((10240 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 18432] = local9;
	Inputs[addresss + 22528] = local11;


	// seventh part
	Butterfly_GS(local12, local14, local12, local14, PsiTable[m + ((12288 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 24576] = local12;
	Inputs[addresss + 28672] = local14;


	// eight part
	Butterfly_GS(local13, local15, local13, local15, PsiTable[m + ((14336 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 26624] = local13;
	Inputs[addresss + 30720] = local15;

	///başla

	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// üçüncü for başlar

	Butterfly_GS(local0, local4, local0, local4, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = local0;
	Inputs[addresss + 8192] = local4;

	// second part

	Butterfly_GS(local1, local5, local1, local5, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = local1;
	Inputs[addresss + 10240] = local5;


}

__global__ void INVERSE_NTT_DIM_NEW_Local_32768_2(unsigned long long* Inputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
	int N, int N_power, int total_array, int M, int T, int q_count, unsigned long long* modinv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];
	unsigned long long invv = modinv[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);


	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	unsigned long long local4 = Inputs[addresss + 8192];
	unsigned long long local5 = Inputs[addresss + 10240];

	unsigned long long local6 = Inputs[addresss + 12288];
	unsigned long long local7 = Inputs[addresss + 14336];

	unsigned long long local8 = Inputs[addresss + 16384];
	unsigned long long local9 = Inputs[addresss + 18432];

	unsigned long long local10 = Inputs[addresss + 20480];
	unsigned long long local11 = Inputs[addresss + 22528];

	unsigned long long local12 = Inputs[addresss + 24576];
	unsigned long long local13 = Inputs[addresss + 26624];

	unsigned long long local14 = Inputs[addresss + 28672];
	unsigned long long local15 = Inputs[addresss + 30720];


	///////////////////////////////////////////////////////////////////////////////////////////////////



	// third part
	Butterfly_GS(local2, local6, local2, local6, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// forth part
	Butterfly_GS(local3, local7, local3, local7, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	// fifth part
	Butterfly_GS(local8, local12, local8, local12, PsiTable[m + ((8192 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// sixth part
	Butterfly_GS(local9, local13, local9, local13, PsiTable[m + ((10240 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// seventh part
	Butterfly_GS(local10, local14, local10, local14, PsiTable[m + ((12288 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	// eight part
	Butterfly_GS(local11, local15, local11, local15, PsiTable[m + ((14336 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m >> 1;
	t_2 = t_2 + 1;
	/// üçüncü for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////


	//dördüncü for başlar
	uint128_t U_prime;
	uint128_t V_prime;

	Butterfly_GS(local0, local8, local0, local8, PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local0, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(local8, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 16384] = V_prime.low;

	// second part
	Butterfly_GS(local1, local9, local1, local9, PsiTable[m + ((2048 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local1, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(local9, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 18432] = V_prime.low;

	// third part
	Butterfly_GS(local2, local10, local2, local10, PsiTable[m + ((4096 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local2, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = U_prime.low;

	mul64(local10, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 20480] = V_prime.low;

	// forth part
	Butterfly_GS(local3, local11, local3, local11, PsiTable[m + ((6144 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local3, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = U_prime.low;

	mul64(local11, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 22528] = V_prime.low;


	// fifth part
	Butterfly_GS(local4, local12, local4, local12, PsiTable[m + ((8192 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local4, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8192] = U_prime.low;

	mul64(local12, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 24576] = V_prime.low;


	// sixth part
	Butterfly_GS(local5, local13, local5, local13, PsiTable[m + ((10240 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local5, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10240] = U_prime.low;

	mul64(local13, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 26624] = V_prime.low;


	// seventh part
	Butterfly_GS(local6, local14, local6, local14, PsiTable[m + ((12288 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local6, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = U_prime.low;

	mul64(local14, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 28672] = V_prime.low;


	// eight part
	Butterfly_GS(local7, local15, local7, local15, PsiTable[m + ((14336 + i) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local7, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14336] = U_prime.low;

	mul64(local15, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 30720] = V_prime.low;




}


// NEW INVERSE
__global__ void INVERSE_NTT_DIM_NEW_Local_32768NEW(unsigned long long* Inputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
	int N, int N_power, int total_array, int M, int T, int q_count, unsigned long long* modinv)
{
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x + (1024 * blockIdx.x);
	int j = blockIdx.y;
	int k = blockIdx.z;

	int n_power = N_power;
	int m = M;
	int t_2 = T;

	unsigned long long q_thread = q_device[j];
	unsigned long long q_mu_thread = mu_device[j];
	unsigned long long q_bit_thread = q_bit_device[j];
	unsigned long long invv = modinv[j];

	int addresss = i + (((gridDim.y * k) + j) << n_power);

	register unsigned long long local[32];



	/// ilk for başlar
	Butterfly_GS(Inputs[addresss], Inputs[addresss + 2048], local[0], local[2], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 512], Inputs[addresss + 2560], local[1], local[3], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 4096], Inputs[addresss + 6144], local[4], local[6], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 4608], Inputs[addresss + 6656], local[5], local[7], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 8192], Inputs[addresss + 10240], local[8], local[10], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 8704], Inputs[addresss + 10752], local[9], local[11], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 12288], Inputs[addresss + 14336], local[12], local[14], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 12800], Inputs[addresss + 14848], local[13], local[15], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 16384], Inputs[addresss + 18432], local[16], local[18], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 16896], Inputs[addresss + 18944], local[17], local[19], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 20480], Inputs[addresss + 22528], local[20], local[22], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 20992], Inputs[addresss + 23040], local[21], local[23], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 24576], Inputs[addresss + 26624], local[24], local[26], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 25088], Inputs[addresss + 27136], local[25], local[27], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(Inputs[addresss + 28672], Inputs[addresss + 30720], local[28], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(Inputs[addresss + 29184], Inputs[addresss + 31232], local[29], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// ikinci for başlar
	Butterfly_GS(local[0], local[4], local[0], local[4], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[1], local[5], local[1], local[5], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[2], local[6], local[2], local[6], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[3], local[7], local[3], local[7], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[8], local[12], local[8], local[12], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[9], local[13], local[9], local[13], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[10], local[14], local[10], local[14], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[11], local[15], local[11], local[15], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[16], local[20], local[16], local[20], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[17], local[21], local[17], local[21], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[18], local[22], local[18], local[22], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[19], local[23], local[19], local[23], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[24], local[28], local[24], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[25], local[29], local[25], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[26], local[30], local[26], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[27], local[31], local[27], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	///başla

	m = m >> 1;
	t_2 = t_2 + 1;

	/// ikinci for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// üçüncü for başlar

	Butterfly_GS(local[0], local[8], local[0], local[8], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[1], local[9], local[1], local[9], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[2], local[10], local[2], local[10], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[3], local[11], local[3], local[11], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[4], local[12], local[4], local[12], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[5], local[13], local[5], local[13], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[6], local[14], local[6], local[14], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[7], local[15], local[7], local[15], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[16], local[24], local[16], local[24], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[17], local[25], local[17], local[25], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[18], local[26], local[18], local[26], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[19], local[27], local[19], local[27], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[20], local[28], local[20], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[21], local[29], local[21], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);


	Butterfly_GS(local[22], local[30], local[22], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);
	Butterfly_GS(local[23], local[31], local[23], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	m = m >> 1;
	t_2 = t_2 + 1;

	/// üçüncü for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// dördüncü for başlar
	uint128_t U_prime;
	uint128_t V_prime;

	Butterfly_GS(local[0], local[16], local[0], local[16], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[0], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(local[16], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 16384] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[1], local[17], local[1], local[17], PsiTable[m + (i >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[1], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 512] = U_prime.low;

	mul64(local[17], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 16896] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[2], local[18], local[2], local[18], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[2], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(local[18], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 18432] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[3], local[19], local[3], local[19], PsiTable[m + ((i + 2048) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[3], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2560] = U_prime.low;

	mul64(local[19], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 18944] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[4], local[20], local[4], local[20], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[4], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = U_prime.low;

	mul64(local[20], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 20480] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[5], local[21], local[5], local[21], PsiTable[m + ((i + 4096) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[5], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4608] = U_prime.low;

	mul64(local[21], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 20992] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[6], local[22], local[6], local[22], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[6], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = U_prime.low;

	mul64(local[22], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 22528] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[7], local[23], local[7], local[23], PsiTable[m + ((i + 6144) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[7], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6656] = U_prime.low;

	mul64(local[23], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 23040] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[8], local[24], local[8], local[24], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[8], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8192] = U_prime.low;

	mul64(local[24], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 24576] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[9], local[25], local[9], local[25], PsiTable[m + ((i + 8192) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[9], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8704] = U_prime.low;

	mul64(local[25], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 25088] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[10], local[26], local[10], local[26], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[10], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10240] = U_prime.low;

	mul64(local[26], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 26624] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[11], local[27], local[11], local[27], PsiTable[m + ((i + 10240) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[11], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10752] = U_prime.low;

	mul64(local[27], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 27136] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[12], local[28], local[12], local[28], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[12], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = U_prime.low;

	mul64(local[28], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 28672] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[13], local[29], local[13], local[29], PsiTable[m + ((i + 12288) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[13], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12800] = U_prime.low;

	mul64(local[29], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 29184] = V_prime.low;

	//---------------------------
	//---------------------------
	Butterfly_GS(local[14], local[30], local[14], local[30], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[14], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14336] = U_prime.low;

	mul64(local[30], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 30720] = V_prime.low;

	//---------------------------
	Butterfly_GS(local[15], local[31], local[15], local[31], PsiTable[m + ((i + 14336) >> t_2) + (j << n_power)], q_thread, q_mu_thread, q_bit_thread);

	mul64(local[15], invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14848] = U_prime.low;

	mul64(local[31], invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 31232] = V_prime.low;

	/// dördüncü for biter

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////- - - - 32768 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////- - - - - HOST FUNCTIONS - - - - -///////////////////////////////////////
// Inplace
__host__ void Forward_NTT(unsigned long long* input_device, unsigned long long* output_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count)
{
	dim3 numBlocks(2, q_count, input_count / q_count);
	dim3 numBlocks_shr(n / (1024 * 2), q_count, input_count / q_count);
	if (n == 4096) {
		int n_power = 12;
		FORWARD_NTT_DIM_LOCAL_4096 << < numBlocks, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		FORWARD_NTT_DIM_NEW_Shared_ALL << < numBlocks_shr, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 2, 2);

	}
	else if (n == 8192) {
		int n_power = 13;
		FORWARD_NTT_DIM_NEW_LOCAL_8192 << < numBlocks, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		FORWARD_NTT_DIM_NEW_Shared_ALL << < numBlocks_shr, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 8, 4);

	}
	else if (n == 16384) {
		int n_power = 14;
		FORWARD_NTT_DIM_NEW_LOCAL_16384 << < numBlocks, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		FORWARD_NTT_DIM_NEW_Shared_ALL << < numBlocks_shr, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 16, 8);
	}
	else if (n == 32768) {
		int n_power = 15;
		//FORWARD_NTT_DIM_NEW_LOCAL_32768_1 << < numBlocks, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		//FORWARD_NTT_DIM_NEW_LOCAL_32768_2 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);

		FORWARD_NTT_DIM_NEW_LOCAL_32768NEW << < numBlocks, 512 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		FORWARD_NTT_DIM_NEW_Shared_ALL << < numBlocks_shr, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 8, 16);
	}
}

// Inplace
__host__ void Inverse_NTT(unsigned long long* input_device, unsigned long long* output_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count, unsigned long long* modinv)
{
	dim3 numBlocks_shr(n / (1024 * 2), q_count, input_count / q_count);
	dim3 numBlocks(2, q_count, input_count / q_count);
	if (n == 4096) {
		int n_power = 12;
		INVERSE_NTT_DIM_NEW_Shared << < numBlocks_shr, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INVERSE_NTT_DIM_NEW_Local_4096 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 1, 11, q_count, modinv);
	}
	else if (n == 8192) {
		int n_power = 13;
		INVERSE_NTT_DIM_NEW_Shared << < numBlocks_shr, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INVERSE_NTT_DIM_NEW_Local_8192 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 2, 11, q_count, modinv);
	}
	else if (n == 16384) {
		int n_power = 14;
		INVERSE_NTT_DIM_NEW_Shared << < numBlocks_shr, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INVERSE_NTT_DIM_NEW_Local_16384 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 4, 11, q_count, modinv);
	}
	else if (n == 32768) {
		int n_power = 15;
		INVERSE_NTT_DIM_NEW_Shared << < numBlocks_shr, 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		//INVERSE_NTT_DIM_NEW_Local_32768_1 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 8, 11, q_count);
		//INVERSE_NTT_DIM_NEW_Local_32768_2 << < numBlocks, 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 2, 13, q_count, modinv);
		INVERSE_NTT_DIM_NEW_Local_32768NEW << < numBlocks, 512 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 8, 11, q_count, modinv);

	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////

__host__ void Forward_NTT_Inplace(unsigned long long* input_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count)
{
	Forward_NTT(input_device, input_device, q_device, mu_device, q_bit_device, n, psitable_device, input_count, q_count);
}

__host__ void Inverse_NTT_Inplace(unsigned long long* input_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count, unsigned long long* modinv)
{
	Inverse_NTT(input_device, input_device, q_device, mu_device, q_bit_device, n, psitable_device, input_count, q_count, modinv);
}

