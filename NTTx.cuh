#pragma once

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "Barret.cuh"

/////////////////////////////////////////////////////START/////////////////////////////////////////////////////*
////////////////////////////////////////////- - - - - 4096 - - - - -///////////////////////////////////////////*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////*
//FORWARD
__global__ void NTT_4096_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	unsigned long long U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;


	int n_power = N_power;
	int t_2 = n_power - 2; /// buraları parametrik yapp
	int t = 1 << t_2;
	int n = N;
	int m = M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = idx >> (n_power - 1);
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];


	int modidx_t = modidx >> t_2;
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = (local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	sharedmemorys[threadIdx.x] = Inputs[address];
	sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;




	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;







	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	Inputs[address] = U_prime;
	Inputs[address + t] = V_prime.low;

}


__global__ void NTT_4096_REG_MULTIx(unsigned long long* Inputs, unsigned long long* Output, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned long long localmemorysx_0, localmemorysx_1;

	unsigned long long U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi; // Psi a ayar çek

	int shrd_read;
	int address_psi;
	int	adress;

	int n_power = N_power;
	int n_2 = N >> 1;
	int m = 1;
	int t_2 = n_power - 1;
	int t = 1 << t_2;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];


	//Input 
	/////////////////////////////////////////
	shrd_read = modidx + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_0 = Inputs[shrd_read];
	localmemorysx_1 = Inputs[shrd_read + 2048];
	/////////////////////////////////////////

	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_0;
	V = localmemorysx_1;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_0 = U_prime;
	localmemorysx_1 = V_prime.low;

	// OUTPUT
	/////////////////////////////////
	shrd_read = modidx + (0 << 12) + (modidx_N << n_power); // 11 ==> 2048
	Output[shrd_read] = localmemorysx_0;
	Output[shrd_read + 2048] = localmemorysx_1;
	/////////////////////////////////


}

//INVERSE
__global__ void INTT_4096_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;

	int n_power = N_power;
	int t_2 = 0;//n_power - 3
	int t = 1 << t_2;
	int n = N;
	int m = n >> 1;//M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = int(idx >> (n_power - 1));
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];


	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	//sharedmemorys[threadIdx.x] = Inputs[address];
	//sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size
	sharedmemorys[shrd_address] = Inputs[address];
	sharedmemorys[shrd_address + t] = Inputs[address + t]; // 1024 block size
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;


	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;


	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Outputs[address] = U_prime.low;
	Outputs[address + t] = V_prime.low;
	__syncthreads();

}



__global__ void INTT_4096_REG_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned long long* modinv, int N, int N_power, int total_array, int M, int T, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//unsigned long long localmemorysx[8];
	unsigned long long localmemorysx_0, localmemorysx_1;

	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long Psi; // Psi a ayar çek

	int shrd_read;
	int address_psi;
	int	adress;


	int n_power = N_power;
	int n_2 = N >> 1;
	int m = M;//1;
	int t_2 = T;//n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];
	unsigned long long invv = modinv[idx_q];

	//Input 
	/////////////////////////////////////////
	shrd_read = modidx + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_0 = Inputs[shrd_read];
	localmemorysx_1 = Inputs[shrd_read + 2048];
	/////////////////////////////////////////

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	//Psi = PsiTable[m + address_psi];

	U_prime = localmemorysx_0 + localmemorysx_1;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = localmemorysx_0 + q_thread * (localmemorysx_0 < localmemorysx_1);
	V_prime = V_prime - localmemorysx_1;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[shrd_read] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[shrd_read + 2048] = V_prime.low;

	// OUTPUT
	/////////////////////////////////
	//shrd_read = modidx + (0 << 12) + (modidx_N << n_power); // 11 ==> 2048
	//Inputs[shrd_read] = localmemorysx_0;
	//Inputs[shrd_read + 2048] = localmemorysx_1;
	/////////////////////////////////



}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////- - - - - 4096 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////




/////////////////////////////////////////////////////START/////////////////////////////////////////////////////
////////////////////////////////////////////- - - - - 8192 - - - - -///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//FORWARD
__global__ void NTT_8192_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	unsigned long long U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;


	int n_power = N_power;
	int t_2 = n_power - 3;
	int t = 1 << t_2;
	int n = N;
	int m = M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = idx >> (n_power - 1);
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];


	int modidx_t = modidx >> t_2;
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = (local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	sharedmemorys[threadIdx.x] = Inputs[address];
	sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;


	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;




	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = modidx >> t_2;
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = (local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;







	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	Inputs[address] = U_prime;
	Inputs[address + t] = V_prime.low;

}








__global__ void NTT_8192_REG_MULTIx(unsigned long long* Inputs, unsigned long long* Output, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned long long U_prime;
	uint128_t V_prime;

	int address_psi;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int addresss = modidx + (modidx_N << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	/// ilk for başlar


	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	mul64(local2, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = local0 + V_prime.low;
	U_prime -= (U_prime >= q_thread) * q_thread;

	local0 = local0 + q_thread;
	V_prime = local0 - V_prime.low;
	V_prime -= (V_prime >= q_thread) * q_thread;

	local0 = U_prime;
	local2 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	mul64(local3, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = local1 + V_prime.low;
	U_prime -= (U_prime >= q_thread) * q_thread;

	local1 = local1 + q_thread;
	V_prime = local1 - V_prime.low;
	V_prime -= (V_prime >= q_thread) * q_thread;

	local1 = U_prime;
	local3 = V_prime.low;

	m = m << 1;
	t_2 = t_2 - 1;

	/// ilk for biter

	/////////////////////////////////

	/// ikinci for başlar


	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	mul64(local1, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = local0 + V_prime.low;
	U_prime -= (U_prime >= q_thread) * q_thread;

	local0 = local0 + q_thread;
	V_prime = local0 - V_prime.low;
	V_prime -= (V_prime >= q_thread) * q_thread;

	Output[addresss] = U_prime;
	Output[addresss + 2048] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	mul64(local3, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = local2 + V_prime.low;
	U_prime -= (U_prime >= q_thread) * q_thread;

	local2 = local2 + q_thread;
	V_prime = local2 - V_prime.low;
	V_prime -= (V_prime >= q_thread) * q_thread;

	Output[addresss + 4096] = U_prime;
	Output[addresss + 6144] = V_prime.low;

	/// ikinci for biter

}

//INVERSE
__global__ void INTT_8192_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;

	int n_power = N_power;
	int t_2 = 0;//n_power - 3
	int t = 1 << t_2;
	int n = N;
	int m = n >> 1;//M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = int(idx >> (n_power - 1));
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	//sharedmemorys[threadIdx.x] = Inputs[address];
	//sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size
	sharedmemorys[shrd_address] = Inputs[address];
	sharedmemorys[shrd_address + t] = Inputs[address + t]; // 1024 block size
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;


	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;


	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Outputs[address] = U_prime.low;
	Outputs[address + t] = V_prime.low;
	__syncthreads();

}


__global__ void INTT_8192_REG_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned long long* modinv, int N, int N_power, int total_array, int M, int T, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint128_t U_prime;
	uint128_t V_prime;

	int address_psi;



	int n_power = N_power;
	int m = M;//1;
	int t_2 = T;//n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];
	unsigned long long invv = modinv[idx_q];

	int addresss = modidx + (modidx_N << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	/// ilk for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local1;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local1);
	V_prime = V_prime - local1;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	local1 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local2 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local2 = U_prime.low;
	local3 = V_prime.low;


	//m = m << 1;
	//t_2 = t_2 - 1;
	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	/////////////////////////////////

	/// ikinci for başlar


	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local2;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local2);
	V_prime = V_prime - local2;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	//Inputs[addresss] = (U_prime.low >> 1) + (((q_thread + 1) >> 1) * (U_prime.low & 1));
	//Inputs[addresss + 2048] = (V_prime.low >> 1) + (((q_thread + 1) >> 1) * (V_prime.low & 1));


	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	//Inputs[addresss + 4096] = (U_prime.low >> 1) + (((q_thread + 1) >> 1) * (U_prime.low & 1));
	//Inputs[addresss + 6144] = (V_prime.low >> 1) + (((q_thread + 1) >> 1) * (V_prime.low & 1));


	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
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

__global__ void NTT_16384_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;


	int n_power = N_power;
	int t_2 = n_power - 4;
	int t = 1 << t_2;
	int n = N;
	int m = M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = int(idx >> (n_power - 1));
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	sharedmemorys[threadIdx.x] = Inputs[address];
	sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;




	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	Inputs[address] = U_prime.low;
	Inputs[address + t] = V_prime.low;

}


__global__ void NTT_16384_REG_MULTIx(unsigned long long* Inputs, unsigned long long* Output, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//unsigned long long localmemorysx[8];
	unsigned long long localmemorysx_0, localmemorysx_1, localmemorysx_2, localmemorysx_3, localmemorysx_4, localmemorysx_5, localmemorysx_6, localmemorysx_7;

	unsigned long long U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi; // Psi a ayar çek

	int shrd_read;
	int address_psi;
	int	adress;
	int n_power = N_power;
	int modidx_N = idx >> 11; // 11 ==> 2048
	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];



	int n_2 = N >> 1;
	int m = 1;
	int t_2 = n_power - 1;
	int t = 1 << t_2;
	int modidx = idx % 2048;


	//Input 
	/////////////////////////////////////////
	shrd_read = modidx + (0) + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_0 = Inputs[shrd_read];
	localmemorysx_1 = Inputs[shrd_read + 2048];


	shrd_read = modidx + (1 << 12) + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_2 = Inputs[shrd_read];
	localmemorysx_3 = Inputs[shrd_read + 2048];


	shrd_read = modidx + (2 << 12) + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_4 = Inputs[shrd_read];
	localmemorysx_5 = Inputs[shrd_read + 2048];


	shrd_read = modidx + (3 << 12) + (modidx_N << n_power); // 11 ==> 2048
	localmemorysx_6 = Inputs[shrd_read];
	localmemorysx_7 = Inputs[shrd_read + 2048];
	/////////////////////////////////////////


	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_0;
	V = localmemorysx_4;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_0 = U_prime;
	localmemorysx_4 = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_1;
	V = localmemorysx_5;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_1 = U_prime;
	localmemorysx_5 = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_2;
	V = localmemorysx_6;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_2 = U_prime;
	localmemorysx_6 = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_3;
	V = localmemorysx_7;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_3 = U_prime;
	localmemorysx_7 = V_prime.low;

	//__syncthreads();
	m = m << 1;
	t_2 = t_2 - 1;

	////////////////////////////////////////////


	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_0;
	V = localmemorysx_2;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_0 = U_prime;
	localmemorysx_2 = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_1;
	V = localmemorysx_3;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_1 = U_prime;
	localmemorysx_3 = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_4;
	V = localmemorysx_6;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_4 = U_prime;
	localmemorysx_6 = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_5;
	V = localmemorysx_7;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_5 = U_prime;
	localmemorysx_7 = V_prime.low;

	//__syncthreads();
	m = m << 1;
	t_2 = t_2 - 1;


	////////////////////////////////////////////



	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_0;
	V = localmemorysx_1;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_0 = U_prime;
	localmemorysx_1 = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_2;
	V = localmemorysx_3;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_2 = U_prime;
	localmemorysx_3 = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_4;
	V = localmemorysx_5;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_4 = U_prime;
	localmemorysx_5 = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx_6;
	V = localmemorysx_7;
	Psi = PsiTable[m + address_psi + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx_6 = U_prime;
	localmemorysx_7 = V_prime.low;


	////////////////////////////////////////////

	// OUTPUT
	/////////////////////////////////
	shrd_read = modidx + (0 << 12) + (modidx_N << n_power); // 11 ==> 2048
	Output[shrd_read] = localmemorysx_0;
	Output[shrd_read + 2048] = localmemorysx_1;

	shrd_read = modidx + (1 << 12) + (modidx_N << n_power); // 11 ==> 2048
	Output[shrd_read] = localmemorysx_2;
	Output[shrd_read + 2048] = localmemorysx_3;

	shrd_read = modidx + (2 << 12) + (modidx_N << n_power); // 11 ==> 2048
	Output[shrd_read] = localmemorysx_4;
	Output[shrd_read + 2048] = localmemorysx_5;

	shrd_read = modidx + (3 << 12) + (modidx_N << n_power); // 11 ==> 2048
	Output[shrd_read] = localmemorysx_6;
	Output[shrd_read + 2048] = localmemorysx_7;
	/////////////////////////////////

}

//INVERSE
__global__ void INTT_16384_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;


	int n_power = N_power;
	int t_2 = 0;//n_power - 3
	int t = 1 << t_2;
	int n = N;
	int m = n >> 1;//M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = int(idx >> (n_power - 1));
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	//sharedmemorys[threadIdx.x] = Inputs[address];
	//sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size
	sharedmemorys[shrd_address] = Inputs[address];
	sharedmemorys[shrd_address + t] = Inputs[address + t]; // 1024 block size
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;


	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;


	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Outputs[address] = U_prime.low;
	Outputs[address + t] = V_prime.low;
	__syncthreads();

}


__global__ void INTT_16384_REG_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned long long* modinv, int N, int N_power, int total_array, int M, int T, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint128_t U_prime;
	uint128_t V_prime;

	int address_psi;


	int n_power = N_power;
	int m = M;//1;
	int t_2 = T;//n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];
	unsigned long long invv = modinv[idx_q];

	int addresss = modidx + (modidx_N << n_power);

	unsigned long long local0 = Inputs[addresss];
	unsigned long long local1 = Inputs[addresss + 2048];

	unsigned long long local2 = Inputs[addresss + 4096];
	unsigned long long local3 = Inputs[addresss + 6144];

	unsigned long long local4 = Inputs[addresss + 8192];
	unsigned long long local5 = Inputs[addresss + 10240];

	unsigned long long local6 = Inputs[addresss + 12288];
	unsigned long long local7 = Inputs[addresss + 14336];

	/// ilk for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local1;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local1);
	V_prime = V_prime - local1;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	local1 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local2 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local2 = U_prime.low;
	local3 = V_prime.low;


	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local4 + local5;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local4 + q_thread * (local4 < local5);
	V_prime = V_prime - local5;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local4 = U_prime.low;
	local5 = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local6 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local6 + q_thread * (local6 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local6 = U_prime.low;
	local7 = V_prime.low;



	//m = m << 1;
	//t_2 = t_2 - 1;
	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// ikinci for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local2;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local2);
	V_prime = V_prime - local2;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	local2 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local1 = U_prime.low;
	local3 = V_prime.low;


	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local4 + local6;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local4 + q_thread * (local4 < local6);
	V_prime = V_prime - local6;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local4 = U_prime.low;
	local6 = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local5 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local5 + q_thread * (local5 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local5 = U_prime.low;
	local7 = V_prime.low;

	//m = m << 1;
	//t_2 = t_2 - 1;
	m = m >> 1;
	t_2 = t_2 + 1;


	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// üçüncü for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local4;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local4);
	V_prime = V_prime - local4;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8192] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local5;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local5);
	V_prime = V_prime - local5;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);


	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10240] = V_prime.low;



	// third part

	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local2 + local6;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local6);
	V_prime = V_prime - local6;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);


	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = V_prime.low;



	// third part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local3 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local3 + q_thread * (local3 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);


	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
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
__global__ void NTT_32768_REG_MULTI_1x(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	register unsigned long long localmemorysx[16];
	//unsigned long long localmemorysx_0, localmemorysx_1, localmemorysx_2, localmemorysx_3, localmemorysx_4, localmemorysx_5, localmemorysx_6, localmemorysx_7, localmemorysx_8, localmemorysx_9, localmemorysx_10, localmemorysx_11, localmemorysx_12, localmemorysx_13, localmemorysx_14, localmemorysx_15;

	unsigned long long U_prime;
	uint128_t V_prime;


	int address_psi;
	int	adress;

	int n_power = N_power;
	int m = 1;
	int t_2 = n_power - 1;
	int modidx = idx % 2048;
	//int modidx_N = idx >> 11; // 11 ==> 2048

	//int idx_q = modidx_N % q_count;
	//int idx_psi = idx_q << n_power;

	//(((idx >> 11) % q_count) << n_power)
	//(idx >> 11)

	unsigned long long q_thread = q_device[((idx >> 11) % q_count)];
	unsigned long long q_mu_thread = mu_device[((idx >> 11) % q_count)];
	unsigned long long q_bit_thread = q_bit_device[((idx >> 11) % q_count)];


	//Input 
	/////////////////////////////////////////
	//int shrd_read = modidx + (0) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[0] = Inputs[modidx + (0) + ((idx >> 11) << n_power)];
	localmemorysx[1] = Inputs[modidx + (0) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (1 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[2] = Inputs[modidx + (1 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[3] = Inputs[modidx + (1 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (2 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[4] = Inputs[modidx + (2 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[5] = Inputs[modidx + (2 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (3 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[6] = Inputs[modidx + (3 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[7] = Inputs[modidx + (3 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (4 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[8] = Inputs[modidx + (4 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[9] = Inputs[modidx + (4 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (5 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[10] = Inputs[modidx + (5 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[11] = Inputs[modidx + (5 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (6 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[12] = Inputs[modidx + (6 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[13] = Inputs[modidx + (6 << 12) + ((idx >> 11) << n_power) + 2048];


	//shrd_read = modidx + (7 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	localmemorysx[14] = Inputs[modidx + (7 << 12) + ((idx >> 11) << n_power)];
	localmemorysx[15] = Inputs[modidx + (7 << 12) + ((idx >> 11) << n_power) + 2048];
	/////////////////////////////////////////


	///////////////////////////////////////////////////////////////////
	//MAIN OPERATION

	// 0 
	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	unsigned long long U = localmemorysx[0];
	unsigned long long V = localmemorysx[8];
	unsigned long long Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[0] = U_prime;
	localmemorysx[8] = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[1];
	V = localmemorysx[9];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[1] = U_prime;
	localmemorysx[9] = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[2];
	V = localmemorysx[10];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[2] = U_prime;
	localmemorysx[10] = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[3];
	V = localmemorysx[11];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[3] = U_prime;
	localmemorysx[11] = V_prime.low;

	//__syncthreads();

	address_psi = ((4 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[4];
	V = localmemorysx[12];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[4] = U_prime;
	localmemorysx[12] = V_prime.low;

	//__syncthreads();

	address_psi = ((5 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[5];
	V = localmemorysx[13];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[5] = U_prime;
	localmemorysx[13] = V_prime.low;

	//__syncthreads();

	address_psi = ((6 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[6];
	V = localmemorysx[14];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[6] = U_prime;
	localmemorysx[14] = V_prime.low;

	//__syncthreads();

	address_psi = ((7 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[7];
	V = localmemorysx[15];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[7] = U_prime;
	localmemorysx[15] = V_prime.low;

	//__syncthreads();
	m = m << 1;
	t_2 = t_2 - 1;

	////////////////////////////////////////////
	// 1

	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[0];
	V = localmemorysx[4];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[0] = U_prime;
	localmemorysx[4] = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[1];
	V = localmemorysx[5];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[1] = U_prime;
	localmemorysx[5] = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[2];
	V = localmemorysx[6];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[2] = U_prime;
	localmemorysx[6] = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[3];
	V = localmemorysx[7];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[3] = U_prime;
	localmemorysx[7] = V_prime.low;

	//__syncthreads();

	address_psi = ((4 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[8];
	V = localmemorysx[12];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[8] = U_prime;
	localmemorysx[12] = V_prime.low;

	//__syncthreads();

	address_psi = ((5 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[9];
	V = localmemorysx[13];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[9] = U_prime;
	localmemorysx[13] = V_prime.low;

	//__syncthreads();

	address_psi = ((6 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[10];
	V = localmemorysx[14];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[10] = U_prime;
	localmemorysx[14] = V_prime.low;

	//__syncthreads();

	address_psi = ((7 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[11];
	V = localmemorysx[15];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[11] = U_prime;
	localmemorysx[15] = V_prime.low;

	//__syncthreads();
	m = m << 1;
	t_2 = t_2 - 1;

	////////////////////////////////////////////
	// 2

	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[0];
	V = localmemorysx[2];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[0] = U_prime;
	localmemorysx[2] = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[1];
	V = localmemorysx[3];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[1] = U_prime;
	localmemorysx[3] = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[4];
	V = localmemorysx[6];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[4] = U_prime;
	localmemorysx[6] = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[5];
	V = localmemorysx[7];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[5] = U_prime;
	localmemorysx[7] = V_prime.low;

	//__syncthreads();

	address_psi = ((4 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[8];
	V = localmemorysx[10];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[8] = U_prime;
	localmemorysx[10] = V_prime.low;

	//__syncthreads();

	address_psi = ((5 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[9];
	V = localmemorysx[11];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[9] = U_prime;
	localmemorysx[11] = V_prime.low;

	//__syncthreads();

	address_psi = ((6 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[12];
	V = localmemorysx[14];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[12] = U_prime;
	localmemorysx[14] = V_prime.low;

	//__syncthreads();

	address_psi = ((7 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[13];
	V = localmemorysx[15];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[13] = U_prime;
	localmemorysx[15] = V_prime.low;

	//__syncthreads();
	/*
	m = m << 1;
	t_2 = t_2 - 1;


	////////////////////////////////////////////
	// 3

	address_psi = ((0 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[0];
	V = localmemorysx[1];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[0] = U_prime;
	localmemorysx[1] = V_prime.low;

	//__syncthreads();

	address_psi = ((1 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[2];
	V = localmemorysx[3];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[2] = U_prime;
	localmemorysx[3] = V_prime.low;

	//__syncthreads();


	address_psi = ((2 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[4];
	V = localmemorysx[5];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[4] = U_prime;
	localmemorysx[5] = V_prime.low;

	//__syncthreads();

	address_psi = ((3 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[6];
	V = localmemorysx[7];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[6] = U_prime;
	localmemorysx[7] = V_prime.low;

	//__syncthreads();

	address_psi = ((4 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[8];
	V = localmemorysx[9];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[8] = U_prime;
	localmemorysx[9] = V_prime.low;

	//__syncthreads();

	address_psi = ((5 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[10];
	V = localmemorysx[11];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[10] = U_prime;
	localmemorysx[11] = V_prime.low;

	//__syncthreads();

	address_psi = ((6 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[12];
	V = localmemorysx[13];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[12] = U_prime;
	localmemorysx[13] = V_prime.low;

	//__syncthreads();

	address_psi = ((7 << 11) + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U = localmemorysx[14];
	V = localmemorysx[15];
	Psi = PsiTable[m + address_psi + (((idx >> 11) % q_count) << n_power)];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	localmemorysx[14] = U_prime;
	localmemorysx[15] = V_prime.low;

	//__syncthreads();
	*/


	// Output 
	/////////////////////////////////////////
	//shrd_read = modidx + (0) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (0) + ((idx >> 11) << n_power)] = localmemorysx[0];
	Outputs[modidx + (0) + ((idx >> 11) << n_power) + 2048] = localmemorysx[1];


	//shrd_read = modidx + (1 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (1 << 12) + ((idx >> 11) << n_power)] = localmemorysx[2];
	Outputs[modidx + (1 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[3];


	//shrd_read = modidx + (2 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (2 << 12) + ((idx >> 11) << n_power)] = localmemorysx[4];
	Outputs[modidx + (2 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[5];


	//shrd_read = modidx + (3 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (3 << 12) + ((idx >> 11) << n_power)] = localmemorysx[6];
	Outputs[modidx + (3 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[7];


	//shrd_read = modidx + (4 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (4 << 12) + ((idx >> 11) << n_power)] = localmemorysx[8];
	Outputs[modidx + (4 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[9];


	//shrd_read = modidx + (5 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (5 << 12) + ((idx >> 11) << n_power)] = localmemorysx[10];
	Outputs[modidx + (5 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[11];


	//shrd_read = modidx + (6 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (6 << 12) + ((idx >> 11) << n_power)] = localmemorysx[12];
	Outputs[modidx + (6 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[13];


	//shrd_read = modidx + (7 << 12) + ((idx >> 11) << n_power); // 11 ==> 2048
	Outputs[modidx + (7 << 12) + ((idx >> 11) << n_power)] = localmemorysx[14];
	Outputs[modidx + (7 << 12) + ((idx >> 11) << n_power) + 2048] = localmemorysx[15];
	/////////////////////////////////////////




}


__global__ void NTT_32768_REG_MULTI_2x(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//__shared__ unsigned long long sharedmemorys[1024];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;

	int t = T;
	int modidx = idx % (N / 2);
	int modidx_N = int(idx / (N / 2));
	int dividx = modidx_N * N;
	int dixidx_t = int(modidx / t) * t;
	int modidx_t = int(modidx / t);


	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << N_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int m = M;
	int address = dividx + dixidx_t + modidx;

	U = Inputs[address];
	V = Inputs[address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	Inputs[address] = U_prime.low;
	Inputs[address + t] = V_prime.low;

}


__global__ void NTT_32768_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* PsiTable, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, int N, int N_power, int total_array, int q_count, int T, int M)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;

	int n_power = N_power;
	int modidx_N = int(idx >> (n_power - 1));
	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];


	int t_2 = n_power - 5;
	int t = 1 << t_2;
	int n = N;
	int m = M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;

	int dividx = modidx_N << n_power;

	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	sharedmemorys[threadIdx.x] = Inputs[address];
	sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;




	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	t = t >> 1;
	m = m << 1;
	t_2 -= 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	mul64(V, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	U_prime = U + V_prime.low;
	U_prime -= q_thread * (U_prime >= q_thread);

	U = U + q_thread;
	V_prime = U - V_prime.low;
	V_prime -= q_thread * (V_prime >= q_thread);

	Inputs[address] = U_prime.low;
	Inputs[address + t] = V_prime.low;

}


//INVERSE


__global__ void INTT_32768_SMEM_MULTIx(unsigned long long* Inputs, unsigned long long* Outputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
	int N, int N_power, int total_array, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int local_idx = threadIdx.x;

	__shared__ unsigned long long sharedmemorys[2048];
	uint128_t U_prime;
	uint128_t V_prime;
	unsigned long long U;
	unsigned long long V;
	unsigned long long Psi;



	int n_power = N_power;
	int t_2 = 0;//n_power - 3
	int t = 1 << t_2;
	int n = N;
	int m = n >> 1;//M;
	int n_2 = n >> 1;

	int modidx = idx % n_2;
	int modidx_N = int(idx >> (n_power - 1));
	int dividx = modidx_N << n_power;

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int modidx_t = int(modidx >> t_2);
	int dixidx_t = modidx_t << t_2;

	int address = dividx + dixidx_t + modidx;

	int shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	int shrd_address = shrd_dixidx_t + local_idx;


	//sharedmemorys[threadIdx.x] = Inputs[address];
	//sharedmemorys[threadIdx.x + 1024] = Inputs[address + t]; // 1024 block size
	sharedmemorys[shrd_address] = Inputs[address];
	sharedmemorys[shrd_address + t] = Inputs[address + t]; // 1024 block size
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;


	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;



	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;


	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();


	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();



	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	sharedmemorys[shrd_address] = U_prime.low;
	sharedmemorys[shrd_address + t] = V_prime.low;

	//t = t >> 1;
	//m = m << 1;
	//t_2 -= 1;
	t = t << 1;
	m = m >> 1;
	t_2 += 1;
	modidx_t = int(modidx >> t_2);
	dixidx_t = modidx_t << t_2;
	address = dividx + dixidx_t + modidx;

	shrd_dixidx_t = int(local_idx >> t_2) << t_2;
	shrd_address = shrd_dixidx_t + local_idx;
	__syncthreads();

	U = sharedmemorys[shrd_address];
	V = sharedmemorys[shrd_address + t];
	Psi = PsiTable[m + modidx_t + idx_psi];

	U_prime = U + V;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = U + q_thread * (U < V);
	V_prime = V_prime - V;

	mul64(V_prime.low, Psi, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Outputs[address] = U_prime.low;
	Outputs[address + t] = V_prime.low;
	__syncthreads();

}


__global__ void INTT_32768_REG_MULTI_1x(unsigned long long* Inputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device,
	int N, int N_power, int total_array, int M, int T, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint128_t U_prime;
	uint128_t V_prime;

	int address_psi;


	int n_power = N_power;
	int m = M;//1;
	int t_2 = T;//n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;
	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];

	int addresss = modidx + (modidx_N << n_power);

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

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local1;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local1);
	V_prime = V_prime - local1;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	local1 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local2 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local2 = U_prime.low;
	local3 = V_prime.low;


	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local4 + local5;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local4 + q_thread * (local4 < local5);
	V_prime = V_prime - local5;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local4 = U_prime.low;
	local5 = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local6 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local6 + q_thread * (local6 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local6 = U_prime.low;
	local7 = V_prime.low;


	// fifth part

	address_psi = (8192 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local8 + local9;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local8 + q_thread * (local8 < local9);
	V_prime = V_prime - local9;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local8 = U_prime.low;
	local9 = V_prime.low;


	// sixth part

	address_psi = (10240 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local10 + local11;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local10 + q_thread * (local10 < local11);
	V_prime = V_prime - local11;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local10 = U_prime.low;
	local11 = V_prime.low;


	// seventh part

	address_psi = (12288 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local12 + local13;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local12 + q_thread * (local12 < local13);
	V_prime = V_prime - local13;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local12 = U_prime.low;
	local13 = V_prime.low;


	// eight part

	address_psi = (14336 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local14 + local15;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local14 + q_thread * (local14 < local15);
	V_prime = V_prime - local15;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local14 = U_prime.low;
	local15 = V_prime.low;


	//m = m << 1;
	//t_2 = t_2 - 1;
	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// ikinci for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local2;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local2);
	V_prime = V_prime - local2;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	Inputs[addresss + 4096] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local3;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local3);
	V_prime = V_prime - local3;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local1 = U_prime.low;
	Inputs[addresss + 6144] = V_prime.low;


	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local4 + local6;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local4 + q_thread * (local4 < local6);
	V_prime = V_prime - local6;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local4 = U_prime.low;
	Inputs[addresss + 12288] = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local5 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local5 + q_thread * (local5 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local5 = U_prime.low;
	Inputs[addresss + 14336] = V_prime.low;


	// fifth part

	address_psi = (8192 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local8 + local10;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local8 + q_thread * (local8 < local10);
	V_prime = V_prime - local10;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 16384] = U_prime.low;
	Inputs[addresss + 20480] = V_prime.low;


	// sixth part

	address_psi = (10240 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local9 + local11;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local9 + q_thread * (local9 < local11);
	V_prime = V_prime - local11;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 18432] = U_prime.low;
	Inputs[addresss + 22528] = V_prime.low;


	// seventh part

	address_psi = (12288 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local12 + local14;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local12 + q_thread * (local12 < local14);
	V_prime = V_prime - local14;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 24576] = U_prime.low;
	Inputs[addresss + 28672] = V_prime.low;


	// eight part

	address_psi = (14336 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local13 + local15;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local13 + q_thread * (local13 < local15);
	V_prime = V_prime - local15;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 26624] = U_prime.low;
	Inputs[addresss + 30720] = V_prime.low;

	///başla

	m = m >> 1;
	t_2 = t_2 + 1;

	/// ilk for biter

	///////////////////////////////////////////////////////////////////////////////////////////////////

	/// üçüncü for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local4;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local4);
	V_prime = V_prime - local4;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss] = U_prime.low;
	Inputs[addresss + 8192] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local5;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local5);
	V_prime = V_prime - local5;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 2048] = U_prime.low;
	Inputs[addresss + 10240] = V_prime.low;

	/*
	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local2 + local6;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local6);
	V_prime = V_prime - local6;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 4096] = U_prime.low;
	Inputs[addresss + 12288] = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local3 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local3 + q_thread * (local3 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 6144] = U_prime.low;
	Inputs[addresss + 14336] = V_prime.low;
	*/

	/// ///////////////////////////////////////////////////////////

	//Inputs[addresss + 4096] = local2;
	//Inputs[addresss + 12288] = local6;

	//Inputs[addresss + 6144] = local3;
	//Inputs[addresss + 14336] = local7;

	//Inputs[addresss + 16384] = local8;
	//Inputs[addresss + 24576] = local12;


	//Inputs[addresss + 18432] = local9;
	//Inputs[addresss + 26624] = local13;


	//Inputs[addresss + 20480] = local10;
	//Inputs[addresss + 28672] = local14;


	//Inputs[addresss + 22528] = local11;
	//Inputs[addresss + 30720] = local15;

	/*
	// fifth part

	address_psi = (8192 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local8 + local10;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local8 + q_thread * (local8 < local10);
	V_prime = V_prime - local10;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 16384] = U_prime.low;
	Inputs[addresss + 20480] = V_prime.low;


	// sixth part

	address_psi = (10240 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local9 + local11;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local9 + q_thread * (local9 < local11);
	V_prime = V_prime - local11;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 18432] = U_prime.low;
	Inputs[addresss + 22528] = V_prime.low;


	// seventh part

	address_psi = (12288 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local12 + local14;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local12 + q_thread * (local12 < local14);
	V_prime = V_prime - local14;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 24576] = U_prime.low;
	Inputs[addresss + 28672] = V_prime.low;


	// eight part

	address_psi = (14336 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local13 + local15;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local13 + q_thread * (local13 < local15);
	V_prime = V_prime - local15;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	Inputs[addresss + 26624] = U_prime.low;
	Inputs[addresss + 30720] = V_prime.low;
	*/
}


__global__ void INTT_32768_REG_MULTI_2x(unsigned long long* Inputs, unsigned long long* PsiTable,
	unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned long long* modinv,
	int N, int N_power, int total_array, int M, int T, int q_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint128_t U_prime;
	uint128_t V_prime;

	int address_psi;



	int n_power = N_power;
	int m = M;//1;
	int t_2 = T;//n_power - 1;
	int modidx = idx % 2048;
	int modidx_N = idx >> 11; // 11 ==> 2048

	int idx_q = modidx_N % q_count;
	int idx_psi = idx_q << n_power;

	unsigned long long q_thread = q_device[idx_q];
	unsigned long long q_mu_thread = mu_device[idx_q];
	unsigned long long q_bit_thread = q_bit_device[idx_q];
	unsigned long long invv = modinv[idx_q];

	int addresss = modidx + (modidx_N << n_power);

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

	/// üçüncü for başlar
	/*
	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local4;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local4);
	V_prime = V_prime - local4;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local0 = U_prime.low;
	local4 = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local5;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local5);
	V_prime = V_prime - local5;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local1 = U_prime.low;
	local5 = V_prime.low;

	*/
	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local2 + local6;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local6);
	V_prime = V_prime - local6;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local2 = U_prime.low;
	local6 = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local3 + local7;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local3 + q_thread * (local3 < local7);
	V_prime = V_prime - local7;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local3 = U_prime.low;
	local7 = V_prime.low;


	// fifth part

	address_psi = (8192 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local8 + local12;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local8 + q_thread * (local8 < local12);
	V_prime = V_prime - local12;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local8 = U_prime.low;
	local12 = V_prime.low;


	// sixth part

	address_psi = (10240 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local9 + local13;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local9 + q_thread * (local9 < local13);
	V_prime = V_prime - local13;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local9 = U_prime.low;
	local13 = V_prime.low;


	// seventh part

	address_psi = (12288 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local10 + local14;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local10 + q_thread * (local10 < local14);
	V_prime = V_prime - local14;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local10 = U_prime.low;
	local14 = V_prime.low;


	// eight part

	address_psi = (14336 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local11 + local15;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local11 + q_thread * (local11 < local15);
	V_prime = V_prime - local15;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	local11 = U_prime.low;
	local15 = V_prime.low;


	//m = m << 1;
	//t_2 = t_2 - 1;
	m = m >> 1;
	t_2 = t_2 + 1;

	/// üçüncü for biter


	///////////////////////////////////////////////////////////////////////////////////////////////////


	//dördüncü for başlar

	address_psi = modidx >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local0 + local8;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local0 + q_thread * (local0 < local8);
	V_prime = V_prime - local8;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 16384] = V_prime.low;

	// second part

	address_psi = (2048 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local1 + local9;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local1 + q_thread * (local1 < local9);
	V_prime = V_prime - local9;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 2048] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 18432] = V_prime.low;

	// third part


	address_psi = (4096 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local2 + local10;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local2 + q_thread * (local2 < local10);
	V_prime = V_prime - local10;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 4096] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 20480] = V_prime.low;

	// forth part

	address_psi = (6144 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //


	U_prime = local3 + local11;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local3 + q_thread * (local3 < local11);
	V_prime = V_prime - local11;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 6144] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 22528] = V_prime.low;


	// fifth part

	address_psi = (8192 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local4 + local12;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local4 + q_thread * (local4 < local12);
	V_prime = V_prime - local12;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 8192] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 24576] = V_prime.low;


	// sixth part

	address_psi = (10240 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local5 + local13;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local5 + q_thread * (local5 < local13);
	V_prime = V_prime - local13;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 10240] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 26624] = V_prime.low;


	// seventh part

	address_psi = (12288 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local6 + local14;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local6 + q_thread * (local6 < local14);
	V_prime = V_prime - local14;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 12288] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 28672] = V_prime.low;


	// eight part

	address_psi = (14336 + modidx) >> t_2;//  t_2 = 13 - 1; // loopy = 0,1; //

	U_prime = local7 + local15;
	U_prime -= q_thread * (U_prime >= q_thread);

	V_prime = local7 + q_thread * (local7 < local15);
	V_prime = V_prime - local15;

	mul64(V_prime.low, PsiTable[m + address_psi + idx_psi], V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);

	mul64(U_prime.low, invv, U_prime);
	singleBarrett(U_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 14336] = U_prime.low;

	mul64(V_prime.low, invv, V_prime);
	singleBarrett(V_prime, q_thread, q_mu_thread, q_bit_thread);
	Inputs[addresss + 30720] = V_prime.low;

}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////- - - - 32768 - - - - -///////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////






///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////- - - - - HOST FUNCTIONS - - - - -///////////////////////////////////////
// Inplace
__host__ void Forward_NTTx(unsigned long long* input_device, unsigned long long* output_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count)
{

	if (n == 4096) {
		int n_power = 12;
		NTT_4096_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		NTT_4096_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 2, 2);
	}
	else if (n == 8192) {
		int n_power = 13;
		NTT_8192_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		NTT_8192_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 8, 4);
	}
	else if (n == 16384) {
		int n_power = 14;
		NTT_16384_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		NTT_16384_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 16, 8);
	}
	else if (n == 32768) {
		int n_power = 15;
		NTT_32768_REG_MULTI_1x << < dim3((input_count * 2), 1, 1), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		NTT_32768_REG_MULTI_2x << < int(input_count * (n / (1024 * 2))), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 16, 8);
		NTT_32768_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count, n / 8, 16);
	}

}

// Inplace
__host__ void Inverse_NTTx(unsigned long long* input_device, unsigned long long* output_device, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* q_bit_device, unsigned n, unsigned long long* psitable_device, unsigned input_count, unsigned q_count, unsigned long long* modinv)
{

	if (n == 4096) {
		int n_power = 12;
		INTT_4096_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INTT_4096_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, modinv, n, n_power, input_count, 1, 11, q_count);
	}
	else if (n == 8192) {
		int n_power = 13;
		INTT_8192_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INTT_8192_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, modinv, n, n_power, input_count, 2, 11, q_count);
	}
	else if (n == 16384) {
		int n_power = 14;
		INTT_16384_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INTT_16384_REG_MULTIx << < dim3((input_count * 2), 1, 1), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, modinv, n, n_power, input_count, 4, 11, q_count);
	}
	else if (n == 32768) {
		int n_power = 15;
		INTT_32768_SMEM_MULTIx << < int(input_count * (n / (1024 * 2))), 1024 >> > (input_device, output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, q_count);
		INTT_32768_REG_MULTI_1x << < dim3((input_count * 2), 1, 1), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, n, n_power, input_count, 8, 11, q_count);
		INTT_32768_REG_MULTI_2x << < dim3((input_count * 2), 1, 1), 1024 >> > (output_device, psitable_device, q_device, mu_device, q_bit_device, modinv, n, n_power, input_count, 2, 13, q_count);

	}

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////END//////////////////////////////////////////////////////


