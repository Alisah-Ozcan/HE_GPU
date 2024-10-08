// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://ieeexplore.ieee.org/document/10097488

#include "ntt.cuh"

__device__ void CooleyTukeyUnit(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = VALUE_GPU::mult(V, root, modulus);

    U = VALUE_GPU::add(u_, v_, modulus);
    V = VALUE_GPU::sub(u_, v_, modulus);
}

__device__ void GentlemanSandeUnit(Data& U, Data& V, Root& root,
                                   Modulus& modulus)
{
    Data u_ = U;
    Data v_ = V;

    U = VALUE_GPU::add(u_, v_, modulus);

    v_ = VALUE_GPU::sub(u_, v_, modulus);
    V = VALUE_GPU::mult(v_, root, modulus);
}

__global__ void FORWARD_NTT_IEEE_REG(Data* Inputs, Data* Outputs, Root* root_of_unity_table, Modulus* modulus, int m_, int t_2_, int k_, int outer_loop, int inner_loop, int N_power, int mod_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;

    int mod_index = j % mod_count;

	int m = m_;
	int t_2 = t_2_;
    int k = k_;

	int addresss = i + (j << N_power);

    unsigned long long local[16];

    for (int inner = 0; inner < (inner_loop << 1); inner++)
    {
        local[inner] = Inputs[addresss + (inner << 11)];
    }

    for (int outer = 0; outer < outer_loop; outer++)
    {
        for (int inner = 0; inner < inner_loop; inner++)
        {   
            int Reg_Location = ((inner / k) * k) + inner;
            CooleyTukeyUnit(local[Reg_Location],
                            local[Reg_Location + k],
                            root_of_unity_table[m + ((i + (inner << 11)) >> t_2) + (mod_index << N_power)], modulus[mod_index]);
        }
        m = m << 1;
	    t_2 = t_2 - 1;
        k = k >> 1;
    }

    for (int inner = 0; inner < (inner_loop << 1); inner++)
    {
        Outputs[addresss + (inner << 11)] = local[inner];
    }

}


__global__ void INVERSE_NTT_IEEE_REG(Data* Inputs, Data* Outputs, Root* root_of_unity_table, Modulus* modulus, int m_, int t_2_, int k_, int outer_loop, int inner_loop, int N_power, Ninverse* n_inverse, int mod_count)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;

    int mod_index = j % mod_count;

	int m = m_;
	int t_2 = t_2_;
    int k = k_;

	int addresss = i + (j << N_power);

    unsigned long long local[16];

    for (int inner = 0; inner < (inner_loop << 1); inner++)
    {
        local[inner] = Inputs[addresss + (inner << 11)];
    }

    for (int outer = 0; outer < outer_loop; outer++)
    {
        for (int inner = 0; inner < inner_loop; inner++)
        {   
            int Reg_Location = ((inner / k) * k) + inner;
            GentlemanSandeUnit(local[Reg_Location],
                            local[Reg_Location + k],
                            root_of_unity_table[m + ((i + (inner << 11)) >> t_2) + (mod_index << N_power)], modulus[mod_index]);
        }
        m = m >> 1;
	    t_2 = t_2 + 1;
        k = k << 1;
    }

    for (int inner = 0; inner < (inner_loop << 1); inner++)
    {
        Outputs[addresss + (inner << 11)] = VALUE_GPU::mult(local[inner], n_inverse[mod_index], modulus[mod_index]);
    }

}


__global__ void FORWARD_NTT_IEEE_SHARED(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus* modulus, int logm, int N_power,
                            bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = 10;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (10)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(2 * block_y * offset) + (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (10)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);


    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

#pragma unroll
    for (int lp = 0; lp < 5; lp++) 
    {
        __syncthreads();
        if (reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        CooleyTukeyUnit(shared_memory[in_shared_address],
                        shared_memory[in_shared_address + t],
                        root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;

        in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < 6; lp++)
    {
        if (reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        CooleyTukeyUnit(shared_memory[in_shared_address],
                        shared_memory[in_shared_address + t],
                        root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;

        in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();
    

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}



__global__ void INVERSE_NTT_IEEE_SHARED(Data* polynomial_in, Data* polynomial_out, Root* inverse_root_of_unity_table,
                            Modulus* modulus, int logm, int k, int N_power,
                            bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);

    int t_ = 0;
    int loops = 11;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (11 - 1)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(2 * block_y * offset) + (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (11 - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address],
                           shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index],
                           modulus[mod_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    
}


__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table,
                      Modulus* modulus, ntt_configuration cfg, int batch_size, int mod_count)
{
    switch (cfg.ntt_type)
    {
        case FORWARD:
            switch (cfg.n_power)
            {
                case 12:
                    FORWARD_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                  0, cfg.stream>>>(device_in, device_out, root_of_unity_table, modulus, 1, 11, 1, 1, 1, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FORWARD_NTT_IEEE_SHARED<<<dim3(1, 2, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 1,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    FORWARD_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                  0, cfg.stream>>>(device_in, device_out, root_of_unity_table, modulus, 1, 12, 2, 2, 2, 13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FORWARD_NTT_IEEE_SHARED<<<dim3(1, 4, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 2,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    FORWARD_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                  0, cfg.stream>>>(device_in, device_out, root_of_unity_table, modulus, 1, 13, 4, 3, 4, 14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FORWARD_NTT_IEEE_SHARED<<<dim3(1, 8, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 3,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    FORWARD_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                  0, cfg.stream>>>(device_in, device_out, root_of_unity_table, modulus, 1, 14, 8, 4, 8, 15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FORWARD_NTT_IEEE_SHARED<<<dim3(1, 16, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 4,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    break;
            }
            break;
        case INVERSE:
            switch (cfg.n_power)
            {
                case 12:
                    INVERSE_NTT_IEEE_SHARED<<<dim3(1, 2, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 11, 1,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    
                    INVERSE_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                                    0, cfg.stream>>>(device_out, device_out, root_of_unity_table, modulus, 1, 11, 1, 1, 1, 12, cfg.mod_inverse, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    INVERSE_NTT_IEEE_SHARED<<<dim3(1, 4, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 12, 2,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    INVERSE_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                                    0, cfg.stream>>>(device_out, device_out, root_of_unity_table, modulus, 2, 11, 1, 2, 2, 13, cfg.mod_inverse, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    INVERSE_NTT_IEEE_SHARED<<<dim3(1, 8, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 13, 3,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    INVERSE_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                                    0, cfg.stream>>>(device_out, device_out, root_of_unity_table, modulus, 4, 11, 1, 3, 4, 14, cfg.mod_inverse, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    INVERSE_NTT_IEEE_SHARED<<<dim3(1, 16, batch_size), dim3(1024, 1),
                                  2048 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 14, 4,
                        cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    INVERSE_NTT_IEEE_REG<<<dim3(2, batch_size), dim3(1024),
                                                    0, cfg.stream>>>(device_out, device_out, root_of_unity_table, modulus, 8, 11, 1, 4, 8, 15, cfg.mod_inverse, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }
}

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table,
                      Modulus* modulus, ntt_configuration cfg, int batch_size, int mod_count){

    GPU_NTT(device_inout, device_inout, root_of_unity_table,
                      modulus, cfg, batch_size, mod_count);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void GPU_ACTIVITY(unsigned long long* output,
                             unsigned long long fix_num)
{
    int idx = blockIdx.x + blockDim.x + threadIdx.x;

    output[idx] = fix_num;
}

__host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                unsigned long long fix_num)
{
    GPU_ACTIVITY<<<64, 512>>>(output, fix_num);
}


__global__ void GPU_ACTIVITY2(unsigned long long* input1, unsigned long long* input2)
{
    int idx = blockIdx.x + blockDim.x + threadIdx.x;

    input1[idx] = input1[idx] + input2[idx];
}

__host__ void GPU_ACTIVITY2_HOST(unsigned long long* input1, unsigned long long* input2,
                                unsigned size)
{
    GPU_ACTIVITY2<<<(size >> 8), 256>>>(input1, input2);
}