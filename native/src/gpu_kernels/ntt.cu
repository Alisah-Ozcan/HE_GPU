// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt.cuh"

#define CC_89 // for RTX 4090
// TODO: All Kernel Initialization will be updated with respect to GPUs. (A100, RTX4090, RTX3060Ti)

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

__global__ void ForwardCore(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus* modulus, int shared_index, int logm,
                            int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel,
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
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(2 * block_y * offset) + (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);


    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if (not_last_kernel)
    {
#pragma unroll
        for (int lp = 0; lp < outer_iteration_count; lp++)
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
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
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
            //__syncthreads();
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
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}



__global__ void InverseCore(Data* polynomial_in, Data* polynomial_out, Root* inverse_root_of_unity_table,
                            Modulus* modulus, int shared_index, int logm, int k,
                            int outer_iteration_count, int N_power,
                            Ninverse* n_inverse, bool last_kernel,
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
    //int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(2 * block_y * offset) + (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
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

    if (last_kernel)
    {
        polynomial_out[global_addresss] =
            VALUE_GPU::mult(shared_memory[shared_addresss], n_inverse[mod_index], modulus[mod_index]);
        polynomial_out[global_addresss + offset] = VALUE_GPU::mult(
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
            n_inverse[mod_index], modulus[mod_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
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
                    ForwardCore<<<dim3(8, 1, batch_size), dim3(64, 4),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 3,
                        cfg.n_power, cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 8, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 9,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    ForwardCore<<<dim3(16, 1, batch_size), dim3(32, 8),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4,
                        cfg.n_power, cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 9,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    ForwardCore<<<dim3(32, 1, batch_size), dim3(16, 16),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5,
                        cfg.n_power, cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 32, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 9,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    ForwardCore<<<dim3(64, 1, batch_size), dim3(8, 32),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6,
                        cfg.n_power, cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 64, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 9,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    ForwardCore<<<dim3(128, 1, batch_size), dim3(4, 64),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7,
                        cfg.n_power, cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 128, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 9,
                        cfg.n_power, false, false,
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
                    InverseCore<<<dim3(1, 8, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 11, 3, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8, 1, batch_size), dim3(64, 4),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 2, 0, 3,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    InverseCore<<<dim3(1, 16, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 12, 4, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16, 1, batch_size), dim3(32, 8),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    InverseCore<<<dim3(1, 32, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 13, 5, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 1, batch_size), dim3(16, 16),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    InverseCore<<<dim3(1, 64, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 14, 6, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 1, batch_size), dim3(8, 32),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    InverseCore<<<dim3(1, 128, batch_size), dim3(256, 1),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 15, 7, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 1, batch_size), dim3(4, 64),
                                  512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
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