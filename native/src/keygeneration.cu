// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "keygeneration.cuh"


__global__ void sk_kernel(Data* secret_key, Modulus* modulus, int n_power, int rns_mod_count, int seed){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

    curandState_t state;
    curand_init((idx * seed) + (seed), 0, 0, &state);
    Data random_number = curand(&state) % 3; // 0,1,2

#pragma unroll
    for (int i = 0; i < rns_mod_count; i++) {

        int location = i << n_power;

        Data result = random_number;

        if (result == 2)
            result = modulus[i].value - 1; // -1,0,1

        secret_key[idx + location] = result;

    }

}


__host__ void HESecretkeygen(Secretkey &sk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned n_power = context.n_power;

    sk_kernel<< < dim3((context.n >> 8), 1, 1), 256 >> >(sk.location, context.modulus_, n_power, rns_mod_count, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(sk.location, context.ntt_table_ , context.modulus_, cfg_ntt, rns_mod_count, rns_mod_count);

}

/////////////////////////////////////////////////////////////////////////////////////

__global__ void error_kernel(Data* a_e, Modulus* modulus, int n_power, int rns_mod_count, int seed){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // a or e

    if(block_y == 0){ // e
        curandState_t state;

        curand_init((idx * seed) + (seed * seed *seed), 0, 0, &state);

        float rn = curand_normal(&state);
        rn = rn * 3.2; // SIGMA
        //rn = rn * 0;

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++) {
            
            signed long long rn_ = rn;
            
            if (rn_ < 0)
                rn_ = rn_ + modulus[i].value;

            Data rn_ULL = rn_;

            int location = i << n_power;

            a_e[idx + location] = rn_ULL % modulus[i].value;
        }

    }
    else{ // a

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++) {

            curandState_t state_lo, state_hi;

            curand_init((idx * seed) + (seed), 0, 0, &state_lo);
            curand_init((idx * seed) + (seed * seed), 0, 0, &state_hi);

            Data rn_lo = curand(&state_lo);
            Data rn_hi = curand(&state_hi);

            rn_hi = rn_hi << 32;
            rn_hi = rn_hi + rn_lo;
    
            int location = i << n_power;

            a_e[idx + location + ((rns_mod_count) << n_power)] = rn_hi % modulus[i].value;
        }

    }

}

__global__ void pk_kernel(Data* public_key, Data* secret_key, Modulus* modulus, int n_power, int rns_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // rns_mod_count
    
    Data sk = secret_key[idx + (block_y << n_power)];
    Data e = public_key[idx + (block_y << n_power)];
    Data a = public_key[idx + (block_y << n_power) + (rns_mod_count << n_power)];
    
    Data temp = VALUE_GPU::mult(sk, a, modulus[block_y]);
    temp = VALUE_GPU::add(temp, e, modulus[block_y]);
    Data zero = 0;

    public_key[idx + (block_y << n_power)] = VALUE_GPU::sub(zero, temp, modulus[block_y]);

}

__host__ void HEPublickeygen(Publickey &pk, Secretkey &sk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned n_power = context.n_power;

    error_kernel<< < dim3((context.n >> 8), 2, 1), 256 >> >(pk.location, context.modulus_, n_power, rns_mod_count, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(pk.location, context.ntt_table_ , context.modulus_, cfg_ntt, 2 * rns_mod_count, rns_mod_count);

    pk_kernel<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(pk.location, sk.location, context.modulus_, n_power, rns_mod_count);

}


/////////////////////////////////////////////////////////////////////////////////////

__global__ void relinkey_kernel(Data* relin_key, Data* secret_key, Data* e_a, Modulus* modulus, Data* factor, int n_power, int rns_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // rns_mod_count

    int location1 = block_y << n_power;

    Data sk = secret_key[idx + (block_y << n_power)];
    Data e = e_a[idx + (block_y << n_power)];
    Data a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];
    
#pragma unroll
    for(int i = 0; i < rns_mod_count - 1; i++){     
        
        Data rk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
        rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
        Data zero = 0;
        
        rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

        
        if(i == block_y){
            Data temp = VALUE_GPU::mult(sk, sk, modulus[block_y]);
            temp = VALUE_GPU::mult(temp, factor[block_y], modulus[block_y]);

            rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
        }


        relin_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1))] = rk_0;
        relin_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1)) + (rns_mod_count << n_power)] = a;

    }

}

__host__ void HERelinkeygen(Relinkey &rk, Secretkey &sk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned n_power = context.n_power;

    error_kernel<< < dim3((context.n >> 8), 2, 1), 256 >> >(rk.e_a, context.modulus_, n_power, rns_mod_count, time(NULL));

    ntt_configuration cfg_ntt = {
        .n_power = context.n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(rk.e_a, context.ntt_table_ , context.modulus_, cfg_ntt, 2 * rns_mod_count, rns_mod_count);

    relinkey_kernel<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(rk.location, sk.location, rk.e_a, context.modulus_, context.factor_, n_power, rns_mod_count);

}


/////////////////////////////////////////////////////////////////////////////////////

int steps_to_galois_elt(int steps, int coeff_count) {

    int n = coeff_count;
    int m32 = n * 2;
    int m = m32;

    if (steps == 0) {
        return m - 1;
    }
    else {
        int sign = steps < 0;
        int pos_steps = abs(steps);

        if(pos_steps >= (n >> 1)){
            std::cout << "Galois Key can not be generated, Step count too large " << std::endl;
            return 0;
        }

        if (sign) {
            steps = (n >> 1) - pos_steps;
        }
        else {
            steps = pos_steps;
        }


        int gen = 3;
        int galois_elt = 1;
        while (steps > 0) {

            galois_elt = galois_elt * gen;
            galois_elt = galois_elt & (m - 1);

            steps = steps - 1;

        }

        return galois_elt;

    }

}

__device__ int bitreverse_gpu(int index, int n_power) {

    int res_1 = 0;
    for (int i = 0; i < n_power; i++)
    {
        res_1 <<= 1;
        res_1 = (index & 1) | res_1;
        index >>= 1;
    }
    return res_1;
}

__device__ int permutation(int index, int galois_elt, int coeff_count, int n_power)
{
    int coeff_count_minus_one = coeff_count - 1;
    int i = index + coeff_count;

    int reversed = bitreverse_gpu(i, n_power + 1);

    int index_raw = (galois_elt * reversed) >> 1;

    index_raw = index_raw & coeff_count_minus_one;

    return bitreverse_gpu(index_raw, n_power);


}

__global__ void galoiskey_kernel(Data* galois_key, Data* secret_key, Data* e_a, Modulus* modulus, Data* factor, int galois_elt, int n_power, int rns_mod_count){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    int block_y = blockIdx.y; // rns_mod_count

    int location1 = block_y << n_power;
    int coeff_count = 1 << n_power;

    Data sk = secret_key[idx + (block_y << n_power)];
    Data e = e_a[idx + (block_y << n_power)];
    Data a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];
    
#pragma unroll
    for(int i = 0; i < rns_mod_count - 1; i++){     
        
        Data gk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
        gk_0 = VALUE_GPU::add(gk_0, e, modulus[block_y]);
        Data zero = 0;
        
        gk_0 = VALUE_GPU::sub(zero, gk_0, modulus[block_y]);

        
        if(i == block_y){

            int permutation_location = permutation(idx, galois_elt, coeff_count, n_power);
            Data sk_permutation = secret_key[(block_y << n_power) + permutation_location];
          
            sk_permutation = VALUE_GPU::mult(sk_permutation, factor[block_y], modulus[block_y]);

            gk_0 = VALUE_GPU::add(gk_0, sk_permutation, modulus[block_y]);
            
        }


        galois_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1))] = gk_0;
        galois_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1)) + (rns_mod_count << n_power)] = a;

    }

}

__host__ void HEGaloiskeygen(Galoiskey &gk, Secretkey &sk, Parameters context){

    unsigned rns_mod_count = context.coeff_modulus;
    unsigned n_power = context.n_power;

    // Positive Shift
    for (int i = 0; i < MAX_SHIFT; i++) {

        gk.galois_elt_pos[i] = steps_to_galois_elt(pow(2, i), context.n);

        error_kernel<< < dim3((context.n >> 8), 2, 1), 256 >> >(gk.e_a, context.modulus_, n_power, rns_mod_count, time(NULL));

        ntt_configuration cfg_ntt = {
            .n_power = context.n_power,
            .ntt_type = FORWARD,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        GPU_NTT_Inplace(gk.e_a, context.ntt_table_ , context.modulus_, cfg_ntt, 2 * rns_mod_count, rns_mod_count);

        galoiskey_kernel<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(gk.positive_location[i], sk.location, gk.e_a, context.modulus_, context.factor_, gk.galois_elt_pos[i], n_power, rns_mod_count);

    }

    // Negative Shift
    for (int i = 0; i < MAX_SHIFT; i++) {

        gk.galois_elt_neg[i] = steps_to_galois_elt(pow(2, i) * (i), context.n);

        error_kernel<< < dim3((context.n >> 8), 2, 1), 256 >> >(gk.e_a, context.modulus_, n_power, rns_mod_count, time(NULL));

        ntt_configuration cfg_ntt = {
            .n_power = context.n_power,
            .ntt_type = FORWARD,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        GPU_NTT_Inplace(gk.e_a, context.ntt_table_ , context.modulus_, cfg_ntt, 2 * rns_mod_count, rns_mod_count);

        galoiskey_kernel<< < dim3((context.n >> 8), rns_mod_count, 1), 256 >> >(gk.negative_location[i], sk.location, gk.e_a, context.modulus_, context.factor_, gk.galois_elt_neg[i], n_power, rns_mod_count);

    }

}