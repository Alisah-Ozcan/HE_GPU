#include "NTT.cuh"
#include "GPU_Context.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

__device__ __forceinline__ void random_0_1(int& idx, unsigned long long seed, unsigned long long& rnd_num)
{
    curandState_t state;

    /* we have to initialize the state */ // 
    curand_init((idx * seed) + (seed * seed * seed), /* the seed controls the sequence of random values that are produced */
        0, /* the sequence number is only important with multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);

    rnd_num = curand(&state) % 3;

}

__device__ __forceinline__ void random_normal_dist(int& idx, unsigned long long seed, float& rnd_num)
{
    curandState_t state;
    curandState_t state2;

    /* we have to initialize the state */ // 
    curand_init((idx * seed) + (seed * seed * seed * seed), /* the seed controls the sequence of random values that are produced */
        0, /* the sequence number is only important with multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);

    rnd_num = curand_normal(&state);

}


__device__ __forceinline__ void random_64(int& idx, unsigned long long seed, unsigned long long& rnd_num)
{
    curandState_t state;
    curandState_t state2;

    /* we have to initialize the state */
    curand_init((idx * seed) + (seed * seed), /* the seed controls the sequence of random values that are produced */
        0, /* the sequence number is only important with multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);

    curand_init((idx + seed), /* the seed controls the sequence of random values that are produced */
        0, /* the sequence number is only important with multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state2);

    unsigned long long MST = curand(&state);
    unsigned long long LST = curand(&state2);

    MST = MST << 32;
    MST = MST + LST;
    rnd_num = MST;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SECRET & PUBLICKEY

__global__ void BFV_RANDOM_KeyGen(unsigned long long* memory, unsigned long long* q, int seed, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Random For Keygeneration
    if (idx < coeff_count) {// for secretkey generation

        unsigned long long RN;

        random_0_1(idx, seed, RN);// it returns 0,1,2

        for (int i = 0; i < q_count; i++) {

            int location = i * coeff_count;
            unsigned long long q_reg = q[i];

            unsigned long long result = RN;

            if (result == 2)
                result = q_reg - 1;

            memory[idx + location] = result;

            /*
            if (RN == 2)
                RN = q_reg - 1;

            memory[idx + location] = RN;
            */
        }

    }
    else if ((coeff_count <= idx) && (idx < (coeff_count << 1))) {// for error generation

        int start_point = (q_count * coeff_count);
        int idx_inside = idx - coeff_count;

        float RN;

        random_normal_dist(idx, seed, RN);

        RN *= 1;// 3.2;

        for (int i = 0; i < q_count; i++) {

            int location = start_point + (i * coeff_count);
            unsigned long long q_reg = q[i];

            unsigned long long result = RN;
            if (result < 0)
                result = q_reg + int(RN);

            memory[idx_inside + location] = result;
            /*
            if (RN < 0)
                RN = q_reg - int(RN);

            memory[idx_inside + location] = RN;
            */
        }

    }
    else {// for "a" generation

        int start_point = (2 * q_count * coeff_count);
        int idx_inside = idx - (2 * coeff_count);
        int index = int(idx_inside / coeff_count);

        unsigned long long q_reg = q[index];
        unsigned long long RN;

        random_64(idx, seed, RN);

        RN = RN % q_reg;

        memory[idx_inside + start_point] = RN;
    }


}


__global__ void keygen_2_new(unsigned long long* pk0, unsigned long long* memory, unsigned long long* q, unsigned long long* qmu, unsigned long long* qbit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = int(idx / coeff_count);
    int loc1 = coeff_count * q_count;

    unsigned long long q_reg = q[index];
    unsigned long long q_mu_reg = qmu[index];
    unsigned long long q_bit_reg = qbit[index];

    unsigned long long sk = memory[idx];
    unsigned long long error = memory[loc1 + idx];
    unsigned long long a = memory[(loc1 * 2) + idx];

    uint128_t result;
    mul64(sk, a, result);
    singleBarrett(result, q_reg, q_mu_reg, q_bit_reg);
    result.low = result.low + error;


    result.low -= (result.low >= q_reg) * q_reg;


    result.low = (q_reg - result.low);

    pk0[idx] = result.low;

}





__host__ void GPU_Keygen(Lib_Parameters context)// secretkey + publickey + errors
{
    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

    unsigned long long* BFV_random_keygen_ = context.BFV_random_keygen;
    unsigned long long* pk0_device_ = context.pk0_device;


    //parameters
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

    unsigned long long* plainmu_device = plain_upper_half_threshold_device + 1;
    unsigned long long* plain_bit_device = plainmu_device + 1;

    BFV_RANDOM_KeyGen << <(((coeff_modulus + 2)) * (n / (1024))), 1024 >> > (BFV_random_keygen_, q_device, time(NULL), n, coeff_modulus); // (sk,e,a)
    Forward_NTT_Inplace(BFV_random_keygen_, q_device, mu_device, q_bit_device, n, ForwardPsi_device, 3 * coeff_modulus, coeff_modulus);// (sk,e,a)

    keygen_2_new << <(coeff_modulus * (n / (1024))), 1024 >> > (pk0_device_, BFV_random_keygen_, q_device, mu_device, q_bit_device, n, coeff_modulus);

}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RELINKEY


__global__ void Relin_Random(unsigned long long* memory, unsigned long long* q, int seed, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 2 * q_count * (q_count - 1) * coeff_count

    int location = q_count * (q_count - 1) * coeff_count;

    if (idx < location) {

        int index = int(idx % (coeff_count * q_count));
        int index2 = int(index / coeff_count);

        unsigned long long q_reg = q[index2];
        unsigned long long RN;

        random_64(idx, seed, RN);

        RN = RN % q_reg;

        memory[idx] = RN;
        //memory[idx] = 0;

    }
    else {

        int idx_inside = idx - location;
        int index = int(idx_inside % (coeff_count * q_count));
        int index2 = int(index / coeff_count);

        unsigned long long q_reg = q[index2];
        float RN;

        random_normal_dist(idx, seed, RN);

        RN *= 1;// 3.2;

        if (RN < 0)
            RN = q_reg - int(RN);

        memory[idx] = RN;
        //memory[idx] = 0;


    }

}

__global__ void Relin_Random_NEW(unsigned long long* memory, unsigned long long* q, int seed, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 2 * q_count * (q_count - 1) * coeff_count

    int location = q_count * (q_count - 1) * coeff_count;

    if (idx < location) {

        int index = int(idx % (coeff_count * q_count));
        int index2 = int(index / coeff_count);

        unsigned long long q_reg = q[index2];
        unsigned long long RN;

        random_64(idx, seed, RN);

        RN = RN % q_reg;

        memory[idx] = RN;
        //memory[idx] = 0;

    }
    else {

        int idx_inside = idx - location;
        int index = int(idx_inside % (coeff_count));
        int index2 = int(idx_inside / coeff_count);
        
        float RN;

        random_normal_dist(idx, seed, RN);

        RN *= 1;// 3.2;

        for (int loop = 0; loop < q_count; loop++) {

            unsigned long long q_reg = q[loop];

            unsigned long long result = RN;

            if (result < 0)
                result = q_reg + int(result);

            memory[(index2 * coeff_count * q_count) + (loop * coeff_count) + index] = result;
            /*
            if (RN < 0)
                RN = q_reg + int(RN);

            memory[(index2 * coeff_count * q_count) + (loop * coeff_count) + index] = RN;
            */
        }
        


    }

}


__global__ void Relin_Gen_Part(unsigned long long* rlk, unsigned long long* sk, unsigned long long* random_memory, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* bit_length,
    unsigned long long* back_value, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    int loc1 = q_count * coeff_count;
    int loc2 = loc1 * (q_count - 1);

    unsigned long long secret_key = sk[(j * coeff_count) + idx];

    unsigned long long q_reg = q_device[j];
    unsigned long long mu_reg = mu_device[j];
    unsigned long long bit_reg = bit_length[j];

    for (int i = 0; i < (q_count - 1); i++) {

        unsigned long long a_reg = random_memory[(i * loc1) + ((j * coeff_count) + idx)];
        unsigned long long e_reg = random_memory[loc2 + (i * loc1) + ((j * coeff_count) + idx)];

        uint128_t rlk_0 = secret_key;
        mul64(rlk_0.low, a_reg, rlk_0);
        rlk_0 = rlk_0 + e_reg;
        singleBarrett(rlk_0, q_reg, mu_reg, bit_reg);

        // AŞAĞI SATIR EKSİKTİ
        rlk_0 = q_reg - rlk_0;


        if (i == j) {
            uint128_t new_sk;
            mul64(secret_key, secret_key, new_sk);
            singleBarrett(new_sk, q_reg, mu_reg, bit_reg);

            mul64(new_sk.low, back_value[j], new_sk);
            singleBarrett(new_sk, q_reg, mu_reg, bit_reg);

            rlk_0.low = rlk_0.low + new_sk.low;
            rlk_0.low -= q_reg * (rlk_0.low >= q_reg);
            
            
            /*
            if (idx == 0) {
                printf("_______> if  ---- i = %u, j = %u \n ", i,j);
            }
            */

            rlk[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx] = rlk_0.low;
            rlk[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx + coeff_count] = a_reg; // rlk1

        }
        else {

            /*
            if (idx == 0) {
                printf("_______> else  ---- i = %u, j = %u \n ", i, j);
            }
            */

            rlk[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx] = rlk_0.low;
            rlk[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx + coeff_count] = a_reg; // rlk1

        }


    }

}



__host__ void GPU_RelinKeyGen(GPU_Relinkey rlk, Lib_Parameters context)// Relinkey
{
    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

    unsigned long long* secret_key = context.BFV_random_keygen;
    unsigned long long* rlk_ = rlk.GPU_Location;
    unsigned long long* rlk_random_ = rlk.GPU_Random_Location;

    //parameters
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

    unsigned long long* plainmu_device = plain_upper_half_threshold_device + 1;
    unsigned long long* plain_bit_device = plainmu_device + 1;

    //for multiply
    unsigned long long* aux_m_tilde_device = plain_bit_device + 1;
    unsigned long long* p_device = aux_m_tilde_device + 1;
    unsigned long long* aux_B_m_sk_device = p_device + (coeff_modulus - 1);
    unsigned long long* base_change_matrix1_device = aux_B_m_sk_device + coeff_modulus;
    unsigned long long* base_change_matrix2_device = base_change_matrix1_device + ((coeff_modulus - 1) * coeff_modulus);
    unsigned long long* inv_prod_q_mod_m_tilde_device = base_change_matrix2_device + (coeff_modulus - 1);
    unsigned long long* inv_m_tilde_mod_Bsk_device = inv_prod_q_mod_m_tilde_device + 1;
    unsigned long long* prod_q_mod_Bsk_device = inv_m_tilde_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_Bsk_elt_mu_device = prod_q_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_Bsk_bitlength_device = base_Bsk_elt_mu_device + coeff_modulus;
    unsigned long long* t_device = base_Bsk_bitlength_device + coeff_modulus;
    unsigned long long* aux_B1_device = t_device + 1;
    unsigned long long* inv_prod_q_mod_Bsk_device = aux_B1_device + coeff_modulus;
    unsigned long long* p1_device = inv_prod_q_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_change_matrix3_device = p1_device + coeff_modulus;
    unsigned long long* base_change_matrix4_device = base_change_matrix3_device + (coeff_modulus - 1) * (coeff_modulus - 1);
    unsigned long long* m_sk_device = base_change_matrix4_device + (coeff_modulus - 1);
    unsigned long long* inv_prod_B_mod_m_sk_device = m_sk_device + 1;
    unsigned long long* prod_B_mod_q_device = inv_prod_B_mod_m_sk_device + 1;
    unsigned long long* ForwardPsi_device_BSK = prod_B_mod_q_device + (coeff_modulus - 1);
    unsigned long long* InversePsi_device_BSK = ForwardPsi_device_BSK + coeff_modulus * n;
    unsigned long long* INTT_inv_bsk = InversePsi_device_BSK + coeff_modulus * n;

    //NEW
    unsigned long long* BACK_VALUE = INTT_inv_bsk + coeff_modulus;

    //İLK YAPTIĞIM:
    //Relin_Random << <((2 * coeff_modulus * (coeff_modulus - 1)) * (n / (1024))), 1024 >> > (rlk_random_, q_device, time(NULL), n, coeff_modulus);

    //2. YAPTIĞIM:
    Relin_Random_NEW << <(((coeff_modulus - 1) + coeff_modulus * (coeff_modulus - 1)) * (n / (1024))), 1024 >> > (rlk_random_, q_device, time(NULL), n, coeff_modulus);


    Forward_NTT_Inplace(rlk_random_, q_device, mu_device, q_bit_device, n, ForwardPsi_device, coeff_modulus * (coeff_modulus - 1) * 2, coeff_modulus);


    dim3 numBlocks((n / 512), coeff_modulus, 1);
    Relin_Gen_Part << < numBlocks, 512 >> > (rlk_, secret_key, rlk_random_, q_device, mu_device, q_bit_device, BACK_VALUE, n, coeff_modulus);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------- ROTATION -------------------------

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void permutation_tables(int* table, int galois_elt, int coeff_count) {

    int n_power = log2l(coeff_count);
    int coeff_count_minus_one = coeff_count - 1;

    for (int i = coeff_count; i < (coeff_count << 1); i++) {

        int reversed = bitreverse(i, n_power + 1);
        int index_raw = (galois_elt * reversed) >> 1;
        index_raw = index_raw & coeff_count_minus_one;
        table[i - coeff_count] = bitreverse(index_raw, n_power);

    }
  
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

__global__ void Galois_Gen_Part(unsigned long long* Galoiskey, int galois_elt, unsigned long long* sk, unsigned long long* random_memory, unsigned long long* q_device, unsigned long long* mu_device, unsigned long long* bit_length,
    unsigned long long* back_value, int coeff_count, int n_power, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    int loc1 = q_count * coeff_count;
    int loc2 = loc1 * (q_count - 1);

    unsigned long long secret_key = sk[(j * coeff_count) + idx];

    unsigned long long q_reg = q_device[j];
    unsigned long long mu_reg = mu_device[j];
    unsigned long long bit_reg = bit_length[j];

    for (int i = 0; i < (q_count - 1); i++) {

        unsigned long long a_reg = random_memory[(i * loc1) + ((j * coeff_count) + idx)];
        unsigned long long e_reg = random_memory[loc2 + (i * loc1) + ((j * coeff_count) + idx)];

        uint128_t rlk_0 = secret_key;
        mul64(rlk_0.low, a_reg, rlk_0);
        rlk_0 = rlk_0 + e_reg;
        singleBarrett(rlk_0, q_reg, mu_reg, bit_reg);

        // AŞAĞI SATIR EKSİKTİ
        rlk_0 = q_reg - rlk_0;


        if (i == j) {

            
            int index_location = permutation(idx, galois_elt, coeff_count, n_power);
            uint128_t new_sk = sk[(j * coeff_count) + index_location];;
            //////////////////////////////////////
           
            mul64(new_sk.low, back_value[j], new_sk);
            singleBarrett(new_sk, q_reg, mu_reg, bit_reg);

            rlk_0.low = rlk_0.low + new_sk.low;
            rlk_0.low -= q_reg * (rlk_0.low >= q_reg);


            /*
            if (idx == 0) {
                printf("_______> if  ---- i = %u, j = %u \n ", i,j);
            }
            */

            Galoiskey[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx] = rlk_0.low;
            Galoiskey[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx + coeff_count] = a_reg; // rlk1

        }
        else {

            /*
            if (idx == 0) {
                printf("_______> else  ---- i = %u, j = %u \n ", i, j);
            }
            */

            Galoiskey[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx] = rlk_0.low;
            Galoiskey[(i * (loc1 * 2)) + (j * coeff_count * 2) + idx + coeff_count] = a_reg; // rlk1

        }


    }

}


__host__ void GPU_GaloisKeyGen(GPU_GaloisKey galoiskey, Lib_Parameters context)// Relinkey
{
    int n = context.n;
    int n_powers = context.n_power;
    int coeff_modulus = context.coeff_modulus;

    unsigned long long* secret_key = context.BFV_random_keygen;
    unsigned long long* galoiskey_positive = galoiskey.GPU_Location_positive;
    unsigned long long* galoiskey_negative = galoiskey.GPU_Location_negative;

    unsigned long long* galoiskey_random_ = galoiskey.GPU_Random_Location;

    int* galois_elt_pos_ = galoiskey.galois_elt_pos;
    int* galois_elt_neg_ = galoiskey.galois_elt_neg;

    int max_power = galoiskey.max_power_shift;
    bool neg_shift = galoiskey.neg_shift_;

    //parameters
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

    unsigned long long* plainmu_device = plain_upper_half_threshold_device + 1;
    unsigned long long* plain_bit_device = plainmu_device + 1;

    //for multiply
    unsigned long long* aux_m_tilde_device = plain_bit_device + 1;
    unsigned long long* p_device = aux_m_tilde_device + 1;
    unsigned long long* aux_B_m_sk_device = p_device + (coeff_modulus - 1);
    unsigned long long* base_change_matrix1_device = aux_B_m_sk_device + coeff_modulus;
    unsigned long long* base_change_matrix2_device = base_change_matrix1_device + ((coeff_modulus - 1) * coeff_modulus);
    unsigned long long* inv_prod_q_mod_m_tilde_device = base_change_matrix2_device + (coeff_modulus - 1);
    unsigned long long* inv_m_tilde_mod_Bsk_device = inv_prod_q_mod_m_tilde_device + 1;
    unsigned long long* prod_q_mod_Bsk_device = inv_m_tilde_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_Bsk_elt_mu_device = prod_q_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_Bsk_bitlength_device = base_Bsk_elt_mu_device + coeff_modulus;
    unsigned long long* t_device = base_Bsk_bitlength_device + coeff_modulus;
    unsigned long long* aux_B1_device = t_device + 1;
    unsigned long long* inv_prod_q_mod_Bsk_device = aux_B1_device + coeff_modulus;
    unsigned long long* p1_device = inv_prod_q_mod_Bsk_device + coeff_modulus;
    unsigned long long* base_change_matrix3_device = p1_device + coeff_modulus;
    unsigned long long* base_change_matrix4_device = base_change_matrix3_device + (coeff_modulus - 1) * (coeff_modulus - 1);
    unsigned long long* m_sk_device = base_change_matrix4_device + (coeff_modulus - 1);
    unsigned long long* inv_prod_B_mod_m_sk_device = m_sk_device + 1;
    unsigned long long* prod_B_mod_q_device = inv_prod_B_mod_m_sk_device + 1;
    unsigned long long* ForwardPsi_device_BSK = prod_B_mod_q_device + (coeff_modulus - 1);
    unsigned long long* InversePsi_device_BSK = ForwardPsi_device_BSK + coeff_modulus * n;
    unsigned long long* INTT_inv_bsk = InversePsi_device_BSK + coeff_modulus * n;

    //NEW
    unsigned long long* BACK_VALUE = INTT_inv_bsk + coeff_modulus;

    for (int k = 0; k < max_power; k++) {

        galois_elt_pos_[k] = steps_to_galois_elt(pow(2, k), n);


        Relin_Random_NEW << <(((coeff_modulus - 1) + coeff_modulus * (coeff_modulus - 1)) * (n / (1024))), 1024 >> > (galoiskey_random_, q_device, time(NULL), n, coeff_modulus);


        Forward_NTT_Inplace(galoiskey_random_, q_device, mu_device, q_bit_device, n, ForwardPsi_device, coeff_modulus * (coeff_modulus - 1) * 2, coeff_modulus);

       
        dim3 numBlocks((n / 512), coeff_modulus, 1);
        Galois_Gen_Part << < numBlocks, 512 >> > (galoiskey_positive + (k * (coeff_modulus * (coeff_modulus - 1) * 2 * n)), galois_elt_pos_[k], secret_key, galoiskey_random_, q_device, mu_device, q_bit_device, BACK_VALUE, n, n_powers, coeff_modulus);

    }
    if (neg_shift) {
        for (int k = 0; k < max_power; k++) {

            galois_elt_neg_[k] = steps_to_galois_elt(pow(2, k) * (-1), n);


            Relin_Random_NEW << <(((coeff_modulus - 1) + coeff_modulus * (coeff_modulus - 1)) * (n / (1024))), 1024 >> > (galoiskey_random_, q_device, time(NULL), n, coeff_modulus);


            Forward_NTT_Inplace(galoiskey_random_, q_device, mu_device, q_bit_device, n, ForwardPsi_device, coeff_modulus * (coeff_modulus - 1) * 2, coeff_modulus);


            dim3 numBlocks((n / 512), coeff_modulus, 1);
            Galois_Gen_Part << < numBlocks, 512 >> > (galoiskey_negative + (k * (coeff_modulus * (coeff_modulus - 1) * 2 * n)), galois_elt_neg_[k], secret_key, galoiskey_random_, q_device, mu_device, q_bit_device, BACK_VALUE, n, n_powers, coeff_modulus);

        }
    }

    
   

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////