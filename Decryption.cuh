#include "NTT.cuh"
#include "GPU_Context.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

//DECRYPTION
__global__ void Dec1(unsigned long long* ct1, unsigned long long* sk, unsigned long long* ct1_sk, unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;// (q-1) * n
    int index = int(idx / coeff_count);

    unsigned long long ct1_reg = ct1[idx];
    uint128_t sk_reg = sk[idx];
    mul64(sk_reg.low, ct1_reg, sk_reg);
    singleBarrett(sk_reg, q[index], mu[index], q_bit[index]);
    ct1_sk[idx] = sk_reg.low;

}



__global__ void DecNEWx3(unsigned long long* ct0, unsigned long long* ct1_sk, unsigned long long* plain,
    unsigned long long* t, unsigned long long* t_mu, unsigned long long* t_bit,
    unsigned long long* gama, unsigned long long* gama_mu, unsigned long long* gama_bit,
    unsigned long long* Qi_inverse_ready, unsigned long long* Qi_ready_t_gama,
    unsigned long long* mul_inv_t, unsigned long long* mul_inv_gama,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit,
    unsigned long long* modinv_gama,
    int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint128_t sum_t = 0;
    uint128_t sum_gama = 0;

    for (int i = 0; i < q_count - 1; i++) {

        int location = idx + (i * coeff_count);

        unsigned long long q_reg = q[i];
        unsigned long long mu_reg = mu[i];
        unsigned long long q_bit_reg = q_bit[i];

        unsigned long long ct0_reg = ct0[location];
        unsigned long long sk_reg = ct1_sk[location];

        unsigned long long mt = ct0_reg + sk_reg;
        mt -= (mt >= q_reg) * q_reg;

        uint128_t result = mt;
        mul64(result.low, t[0], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(result.low, gama[0] % q_reg, result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(result.low, Qi_inverse_ready[i], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        uint128_t mt_in_t;
        uint128_t mt_in_gama;
        unsigned long long Qi_ready_t_gama_reg_for_t = Qi_ready_t_gama[i];
        unsigned long long Qi_ready_t_gama_reg_for_gama = Qi_ready_t_gama[(q_count - 1) + i];

        mul64((result.low % t[0]), Qi_ready_t_gama_reg_for_t, mt_in_t); 
        mul64(result.low, Qi_ready_t_gama_reg_for_gama, mt_in_gama);

        singleBarrett(mt_in_t, t[0], t_mu[0], t_bit[0]);
        singleBarrett(mt_in_gama, gama[0], gama_mu[0], gama_bit[0]);


        sum_t = sum_t + mt_in_t;
        singleBarrett(sum_t, t[0], t_mu[0], t_bit[0]);

        sum_gama = sum_gama + mt_in_gama;
        singleBarrett(sum_gama, gama[0], gama_mu[0], gama_bit[0]);

    }


    mul64(sum_t.low, mul_inv_t[0], sum_t);
    mul64(sum_gama.low, mul_inv_gama[0], sum_gama);

    singleBarrett(sum_t, t[0], t_mu[0], t_bit[0]);
    singleBarrett(sum_gama, gama[0], gama_mu[0], gama_bit[0]);


    unsigned long long gama_reg = gama[0] >> 1;
    unsigned long long resultss = 0;

    unsigned t_reg = t[0];
    unsigned t_mu_reg = t_mu[0];
    unsigned t_bit_reg = t_bit[0];

    if (sum_gama.low > gama_reg) {

        resultss = (sum_t.low + (gama[0] - sum_gama.low)) % t_reg; 


    }
    else {
        resultss = (sum_gama.low) % t[0];
        sum_t.low = sum_t.low + t[0];
        resultss = (sum_t.low - resultss) % t[0];

    }

    resultss = (resultss * modinv_gama[0]);
    singleBarrett_64(resultss, t_reg, t_mu_reg, t_bit_reg);
    plain[idx] = resultss;

}


__global__ void DecNEWx3_deneme(unsigned long long* TEST_, unsigned long long* ct0, unsigned long long* ct1_sk, unsigned long long* plain,
    unsigned long long* t, unsigned long long* t_mu, unsigned long long* t_bit,
    unsigned long long* gama, unsigned long long* gama_mu, unsigned long long* gama_bit,
    unsigned long long* Qi_inverse_ready, unsigned long long* Qi_ready_t_gama,
    unsigned long long* mul_inv_t, unsigned long long* mul_inv_gama,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit,
    unsigned long long* modinv_gama,
    int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint128_t sum_t = 0;
    uint128_t sum_gama = 0;

    for (int i = 0; i < q_count - 1; i++) {

        int location = idx + (i * coeff_count);

        unsigned long long q_reg = q[i];
        unsigned long long mu_reg = mu[i];
        unsigned long long q_bit_reg = q_bit[i];

        unsigned long long ct0_reg = ct0[location];
        unsigned long long sk_reg = ct1_sk[location];

        unsigned long long mt = ct0_reg + sk_reg;
        mt -= (mt >= q_reg) * q_reg;


        uint128_t result = mt;
        mul64(result.low, t[0], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(gama[0] % q_reg, result.low, result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(result.low, Qi_inverse_ready[i], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        uint128_t mt_in_t;
        uint128_t mt_in_gama;
        unsigned long long Qi_ready_t_gama_reg_for_t = Qi_ready_t_gama[i];
        unsigned long long Qi_ready_t_gama_reg_for_gama = Qi_ready_t_gama[(q_count - 1) + i];

    
        unsigned t_reg = t[0];
        unsigned t_mu_reg = t_mu[0];
        unsigned t_bit_reg = t_bit[0];

        mul64((result.low % t[0]), Qi_ready_t_gama_reg_for_t, mt_in_t); 
        mul64(result.low, Qi_ready_t_gama_reg_for_gama, mt_in_gama);
        singleBarrett(mt_in_t, t[0], t_mu[0], t_bit[0]);
        singleBarrett(mt_in_gama, gama[0], gama_mu[0], gama_bit[0]);

        sum_t = sum_t + mt_in_t;
        singleBarrett(sum_t, t[0], t_mu[0], t_bit[0]);



        sum_gama = sum_gama + mt_in_gama;
        singleBarrett(sum_gama, gama[0], gama_mu[0], gama_bit[0]);



    }

    TEST_[idx] = sum_t.low;
    TEST_[coeff_count + idx] = sum_gama.low;


}


__global__ void DecNEWx3_deneme_2(unsigned long long* TEST_, unsigned long long* ct0, unsigned long long* ct1_sk, unsigned long long* plain,
    unsigned long long* t, unsigned long long* t_mu, unsigned long long* t_bit,
    unsigned long long* gama, unsigned long long* gama_mu, unsigned long long* gama_bit,
    unsigned long long* Qi_inverse_ready, unsigned long long* Qi_ready_t_gama,
    unsigned long long* mul_inv_t, unsigned long long* mul_inv_gama,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit,
    unsigned long long* modinv_gama,
    int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint128_t sum_t = 0;
    uint128_t sum_gama = 0;

    for (int i = 0; i < q_count - 1; i++) {

        int location = idx + (i * coeff_count);

        unsigned long long q_reg = q[i];
        unsigned long long mu_reg = mu[i];
        unsigned long long q_bit_reg = q_bit[i];

        unsigned long long ct0_reg = ct0[location];
        unsigned long long sk_reg = ct1_sk[location];

        unsigned long long mt = ct0_reg + sk_reg;
        mt -= (mt >= q_reg) * q_reg;


        uint128_t result = mt;
        mul64(result.low, t[0], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        //mul64(gama[0], result.low, result);
        mul64(result.low, gama[0] % q_reg, result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(result.low, Qi_inverse_ready[i], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        TEST_[(i * coeff_count) + idx] = result.low;

    }

}

__global__ void DecNEWx3_deneme_3(unsigned long long* TEST_, unsigned long long* ct0, unsigned long long* ct1_sk, unsigned long long* plain,
    unsigned long long* t, unsigned long long* t_mu, unsigned long long* t_bit,
    unsigned long long* gama, unsigned long long* gama_mu, unsigned long long* gama_bit,
    unsigned long long* Qi_inverse_ready, unsigned long long* Qi_ready_t_gama,
    unsigned long long* mul_inv_t, unsigned long long* mul_inv_gama,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit,
    unsigned long long* modinv_gama,
    int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint128_t sum_t = 0;
    uint128_t sum_gama = 0;

    for (int i = 0; i < q_count - 1; i++) {

        int location = idx + (i * coeff_count);

        unsigned long long q_reg = q[i];
        unsigned long long mu_reg = mu[i];
        unsigned long long q_bit_reg = q_bit[i];

        unsigned long long ct0_reg = ct0[location];
        unsigned long long sk_reg = ct1_sk[location];

        unsigned long long mt = ct0_reg + sk_reg;
        mt -= (mt >= q_reg) * q_reg;


        uint128_t result = mt;
        mul64(result.low, t[0], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        //mul64(gama[0], result.low, result);
        mul64(gama[0] % q_reg, result.low, result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        mul64(result.low, Qi_inverse_ready[i], result);
        singleBarrett(result, q_reg, mu_reg, q_bit_reg);

        uint128_t mt_in_t;
        uint128_t mt_in_gama;
        unsigned long long Qi_ready_t_gama_reg_for_t = Qi_ready_t_gama[i];
        unsigned long long Qi_ready_t_gama_reg_for_gama = Qi_ready_t_gama[(q_count - 1) + i];

      
        unsigned t_reg = t[0];
        unsigned t_mu_reg = t_mu[0];
        unsigned t_bit_reg = t_bit[0];

        mul64((result.low % t[0]), Qi_ready_t_gama_reg_for_t, mt_in_t);
        mul64(result.low, Qi_ready_t_gama_reg_for_gama, mt_in_gama);
        singleBarrett(mt_in_t, t[0], t_mu[0], t_bit[0]);
        singleBarrett(mt_in_gama, gama[0], gama_mu[0], gama_bit[0]);

        sum_t = sum_t + mt_in_t;
        singleBarrett(sum_t, t[0], t_mu[0], t_bit[0]);



        sum_gama = sum_gama + mt_in_gama;
        singleBarrett(sum_gama, gama[0], gama_mu[0], gama_bit[0]);



    }




    mul64(sum_t.low, mul_inv_t[0], sum_t);
    mul64(sum_gama.low, mul_inv_gama[0], sum_gama);

    singleBarrett(sum_t, t[0], t_mu[0], t_bit[0]);
    singleBarrett(sum_gama, gama[0], gama_mu[0], gama_bit[0]);

    unsigned long long sum_t_reg = sum_t.low;
    unsigned long long sum_gama_reg = sum_gama.low;

    unsigned long long gama_reg = gama[0] >> 1;
    unsigned long long resultss = 0;

    if (sum_gama_reg > gama_reg) {

        resultss = (sum_t_reg + (gama[0] - sum_gama_reg)) % t[0];
    

    }
    else {

        resultss = (sum_gama_reg) % t[0];
        sum_t_reg = sum_t_reg + t[0];
        resultss = (sum_t_reg - resultss) % t[0];

    }

    TEST_[idx] = resultss;

    unsigned long long t_reg = t[0];
    unsigned long long t_mu_reg = t_mu[0];
    unsigned long long t_bit_reg = t_bit[0];

}


__host__ void GPU_Dec(GPU_Ciphertext cipher, GPU_Plaintext plain, Lib_Parameters context)
{
    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

    unsigned long long* ciphertext_device = cipher.GPU_Location;
    unsigned long long* plain_device = plain.GPU_Location;


    unsigned long long* BFV_random_keygen_ = context.BFV_random_keygen;
    unsigned long long* ciphertext1_sk_ = context.ciphertext1_sk;


    // TEST DEC
    unsigned long long* TESTTTTTT = context.DEC_TESTTTT;

    // -------------------------------------------------


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


    //////////////////////////
    //parameters2
    unsigned long long* _2_gamma_device = context.Context_GPU2;
    unsigned long long* _2_gamma_mu_device = _2_gamma_device + 1;
    unsigned long long* _2_gamma_bit_device = _2_gamma_mu_device + 1;

    unsigned long long* _2_coeff_div_plainmod_device = _2_gamma_bit_device + 1;
    unsigned long long* _2_Qi_ready_t_gama_device = _2_coeff_div_plainmod_device + (coeff_modulus - 1);
    unsigned long long* _2_qinverse_ready_device = _2_Qi_ready_t_gama_device + 2 * (coeff_modulus - 1);

    unsigned long long* _2_mul_inv_t_device = _2_qinverse_ready_device + (coeff_modulus - 1);
    unsigned long long* _2_mul_inv_gama_device = _2_mul_inv_t_device + 1;
    unsigned long long* _2_modinv_gama_device = _2_mul_inv_gama_device + 1;


    unsigned long long* CT1 = ciphertext_device + (coeff_modulus - 1) * n;
    Forward_NTT_Inplace(CT1, q_device, mu_device, q_bit_device, n, ForwardPsi_device, (coeff_modulus - 1), (coeff_modulus - 1));

    unsigned long long* secretkey = BFV_random_keygen_;

    Dec1 << <((coeff_modulus - 1) * (n / (1024))), 1024 >> > (CT1, secretkey, ciphertext1_sk_, q_device, mu_device, q_bit_device, n, coeff_modulus);

    Inverse_NTT_Inplace(ciphertext1_sk_, q_device, mu_device, q_bit_device, n, InversePsi_device, (coeff_modulus - 1), (coeff_modulus - 1), INTT_inv_q);



    DecNEWx3 << <((n / 256)), 256 >> > (ciphertext_device, ciphertext1_sk_, plain_device,
        plainmod_device, plainmu_device, plain_bit_device,
        _2_gamma_device, _2_gamma_mu_device, _2_gamma_bit_device,
        _2_qinverse_ready_device, _2_Qi_ready_t_gama_device,
        _2_mul_inv_t_device, _2_mul_inv_gama_device,
        q_device, mu_device, q_bit_device,
        _2_modinv_gama_device,
        n, coeff_modulus);

   


}
