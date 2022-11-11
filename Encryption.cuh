#include "Key_Generation.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

__global__ void BFV_RANDOM_Enc(unsigned long long* memory, unsigned long long* q, int seed, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Random For Encryption
    if (idx < coeff_count) {

        unsigned long long RN;

        random_0_1(idx, seed, RN);// it returns 0,1,2

        for (int i = 0; i < q_count; i++) {

            int location = i * coeff_count;
            unsigned long long q_reg = q[i];

            unsigned long long result = RN;

            if (result == 2)
                result = q_reg - 1;

            memory[idx + location] = result;
           
        }

    }
    else if ((coeff_count <= idx) && (idx < (coeff_count << 1))) {

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
                result = q_reg - int(result);

            memory[idx_inside + location] = result;
           
        }

    }
    else { // e2

        int start_point = (2 * q_count * coeff_count);
        int idx_inside = idx - (2 * coeff_count);

        float RN;

        random_normal_dist(idx, seed, RN);

        RN *= 1;// 3.2;

        for (int i = 0; i < q_count; i++) {

            int location = start_point + (i * coeff_count);
            unsigned long long q_reg = q[i];

            unsigned long long result = RN;
            if (result < 0)
                result = q_reg - int(result);

            memory[idx_inside + location] = result;
          
        }

    }

}


//Encryption --------------------------------------------------


__global__ void Enc1(unsigned long long* pk0, unsigned long long* pk1, unsigned long long* u, unsigned long long* pk_u, unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * q_count;

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        uint128_t pk_reg = pk0[idx];
        mul64(pk_reg.low, u[idx], pk_reg);
        singleBarrett(pk_reg, q[index], mu[index], q_bit[index]);
        pk_u[idx] = pk_reg.low;
    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        uint128_t pk_reg = pk1[idx_in];
        mul64(pk_reg.low, u[idx_in], pk_reg);
        singleBarrett(pk_reg, q[index], mu[index], q_bit[index]);
        pk_u[idx] = pk_reg.low;
    }

}


__global__ void Enc2(unsigned long long* ciphertext, unsigned long long* plaintext, unsigned long long* pk_u, unsigned long long* e_0, unsigned long long* e_1, unsigned long long* half, unsigned long long* halfmod,
    unsigned long long* lastq_modinv, unsigned long long* divplainmod, unsigned long long* q_mod_t, unsigned long long* upper_threshold, unsigned long long* plain,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * (q_count - 1);

    unsigned long long half_reg = half[0];
    unsigned long long qlast_reg = q[q_count - 1];

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        int loc2 = idx % coeff_count;

        unsigned long long pk0_u = pk_u[idx];
        unsigned long long e0_reg = e_0[idx];

        unsigned long long pk0_u_last = pk_u[loc1 + loc2];
        unsigned long long e0_reg_last = e_0[loc1 + loc2];

        pk0_u = pk0_u + e0_reg;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        pk0_u_last = pk0_u_last + e0_reg_last;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last = pk0_u_last + half_reg;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last += q[index];
        pk0_u_last = pk0_u_last - halfmod[index];
        pk0_u_last -= (pk0_u_last >= q[index]) * q[index];

        pk0_u += q[index];
        pk0_u = pk0_u - pk0_u_last;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk0_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        //---------------------------
        uint128_t messagge = plaintext[loc2];

        unsigned long long prod_ = messagge.low;
        prod_ = prod_ * q_mod_t[0] % plain[0];
        prod_ = prod_ + upper_threshold[0];
        prod_ = int(prod_ / plain[0]);

        //---------------------------
        mul64(messagge.low, divplainmod[index], messagge);
        singleBarrett(messagge, q[index], mu[index], q_bit[index]);

        pk0_u = mult.low + messagge.low + prod_;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        ciphertext[idx] = pk0_u;

    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        int loc2 = idx_in % coeff_count;

        unsigned long long pk1_u = pk_u[coeff_count + idx];
        unsigned long long e1_reg = e_1[idx_in];

        unsigned long long pk1_u_last = pk_u[loc1 + (coeff_count * (q_count)) + loc2];
        unsigned long long e1_reg_last = e_1[loc1 + loc2];

        pk1_u = pk1_u + e1_reg;
        pk1_u -= (pk1_u >= q[index]) * q[index];

        pk1_u_last = (pk1_u_last + e1_reg_last);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last = (pk1_u_last + half_reg);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last += q[index];
        pk1_u_last = (pk1_u_last - halfmod[index]);
        pk1_u_last -= (pk1_u_last >= q[index]) * q[index];

        pk1_u += q[index];
        pk1_u = pk1_u - pk1_u_last;
        pk1_u -= (pk1_u >= q[index]) * q[index];


        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk1_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = mult.low;

    }
}


__global__ void Enc2x4(unsigned long long* ciphertext, unsigned long long* plaintext, unsigned long long* pk_u, unsigned long long* e_0, unsigned long long* e_1, unsigned long long* half, unsigned long long* halfmod,
    unsigned long long* lastq_modinv, unsigned long long* divplainmod, unsigned long long* q_mod_t, unsigned long long* upper_threshold, unsigned long long* plain,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * (q_count - 1);

    unsigned long long half_reg = half[0];
    unsigned long long qlast_reg = q[q_count - 1];

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        int loc2 = idx % coeff_count;

        unsigned long long pk0_u = pk_u[idx];
        unsigned long long e0_reg = e_0[idx];

        unsigned long long pk0_u_last = pk_u[loc1 + loc2];
        unsigned long long e0_reg_last = e_0[loc1 + loc2];

        pk0_u = pk0_u + e0_reg;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        pk0_u_last = pk0_u_last + e0_reg_last;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last = pk0_u_last + half_reg;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last += q[index];
        pk0_u_last = pk0_u_last - halfmod[index];
  

        uint128_t pk0_u_last_128 = pk0_u_last;
        singleBarrett(pk0_u_last_128, q[index], mu[index], q_bit[index]);


        pk0_u += q[index];
        pk0_u = pk0_u - pk0_u_last_128.low;

        pk0_u -= (pk0_u >= q[index]) * q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk0_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        uint128_t messagge = plaintext[loc2];

        unsigned long long prod_ = messagge.low;
        prod_ = prod_ * q_mod_t[0];
        prod_ = prod_ + upper_threshold[0];
        prod_ = int(prod_ / plain[0]);

        if (idx == 0) {
            printf("fix: %llu \n", prod_);
        }


        mul64(messagge.low, divplainmod[index], messagge);
        singleBarrett(messagge, q[index], mu[index], q_bit[index]);

        pk0_u = mult.low + messagge.low + prod_;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        ciphertext[idx] = pk0_u;


    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        int loc2 = idx_in % coeff_count;

        unsigned long long pk1_u = pk_u[coeff_count + idx];
        unsigned long long e1_reg = e_1[idx_in];

        unsigned long long pk1_u_last = pk_u[loc1 + (coeff_count * (q_count)) + loc2];
        unsigned long long e1_reg_last = e_1[loc1 + loc2];

        pk1_u = pk1_u + e1_reg;
        pk1_u -= (pk1_u >= q[index]) * q[index];

        pk1_u_last = (pk1_u_last + e1_reg_last);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last = (pk1_u_last + half_reg);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last += q[index];
        pk1_u_last = (pk1_u_last - halfmod[index]);


        uint128_t pk1_u_last_128 = pk1_u_last;
        singleBarrett(pk1_u_last_128, q[index], mu[index], q_bit[index]);

        pk1_u += q[index];
        pk1_u = pk1_u - pk1_u_last_128.low;
        pk1_u -= (pk1_u >= q[index]) * q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk1_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = mult.low;

    }
}


__global__ void Enc2_Last1(unsigned long long* ciphertext, unsigned long long* plaintext, unsigned long long* pk_u, unsigned long long* e_0, unsigned long long* e_1, unsigned long long* half, unsigned long long* halfmod,
    unsigned long long* lastq_modinv, unsigned long long* divplainmod, unsigned long long* q_mod_t, unsigned long long* upper_threshold, unsigned long long* plain,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * (q_count - 1);

    unsigned long long half_reg = half[0];
    unsigned long long qlast_reg = q[q_count - 1];

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        int loc2 = idx % coeff_count;

        unsigned long long pk0_u = pk_u[idx];
        unsigned long long e0_reg = e_0[idx];

        unsigned long long pk0_u_last = pk_u[loc1 + loc2];
        unsigned long long e0_reg_last = e_0[loc1 + loc2];

        pk0_u = pk0_u + e0_reg;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        pk0_u_last = pk0_u_last + e0_reg_last;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;

        pk0_u_last = pk0_u_last + half_reg;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;

        pk0_u_last += q[index];
        pk0_u_last = pk0_u_last - halfmod[index];
        uint128_t pk0_u_last_128 = pk0_u_last;
        singleBarrett(pk0_u_last_128, q[index], mu[index], q_bit[index]);


        pk0_u += q[index];
        pk0_u = pk0_u - pk0_u_last_128.low;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk0_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);


        //--------------------------------------------------------

        //------------------------------

        //---------------------------
        uint128_t messagge = plaintext[loc2];

        unsigned long long prod_ = messagge.low;
        prod_ = prod_ * q_mod_t[0];// % plain[0];
        prod_ = prod_ + upper_threshold[0];
        prod_ = int(prod_ / plain[0]);

        


        
        mul64(messagge.low, divplainmod[index], messagge);

        messagge = messagge + prod_;
        messagge = messagge + mult.low;
        singleBarrett(messagge, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = messagge.low;
        


        

    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        int loc2 = idx_in % coeff_count;

        unsigned long long pk1_u = pk_u[coeff_count + idx];
        unsigned long long e1_reg = e_1[idx_in];

        unsigned long long pk1_u_last = pk_u[loc1 + (coeff_count * (q_count)) + loc2];
        unsigned long long e1_reg_last = e_1[loc1 + loc2];

        pk1_u = pk1_u + e1_reg;
        pk1_u -= (pk1_u >= q[index]) * q[index];

        pk1_u_last = (pk1_u_last + e1_reg_last);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last = (pk1_u_last + half_reg);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last += q[index];
        pk1_u_last = (pk1_u_last - halfmod[index]);
        uint128_t pk1_u_last_128 = pk1_u_last;
        singleBarrett(pk1_u_last_128, q[index], mu[index], q_bit[index]);


        pk1_u += q[index];
        pk1_u = pk1_u - pk1_u_last_128.low;
        pk1_u -= (pk1_u >= q[index]) * q[index];


        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk1_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = mult.low;

    }
}



__global__ void Enc2x(unsigned long long* ciphertext, unsigned long long* plaintext, unsigned long long* pk_u, unsigned long long* e_0, unsigned long long* e_1, unsigned long long* half, unsigned long long* halfmod,
    unsigned long long* lastq_modinv, unsigned long long* divplainmod, unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * (q_count - 1);

    unsigned long long half_reg = half[0];
    unsigned long long qlast_reg = q[q_count - 1];

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        int loc2 = idx % coeff_count;

        unsigned long long pk0_u = pk_u[idx];
        unsigned long long e0_reg = e_0[idx];

        unsigned long long pk0_u_last = pk_u[loc1 + loc2];
        unsigned long long e0_reg_last = e_0[loc1 + loc2];

        pk0_u = pk0_u + e0_reg;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        pk0_u_last = pk0_u_last + e0_reg_last;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last = pk0_u_last + half_reg;
        pk0_u_last -= (pk0_u_last >= qlast_reg) * qlast_reg;
        pk0_u_last += q[index];
        pk0_u_last = pk0_u_last - halfmod[index];
 

        uint128_t pk0_u_last_128 = pk0_u_last;
        singleBarrett(pk0_u_last_128, q[index], mu[index], q_bit[index]);


        pk0_u += q[index];
        pk0_u = pk0_u - pk0_u_last_128.low;

        pk0_u -= (pk0_u >= q[index]) * q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk0_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);


        uint128_t messagge = plaintext[loc2];
        mul64(messagge.low, divplainmod[index], messagge);
        singleBarrett(messagge, q[index], mu[index], q_bit[index]);

        pk0_u = mult.low + messagge.low;
        pk0_u -= (pk0_u >= q[index]) * q[index];

        ciphertext[idx] = pk0_u;

    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        int loc2 = idx_in % coeff_count;

        unsigned long long pk1_u = pk_u[coeff_count + idx];
        unsigned long long e1_reg = e_1[idx_in];

        unsigned long long pk1_u_last = pk_u[loc1 + (coeff_count * (q_count)) + loc2];
        unsigned long long e1_reg_last = e_1[loc1 + loc2];

        pk1_u = pk1_u + e1_reg;
        pk1_u -= (pk1_u >= q[index]) * q[index];

        pk1_u_last = (pk1_u_last + e1_reg_last);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last = (pk1_u_last + half_reg);
        pk1_u_last -= (pk1_u_last >= qlast_reg) * qlast_reg;
        pk1_u_last += q[index];
        pk1_u_last = (pk1_u_last - halfmod[index]);

        uint128_t pk1_u_last_128 = pk1_u_last;
        singleBarrett(pk1_u_last_128, q[index], mu[index], q_bit[index]);

        pk1_u += q[index];
        pk1_u = pk1_u - pk1_u_last_128.low;
        pk1_u -= (pk1_u >= q[index]) * q[index];


        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk1_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = mult.low;

    }
}



__global__ void Enc2_Last2__(unsigned long long* ciphertext, unsigned long long* plaintext, unsigned long long* pk_u, unsigned long long* e_0, unsigned long long* e_1, unsigned long long* half, unsigned long long* halfmod,
    unsigned long long* lastq_modinv, unsigned long long* divplainmod, unsigned long long* q_mod_t, unsigned long long* upper_threshold, unsigned long long* plain,
    unsigned long long* q, unsigned long long* mu, unsigned long long* q_bit, int coeff_count, int q_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loc1 = coeff_count * (q_count - 1);

    unsigned long long half_reg = half[0];
    unsigned long long qlast_reg = q[q_count - 1];

    if (idx < loc1) {
        int index = int(idx / coeff_count);
        int loc2 = idx % coeff_count;

        unsigned long long pk0_u = pk_u[idx];
        unsigned long long e0_reg = e_0[idx];

        unsigned long long pk0_u_last = pk_u[loc1 + loc2];
        unsigned long long e0_reg_last = e_0[loc1 + loc2];

        pk0_u = pk0_u + e0_reg % q[index];
       
        pk0_u_last = pk0_u_last + e0_reg_last % qlast_reg;
        pk0_u_last = pk0_u_last + half_reg % qlast_reg;;

        pk0_u_last += q[index];
        pk0_u_last = pk0_u_last - halfmod[index] % q[index];
       
        pk0_u += q[index];
        pk0_u = pk0_u - pk0_u_last % q[index];
 
        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk0_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        //--------------------------------------------------------

        //------------------------------

        //---------------------------
        uint128_t messagge = plaintext[loc2];

        unsigned long long prod_ = messagge.low;
        prod_ = prod_ * q_mod_t[0];// % plain[0];
        prod_ = prod_ + upper_threshold[0];
        prod_ = int(prod_ / plain[0]);

        //---------------------------

   
        mul64(messagge.low, divplainmod[index], messagge);

        messagge = messagge + prod_;
        messagge = messagge + mult;
        singleBarrett(messagge, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = messagge.low;





    }
    else {
        int idx_in = idx - loc1;
        int index = int(idx_in / coeff_count);
        int loc2 = idx_in % coeff_count;

        unsigned long long pk1_u = pk_u[coeff_count + idx];
        unsigned long long e1_reg = e_1[idx_in];

        unsigned long long pk1_u_last = pk_u[loc1 + (coeff_count * (q_count)) + loc2];
        unsigned long long e1_reg_last = e_1[loc1 + loc2];

        pk1_u = pk1_u + e1_reg % q[index];

        pk1_u_last = (pk1_u_last + e1_reg_last) % qlast_reg;
        pk1_u_last = (pk1_u_last + half_reg) % qlast_reg;

        pk1_u_last += q[index];
        pk1_u_last = (pk1_u_last - halfmod[index]) % q[index];

    
        pk1_u += q[index];
        pk1_u = pk1_u - pk1_u_last % q[index];

        uint128_t mult = lastq_modinv[index];
        mul64(mult.low, pk1_u, mult);
        singleBarrett(mult, q[index], mu[index], q_bit[index]);

        ciphertext[idx] = mult.low;

    }
}



__host__ void GPU_Enc(GPU_Ciphertext cipher, GPU_Plaintext plain, Lib_Parameters context)
{
    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

    unsigned long long* ciphertext_device = cipher.GPU_Location;
    unsigned long long* plain_device = plain.GPU_Location;

    unsigned long long* BFV_random_keygen_ = context.BFV_random_keygen;
    unsigned long long* pk0_device_ = context.pk0_device;


    unsigned long long* BFV_random_enc_ = context.BFV_random_enc;
    unsigned long long* pk_u_device_ = context.pk_u_device;

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
    unsigned long long* _2_Q_mod_t_device = _2_modinv_gama_device + 1;


    BFV_RANDOM_Enc << <(3 * (n / (1024))), 1024 >> > (BFV_random_enc_, q_device, time(NULL), n, coeff_modulus);// (u,e1,e2)

    unsigned long long* u_device = BFV_random_enc_;
    unsigned long long* e0_device = BFV_random_enc_ + ((coeff_modulus)*n);
    unsigned long long* e1_device = BFV_random_enc_ + (2 * (coeff_modulus)*n);

    Forward_NTT_Inplace(u_device, q_device, mu_device, q_bit_device, n, ForwardPsi_device, coeff_modulus, coeff_modulus);// (u)

    unsigned long long* pk1_device = BFV_random_keygen_ + (2 * (coeff_modulus)*n);

    Enc1 << <(2 * coeff_modulus * (n / (1024))), 1024 >> > (pk0_device_, pk1_device, u_device, pk_u_device_, q_device, mu_device, q_bit_device, n, coeff_modulus);
    Inverse_NTT_Inplace(pk_u_device_, q_device, mu_device, q_bit_device, n, InversePsi_device, 2 * coeff_modulus, coeff_modulus, INTT_inv_q); // normal ciphertext'lerin inverse ntt sonuçları
    
    Enc2_Last1 << <(2 * (coeff_modulus - 1) * (n / (1024))), 1024 >> > (ciphertext_device, plain_device, pk_u_device_, e0_device, e1_device, half_device, half_mod_device, lastq_modinv_device, _2_coeff_div_plainmod_device,
        _2_Q_mod_t_device, plain_upper_half_threshold_device, plainmod_device,
        q_device, mu_device, q_bit_device, n, coeff_modulus);

    
}



