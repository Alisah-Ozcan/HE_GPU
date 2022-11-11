#include "NTT.cuh"
#include "GPU_Context.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

__global__ void encode(unsigned long long* messagge, unsigned long long* messagge_encoded, unsigned long long* location_info) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int location = location_info[idx];
    messagge_encoded[location] = messagge[idx];

}

__global__ void decode(unsigned long long* plaintext, unsigned long long* messagge, unsigned long long* location_info) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int location = location_info[idx];
    messagge[idx] = plaintext[location];

}


__host__ void GPU_Encode(GPU_Messagge input, GPU_Plaintext output, Lib_Parameters context)
{
    //class features
    unsigned long long* messagge = input.GPU_Location;
    unsigned long long* messagge_encoded = output.GPU_Location;
    unsigned long long* location_info = context.Encode_addr;
    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

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


    encode << <(n / 1024), 1024 >> > (messagge, messagge_encoded, location_info);

    Inverse_NTT_Inplace(messagge_encoded, plainmod_device, plainmu_device, plain_bit_device, n, InversePlainPsi_device, 1, 1, plain_ninverse);

}

__host__ void GPU_Decode(GPU_Messagge output, GPU_Plaintext input, Lib_Parameters context)
{
    //class features
    unsigned long long* messagge = output.GPU_Location;
    unsigned long long* plaintxt = input.GPU_Location;
    unsigned long long* location_info = context.Encode_addr;

    int n = context.n;
    int coeff_modulus = context.coeff_modulus;

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

    Forward_NTT_Inplace(plaintxt, plainmod_device, plainmu_device, plain_bit_device, n, ForwardPlainPsi_device, 1, 1);

    decode << <(n / 1024), 1024 >> > (plaintxt, messagge, location_info);

}