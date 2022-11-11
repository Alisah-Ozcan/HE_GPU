#pragma once

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include <string>
#include <iostream>


#include "CPU_GPU_128.cuh"
#include "Context_Pool.cuh"

using namespace std;



class Lib_Parameters {

public:


    unsigned long long* Context_GPU;
    unsigned long long* Context_GPU2;
    unsigned long long* Encode_addr;


    unsigned long long* Multiplication_Pool;
    unsigned long long* adjust_poly;// for multiply plain



    // Buralara ayrı class olacak sonrasında
    unsigned long long* BFV_random_keygen;
    unsigned long long* BFV_random_enc;
    unsigned long long* pk0_device;
    unsigned long long* pk_u_device;

    // decryption temporary
    unsigned long long* ciphertext1_sk;

    // TEST FOR DEC 
    unsigned long long* DEC_TESTTTT;


    int n;
    int total_bits;
    int coeff_modulus;
    long double n_power;
    unsigned long long plain_mod;
    string scheme;
    security_level sec;


    __host__ Lib_Parameters(string scheme_type, int poly_degree, security_level sec_level = security_level::HES_128) {


        if (scheme_type != "BFV")
            throw("Invalid Scheme Type");

        scheme = scheme_type;
        n = poly_degree;
        n_power = log2l(n);
        sec = sec_level;


        Prime_Pool POOL(n, sec);
        coeff_modulus = POOL.Prime_Count();
        total_bits = POOL.Total_Primes_Bits();
        plain_mod = POOL.Plain_Modulus()[0];
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Önemli !!

        cudaMalloc(&BFV_random_keygen, 3 * coeff_modulus * n * sizeof(unsigned long long));// (sk,e,a)   // ---> OK
        cudaMalloc(&BFV_random_enc, 3 * coeff_modulus * n * sizeof(unsigned long long));// (u,e1,e2)     // ---> OK
        cudaMalloc(&pk0_device, coeff_modulus * n * sizeof(unsigned long long));                         // ---> OK
        cudaMalloc(&pk_u_device, 2 * coeff_modulus * n * sizeof(unsigned long long));                    // ---> OK

        cudaMalloc(&ciphertext1_sk, (coeff_modulus - 1) * n * sizeof(unsigned long long)); // temporary  // ---> OK

        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        cudaMalloc(&DEC_TESTTTT, 2 * n * sizeof(unsigned long long));


        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        unsigned long long* en_de_L = (unsigned long long*)malloc(sizeof(unsigned long long) * n);

        int m = n << 1;
        int gen = 3;
        int pos = 1;
        int index = 0;
        int location = 0;
        for (int i = 0; i < int(n / 2); i++) {

            index = (pos - 1) >> 1;
            location = bitreverse(index, n_power);
            en_de_L[i] = location;
            pos *= gen;
            pos &= (m - 1);

        }
        for (int i = int(n / 2); i < n; i++) {

            index = (m - pos - 1) >> 1;
            location = bitreverse(index, n_power);
            en_de_L[i] = location;
            pos *= gen;
            pos &= (m - 1);

        }


        cudaMalloc(&Encode_addr, n * sizeof(unsigned long long));
        cudaMemcpy(Encode_addr, en_de_L, n * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        free(en_de_L);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        cudaMalloc(&Multiplication_Pool, 4 * (coeff_modulus - 1) * n * sizeof(unsigned long long) +
            4 * (coeff_modulus)*n * sizeof(unsigned long long) +
            4 * n * sizeof(unsigned long long) +
            2 * coeff_modulus * n * sizeof(unsigned long long) +
            2 * coeff_modulus * n * sizeof(unsigned long long) +
            3 * (coeff_modulus - 1) * n * sizeof(unsigned long long) +
            3 * ((coeff_modulus - 1) + coeff_modulus) * n * sizeof(unsigned long long) +
            3 * (coeff_modulus - 1) * n * sizeof(unsigned long long) +
            2 * (coeff_modulus - 1) * n * sizeof(unsigned long long) +
            2 * (coeff_modulus - 1) * n * sizeof(unsigned long long));

        cudaMalloc(&Context_GPU,
            coeff_modulus * sizeof(unsigned long long) + // q
            coeff_modulus * sizeof(unsigned long long) + // mu
            coeff_modulus * sizeof(unsigned long long) + // qbit
            coeff_modulus * 2 * sizeof(unsigned long long) + // qıntt
            coeff_modulus * 2 * sizeof(unsigned long long) + // muıntt
            coeff_modulus * 2 * sizeof(unsigned long long) + // bitıntt
            coeff_modulus * n * sizeof(unsigned long long) + // forwardpsi
            coeff_modulus * n * sizeof(unsigned long long) + // ınversepsi
            coeff_modulus * 2 * n * sizeof(unsigned long long) + // doubleınversepsi
            (coeff_modulus - 1) * sizeof(unsigned long long) + // lastqmodinv
            sizeof(unsigned long long) + // half
            (coeff_modulus - 1) * sizeof(unsigned long long) + // halfmod
            coeff_modulus * sizeof(unsigned long long) + // ninverseq
            (2 * coeff_modulus) * sizeof(unsigned long long) + // doubleninverseq
            sizeof(unsigned long long) + // plainmod
            sizeof(unsigned long long) + // plainpsi
            sizeof(unsigned long long) + // plain_ninverse
            n * sizeof(unsigned long long) + //forward plain psitable
            n * sizeof(unsigned long long) + //inverse plain psitable
            (coeff_modulus - 1) * sizeof(unsigned long long) + //plain_upper_half_increment_device
            sizeof(unsigned long long) + //plain_upper_half_threshold_device
            sizeof(unsigned long long) + // plain mu
            sizeof(unsigned long long) + // plain bit

            sizeof(unsigned long long) + // aux_m_tilde
            (coeff_modulus - 1) * sizeof(unsigned long long) + // p
            coeff_modulus * sizeof(unsigned long long) + // aux_B_m_sk
            ((coeff_modulus - 1) * coeff_modulus) * sizeof(unsigned long long) + //base_change_matrix1_device
            (coeff_modulus - 1) * sizeof(unsigned long long) + //base_change_matrix2_device
            sizeof(unsigned long long) + // inv_prod_q_mod_m_tilde
            coeff_modulus * sizeof(unsigned long long) + //inv_m_tilde_mod_Bsk
            coeff_modulus * sizeof(unsigned long long) + //prod_q_mod_Bsk
            coeff_modulus * sizeof(unsigned long long) + // base_Bsk_elt_mu
            coeff_modulus * sizeof(unsigned long long) + // base_Bsk_bitlength
            sizeof(unsigned long long) + // t
            coeff_modulus * sizeof(unsigned long long) + // aux_B1
            coeff_modulus * sizeof(unsigned long long) + // inv_prod_q_mod_Bsk
            coeff_modulus * sizeof(unsigned long long) + // p1
            (coeff_modulus - 1) * (coeff_modulus - 1) * sizeof(unsigned long long) + // base_change_matrix3
            (coeff_modulus - 1) * sizeof(unsigned long long) + // base_change_matrix4
            sizeof(unsigned long long) + // m_sk
            sizeof(unsigned long long) + // inv_prod_B_mod_m_sk
            (coeff_modulus - 1) * sizeof(unsigned long long) + // prod_B_mod_q
            coeff_modulus * n * sizeof(unsigned long long) + // ForwardPsi_BSK
            coeff_modulus * n * sizeof(unsigned long long) + // InversePsi_BSK
            coeff_modulus * sizeof(unsigned long long) + // INTT_inv_bsk

            (coeff_modulus - 1) * sizeof(unsigned long long) // BACKVALUE
        );

        cudaMalloc(&adjust_poly, n * (coeff_modulus - 1) * sizeof(unsigned long long));

        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        //Extract Context Parameters

        unsigned long long* q_device = Context_GPU; // Tık
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
        cudaMemcpy(BACK_VALUE, POOL.BackValue(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        //Multiplication: -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- 


        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // FOR OPERATION CONTEXT


        cudaMemcpy(q_device, POOL.Primes(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(mu_device, POOL.Primes_Mu(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(q_bit_device, POOL.Primes_Bits(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(lastq_modinv_device, POOL.Last_q_modinv(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(q_INTT_device, POOL.Primes_double(), coeff_modulus * 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(q_mu_INTT_device, POOL.Primes_Mu_double(), coeff_modulus * 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(q_bit_INTT_device, POOL.Primes_Bits_double(), coeff_modulus * 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        unsigned long long* psi_device;
        cudaMalloc(&psi_device, coeff_modulus * sizeof(unsigned long long));
        cudaMemcpy(psi_device, POOL.Primes_Psi(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        NTT_Table << < coeff_modulus * int(n / 1024), 1024 >> > (q_device, psi_device, ForwardPsi_device, InversePsi_device, DoubleInversePsi_device, n, coeff_modulus, n_power);

        cudaFree(psi_device);

        cudaMemcpy(half_device, POOL.Half(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(half_mod_device, POOL.Halfmod(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        //INTT
        cudaMemcpy(INTT_inv_q, POOL.n_inverse(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(INTT_inv_double_q, POOL.n_inverse_double(), (2 * coeff_modulus) * sizeof(unsigned long long), cudaMemcpyHostToDevice);


        // -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- 


        cudaMemcpy(plainmod_device, POOL.Plain_Modulus(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(plainmu_device, POOL.Plain_Modulus() + 1, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(plain_bit_device, POOL.Plain_Modulus() + 2, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(plainpsi_device, POOL.Plain_Modulus() + 3, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(plain_ninverse, POOL.n_plain_inverse(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        NTT_Table_plain << < int(n / 1024), 1024 >> > (plainmod_device, plainpsi_device, ForwardPlainPsi_device, InversePlainPsi_device, n, n_power);
        cudaMemcpy(plain_upper_half_increment_device, POOL.Upper_half_increment(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(plain_upper_half_threshold_device, POOL.Upper_half_threshold(), sizeof(unsigned long long), cudaMemcpyHostToDevice);


        // -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----

        //Multiplication: -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- 


        //cudaMemcpy(INTT_inv_bsk, INTT_MODINV_BSK, coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(INTT_inv_bsk, POOL.Auxiliary_Bases_inverse(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        //Multiplication Copy:  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- 


        cudaMemcpy(aux_m_tilde_device, POOL.M_Tilde(), sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(p_device, POOL.p(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(aux_B_m_sk_device, POOL.Auxiliary_Bases(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_change_matrix1_device, POOL.base_change_matrix1(), ((coeff_modulus - 1) * coeff_modulus) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_change_matrix2_device, POOL.base_change_matrix2(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(inv_prod_q_mod_m_tilde_device, POOL.inv_prod_q_mod_m_tilde(), sizeof(unsigned long long), cudaMemcpyHostToDevice);


        cudaMemcpy(inv_m_tilde_mod_Bsk_device, POOL.inv_m_tilde_mod_Bsk(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(prod_q_mod_Bsk_device, POOL.prod_q_mod_Bsk(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_Bsk_elt_mu_device, POOL.Auxiliary_Bases_mu(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_Bsk_bitlength_device, POOL.Auxiliary_Bases_bit(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(t_device, POOL.Plain_Modulus(), sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(aux_B1_device, POOL.Auxiliary_Bases(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(inv_prod_q_mod_Bsk_device, POOL.inv_prod_q_mod_Bsk(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(p1_device, POOL.p1(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_change_matrix3_device, POOL.base_change_matrix3(), (coeff_modulus - 1) * (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(base_change_matrix4_device, POOL.base_change_matrix4(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(m_sk_device, POOL.M_SK(), sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(inv_prod_B_mod_m_sk_device, POOL.inv_prod_B_mod_m_sk(), sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(prod_B_mod_q_device, POOL.prod_B_mod_q(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        ///////////////////////
        unsigned long long* psi_device_bsk;
        cudaMalloc(&psi_device_bsk, coeff_modulus * sizeof(unsigned long long));
        cudaMemcpy(psi_device_bsk, POOL.Auxiliary_Bases_Psi(), coeff_modulus * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        NTT_Table_BSK << < coeff_modulus * int(n / 1024), 1024 >> > (aux_B_m_sk_device, psi_device_bsk, ForwardPsi_device_BSK, InversePsi_device_BSK, n, coeff_modulus, n_power);



        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cudaMalloc(&Context_GPU2,

            sizeof(unsigned long long) + // gamma
            sizeof(unsigned long long) + // gamma_mu
            sizeof(unsigned long long) + // gamma_bit

            (coeff_modulus - 1) * sizeof(unsigned long long) + //coeff_div_plainmod
            2 * (coeff_modulus - 1) * sizeof(unsigned long long) + //Qi_ready_t_gama
            (coeff_modulus - 1) * sizeof(unsigned long long) + //qinverse_ready

            sizeof(unsigned long long) + // mul_inv_t
            sizeof(unsigned long long) + // mul_inv_gama
            sizeof(unsigned long long) + // modinv_gama

            sizeof(unsigned long long) // Q_mod_t

        );



        unsigned long long* _2_gamma_device = Context_GPU2;
        unsigned long long* _2_gamma_mu_device = _2_gamma_device + 1;
        unsigned long long* _2_gamma_bit_device = _2_gamma_mu_device + 1;

        unsigned long long* _2_coeff_div_plainmod_device = _2_gamma_bit_device + 1;
        unsigned long long* _2_Qi_ready_t_gama_device = _2_coeff_div_plainmod_device + (coeff_modulus - 1);
        unsigned long long* _2_qinverse_ready_device = _2_Qi_ready_t_gama_device + 2 * (coeff_modulus - 1);

        unsigned long long* _2_mul_inv_t_device = _2_qinverse_ready_device + (coeff_modulus - 1);
        unsigned long long* _2_mul_inv_gama_device = _2_mul_inv_t_device + 1;
        unsigned long long* _2_modinv_gama_device = _2_mul_inv_gama_device + 1;
        unsigned long long* _2_Q_mod_t_device = _2_modinv_gama_device + 1;


        cudaMemcpy(_2_gamma_device, POOL.Gamma(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_gamma_mu_device, POOL.Gamma() + 1, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_gamma_bit_device, POOL.Gamma() + 2, sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(_2_coeff_div_plainmod_device, POOL.Coeff_div_plain_modulus(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_Qi_ready_t_gama_device, POOL.Qi_ready_t_gama(), 2 * (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_qinverse_ready_device, POOL.Qi_inverse_ready_t_gama(), (coeff_modulus - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(_2_mul_inv_t_device, POOL.Mul_inv_t(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_mul_inv_gama_device, POOL.Mul_inv_gama(), sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(_2_modinv_gama_device, POOL.Mod_inv_gama(), sizeof(unsigned long long), cudaMemcpyHostToDevice);

        cudaMemcpy(_2_Q_mod_t_device, POOL.Q_mod_t(), sizeof(unsigned long long), cudaMemcpyHostToDevice);



        // ------------------------------------------------ TEST ------------------------------------------------ \\

        /*
        unsigned long long* TEST_FIELD = (unsigned long long*)malloc( sizeof(unsigned long long));
        cudaMemcpy(TEST_FIELD, _2_modinv_gama_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cout << " --- TEST FIELD --- " << endl;
        for (int i = 0; i < 1; i++) {
            cout << "Index: " << i << ": " << TEST_FIELD[i] << endl;

        }
        cout << " --- --- -- --- --- " << endl;
        */

        // ------------------------------------------------------------------------------------------------------ \\


    }


};



class GPU_Messagge
{
public:

    unsigned long long* GPU_Location;

    unsigned long long ring_size;

    __host__ __device__ __forceinline__ GPU_Messagge()
    {
        ring_size = 0;
    }

    __host__ __device__ __forceinline__ GPU_Messagge(Lib_Parameters context)// empty message allocate
    {
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, ring_size * sizeof(unsigned long long));
    }

    __host__ __device__ __forceinline__ GPU_Messagge(unsigned long long* messagge, Lib_Parameters context)
    {
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, ring_size * sizeof(unsigned long long));
        cudaMemcpy(GPU_Location, messagge, ring_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }

    __host__ __device__ __forceinline__ void kill()
    {
        cudaFree(GPU_Location);
    }

};


class GPU_Plaintext
{
public:

    unsigned long long* GPU_Location;

    unsigned long long ring_size;

    __host__ __device__ __forceinline__ GPU_Plaintext()
    {
        ring_size = 0;
    }

    __host__ __device__ __forceinline__ GPU_Plaintext(Lib_Parameters context)// empty plaintext allocate
    {
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, ring_size * sizeof(unsigned long long));

    }

    __host__ __device__ __forceinline__ GPU_Plaintext(unsigned long long* plain, Lib_Parameters context)
    {
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, ring_size * sizeof(unsigned long long));
        cudaMemcpy(GPU_Location, plain, ring_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }

    __host__ __device__ __forceinline__ void kill()
    {
        cudaFree(GPU_Location);
    }

};


class GPU_Ciphertext
{
public:

    unsigned long long* GPU_Location;
    unsigned long long* GPU_Location2;

    unsigned long long coeff_mod_count;
    unsigned long long cipher_size;
    unsigned long long ring_size;

    __host__ __device__ __forceinline__ GPU_Ciphertext()
    {
        coeff_mod_count = 0;
        cipher_size = 0;
        ring_size = 0;

    }

    __host__ __device__ __forceinline__ GPU_Ciphertext(Lib_Parameters context)// empty cipher allocate
    {

        coeff_mod_count = context.coeff_modulus - 1; 
        cipher_size = 2;//default
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, cipher_size * coeff_mod_count * ring_size * sizeof(unsigned long long));

    }

    __host__ __device__ __forceinline__ GPU_Ciphertext(unsigned long long* cipher, Lib_Parameters context)
    {
        coeff_mod_count = context.coeff_modulus - 1; 
        cipher_size = 2;//default
        ring_size = context.n; // n

        cudaMalloc(&GPU_Location, cipher_size * coeff_mod_count * ring_size * sizeof(unsigned long long));
        cudaMemcpy(GPU_Location, cipher, cipher_size * coeff_mod_count * ring_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }


    __host__ __device__ __forceinline__ void pre_mult()
    {
        cipher_size = cipher_size + 1;

        cudaMalloc(&GPU_Location2, coeff_mod_count * ring_size * sizeof(unsigned long long));

    }

    __host__ __device__ __forceinline__ void post_relin()
    {
        cipher_size = cipher_size - 1;

        //cudaMalloc(&GPU_Location2, coeff_mod_count * ring_size * sizeof(unsigned long long));
        cudaFree(GPU_Location2);

    }

    __host__ __device__ __forceinline__ void kill()
    {
        cudaFree(GPU_Location);
        cudaFree(GPU_Location2);
    }

};



class GPU_Relinkey
{
public:

    unsigned long long* GPU_Location;
    unsigned long long* GPU_Random_Location;// for random memory

    unsigned long long* GPU_temp; // for relinearization operation

    unsigned long long coeff_mod_count;
    unsigned long long decompmod_count;
    unsigned long long ring_size;

    __host__ __device__ __forceinline__ GPU_Relinkey()
    {
        coeff_mod_count = 0;
        decompmod_count = 0;
        ring_size = 0;

    }

    __host__ __device__ __forceinline__ GPU_Relinkey(Lib_Parameters context)
    {

        ring_size = context.n;
        coeff_mod_count = context.coeff_modulus; // q count
        decompmod_count = coeff_mod_count - 1;

        cudaMalloc(&GPU_Random_Location, 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));

        cudaMalloc(&GPU_Location, 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));

        cudaMalloc(&GPU_temp, coeff_mod_count * 2 * ring_size * sizeof(unsigned long long) + decompmod_count * coeff_mod_count * ring_size * sizeof(unsigned long long));

    }

    __host__ __device__ __forceinline__ void kill()
    {
        cudaFree(GPU_Location);
        cudaFree(GPU_temp);
    }

};


class GPU_GaloisKey // Yeniden tasarlanması gerekli
{
public:

    unsigned long long* GPU_Location_positive;
    unsigned long long* GPU_Location_negative;

    unsigned long long* GPU_Random_Location;// for random memory
    unsigned long long* GPU_temp;

    int* galois_elt_pos;
    int* galois_elt_neg;

    unsigned long long coeff_mod_count;
    unsigned long long decompmod_count;
    unsigned long long ring_size;

    int max_power_shift;
    bool neg_shift_;

    __host__ __device__ __forceinline__ GPU_GaloisKey()
    {

        coeff_mod_count = 0;
        decompmod_count = 0;
        ring_size = 0;

    }

    __host__ __device__ __forceinline__ GPU_GaloisKey(Lib_Parameters context, int max_power = 5, bool neg_shift = false)
    {

        ring_size = context.n;
        coeff_mod_count = context.coeff_modulus; // q count
        decompmod_count = coeff_mod_count - 1;

        max_power_shift = max_power;
        neg_shift_ = neg_shift;

        cudaMalloc(&GPU_Location_positive, max_power * 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));
        galois_elt_pos = (int*)malloc(max_power * sizeof(int));

      
        if (neg_shift) {

            cudaMalloc(&GPU_Location_negative, max_power * 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));
            galois_elt_neg = (int*)malloc(max_power * sizeof(int));

        }

        cudaMalloc(&GPU_Random_Location, 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));

        cudaMalloc(&GPU_temp, coeff_mod_count * 2 * ring_size * sizeof(unsigned long long) + decompmod_count * 2 * ring_size * sizeof(unsigned long long) + decompmod_count * coeff_mod_count * ring_size * sizeof(unsigned long long));

    }




    /*
    __host__ __device__ __forceinline__ GPU_GaloisKey(GaloisKeys evk, SEALContext context, int max_power = 5, bool neg_shift = false)
    {

        auto context_data = context.key_context_data();
        auto galois_tool = context_data->galois_tool();

        ring_size = context_data->parms().poly_modulus_degree(); // n
        coeff_mod_count = context_data->parms().coeff_modulus().size(); // q count
        decompmod_count = coeff_mod_count - 1;

        cudaMalloc(&GPU_Location_positive, max_power * 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));
        galois_elt_pos = (int*)malloc(max_power * sizeof(int));

        for (int k = 0; k < max_power; k++) {

            int galois_elt = galois_tool->get_elt_from_step(pow(2, k));
            galois_elt_pos[k] = galois_elt;
            KeySet_to_GPU(ring_size, coeff_mod_count, GPU_Location_positive + (k * (coeff_mod_count * decompmod_count * 2 * ring_size)), evk, galois_elt);

        }

        if (neg_shift) {
            cudaMalloc(&GPU_Location_negative, max_power * 2 * coeff_mod_count * ring_size * decompmod_count * sizeof(unsigned long long));
            galois_elt_neg = (int*)malloc(max_power * sizeof(int));

            for (int k = 0; k < max_power; k++) {

                int galois_elt = galois_tool->get_elt_from_step((pow(2, k) * (-1)));
                galois_elt_neg[k] = galois_elt;
                KeySet_to_GPU(ring_size, coeff_mod_count, GPU_Location_negative + (k * (coeff_mod_count * decompmod_count * 2 * ring_size)), evk, galois_elt);

            }
        }

        cudaMalloc(&GPU_temp, coeff_mod_count * 2 * ring_size * sizeof(unsigned long long) + decompmod_count * 2 * ring_size * sizeof(unsigned long long) + decompmod_count * coeff_mod_count * ring_size * sizeof(unsigned long long));

    }
    */
    __host__ __device__ __forceinline__ void kill()
    {
        cudaFree(GPU_Location_positive);
        cudaFree(GPU_Location_negative);
        cudaFree(GPU_temp);
        free(galois_elt_pos);
        free(galois_elt_neg);

    }

};