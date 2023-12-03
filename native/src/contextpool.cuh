// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include <iostream>
#include <vector>
#include "gpu_kernels/nttparameters.cuh"
#ifndef PRIMEPOOL_H
#define PRIMEPOOL_H

Data extendedGCD(Data a, Data b, Data &x, Data &y);
Data modInverse(Data a, Data m);

class PrimePool {

public:

    enum class security_level : int
    {
        //128-bit classical security level according to HomomorphicEncryption.org standard.
        HES_128 = 128,

        //192-bit classical security level according to HomomorphicEncryption.org standard.
        HES_192 = 192,

        //256-bit classical security level according to HomomorphicEncryption.org standard.
        HES_256 = 256
    };

    static int n;
    static security_level sec;

    PrimePool(int poly_degree, security_level sec_level);

    // Returns prime count for n = 4096, 8192, 16384, 32768
    static int prime_count();

    // Returns main modulus bit count for n = 4096, 8192, 16384, 32768
    static int total_primes_bits();

    // Returns all RNS modulus as vector for n = 4096, 8192, 16384, 32768
    std::vector<Modulus> base_modulus();

    std::vector<Root> ntt_tables();

    std::vector<Root> intt_tables();

    std::vector<Ninverse> n_inverse();

    std::vector<Data> last_q_modinv();


    std::vector<Modulus> base_Bsk(); // with msk

    std::vector<Root> bsk_ntt_tables(); // with msk

    std::vector<Root> bsk_intt_tables(); // with msk

    std::vector<Ninverse> bsk_n_inverse();

    Modulus m_tilde();

    std::vector<Data> base_change_matrix_Bsk();
    
    std::vector<Data> inv_punctured_prod_mod_base_array();
    
    std::vector<Data> base_change_matrix_m_tilde();

    Data inv_prod_q_mod_m_tilde();

    std::vector<Data> inv_m_tilde_mod_Bsk();

    std::vector<Data> prod_q_mod_Bsk();

    std::vector<Data> inv_prod_q_mod_Bsk();

    Modulus plain_modulus();

    std::vector<Modulus> plain_modulus2();

    Data plain_psi();

    std::vector<Ninverse> n_plain_inverse();

    std::vector<Root> plain_ntt_tables();

    std::vector<Root> plain_intt_tables();
    
    std::vector<Data> base_change_matrix_q();

    std::vector<Data> base_change_matrix_msk();

    std::vector<Data> inv_punctured_prod_mod_B_array();

    Data inv_prod_B_mod_m_sk();
    
    std::vector<Data> prod_B_mod_q();



    std::vector<Modulus> q_Bsk_merge_modulus();

    std::vector<Root> q_Bsk_merge_ntt_tables();

    std::vector<Root> q_Bsk_merge_intt_tables();

    std::vector<Ninverse> q_Bsk_n_inverse();


    Data half();

    std::vector<Data> half_mod();

    std::vector<Data> factor();
    
    
    Modulus gamma();
    
    std::vector<Data> coeeff_div_plainmod();

    Data Q_mod_t();

    Data upper_threshold();

    std::vector<Data> upper_halfincrement();


    std::vector<Data> Qi_t();

    std::vector<Data> Qi_gamma();

    std::vector<Data> Qi_inverse();

    Data mulq_inv_t();

    Data mulq_inv_gamma();

    Data inv_gamma();

/*
    std::vector<unsigned long long> primes_double();

    // Returns all RNS modulus's mu as vector for n = 4096, 8192, 16384, 32768
    std::vector<unsigned long long> primes_mu();
    std::vector<unsigned long long> primes_mu_double();

    // Returns all RNS modulus's bits as vector for n = 4096, 8192, 16384, 32768
    std::vector<unsigned long long> primes_bits();
    std::vector<unsigned long long> primes_bits_double();

    // Returns plaintext modulus for n = 4096, 8192, 16384, 32768
    std::vector<unsigned long long> plain_modulus();
    
    std::vector<unsigned long long> primes_psi();

    std::vector<unsigned long long> half();
    std::vector<unsigned long long> halfmod();
    std::vector<unsigned long long> last_q_modinv();
*/
};

#endif // PRIMEPOOL_H

