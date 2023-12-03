// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "context.cuh"


__host__ Parameters::Parameters(std::string scheme_type, int poly_degree, PrimePool::security_level sec_level) {

    if (scheme_type != "BFV")
        throw("Invalid Scheme Type");

    scheme = scheme_type;
    n = poly_degree;
    n_power = int(log2l(n));
    sec = sec_level;

    PrimePool pool(n, sec);
    coeff_modulus = pool.prime_count();
    bsk_modulus = pool.base_Bsk().size();
    total_bits = pool.total_primes_bits();


    cudaMalloc(&modulus_, coeff_modulus * sizeof(Modulus));
    cudaMemcpy(modulus_, pool.base_modulus().data(), coeff_modulus * sizeof(Modulus), cudaMemcpyHostToDevice);

    cudaMalloc(&ntt_table_, coeff_modulus * n * sizeof(Root));
    cudaMemcpy(ntt_table_, pool.ntt_tables().data(), coeff_modulus * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&intt_table_, coeff_modulus * n * sizeof(Root));
    cudaMemcpy(intt_table_, pool.intt_tables().data(), coeff_modulus * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&n_inverse_, coeff_modulus * sizeof(Ninverse));
    cudaMemcpy(n_inverse_, pool.n_inverse().data(), coeff_modulus * sizeof(Ninverse), cudaMemcpyHostToDevice);

    cudaMalloc(&last_q_modinv_, (coeff_modulus - 1) * sizeof(Data));
    cudaMemcpy(last_q_modinv_, pool.last_q_modinv().data(), (coeff_modulus - 1) * sizeof(Data), cudaMemcpyHostToDevice);


    cudaMalloc(&base_Bsk_, pool.base_Bsk().size() * sizeof(Modulus));
    cudaMemcpy(base_Bsk_, pool.base_Bsk().data(), pool.base_Bsk().size() * sizeof(Modulus), cudaMemcpyHostToDevice);

    cudaMalloc(&bsk_ntt_tables_, pool.base_Bsk().size() * n * sizeof(Root));
    cudaMemcpy(bsk_ntt_tables_, pool.bsk_ntt_tables().data(), pool.base_Bsk().size() * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&bsk_intt_tables_, pool.base_Bsk().size() * n * sizeof(Root));
    cudaMemcpy(bsk_intt_tables_, pool.bsk_intt_tables().data(), pool.base_Bsk().size() * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&bsk_n_inverse_, pool.base_Bsk().size() * sizeof(Ninverse));
    cudaMemcpy(bsk_n_inverse_, pool.bsk_n_inverse().data(), pool.base_Bsk().size() * sizeof(Ninverse), cudaMemcpyHostToDevice);
    
    m_tilde_ = pool.m_tilde();

    cudaMalloc(&base_change_matrix_Bsk_, pool.base_change_matrix_Bsk().size() * sizeof(Data));
    cudaMemcpy(base_change_matrix_Bsk_, pool.base_change_matrix_Bsk().data(), pool.base_change_matrix_Bsk().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&inv_punctured_prod_mod_base_array_, pool.inv_punctured_prod_mod_base_array().size() * sizeof(Data));
    cudaMemcpy(inv_punctured_prod_mod_base_array_, pool.inv_punctured_prod_mod_base_array().data(), pool.inv_punctured_prod_mod_base_array().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&base_change_matrix_m_tilde_, pool.base_change_matrix_m_tilde().size() * sizeof(Data));
    cudaMemcpy(base_change_matrix_m_tilde_, pool.base_change_matrix_m_tilde().data(), pool.base_change_matrix_m_tilde().size() * sizeof(Data), cudaMemcpyHostToDevice);

    inv_prod_q_mod_m_tilde_ = pool.inv_prod_q_mod_m_tilde();

    cudaMalloc(&inv_m_tilde_mod_Bsk_, pool.inv_m_tilde_mod_Bsk().size() * sizeof(Data));
    cudaMemcpy(inv_m_tilde_mod_Bsk_, pool.inv_m_tilde_mod_Bsk().data(), pool.inv_m_tilde_mod_Bsk().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&prod_q_mod_Bsk_, pool.prod_q_mod_Bsk().size() * sizeof(Data));
    cudaMemcpy(prod_q_mod_Bsk_, pool.prod_q_mod_Bsk().data(), pool.prod_q_mod_Bsk().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&inv_prod_q_mod_Bsk_, pool.inv_prod_q_mod_Bsk().size() * sizeof(Data));
    cudaMemcpy(inv_prod_q_mod_Bsk_, pool.inv_prod_q_mod_Bsk().data(), pool.inv_prod_q_mod_Bsk().size() * sizeof(Data), cudaMemcpyHostToDevice);

    plain_modulus_ = pool.plain_modulus();

    cudaMalloc(&base_change_matrix_q_, pool.base_change_matrix_q().size() * sizeof(Data));
    cudaMemcpy(base_change_matrix_q_, pool.base_change_matrix_q().data(), pool.base_change_matrix_q().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&base_change_matrix_msk_, pool.base_change_matrix_msk().size() * sizeof(Data));
    cudaMemcpy(base_change_matrix_msk_, pool.base_change_matrix_msk().data(), pool.base_change_matrix_msk().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&inv_punctured_prod_mod_B_array_, pool.inv_punctured_prod_mod_B_array().size() * sizeof(Data));
    cudaMemcpy(inv_punctured_prod_mod_B_array_, pool.inv_punctured_prod_mod_B_array().data(), pool.inv_punctured_prod_mod_B_array().size() * sizeof(Data), cudaMemcpyHostToDevice);

    inv_prod_B_mod_m_sk_ = pool.inv_prod_B_mod_m_sk();

    cudaMalloc(&prod_B_mod_q_, pool.prod_B_mod_q().size() * sizeof(Data));
    cudaMemcpy(prod_B_mod_q_, pool.prod_B_mod_q().data(), pool.prod_B_mod_q().size() * sizeof(Data), cudaMemcpyHostToDevice);



    // For new
    cudaMalloc(&q_Bsk_merge_modulus_, pool.q_Bsk_merge_modulus().size() * sizeof(Modulus));
    cudaMemcpy(q_Bsk_merge_modulus_, pool.q_Bsk_merge_modulus().data(), pool.q_Bsk_merge_modulus().size() * sizeof(Modulus), cudaMemcpyHostToDevice);
    
    cudaMalloc(&q_Bsk_merge_ntt_tables_, pool.q_Bsk_merge_modulus().size() * n * sizeof(Root));
    cudaMemcpy(q_Bsk_merge_ntt_tables_, pool.q_Bsk_merge_ntt_tables().data(), pool.q_Bsk_merge_modulus().size() * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&q_Bsk_merge_intt_tables_, pool.q_Bsk_merge_modulus().size() * n * sizeof(Root));
    cudaMemcpy(q_Bsk_merge_intt_tables_, pool.q_Bsk_merge_intt_tables().data(), pool.q_Bsk_merge_modulus().size() * n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&q_Bsk_n_inverse_, pool.q_Bsk_merge_modulus().size() * sizeof(Ninverse));
    cudaMemcpy(q_Bsk_n_inverse_, pool.q_Bsk_n_inverse().data(), pool.q_Bsk_merge_modulus().size() * sizeof(Ninverse), cudaMemcpyHostToDevice);


    half_ = pool.half();

    cudaMalloc(&half_mod_, pool.half_mod().size() * sizeof(Data));
    cudaMemcpy(half_mod_, pool.half_mod().data(), pool.half_mod().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&factor_, pool.factor().size() * sizeof(Data));
    cudaMemcpy(factor_, pool.factor().data(), pool.factor().size() * sizeof(Data), cudaMemcpyHostToDevice);


    cudaMalloc(&plain_modulus2_, 1 * sizeof(Modulus));
    cudaMemcpy(plain_modulus2_, pool.plain_modulus2().data(), 1 * sizeof(Modulus), cudaMemcpyHostToDevice);

    cudaMalloc(&n_plain_inverse_, 1 * sizeof(Ninverse));
    cudaMemcpy(n_plain_inverse_, pool.n_plain_inverse().data(), 1 * sizeof(Ninverse), cudaMemcpyHostToDevice);
    
    cudaMalloc(&plain_ntt_tables_, n * sizeof(Root));
    cudaMemcpy(plain_ntt_tables_, pool.plain_ntt_tables().data(), n * sizeof(Root), cudaMemcpyHostToDevice);

    cudaMalloc(&plain_intt_tables_, n * sizeof(Root));
    cudaMemcpy(plain_intt_tables_, pool.plain_intt_tables().data(), n * sizeof(Root), cudaMemcpyHostToDevice);



    gamma_ = pool.gamma();

    cudaMalloc(&coeeff_div_plainmod_, pool.coeeff_div_plainmod().size() * sizeof(Data));
    cudaMemcpy(coeeff_div_plainmod_, pool.coeeff_div_plainmod().data(), pool.coeeff_div_plainmod().size() * sizeof(Data), cudaMemcpyHostToDevice);

    Q_mod_t_ = pool.Q_mod_t();

    upper_threshold_ = pool.upper_threshold();

    cudaMalloc(&upper_halfincrement_, pool.upper_halfincrement().size() * sizeof(Data));
    cudaMemcpy(upper_halfincrement_, pool.upper_halfincrement().data(), pool.upper_halfincrement().size() * sizeof(Data), cudaMemcpyHostToDevice);



    cudaMalloc(&Qi_t_, pool.Qi_t().size() * sizeof(Data));
    cudaMemcpy(Qi_t_, pool.Qi_t().data(), pool.Qi_t().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&Qi_gamma_, pool.Qi_gamma().size() * sizeof(Data));
    cudaMemcpy(Qi_gamma_, pool.Qi_gamma().data(), pool.Qi_gamma().size() * sizeof(Data), cudaMemcpyHostToDevice);

    cudaMalloc(&Qi_inverse_, pool.Qi_inverse().size() * sizeof(Data));
    cudaMemcpy(Qi_inverse_, pool.Qi_inverse().data(), pool.Qi_inverse().size() * sizeof(Data), cudaMemcpyHostToDevice);

    mulq_inv_t_ = pool.mulq_inv_t();
    mulq_inv_gamma_ = pool.mulq_inv_gamma();
    inv_gamma_ = pool.inv_gamma();


    /////////////////////////////////////////////////////////////////////////////
    
    // Encode -Decode Index
    std::vector<Data> encode_index;

    int m = n << 1;
    int gen = 3;
    int pos = 1;
    int index = 0;
    int location = 0;
    for (int i = 0; i < int(n / 2); i++) {

        index = (pos - 1) >> 1;
        location = bitreverse(index, n_power);
        //encoding_location_[i] = location;
        encode_index.push_back(location);
        pos *= gen;
        pos &= (m - 1);

    }
    for (int i = int(n / 2); i < n; i++) {

        index = (m - pos - 1) >> 1;
        location = bitreverse(index, n_power);
        //encoding_location_[i] = location;
        encode_index.push_back(location);
        pos *= gen;
        pos &= (m - 1);

    }

    cudaMalloc(&encoding_location_, n * sizeof(Data));
    cudaMemcpy(encoding_location_, encode_index.data(), n * sizeof(Data), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////////////////////////


    cudaMalloc(&temp1_enc, 3 * n * coeff_modulus * sizeof(Data));
    cudaMalloc(&temp2_enc, 2 * n * coeff_modulus * sizeof(Data));


    cudaMalloc(&temp1_mul, 4 * n * (bsk_modulus+(coeff_modulus - 1)) * sizeof(Data)); 
    cudaMalloc(&temp2_mul, 3 * n * (bsk_modulus+(coeff_modulus - 1)) * sizeof(Data)); 
    cudaMalloc(&temp3_mul, 3 * n * (coeff_modulus - 1) * sizeof(Data));


    cudaMalloc(&temp1_relin, n * (coeff_modulus - 1) * coeff_modulus * sizeof(Data));
    cudaMalloc(&temp2_relin, 2 * n * coeff_modulus * sizeof(Data));


    cudaMalloc(&temp0_rotation, 2 * n * (coeff_modulus - 1) * sizeof(Data));
    cudaMalloc(&temp1_rotation, n * (coeff_modulus - 1) * coeff_modulus * sizeof(Data));
    cudaMalloc(&temp2_rotation, 2 * n * coeff_modulus * sizeof(Data));


    cudaMalloc(&temp1_plain_mul, n * (coeff_modulus - 1) * sizeof(Data));


}

///////////////////////////////////////////////////////////////////////////



__host__ HEStream::HEStream(Parameters context){

    ring_size = context.n;
    coeff_modulus_count = context.coeff_modulus;
    bsk_modulus_count = context.bsk_modulus;

    cudaStreamCreate(&stream); 

    cudaMallocAsync(&temp1_enc, 3 * ring_size * coeff_modulus_count * sizeof(Data), stream);
    cudaMallocAsync(&temp2_enc, 2 * ring_size * coeff_modulus_count * sizeof(Data), stream);

    cudaMallocAsync(&temp1_mul, 4 * ring_size * (bsk_modulus_count+(coeff_modulus_count - 1)) * sizeof(Data), stream); 
    cudaMallocAsync(&temp2_mul, 3 * ring_size * (bsk_modulus_count+(coeff_modulus_count - 1)) * sizeof(Data), stream); 
    //cudaMallocAsync(&temp3_mul, 3 * ring_size * (coeff_modulus_count - 1) * sizeof(Data), stream);

    cudaMallocAsync(&temp1_relin, ring_size * (coeff_modulus_count - 1) * coeff_modulus_count * sizeof(Data), stream);
    cudaMallocAsync(&temp2_relin, 2 * ring_size * coeff_modulus_count * sizeof(Data), stream);

    cudaMallocAsync(&temp0_rotation, 2 * ring_size * (coeff_modulus_count - 1) * sizeof(Data), stream);
    cudaMallocAsync(&temp1_rotation, ring_size * (coeff_modulus_count - 1) * coeff_modulus_count * sizeof(Data), stream);
    cudaMallocAsync(&temp2_rotation, 2 * ring_size * coeff_modulus_count * sizeof(Data), stream);

    cudaMallocAsync(&temp1_plain_mul, ring_size * (coeff_modulus_count - 1) * sizeof(Data), stream);

}

__host__ void HEStream::kill(){

    cudaStreamDestroy(stream);

    cudaFree(temp1_enc);
    cudaFree(temp2_enc);

    cudaFree(temp1_mul);
    cudaFree(temp2_mul);
    cudaFree(temp3_mul);

    cudaFree(temp1_relin);
    cudaFree(temp2_relin);

    cudaFree(temp0_rotation);
    cudaFree(temp1_rotation);
    cudaFree(temp2_rotation);

    cudaFree(temp1_plain_mul);

}

///////////////////////////////////////////////////////////////////////////
 
__host__ Ciphertext::Ciphertext()
{

    ring_size = 0;
    coeff_modulus_count = 0;
    cipher_size = 0;

}

__host__ Ciphertext::Ciphertext(Parameters context)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    cipher_size = 3; //default
    ring_size = context.n; // n

    cudaMalloc(&location, cipher_size * coeff_modulus_count * ring_size * sizeof(Data));

}

__host__ Ciphertext::Ciphertext(Parameters context, HEStream stream)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    cipher_size = 3; //default
    ring_size = context.n; // n

    cudaMallocAsync(&location, cipher_size * coeff_modulus_count * ring_size * sizeof(Data), stream.stream);

}

__host__ Ciphertext::Ciphertext(Data* cipher, Parameters context)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    cipher_size = 3; //default
    ring_size = context.n; // n

    cudaMalloc(&location, cipher_size * coeff_modulus_count * ring_size * sizeof(Data));
    cudaMemcpy(location, cipher, cipher_size * coeff_modulus_count * ring_size * sizeof(Data), cudaMemcpyHostToDevice);

}

__host__ Ciphertext::Ciphertext(Data* cipher, Parameters context, HEStream stream)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    cipher_size = 3; //default
    ring_size = context.n; // n

    cudaMallocAsync(&location, cipher_size * coeff_modulus_count * ring_size * sizeof(Data), stream.stream);
    cudaMemcpyAsync(location, cipher, cipher_size * coeff_modulus_count * ring_size * sizeof(Data), cudaMemcpyHostToDevice, stream.stream);

}

__host__ void Ciphertext::kill()
{
    cudaFree(location);
}

///////////////////////////////////////////////////////////////////////////

__host__ Message::Message()
{

    ring_size = 0;

}

__host__ Message::Message(Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));

}


__host__ Message::Message(Parameters context, HEStream stream){

    ring_size = context.n; // n

    cudaMallocAsync(&location, ring_size * sizeof(Data), stream.stream);

}

__host__ Message::Message(Data* message, Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));
    cudaMemcpy(location, message, ring_size * sizeof(Data), cudaMemcpyHostToDevice);

}

__host__ Message::Message(const std::vector<uint64_t> &message, Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));
    cudaMemcpy(location, message.data(), ring_size * sizeof(Data), cudaMemcpyHostToDevice);

}

 __host__ Message::Message(Data* message, Parameters context, HEStream stream){

    ring_size = context.n; // n

    cudaMallocAsync(&location, ring_size * sizeof(Data), stream.stream);
    cudaMemcpyAsync(location, message, ring_size * sizeof(Data), cudaMemcpyHostToDevice, stream.stream);

 }

__host__ Message::Message(Data* message, int size, Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));
    cudaMemcpy(location, message, size * sizeof(Data), cudaMemcpyHostToDevice);

}

__host__ Message::Message(const std::vector<uint64_t> &message, int size, Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));
    cudaMemcpy(location, message.data(), size * sizeof(Data), cudaMemcpyHostToDevice);

}

__host__ Message::Message(Data* message, int size, Parameters context, HEStream stream){

    ring_size = context.n; // n

    cudaMallocAsync(&location, ring_size * sizeof(Data), stream.stream);
    cudaMemcpyAsync(location, message, size * sizeof(Data), cudaMemcpyHostToDevice, stream.stream);

}

__host__ void Message::kill()
{
    cudaFree(location);
}

///////////////////////////////////////////////////////////////////////////

__host__ Plaintext::Plaintext()
{

    ring_size = 0;

}

__host__ Plaintext::Plaintext(Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));

}

__host__ Plaintext::Plaintext(Parameters context, HEStream stream){

    ring_size = context.n; // n

    cudaMallocAsync(&location, ring_size * sizeof(Data), stream.stream);

}

__host__ Plaintext::Plaintext(Message message, Parameters context)
{

    ring_size = context.n; // n

    cudaMalloc(&location, ring_size * sizeof(Data));
    cudaMemcpy(location, message.location, ring_size * sizeof(Data), cudaMemcpyHostToDevice);

}

__host__ Plaintext::Plaintext(Message message, Parameters context, HEStream stream){

    ring_size = context.n; // n

    cudaMallocAsync(&location, ring_size * sizeof(Data), stream.stream);
    cudaMemcpyAsync(location, message.location, ring_size * sizeof(Data), cudaMemcpyHostToDevice, stream.stream);

}

__host__ void Plaintext::kill()
{
    cudaFree(location);
}




///////////////////////////////////////////////////////////////////////////

__host__ Relinkey::Relinkey()
{

    ring_size = 0;
    coeff_modulus_count = 0;

}

__host__ Relinkey::Relinkey(Parameters context)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    ring_size = context.n; // n

    cudaMalloc(&location, 2 * coeff_modulus_count * (coeff_modulus_count + 1) * ring_size * sizeof(Data));

    cudaMalloc(&e_a, 2 * (coeff_modulus_count + 1) * ring_size * sizeof(Data));
    
}

__host__ void Relinkey::kill()
{
    cudaFree(location);
    cudaFree(e_a);
}

///////////////////////////////////////////////////////////////////////////

__host__ Galoiskey::Galoiskey()
{

    ring_size = 0;
    coeff_modulus_count = 0;

}

__host__ Galoiskey::Galoiskey(Parameters context)
{

    coeff_modulus_count = context.coeff_modulus - 1; 
    ring_size = context.n; // n

    galois_elt_pos = (int*)malloc(MAX_SHIFT * sizeof(int));
    galois_elt_neg = (int*)malloc(MAX_SHIFT * sizeof(int));

    for(int i = 0; i < MAX_SHIFT; i++){
        cudaMalloc(&positive_location[i], 2 * coeff_modulus_count * (coeff_modulus_count + 1) * ring_size * sizeof(Data));
        cudaMalloc(&negative_location[i], 2 * coeff_modulus_count * (coeff_modulus_count + 1) * ring_size * sizeof(Data));
    }
    
    cudaMalloc(&e_a, 2 * (coeff_modulus_count + 1) * ring_size * sizeof(Data));
    
}

__host__ void Galoiskey::kill()
{   
    for(int i = 0; i < MAX_SHIFT; i++){
        cudaFree(positive_location[i]);
        cudaFree(negative_location[i]);
    }

    free(galois_elt_pos);
    free(galois_elt_neg);

    cudaFree(e_a);
}


///////////////////////////////////////////////////////////////////////////

__host__ Secretkey::Secretkey()
{

    ring_size = 0;
    coeff_modulus_count = 0;

}

__host__ Secretkey::Secretkey(Parameters context)
{

    coeff_modulus_count = context.coeff_modulus;
    ring_size = context.n; // n

    cudaMalloc(&location, coeff_modulus_count * ring_size * sizeof(Data));

}

__host__ void Secretkey::kill()
{
    cudaFree(location);
}

///////////////////////////////////////////////////////////////////////////

__host__ Publickey::Publickey()
{

    ring_size = 0;
    coeff_modulus_count = 0;

}

__host__ Publickey::Publickey(Parameters context)
{

    coeff_modulus_count = context.coeff_modulus;
    ring_size = context.n; // n

    cudaMalloc(&location, 2 * coeff_modulus_count * ring_size * sizeof(Data));

}

__host__ void Publickey::kill()
{
    cudaFree(location);
}