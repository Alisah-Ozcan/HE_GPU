// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "encoder.cuh"

__host__ HEEncoder::HEEncoder(Parameters &context){

    n = context.n;
    n_power = context.n_power;

    plain_modulus_ = context.plain_modulus2_;

    plain_ntt_tables_ = context.plain_ntt_tables_;
    plain_intt_tables_ = context.plain_intt_tables_;

    n_plain_inverse_ = context.n_plain_inverse_;

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
        encode_index.push_back(location);
        pos *= gen;
        pos &= (m - 1);

    }
    for (int i = int(n / 2); i < n; i++) {

        index = (m - pos - 1) >> 1;
        location = bitreverse(index, n_power);
        encode_index.push_back(location);
        pos *= gen;
        pos &= (m - 1);

    }

    cudaMalloc(&encoding_location_, n * sizeof(Data));
    cudaMemcpy(encoding_location_, encode_index.data(), n * sizeof(Data), cudaMemcpyHostToDevice);

    //encode_index.clear();

}

///////////////////////////////////////////////////

__host__ void HEEncoder::encode(Plaintext &plain, const Message message){

    encode_kernel<< < dim3((n >> 8), 1, 1), 256 >> >(plain.location, message.location, encoding_location_);

    ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_plain_inverse_,
        .stream = 0};

    GPU_NTT_Inplace(plain.location, plain_intt_tables_ , plain_modulus_, cfg_intt, 1, 1);

}


__host__ void HEEncoder::encode(Plaintext &plain, const Message message, HEStream stream){

    encode_kernel<< < dim3((n >> 8), 1, 1), 256, 0, stream.stream >> >(plain.location, message.location, encoding_location_);

     ntt_configuration cfg_intt = {
        .n_power = n_power,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = n_plain_inverse_,
        .stream = stream.stream};

    GPU_NTT_Inplace(plain.location, plain_intt_tables_ , plain_modulus_, cfg_intt, 1, 1);

}

///////////////////////////////////////////////////

__host__ void HEEncoder::decode(Message &message, const Plaintext plain){

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = 0};

    GPU_NTT_Inplace(plain.location, plain_ntt_tables_ , plain_modulus_, cfg_ntt, 1, 1); 

    decode_kernel<< < dim3((n >> 8), 1, 1), 256 >> >(message.location, plain.location, encoding_location_);

}

__host__ void HEEncoder::decode(Message &message, const Plaintext plain, HEStream stream){

    ntt_configuration cfg_ntt = {
        .n_power = n_power,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream.stream};

    GPU_NTT_Inplace(plain.location, plain_ntt_tables_ , plain_modulus_, cfg_ntt, 1, 1); 

    decode_kernel<< < dim3((n >> 8), 1, 1), 256, 0, stream.stream >> >(message.location, plain.location, encoding_location_);

}

