#include <cstdlib>  // For atoi or atof functions
#include <random>

#include "../src/keygeneration.cuh"
#include "../src/encoder.cuh"
#include "../src/encryptor.cuh"
#include "../src/decryptor.cuh"
#include "../src/operator.cuh"
#include <gtest/gtest.h>
#include "seal/seal.h"  // Include the SEAL library headers
#include <fstream>

#define DEFAULT_MODULUS

using namespace std;

TEST(HEonGPU, Test_Encoder_HEEncryptor_HEDecryptor) {

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Microsoft SEAL 

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));

    seal::SEALContext cpu_context(parms);

    seal::KeyGenerator keygen(cpu_context);
    seal::SecretKey cpu_secret_key = keygen.secret_key();
    seal::PublicKey cpu_public_key;
    keygen.create_public_key(cpu_public_key);

    seal::Encryptor encryptor(cpu_context, cpu_public_key);
    seal::Evaluator evaluator(cpu_context);
    seal::Decryptor decryptor(cpu_context, cpu_secret_key);
    seal::BatchEncoder cpu_batch_encoder(cpu_context);
    
    size_t slot_count = cpu_batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    pod_matrix[0] = 0ULL;
    pod_matrix[1] = 12ULL;
    pod_matrix[2] = 23ULL;
    pod_matrix[3] = 31ULL;
    pod_matrix[row_size] = 41ULL;
    pod_matrix[row_size + 1] = 54ULL;
    pod_matrix[row_size + 2] = 6ULL;
    pod_matrix[row_size + 3] = 100ULL;


    seal::Plaintext plain_matrix;
    cpu_batch_encoder.encode(pod_matrix, plain_matrix);

    seal::Ciphertext encrypted_matrix;
    encryptor.encrypt(plain_matrix, encrypted_matrix);

    seal::Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix, plain_result);

    vector<uint64_t> pod_result;
    cpu_batch_encoder.decode(plain_result, pod_result);

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // HEonGPU 

    Parameters gpu_context("BFV", poly_modulus_degree, PrimePool::security_level::HES_128);

    Secretkey gpu_secret_key(gpu_context);
    Publickey gpu_public_key(gpu_context);
    Relinkey gpu_relin_key(gpu_context);
    Galoiskey gpu_galois_key(gpu_context);

    HESecretkeygen(gpu_secret_key, gpu_context);
    HEPublickeygen(gpu_public_key, gpu_secret_key, gpu_context);
    HERelinkeygen(gpu_relin_key, gpu_secret_key, gpu_context);
    HEGaloiskeygen(gpu_galois_key, gpu_secret_key, gpu_context);

    Message message1(pod_matrix, gpu_context);
    Plaintext plaintext1(gpu_context);

    HEEncoder gpu_encoder(gpu_context);
    gpu_encoder.encode(plaintext1, message1);


    Ciphertext ciphertext1(gpu_context);
    HEEncryptor gpu_encryptor(gpu_context, gpu_public_key);
    gpu_encryptor.encrypt(ciphertext1, plaintext1);

    Plaintext plaintext2(gpu_context);
    HEDecryptor gpu_decryptor(gpu_context, gpu_secret_key);
    gpu_decryptor.decrypt(plaintext2, ciphertext1);

    
    Message message_result(gpu_context);
    gpu_encoder.decode(message_result, plaintext2);

    vector<Data> gpu_result(slot_count);
    cudaMemcpy(gpu_result.data(), message_result.location, slot_count * sizeof(Data), cudaMemcpyDeviceToHost);  


    EXPECT_EQ(std::equal(pod_result.begin(), pod_result.end(), gpu_result.begin()), true);

}



TEST(HEonGPU, Test_HEAddition) {

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Microsoft SEAL 

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));

    seal::SEALContext cpu_context(parms);

    seal::KeyGenerator keygen(cpu_context);
    seal::SecretKey cpu_secret_key = keygen.secret_key();
    seal::PublicKey cpu_public_key;
    keygen.create_public_key(cpu_public_key);

    seal::Encryptor encryptor(cpu_context, cpu_public_key);
    seal::Evaluator evaluator(cpu_context);
    seal::Decryptor decryptor(cpu_context, cpu_secret_key);
    seal::BatchEncoder cpu_batch_encoder(cpu_context);
    
    size_t slot_count = cpu_batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    vector<uint64_t> pod_matrix1(slot_count, 0ULL);
    pod_matrix1[0] = 0ULL;
    pod_matrix1[1] = 12ULL;
    pod_matrix1[2] = 23ULL;
    pod_matrix1[3] = 31ULL;
    pod_matrix1[row_size] = 41ULL;
    pod_matrix1[row_size + 1] = 54ULL;
    pod_matrix1[row_size + 2] = 6ULL;
    pod_matrix1[row_size + 3] = 100ULL;

    vector<uint64_t> pod_matrix2(slot_count, 1ULL);

    seal::Plaintext plain_matrix1, plain_matrix2;
    cpu_batch_encoder.encode(pod_matrix1, plain_matrix1);
    cpu_batch_encoder.encode(pod_matrix2, plain_matrix2);

    seal::Ciphertext encrypted_matrix1, encrypted_matrix2;
    encryptor.encrypt(plain_matrix1, encrypted_matrix1);
    encryptor.encrypt(plain_matrix2, encrypted_matrix2);

    evaluator.add_inplace(encrypted_matrix1, encrypted_matrix2);

    seal::Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix1, plain_result);

    vector<uint64_t> pod_result;
    cpu_batch_encoder.decode(plain_result, pod_result);

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // HEonGPU 

    Parameters gpu_context("BFV", poly_modulus_degree, PrimePool::security_level::HES_128);

    Secretkey gpu_secret_key(gpu_context);
    Publickey gpu_public_key(gpu_context);
    Relinkey gpu_relin_key(gpu_context);
    Galoiskey gpu_galois_key(gpu_context);

    HESecretkeygen(gpu_secret_key, gpu_context);
    HEPublickeygen(gpu_public_key, gpu_secret_key, gpu_context);
    HERelinkeygen(gpu_relin_key, gpu_secret_key, gpu_context);
    HEGaloiskeygen(gpu_galois_key, gpu_secret_key, gpu_context);

    Message message1(pod_matrix1, gpu_context);
    Message message2(pod_matrix2, gpu_context);
    Plaintext plaintext1(gpu_context);
    Plaintext plaintext2(gpu_context);

    HEEncoder gpu_encoder(gpu_context);
    gpu_encoder.encode(plaintext1, message1);
    gpu_encoder.encode(plaintext2, message2);


    Ciphertext ciphertext1(gpu_context);
    Ciphertext ciphertext2(gpu_context);
    HEEncryptor gpu_encryptor(gpu_context, gpu_public_key);
    gpu_encryptor.encrypt(ciphertext1, plaintext1);
    gpu_encryptor.encrypt(ciphertext2, plaintext2);

    HEOperator operators(gpu_context);    
    operators.add_inplace(ciphertext1, ciphertext2);

    Plaintext plaintext3(gpu_context);
    HEDecryptor gpu_decryptor(gpu_context, gpu_secret_key);
    gpu_decryptor.decrypt(plaintext3, ciphertext1);

    
    Message message_result(gpu_context);
    gpu_encoder.decode(message_result, plaintext3);

    vector<Data> gpu_result(slot_count);
    cudaMemcpy(gpu_result.data(), message_result.location, slot_count * sizeof(Data), cudaMemcpyDeviceToHost);  

    EXPECT_EQ(std::equal(pod_result.begin(), pod_result.end(), gpu_result.begin()), true);

}

TEST(HEonGPU, Test_HEMultiplication_HERelinearization) {

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Microsoft SEAL 

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));

    seal::SEALContext cpu_context(parms);

    seal::KeyGenerator keygen(cpu_context);
    seal::SecretKey cpu_secret_key = keygen.secret_key();
    seal::PublicKey cpu_public_key;
    keygen.create_public_key(cpu_public_key);

    seal::RelinKeys cpu_relin_keys;
    keygen.create_relin_keys(cpu_relin_keys);

    seal::Encryptor encryptor(cpu_context, cpu_public_key);
    seal::Evaluator evaluator(cpu_context);
    seal::Decryptor decryptor(cpu_context, cpu_secret_key);
    seal::BatchEncoder cpu_batch_encoder(cpu_context);
    
    size_t slot_count = cpu_batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    vector<uint64_t> pod_matrix1(slot_count, 0ULL);
    pod_matrix1[0] = 0ULL;
    pod_matrix1[1] = 12ULL;
    pod_matrix1[2] = 23ULL;
    pod_matrix1[3] = 31ULL;
    pod_matrix1[row_size] = 41ULL;
    pod_matrix1[row_size + 1] = 54ULL;
    pod_matrix1[row_size + 2] = 6ULL;
    pod_matrix1[row_size + 3] = 100ULL;

    vector<uint64_t> pod_matrix2(slot_count, 1ULL);

    seal::Plaintext plain_matrix1, plain_matrix2;
    cpu_batch_encoder.encode(pod_matrix1, plain_matrix1);
    cpu_batch_encoder.encode(pod_matrix2, plain_matrix2);

    seal::Ciphertext encrypted_matrix1, encrypted_matrix2;
    encryptor.encrypt(plain_matrix1, encrypted_matrix1);
    encryptor.encrypt(plain_matrix2, encrypted_matrix2);

    evaluator.multiply_inplace(encrypted_matrix1, encrypted_matrix2);
    evaluator.relinearize_inplace(encrypted_matrix1, cpu_relin_keys);

    seal::Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix1, plain_result);

    vector<uint64_t> pod_result;
    cpu_batch_encoder.decode(plain_result, pod_result);

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // HEonGPU 

    Parameters gpu_context("BFV", poly_modulus_degree, PrimePool::security_level::HES_128);

    Secretkey gpu_secret_key(gpu_context);
    Publickey gpu_public_key(gpu_context);
    Relinkey gpu_relin_key(gpu_context);
    Galoiskey gpu_galois_key(gpu_context);

    HESecretkeygen(gpu_secret_key, gpu_context);
    HEPublickeygen(gpu_public_key, gpu_secret_key, gpu_context);
    HERelinkeygen(gpu_relin_key, gpu_secret_key, gpu_context);
    HEGaloiskeygen(gpu_galois_key, gpu_secret_key, gpu_context);

    Message message1(pod_matrix1, gpu_context);
    Message message2(pod_matrix2, gpu_context);
    Plaintext plaintext1(gpu_context);
    Plaintext plaintext2(gpu_context);

    HEEncoder gpu_encoder(gpu_context);
    gpu_encoder.encode(plaintext1, message1);
    gpu_encoder.encode(plaintext2, message2);


    Ciphertext ciphertext1(gpu_context);
    Ciphertext ciphertext2(gpu_context);
    HEEncryptor gpu_encryptor(gpu_context, gpu_public_key);
    gpu_encryptor.encrypt(ciphertext1, plaintext1);
    gpu_encryptor.encrypt(ciphertext2, plaintext2);

    HEOperator operators(gpu_context);    
    operators.multiply_inplace(ciphertext1, ciphertext2);
    operators.relinearize_inplace(ciphertext1, gpu_relin_key);

    Plaintext plaintext3(gpu_context);
    HEDecryptor gpu_decryptor(gpu_context, gpu_secret_key);
    gpu_decryptor.decrypt(plaintext3, ciphertext1);

    
    Message message_result(gpu_context);
    gpu_encoder.decode(message_result, plaintext3);

    vector<Data> gpu_result(slot_count);
    cudaMemcpy(gpu_result.data(), message_result.location, slot_count * sizeof(Data), cudaMemcpyDeviceToHost);  

    EXPECT_EQ(std::equal(pod_result.begin(), pod_result.end(), gpu_result.begin()), true);

}

TEST(HEonGPU, Test_HERotation) {

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Microsoft SEAL 

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));

    seal::SEALContext cpu_context(parms);

    seal::KeyGenerator keygen(cpu_context);
    seal::SecretKey cpu_secret_key = keygen.secret_key();
    seal::PublicKey cpu_public_key;
    keygen.create_public_key(cpu_public_key);

    seal::GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    seal::Encryptor encryptor(cpu_context, cpu_public_key);
    seal::Evaluator evaluator(cpu_context);
    seal::Decryptor decryptor(cpu_context, cpu_secret_key);
    seal::BatchEncoder cpu_batch_encoder(cpu_context);
    
    size_t slot_count = cpu_batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    vector<uint64_t> pod_matrix1(slot_count, 0ULL);
    pod_matrix1[0] = 0ULL;
    pod_matrix1[1] = 12ULL;
    pod_matrix1[2] = 23ULL;
    pod_matrix1[3] = 31ULL;
    pod_matrix1[row_size] = 41ULL;
    pod_matrix1[row_size + 1] = 54ULL;
    pod_matrix1[row_size + 2] = 6ULL;
    pod_matrix1[row_size + 3] = 100ULL;

    seal::Plaintext plain_matrix1;
    cpu_batch_encoder.encode(pod_matrix1, plain_matrix1);

    seal::Ciphertext encrypted_matrix1;
    encryptor.encrypt(plain_matrix1, encrypted_matrix1);

    evaluator.rotate_rows_inplace(encrypted_matrix1, 3, galois_keys);

    seal::Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix1, plain_result);

    vector<uint64_t> pod_result;
    cpu_batch_encoder.decode(plain_result, pod_result);

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // HEonGPU 

    Parameters gpu_context("BFV", poly_modulus_degree, PrimePool::security_level::HES_128);

    Secretkey gpu_secret_key(gpu_context);
    Publickey gpu_public_key(gpu_context);
    Relinkey gpu_relin_key(gpu_context);
    Galoiskey gpu_galois_key(gpu_context);

    HESecretkeygen(gpu_secret_key, gpu_context);
    HEPublickeygen(gpu_public_key, gpu_secret_key, gpu_context);
    HERelinkeygen(gpu_relin_key, gpu_secret_key, gpu_context);
    HEGaloiskeygen(gpu_galois_key, gpu_secret_key, gpu_context);

    Message message1(pod_matrix1, gpu_context);
    Plaintext plaintext1(gpu_context);

    HEEncoder gpu_encoder(gpu_context);
    gpu_encoder.encode(plaintext1, message1);

    Ciphertext ciphertext1(gpu_context);
    HEEncryptor gpu_encryptor(gpu_context, gpu_public_key);
    gpu_encryptor.encrypt(ciphertext1, plaintext1);

    HEOperator operators(gpu_context);    
    operators.rotate(ciphertext1, ciphertext1, gpu_galois_key, 3);

    Plaintext plaintext3(gpu_context);
    HEDecryptor gpu_decryptor(gpu_context, gpu_secret_key);
    gpu_decryptor.decrypt(plaintext3, ciphertext1);

    Message message_result(gpu_context);
    gpu_encoder.decode(message_result, plaintext3);

    vector<Data> gpu_result(slot_count);
    cudaMemcpy(gpu_result.data(), message_result.location, slot_count * sizeof(Data), cudaMemcpyDeviceToHost);  

    EXPECT_EQ(std::equal(pod_result.begin(), pod_result.end(), gpu_result.begin()), true);

}


TEST(HEonGPU, Test_HEMultiplyplain) {

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Microsoft SEAL 

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));

    seal::SEALContext cpu_context(parms);

    seal::KeyGenerator keygen(cpu_context);
    seal::SecretKey cpu_secret_key = keygen.secret_key();
    seal::PublicKey cpu_public_key;
    keygen.create_public_key(cpu_public_key);

    seal::RelinKeys cpu_relin_keys;
    keygen.create_relin_keys(cpu_relin_keys);

    seal::Encryptor encryptor(cpu_context, cpu_public_key);
    seal::Evaluator evaluator(cpu_context);
    seal::Decryptor decryptor(cpu_context, cpu_secret_key);
    seal::BatchEncoder cpu_batch_encoder(cpu_context);
    
    size_t slot_count = cpu_batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    vector<uint64_t> pod_matrix1(slot_count, 0ULL);
    pod_matrix1[0] = 0ULL;
    pod_matrix1[1] = 12ULL;
    pod_matrix1[2] = 23ULL;
    pod_matrix1[3] = 31ULL;
    pod_matrix1[row_size] = 41ULL;
    pod_matrix1[row_size + 1] = 54ULL;
    pod_matrix1[row_size + 2] = 6ULL;
    pod_matrix1[row_size + 3] = 100ULL;

    vector<uint64_t> pod_matrix2(slot_count, 1ULL);

    seal::Plaintext plain_matrix1, plain_matrix2;
    cpu_batch_encoder.encode(pod_matrix1, plain_matrix1);
    cpu_batch_encoder.encode(pod_matrix2, plain_matrix2);

    seal::Ciphertext encrypted_matrix1, encrypted_matrix2;
    encryptor.encrypt(plain_matrix1, encrypted_matrix1);

    evaluator.multiply_plain_inplace(encrypted_matrix1, plain_matrix2);

    seal::Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix1, plain_result);

    vector<uint64_t> pod_result;
    cpu_batch_encoder.decode(plain_result, pod_result);

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // HEonGPU 

    Parameters gpu_context("BFV", poly_modulus_degree, PrimePool::security_level::HES_128);

    Secretkey gpu_secret_key(gpu_context);
    Publickey gpu_public_key(gpu_context);
    Relinkey gpu_relin_key(gpu_context);
    Galoiskey gpu_galois_key(gpu_context);

    HESecretkeygen(gpu_secret_key, gpu_context);
    HEPublickeygen(gpu_public_key, gpu_secret_key, gpu_context);
    HERelinkeygen(gpu_relin_key, gpu_secret_key, gpu_context);
    HEGaloiskeygen(gpu_galois_key, gpu_secret_key, gpu_context);

    Message message1(pod_matrix1, gpu_context);
    Message message2(pod_matrix2, gpu_context);
    Plaintext plaintext1(gpu_context);
    Plaintext plaintext2(gpu_context);

    HEEncoder gpu_encoder(gpu_context);
    gpu_encoder.encode(plaintext1, message1);
    gpu_encoder.encode(plaintext2, message2);

    Ciphertext ciphertext1(gpu_context);
    HEEncryptor gpu_encryptor(gpu_context, gpu_public_key);
    gpu_encryptor.encrypt(ciphertext1, plaintext1);

    HEOperator operators(gpu_context);    
    operators.multiply_plain_inplace(ciphertext1, plaintext2);

    Plaintext plaintext3(gpu_context);
    HEDecryptor gpu_decryptor(gpu_context, gpu_secret_key);
    gpu_decryptor.decrypt(plaintext3, ciphertext1);
    
    Message message_result(gpu_context);
    gpu_encoder.decode(message_result, plaintext3);

    vector<Data> gpu_result(slot_count);
    cudaMemcpy(gpu_result.data(), message_result.location, slot_count * sizeof(Data), cudaMemcpyDeviceToHost);  

    EXPECT_EQ(std::equal(pod_result.begin(), pod_result.end(), gpu_result.begin()), true);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
// tests

cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 

cmake --build ./cmake-build/ --target HE_test1 --parallel
./cmake-build/HE_test1

*/

