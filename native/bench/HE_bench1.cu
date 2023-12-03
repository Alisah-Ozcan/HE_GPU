#include <cstdlib>  // For atoi or atof functions
#include <random>

#include "../src/keygeneration.cuh"
#include "../src/encoder.cuh"
#include "../src/encryptor.cuh"
#include "../src/decryptor.cuh"
#include "../src/operator.cuh"

#include <fstream>

#define DEFAULT_MODULUS

using namespace std;


/*
// tests

cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 

cmake --build ./cmake-build/ --target HE_bench1 --parallel
./cmake-build/HE_bench1 12 1

*/



int main(int argc, char* argv[])
{
    Parameters contxt("BFV", 8192, PrimePool::security_level::HES_128);

    cout << "/ --------------------------------------------------------- /" << endl;
    cout << "|                 ~ WELCOME TO BFV GPU LIBRARY ~                 " << endl;
    cout << "| Encryption Parameters :" << endl;
    cout << "|  - Scheme: " << contxt.scheme << endl;
    cout << "|  - Poly Modulus Degree: " << contxt.n << endl;
    cout << "|  - Coeff Modulus Size: " << contxt.total_bits << " bits & Coeff Modulus Count: " << contxt.coeff_modulus << endl;
    cout << "|  - Plain Modulus: " << contxt.plain_modulus_.value << endl;
    cout << "/ --------------------------------------------------------- /" << endl;

    int coeff_modulus = contxt.coeff_modulus;
    const int n = contxt.n;
    const int row_size = n / 2;

    Secretkey secret_key(contxt);
    Publickey public_key(contxt);
    Relinkey relin_key(contxt);
    Galoiskey galois_key(contxt);

    HESecretkeygen(secret_key, contxt);
    HEPublickeygen(public_key, secret_key, contxt);
    HERelinkeygen(relin_key, secret_key, contxt);
    HEGaloiskeygen(galois_key, secret_key, contxt);

    Data* message = (Data*)malloc(sizeof(Data) * n);
    for (int i = 0; i < n; i++) {

        message[i] = 8;
    }
    
    message[0] = 1;
    message[1] = 12;
    message[2] = 23;
    message[3] = 31;
    message[row_size] = 41;
    message[row_size + 1] = 54;
    message[row_size + 2] = 6;
    message[row_size + 3] = 100;

    Message M1(message, contxt);
    Plaintext P1(contxt);
    Plaintext P3(contxt);

    const int test_count = 50;
    float time_encoding_measurements[test_count];
    float time_decoding_measurements[test_count];
    float time_encryption_measurements[test_count];
    float time_decryption_measurements[test_count];
    float time_addition_measurements[test_count];
    float time_multiplication_measurements[test_count];
    float time_relinearization_measurements[test_count];
    float time_rotation_measurements[test_count];
    float time_plain_multiplication_measurements[test_count];
    for (int loop = 0; loop < test_count; loop++)
    {
        float time = 0;
        cudaEvent_t start_encoding, stop_encoding;
        cudaEventCreate(&start_encoding); cudaEventCreate(&stop_encoding);
    
        HEEncoder encoder(contxt);
        cudaEventRecord(start_encoding);

        encoder.encode(P1, M1);
        
        cudaEventRecord(stop_encoding);
        cudaEventSynchronize(stop_encoding);
        cudaEventElapsedTime(&time, start_encoding, stop_encoding);
        time_encoding_measurements[loop] = time;
        
        ///////////////////////////////////////////////////////////////////////////////
    
        Ciphertext C1(contxt);

        cudaEvent_t start_encryption, stop_encryption;
        cudaEventCreate(&start_encryption); cudaEventCreate(&stop_encryption);

        HEEncryptor encryptor(contxt, public_key);

        cudaEventRecord(start_encryption);

        encryptor.encrypt(C1, P1);

        cudaEventRecord(stop_encryption);
        cudaEventSynchronize(stop_encryption);
        cudaEventElapsedTime(&time, start_encryption, stop_encryption);
        time_encryption_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        Ciphertext C_mul(contxt);

        cudaEvent_t start_multiplication, stop_multiplication;
        cudaEventCreate(&start_multiplication); cudaEventCreate(&stop_multiplication);

        HEOperator operators(contxt);

        cudaEventRecord(start_multiplication);

        operators.multiply(C1, C1, C_mul);

        cudaEventRecord(stop_multiplication);
        cudaEventSynchronize(stop_multiplication);
        cudaEventElapsedTime(&time, start_multiplication, stop_multiplication);
        time_multiplication_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        cudaEvent_t start_relinearization, stop_relinearization;
        cudaEventCreate(&start_relinearization); cudaEventCreate(&stop_relinearization);

        cudaEventRecord(start_relinearization);

        operators.relinearize_inplace(C_mul, relin_key);

        cudaEventRecord(stop_relinearization);
        cudaEventSynchronize(stop_relinearization);
        cudaEventElapsedTime(&time, start_relinearization, stop_relinearization);
        time_relinearization_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        Ciphertext C2(contxt);

        cudaEvent_t start_rotation, stop_rotation;
        cudaEventCreate(&start_rotation); cudaEventCreate(&stop_rotation);

        cudaEventRecord(start_rotation);

        operators.rotate(C_mul, C2, galois_key, 1);

        cudaEventRecord(stop_rotation);
        cudaEventSynchronize(stop_rotation);
        cudaEventElapsedTime(&time, start_rotation, stop_rotation);
        time_rotation_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        cudaEvent_t start_addition, stop_addition;
        cudaEventCreate(&start_addition); cudaEventCreate(&stop_addition);

        cudaEventRecord(start_addition);

        operators.add_inplace(C2, C2);

        cudaEventRecord(stop_addition);
        cudaEventSynchronize(stop_addition);
        cudaEventElapsedTime(&time, start_addition, stop_addition);
        time_addition_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        cudaEvent_t start_plain_multiplication, stop_plain_multiplication;
        cudaEventCreate(&start_plain_multiplication); cudaEventCreate(&stop_plain_multiplication);

        cudaEventRecord(start_plain_multiplication);

        operators.multiply_plain(C2, P3, C2);

        cudaEventRecord(stop_plain_multiplication);
        cudaEventSynchronize(stop_plain_multiplication);
        cudaEventElapsedTime(&time, start_plain_multiplication, stop_plain_multiplication);
        time_plain_multiplication_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        Message M2(contxt);
        Plaintext P2(contxt);
        
        HEDecryptor decryptor(contxt, secret_key);

        cudaEvent_t start_decryption, stop_decryption;
        cudaEventCreate(&start_decryption); cudaEventCreate(&stop_decryption);

        cudaEventRecord(start_decryption);

        decryptor.decrypt(P2, C2);

        cudaEventRecord(stop_decryption);
        cudaEventSynchronize(stop_decryption);
        cudaEventElapsedTime(&time, start_decryption, stop_decryption);
        time_decryption_measurements[loop] = time;

        ///////////////////////////////////////////////////////////////////////////////

        cudaEvent_t start_decoding, stop_decoding;
        cudaEventCreate(&start_decoding); cudaEventCreate(&stop_decoding);

        cudaEventRecord(start_decoding);

        encoder.decode(M2, P2);

        cudaEventRecord(stop_decoding);
        cudaEventSynchronize(stop_decoding);
        cudaEventElapsedTime(&time, start_decoding, stop_decoding);
        time_decoding_measurements[loop] = time;
        

    }

    float sum_encoding = 0.0;
    float sum_decoding = 0.0;
    float sum_encryption = 0.0;
    float sum_decryption = 0.0;
    float sum_addition = 0.0;
    float sum_multiplication = 0.0;
    float sum_relinearization = 0.0;
    float sum_rotation = 0.0;
    float sum_plain_multiplication = 0.0;

    for (int i = 0; i < test_count; ++i)
    {
        sum_encoding += time_encoding_measurements[i];
        sum_decoding += time_decoding_measurements[i];
        sum_encryption += time_encryption_measurements[i];
        sum_decryption += time_decryption_measurements[i];
        sum_addition += time_addition_measurements[i];
        sum_multiplication += time_multiplication_measurements[i];
        sum_relinearization += time_relinearization_measurements[i];
        sum_rotation += time_rotation_measurements[i];
        sum_plain_multiplication += time_plain_multiplication_measurements[i];
    }

    cout << endl << "/ ----------------------- Timings ------------------------- /" << endl;
    cout << "Encoding timing:             " << sum_encoding / test_count << " us"<< endl;
    cout << "Decoding timing:             " << sum_decoding / test_count << " us"<< endl;
    cout << "Encryption timing:           " << sum_encryption / test_count << " us"<< endl;
    cout << "Decryption timing:           " << sum_decryption / test_count << " us"<< endl;
    cout << "Addition timing:             " << sum_addition / test_count << " us"<< endl;
    cout << "Multiplication timing:       " << sum_multiplication / test_count << " us"<< endl;
    cout << "Relinearization timing:      " << sum_relinearization / test_count << " us"<< endl;
    cout << "Rotation timing:             " << sum_rotation / test_count << " us"<< endl;
    cout << "Plain_Multiplication timing: " << sum_plain_multiplication / test_count << " us"<< endl;
    cout << "/ --------------------------------------------------------- /" << endl;

    return EXIT_SUCCESS;
}



