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

int main(int argc, char* argv[])
{
    CudaDevice();

    int device = 0; // Assuming you are using device 0
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;

    cout << "-----------------------------------" << endl;

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
    cout << "Ring Size: " << n << endl;
    cout << "N_power: " << contxt.n_power << endl;
    cout << "RNS_SIZE: " << coeff_modulus << endl;
    cout << "DECOMP_SIZE: " << coeff_modulus - 1 << endl;
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
    

    HEEncoder encoder(contxt);
    encoder.encode(P1, M1);
    encoder.encode(P3, M1);

    Ciphertext C1(contxt);

    ///////////////////////////////////////////////////////////
 


    HEEncryptor encryptor(contxt, public_key);
    encryptor.encrypt(C1, P1);

    ///////////////////////////////////////////////////////////////////////////////

    Ciphertext C_mul(contxt);



    HEOperator operators(contxt);
    operators.multiply(C1, C1, C_mul);

    ///////////////////////////////////////////////////////////////////////////////

    operators.relinearize_inplace(C_mul, relin_key);

    ///////////////////////////////////////////////////////////////////////////////

    Ciphertext C2(contxt);


    operators.rotate(C_mul, C2, galois_key, 1);

    ///////////////////////////////////////////////////////////////////////////////

    

    
    operators.add_inplace(C2, C2);

    ///////////////////////////////////////////////////////////////////////////////


    operators.multiply_plain(C2, P3, C2);

    ///////////////////////////////////////////////////////////////////////////////

    Message M2(contxt);
    Plaintext P2(contxt);


    
    HEDecryptor decryptor(contxt, secret_key);
    decryptor.decrypt(P2, C2);


    encoder.decode(M2, P2);

    Data* check = (Data*)malloc(contxt.n * sizeof(Data));
    cudaMemcpy(check, M2.location, contxt.n * sizeof(Data), cudaMemcpyDeviceToHost);  


    for(int i = 0; i < 20; i++){
        cout << i << ": " << check[i] << endl;
    }

    secret_key.kill();
    public_key.kill();
    relin_key.kill();

    return EXIT_SUCCESS;
}



/*
// tests

cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 

cmake --build ./cmake-build/ --target HE_test2 --parallel
./cmake-build/HE_test2

*/

