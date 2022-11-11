#include <fstream>
#include "Lib.cuh"
// --------------------- //
// Author: Alisah Ozcan
// --------------------- //


/*
      This study is a part of Alişah Özcan's master's thesis and was done for educational purposes. There may be shortcomings.

      In this Library, BFV operations are performed with using GPU. Since the library uses multiple coefficient modulus with RNS,
      it is very suitable for parallel operation. That's why the library has an NTT GPU implementation that can pereform batch-NTT.

      The library currently only performing for 128 bit security level and certain ring sizes(4096, 8192, 16384, 32768).
      Since this library is still a prototype, it works with precomputed parameters for now. For example,
      the user cannot use own plain modulus or own coeffmodulus, because all values are pre-calculated and written in Contect_Pool.
      For this reason, the user should use the library with the currently existing parameters. In the future,
      the library will be updated and made suitable for users to use the parameters they want within the rules.
      Also, CKKS scheme and Bootstrapping for BFV will be added to the library in the future.
*/

int main() {

  
    /*
        Lib_Parameters generates context with certain ringsize.
        All parameters to be used are pre-calculated. You cannot use your own parameters for this version.
    */

    Lib_Parameters contxt("BFV", 16384, security_level::HES_128);

    cout << "/ --------------------------------------------------------- /" << endl;
    cout << "|                 ~ WELCOME TO BFV GPU LIBRARY ~                 " << endl;
    cout << "| Encryption Parameters :" << endl;
    cout << "|  - Scheme: " << contxt.scheme << endl;
    cout << "|  - Poly Modulus Degree: " << contxt.n << endl;
    cout << "|  - Coeff Modulus Size: " << contxt.total_bits << " bits & Coeff Modulus Count: " << contxt.coeff_modulus << endl;
    cout << "|  - Plain Modulus: " << contxt.plain_mod << endl;
    cout << "/ --------------------------------------------------------- /" << endl;

    int coeff_modulus = contxt.coeff_modulus;
    const int n = contxt.n;
    cout << "Ring Size: " << n << endl;
    cout << "N_power " << contxt.n_power << endl;
    cout << "RNS_SIZE: " << coeff_modulus << endl;
    cout << "DECOMP_SIZE: " << coeff_modulus - 1 << endl;
    const int row_size = n / 2;

    /*
        GPU_Keygen creates secret key and public key on GPU using context.
        The created data is stored in a pointer in the context object.
    */
    GPU_Keygen(contxt); // Key Generation (pk,sk)


    /*
        First, a memory is set on the GPU using GPU_Relinkey.
        Then a relinkey is generated on the GPU using GPU_RelinKeyGen.
    */
    GPU_Relinkey relin_key(contxt);
    GPU_RelinKeyGen(relin_key, contxt);


    /*
    Creating a galoiskey is a little different than creating a relinkkey.
    Because it may be necessary to create for every power of 2. For example, if you want to shift a ciphertext 13 units to the right,
    you have to do it with the galois keys created for 2^0, 2^2 and 2^3(13 = 1 + 4 + 8). For this reason,
    you should write a value here according to the largest scrolling operation you will make.
    Also, if you want to shift the ciphertext to the left, you must activate the negative side.
    */
    GPU_GaloisKey galois_key(contxt, 2, true);
    GPU_GaloisKeyGen(galois_key, contxt);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    /*
        Messages are created by the CPU.
    */

    unsigned long long* messagge = (unsigned long long*)malloc(sizeof(unsigned long long) * n);
    for (int i = 0; i < n; i++) {

        messagge[i] = 0;
    }

    messagge[0] = 1;
    messagge[1] = 10;
    messagge[2] = 20;
    messagge[3] = 31;
    messagge[row_size] = 44;
    messagge[row_size + 1] = 33;
    messagge[row_size + 2] = 22;
    messagge[row_size + 3] = 120;

    /*
        The message created in the CPU is sent to the GPU. 
        The GPU_Messagge class send your message to GPU. ,
    */

    GPU_Messagge MSG(messagge, contxt);

    //messagge generation
    unsigned long long* messagge2 = (unsigned long long*)malloc(sizeof(unsigned long long) * n);
    for (int i = 0; i < n; i++) {

        messagge2[i] = 5;
    }

    GPU_Messagge MSG2(messagge2, contxt);

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    
    /*
        GPU_Plaintext & GPU_Ciphertext allocates memory in gpu for plaintext & ciphertext
    */
    GPU_Plaintext PLAINTXT(contxt);
    GPU_Ciphertext CIPHERTXT(contxt);


    /*
        GPU_Encode encodes the message and writes it to plaintext in GPU.  
        GPU_Enc encrypts the plaintext and writes it to ciphertext in GPU.   
    */
    GPU_Encode(MSG, PLAINTXT, contxt);
    GPU_Enc(CIPHERTXT, PLAINTXT, contxt);


    GPU_Plaintext PLAINTXT2(contxt);
    GPU_Ciphertext CIPHERTXT2(contxt);
    GPU_Encode(MSG2, PLAINTXT2, contxt);
    GPU_Enc(CIPHERTXT2, PLAINTXT2, contxt);

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /*
        Add/Sub Functions:
        * GPU_Addition              : performs homomorphic addition in GPU.
        * GPU_Addition_Inplace      : performs homomorphic addition inplace in GPU.
        * GPU_Addition_x3           : performs homomorphic addition after the multiplication without relinearization in GPU.
        * GPU_Addition_Inplace_x3   : performs homomorphic addition inplace after the multiplication without relinearization in GPU.
        * GPU_Subtraction           : performs homomorphic subtraction in GPU.
        * GPU_Subtraction_Inplace   : performs homomorphic subtraction inplace in GPU.
        * GPU_Subtraction_x3        : performs homomorphic subtraction after the multiplication without relinearization in GPU.
        * GPU_Subtraction_Inplace_x3: performs homomorphic subtraction inplace after the multiplication without relinearization in GPU.
    */
    GPU_Addition_Inplace(CIPHERTXT, CIPHERTXT2, contxt);


    /*
        The ciphertext size is 2 before multiplication, and the ciphertext size increases to 3 after multiplication.
        For this reason, it is necessary to increase the memory size of the output.
    */
    GPU_Ciphertext CIPHERTXT_result_mult(contxt);
    CIPHERTXT_result_mult.pre_mult(); // Increase the ciphertext size. 2 --> 3


    /*
        GPU_Multiplication performs homomorphic multiplication in GPU.
    */
    GPU_Multiplication(CIPHERTXT, CIPHERTXT, CIPHERTXT_result_mult, contxt);


    /*
        In this library, you cannot perform multiplication again without relinearization operation after multiplication.
        (This feature will be added in future versions.)
        Therefore, relinearization operation should be performed to reduce the ciphertext size.
    */
    GPU_Relinearization_Inplace(CIPHERTXT_result_mult, relin_key, contxt);
        

    /*
        GPU_Rotation performs homomorphic rotation operation in GPU.
    */
    GPU_Rotation(CIPHERTXT_result_mult, CIPHERTXT_result_mult, -2, galois_key, contxt);

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /*
        GPU_Dec performs decryption in GPU.
        GPU_Decode performs decode in GPU.
    */
    GPU_Plaintext PLAINTXT3(contxt);
    GPU_Messagge MSG_OUT(contxt);

    GPU_Dec(CIPHERTXT_result_mult, PLAINTXT3, contxt);
    GPU_Decode(MSG_OUT, PLAINTXT3, contxt);

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned long long* MSG_CHK = (unsigned long long*)malloc(n * sizeof(unsigned long long));
    cudaMemcpy(MSG_CHK, MSG_OUT.GPU_Location, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; i++) {

        int loc = n / 2;

        cout << MSG_CHK[loc * i] << ", " << MSG_CHK[loc * i + 1] << ", " << MSG_CHK[loc * i + 2] << ", " << MSG_CHK[loc * i + 3] << ", " << MSG_CHK[loc * i + 4] << ", " << MSG_CHK[loc * i + 5] << ", " << MSG_CHK[loc * i + 6] << endl;
    }

    return 0;
}
