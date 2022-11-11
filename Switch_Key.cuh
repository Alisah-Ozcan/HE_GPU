#pragma once

#include "NTT.cuh"
#include "GPU_Context.cuh"

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//--//--------------------------------------------------------------SWITCH_KEY FUNCTIONS--------------------------------------------------------------//--//
// --------------------------------------------------------------------------------------------------------------------------------------------------------



__global__ void Relin_Part1_Part2_renewed(unsigned long long temp_poly[], unsigned long long a[], unsigned long long rk[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    register uint128_t temp_add = 0;

#pragma unroll
    for (int piece = 0; piece < (modcount - 1); piece++) {

#pragma unroll
        for (int loop = 0; loop < modcount; loop++) {

            if ((((loop * 2) * n) <= idx) && (idx < ((loop * 2) + 2) * n)) {

                unsigned long long q = q_cons[loop];
                unsigned long long mu = mu_cons[loop];
                unsigned long long qbit = q_bit_cons[loop];

                uint128_t a_thread = a[(piece * modcount * n) + (loop * n) + idx % n];
                unsigned long long rk_thread = rk[idx + (2 * modcount * n * piece)];
               

                mul64(a_thread.low, rk_thread, a_thread);
                singleBarrett(a_thread, q, mu, qbit);


                temp_add = temp_add + a_thread.low;

            }

        }

    }
    singleBarrett(temp_add, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]);
    temp_poly[idx] = temp_add.low;

}

// for nx kısmı
__global__ void Relin_Part_Last_renewed_rotation(unsigned long long result[], unsigned long long ct[], unsigned long long input1[], unsigned long long half[], unsigned long long half_mod[], unsigned long long last_q_modinv[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long last_q = q_cons[modcount - 1];
                                      
    register unsigned long long allprocess = input1[n * (modcount - 1) * 2 + (idx % (2 * n))]; 
    register uint128_t allprocess_128 = allprocess + half[0];


    singleBarrett(allprocess_128, q_cons[modcount - 1], mu_cons[modcount - 1], q_bit_cons[modcount - 1]);
    // done



    allprocess = allprocess_128.low + q_cons[int(idx / (2 * n))];
    allprocess_128 = allprocess - half_mod[int(idx / (2 * n))];
    singleBarrett(allprocess_128, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]);
    // done


    register unsigned long long allprocess_2 = input1[idx]; 
    allprocess_2 = allprocess_2 + q_cons[int(idx / (2 * n))];
    register uint128_t allprocess_2_128 = allprocess_2 - allprocess_128.low;
    singleBarrett(allprocess_2_128, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]); 
  


    mul64(allprocess_2_128.low, last_q_modinv[int(idx / (2 * n))], allprocess_2_128);

    singleBarrett(allprocess_2_128, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]);


#pragma unroll                  
    for (int loop_x = 0; loop_x < (modcount - 1); loop_x++) {

        if ((((loop_x * 2) * n) <= idx) && (idx < (((loop_x * 2) + 1) * n))) {

            allprocess_2_128 = allprocess_2_128.low + ct[idx - (loop_x * n)];
            singleBarrett(allprocess_2_128, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]);  
            result[(loop_x * n) + (idx % n)] = allprocess_2_128.low;
        }
        else if (((((loop_x * 2) + 1) * n) <= idx) && (idx < (((loop_x * 2) + 2) * n))) {

            allprocess_2_128 = allprocess_2_128.low + ct[idx + (((modcount - 2) - loop_x) * n)]; 
            singleBarrett(allprocess_2_128, q_cons[int(idx / (2 * n))], mu_cons[int(idx / (2 * n))], q_bit_cons[int(idx / (2 * n))]);
            result[((modcount - 1) * n) + (loop_x * n) + (idx % n)] = allprocess_2_128.low;
        }

    }

}


// for nx kısmı
__global__ void Relin_Part_Last_renewed(unsigned long long ct[], unsigned long long input1[], unsigned long long half[], unsigned long long half_mod[], unsigned long long last_q_modinv[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long last_q = q_cons[modcount - 1];
                                        
    register unsigned long long allprocess = input1[n * (modcount - 1) * 2 + (idx % (2 * n))];
    register uint128_t allprocess_128 = allprocess + half[0];

    unsigned long long q_thread = q_cons[int(idx / (2 * n))];
    unsigned long long mu_thread = mu_cons[int(idx / (2 * n))];
    unsigned long long qbit_thread = q_bit_cons[int(idx / (2 * n))];


    singleBarrett(allprocess_128, q_cons[modcount - 1], mu_cons[modcount - 1], q_bit_cons[modcount - 1]);
    // done



    allprocess = allprocess_128.low + q_thread;
    allprocess_128 = allprocess - half_mod[int(idx / (2 * n))];
    singleBarrett(allprocess_128, q_thread, mu_thread, qbit_thread);
    // done


    register unsigned long long allprocess_2 = input1[idx]; 
    allprocess_2 = allprocess_2 + q_thread;
    register uint128_t allprocess_2_128 = allprocess_2 - allprocess_128.low;
    singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread); 
    //done


    mul64(allprocess_2_128.low, last_q_modinv[int(idx / (2 * n))], allprocess_2_128);

    singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
    // done

#pragma unroll                  
    for (int loop_x = 0; loop_x < (modcount - 1); loop_x++) {

        if ((((loop_x * 2) * n) <= idx) && (idx < (((loop_x * 2) + 1) * n))) {

            allprocess_2_128 = allprocess_2_128.low + ct[idx - (loop_x * n)];
            singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
            ct[(loop_x * n) + (idx % n)] = allprocess_2_128.low;
        }
        else if (((((loop_x * 2) + 1) * n) <= idx) && (idx < (((loop_x * 2) + 2) * n))) {

            allprocess_2_128 = allprocess_2_128.low + ct[idx + (((modcount - 2) - loop_x) * n)]; 
            singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
            ct[((modcount - 1) * n) + (loop_x * n) + (idx % n)] = allprocess_2_128.low;
        }

    }

}



__global__ void apply_galois(unsigned long long input[], unsigned long long modulus[], unsigned long long result0[], unsigned long long result1[], unsigned coeff_count_power,
    unsigned galois_elt, int n, int de_modcount)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n * de_modcount) {
        unsigned long long modulus_value = modulus[int(idx / n)];
        unsigned coeff_count_minus_one = (1 << coeff_count_power) - 1;

        unsigned index_raw = (idx % n) * galois_elt;
        unsigned index = index_raw & coeff_count_minus_one;
        unsigned long long result_value = input[idx];

        if ((index_raw >> coeff_count_power) & 1) {
            unsigned long long non_zero = unsigned long long int(result_value != 0);
            result_value = (modulus_value - result_value) & (-non_zero);
        }

#pragma unroll 
        for (int loop = 0; loop < de_modcount; loop++) {

            if (((n * loop) <= idx) && (idx < (n * (loop + 1)))) {

                result0[index + (n * loop)] = result_value;

            }
        }
    }


    else if (n * de_modcount <= idx) {//                           
        unsigned long long modulus_value = modulus[int(idx / n) % de_modcount];
        unsigned coeff_count_minus_one = (1 << coeff_count_power) - 1;

        unsigned index_raw = (idx % n) * galois_elt;
        unsigned index = index_raw & coeff_count_minus_one;
        unsigned long long result_value = input[idx];

        if ((index_raw >> coeff_count_power) & 1) {
            result_value = modulus_value - result_value;
        }// Buraya bir matematik düşünülebilir.

#pragma unroll 
        for (int loop1 = de_modcount; loop1 < (2 * de_modcount); loop1++) {

            if (((n * loop1) <= idx) && (idx < (n * (loop1 + 1)))) {

                for (int loop2 = 0; loop2 < (de_modcount + 1); loop2++) {
                  
                    result1[index + (n * (loop1 % de_modcount) * (de_modcount + 1)) + (loop2 * n)] = result_value; 
                }
            }
        }
    }
}



__global__ void Ct2_Duplication(int n, int q_length, unsigned long long origin[], unsigned long long created[])
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // q_length * n lik thread halleder
    unsigned long long temp_origin;
    for (int i = 0; i < q_length - 1; i++) {

        temp_origin = origin[(idx % n) + (n * i)];
        created[idx + (i * q_length * n)] = temp_origin;

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




//-------------------------------------------------------------------------------------------------------------------------------------------\\
//-------------------------------------------------------------------------------------------------------------------------------------------\\

__global__ void Relin_Part1_Part2_renewed_NEW(unsigned long long temp_poly[], unsigned long long a[], unsigned long long rk[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    register uint128_t temp_add = 0;

    for (int piece = 0; piece < (modcount - 1); piece++) {

        unsigned long long q = q_cons[j];
        unsigned long long mu = mu_cons[j];
        unsigned long long qbit = q_bit_cons[j];

        uint128_t a_thread = a[(piece * gridDim.y * n) + (j * n) + idx];
        unsigned long long rk_thread = rk[(k * n) + (j * n * 2) + idx + (gridDim.z * gridDim.y * n * piece)];

        mul64(a_thread.low, rk_thread, a_thread);
        singleBarrett(a_thread, q, mu, qbit);

        temp_add = temp_add + a_thread.low;

        temp_add.low -= (temp_add >= q) * q;

    }
    //singleBarrett(temp_add, q_cons[j], mu_cons[j], q_bit_cons[j]);
    temp_poly[(k * n) + (j * n * 2) + idx] = temp_add.low;

}


//--
__global__ void Relin_Part_Last_renewed_NEW(unsigned long long result[], unsigned long long ct[], unsigned long long input1[], unsigned long long half[], unsigned long long half_mod[], unsigned long long last_q_modinv[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    unsigned long long last_q = q_cons[gridDim.y];

    register unsigned long long allprocess = input1[(gridDim.y * n * 2) + (k * n) + idx];
    register uint128_t allprocess_128 = allprocess + half[0];

    unsigned long long q_thread = q_cons[j];
    unsigned long long mu_thread = mu_cons[j];
    unsigned long long qbit_thread = q_bit_cons[j];

    singleBarrett(allprocess_128, q_cons[gridDim.y], mu_cons[gridDim.y], q_bit_cons[gridDim.y]);
    // done

    allprocess = allprocess_128.low + q_thread;
    //allprocess_128 = allprocess - half_mod[int(idx / (2 * n))];
    allprocess_128 = allprocess - half_mod[j];
    singleBarrett(allprocess_128, q_thread, mu_thread, qbit_thread);
    // done

    register unsigned long long allprocess_2 = input1[(k * n) + (j * n * 2) + idx];
    allprocess_2 = allprocess_2 + q_thread;
    register uint128_t allprocess_2_128 = allprocess_2 - allprocess_128.low;
    singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
    //done


    mul64(allprocess_2_128.low, last_q_modinv[j], allprocess_2_128);

    singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
    // done

    // new code --- --- ---
    allprocess_2_128 = allprocess_2_128.low + ct[(k * gridDim.y * n) + (j * n) + idx];
    singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
    result[(k * gridDim.y * n) + (j * n) + idx] = allprocess_2_128.low;


}

//----------------------------------------------------------------------------------------
// Deneme2

__global__ void Relin_Part_Last_renewed_NEW2(unsigned long long ct[], unsigned long long input1[], unsigned long long half[], unsigned long long half_mod[], unsigned long long last_q_modinv[], unsigned long long q_cons[], unsigned long long mu_cons[], unsigned long long q_bit_cons[], int n, int modcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;

    int loc1 = modcount - 1;
    //unsigned long long last_q = q_cons[loc1];

    register unsigned long long allprocess = input1[(loc1 * n * 2) + (j * n) + idx];
    register uint128_t allprocess_128 = allprocess + half[0];

    //allprocess_128.low -= (allprocess_128 >= last_q) * last_q;
    singleBarrett(allprocess_128, q_cons[loc1], mu_cons[loc1], q_bit_cons[loc1]);
    // done

    for (int lp = 0; lp < loc1; lp++) {

        unsigned long long q_thread = q_cons[lp];
        unsigned long long mu_thread = mu_cons[lp];
        unsigned long long qbit_thread = q_bit_cons[lp];

        allprocess = allprocess_128.low + q_thread;
        allprocess_128 = allprocess - half_mod[lp];
        singleBarrett(allprocess_128, q_thread, mu_thread, qbit_thread);
        // done


        register unsigned long long allprocess_2 = input1[(j * n) + (lp * n * 2) + idx];
        allprocess_2 = allprocess_2 + q_thread;
        register uint128_t allprocess_2_128 = allprocess_2 - allprocess_128.low;
        //allprocess_2_128.low -= (allprocess_2_128 >= q_thread) * q_thread;
        singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
        //done

        mul64(allprocess_2_128.low, last_q_modinv[lp], allprocess_2_128);

        singleBarrett(allprocess_2_128, q_thread, mu_thread, qbit_thread);
        // done

         // new code --- --- ---
        allprocess_2_128 = allprocess_2_128.low + ct[(j * loc1 * n) + (lp * n) + idx];
        allprocess_2_128.low -= (allprocess_2_128 >= q_thread) * q_thread;
        ct[(j * loc1 * n) + (lp * n) + idx] = allprocess_2_128.low;

    }

}


__host__ void GPU_Relinearization_Inplace(GPU_Ciphertext input, GPU_Relinkey evk,
    Lib_Parameters context
)
{
    // class features
    unsigned long long* Relinearization_Pool = evk.GPU_temp;
    int n = input.ring_size;
    int q_length = input.coeff_mod_count + 1;

    //Extract Input Parameters
    unsigned long long* Input1 = input.GPU_Location2;


    //Extract Context Parameters
    unsigned long long* q_cons = context.Context_GPU;// Tık
    unsigned long long* mu_cons = q_cons + q_length;// Tık
    unsigned long long* q_bit_cons = mu_cons + q_length;// Tık
    unsigned long long* q_INTT_cons = q_bit_cons + q_length;// Tık
    unsigned long long* mu_INTT_cons = q_INTT_cons + q_length * 2;// Tık
    unsigned long long* q_bit_INTT_cons = mu_INTT_cons + q_length * 2;// Tık
    unsigned long long* psi_powers = q_bit_INTT_cons + q_length * 2;// Tık
    unsigned long long* psi_inv_double = psi_powers + q_length * n * 2;// Tık(2 adım atlıyor)
    unsigned long long* last_q_modinv = psi_inv_double + q_length * n * 2;// Tık
    unsigned long long* half = last_q_modinv + (q_length - 1);// Tık
    unsigned long long* half_mod = half + 1;// Tık
    unsigned long long* INTT_modinv_q = half_mod + (q_length - 1); // Tık
    unsigned long long* INTT_inv_double_q = INTT_modinv_q + q_length; // Tık

    //Extract Temporary Parameters
    unsigned long long* ct2_dublicate = Relinearization_Pool + q_length * 2 * n;


    Ct2_Duplication << < ((q_length) * (n / 1024)), 1024 >> > (n, q_length, Input1, ct2_dublicate);

    // Forward NTT Part-------------------------------------------------------------------------------------------------------------------------


    Forward_NTT_Inplace(ct2_dublicate, q_cons, mu_cons, q_bit_cons, n, psi_powers, (q_length * (q_length - 1)), q_length);

    //-------------------------------------------------------------------------------------------------------------------------------------------

    //Relin_Part1_Part2_renewed << <(q_length * 2 * (n / 1024)), 1024, 0, 0 >> > (Relinearization_Pool, ct2_dublicate, evk.GPU_Location, q_cons, mu_cons, q_bit_cons, n, q_length);
    dim3 numBlockss((n / 1024), q_length, 2);
    Relin_Part1_Part2_renewed_NEW << <numBlockss, 1024 >> > (Relinearization_Pool, ct2_dublicate, evk.GPU_Location, q_cons, mu_cons, q_bit_cons, n, q_length);

    // Inverse NTT Part-------------------------------------------------------------------------------------------------------------------------

    Inverse_NTT_Inplace(Relinearization_Pool, q_INTT_cons, mu_INTT_cons, q_bit_INTT_cons, n, psi_inv_double, (q_length * 2), (q_length * 2), INTT_inv_double_q);

    //-------------------------------------------------------------------------------------------------------------------------------------------
    dim3 numBlocks((n / 1024), q_length - 1, 2);
    Relin_Part_Last_renewed_NEW << < numBlocks, 1024 >> > (input.GPU_Location, input.GPU_Location, Relinearization_Pool, half, half_mod, last_q_modinv, q_cons, mu_cons, q_bit_cons, n, q_length); //2.4
    //dim3 numBlocks((n / 1024), 2, 1);
    //Relin_Part_Last_renewed_NEW2 << < numBlocks, 1024 >> > (input.GPU_Location, Relinearization_Pool, half, half_mod, last_q_modinv, q_cons, mu_cons, q_bit_cons, n, q_length); //2.4
    //-------------------------------------------------------------------------------------------------------------------------------------------


}


//-------------------------------------------------------------------------------------------------------------------------------------------\\
//-------------------------------------------------------------------------------------------------------------------------------------------\\

//-------------------------------------------------------------------------------------------------------------------------------------------\\
//-------------------------------------------------------------------------------------------------------------------------------------------\\

__host__ void GPU_Rotation(GPU_Ciphertext input, GPU_Ciphertext output, int shift, GPU_GaloisKey evk, Lib_Parameters context)
{

    // Class Features
    int n = input.ring_size;
    int q_length = input.coeff_mod_count + 1;
    int n_power = context.n_power;

    unsigned long long* GaloisKey_positive = evk.GPU_Location_positive;
    unsigned long long* GaloisKey_negative = evk.GPU_Location_negative;
    int* galoiselt_positive = evk.galois_elt_pos;
    int* galoiselt_negative = evk.galois_elt_neg;


    //Extract Context Parameters
    unsigned long long* q_cons = context.Context_GPU;// Tık
    unsigned long long* mu_cons = q_cons + q_length;// Tık
    unsigned long long* q_bit_cons = mu_cons + q_length;// Tık
    unsigned long long* q_INTT_cons = q_bit_cons + q_length;// Tık
    unsigned long long* mu_INTT_cons = q_INTT_cons + q_length * 2;// Tık
    unsigned long long* q_bit_INTT_cons = mu_INTT_cons + q_length * 2;// Tık
    unsigned long long* psi_powers = q_bit_INTT_cons + q_length * 2;// Tık
    unsigned long long* psi_inv_double = psi_powers + q_length * n * 2;// Tık(2 adım atlıyor)
    unsigned long long* last_q_modinv = psi_inv_double + q_length * n * 2;// Tık
    unsigned long long* half = last_q_modinv + (q_length - 1);// Tık
    unsigned long long* half_mod = half + 1;// Tık
    unsigned long long* INTT_modinv_q = half_mod + (q_length - 1); // Tık
    unsigned long long* INTT_inv_double_q = INTT_modinv_q + q_length; // Tık


    //Extract Temporary Parameters
    unsigned long long* temp0 = evk.GPU_temp + (q_length * 2 * n);
    unsigned long long* temp1 = temp0 + ((q_length - 1) * 2 * n);

    unsigned long long* GaloisKey;
    int* galoiselt;
    // positive or negative
    if (shift > 0) {
        GaloisKey = GaloisKey_positive;
        galoiselt = galoiselt_positive;
    }
    else {
        GaloisKey = GaloisKey_negative;
        galoiselt = galoiselt_negative;
    }

    unsigned long long* last_results_host = input.GPU_Location;

    int shift_num = abs(shift);
    while (shift_num != 0) {
        int power = int(log2(shift_num));
        int power_2 = pow(2, power);
        shift_num = shift_num - power_2;

        if (shift_num == 0) {
            last_results_host = output.GPU_Location;
        }

        // Apply Galois --------------------------------------------------------------------------temp0---temp1---------------------------
        apply_galois << <((q_length - 1) * 2 * (n / 1024)), 1024, 0, 0 >> > (input.GPU_Location, q_cons, temp0, temp1, n_power, galoiselt[power], n, (q_length - 1)); // burası OK
        //cudaDeviceSynchronize();

        // Forward NTT Part-------------------------------------------------------------------------------------------------------------------------


        Forward_NTT_Inplace(temp1, q_cons, mu_cons, q_bit_cons, n, psi_powers, (q_length * (q_length - 1)), q_length); //BSK ciphertext'lerinin ntt sonuçları


        //cudaDeviceSynchronize();

        //-------------------------------------------------------------------------------------------------------------------------------------------

        //Relin_Part1_Part2_renewed << <(q_length * 2 * (n / 1024)), 1024, 0, 0 >> > (evk.GPU_temp, temp1, GaloisKey + (power * (q_length * (q_length - 1) * 2 * n)), q_cons, mu_cons, q_bit_cons, n, q_length);

        dim3 numBlockss((n / 1024), q_length, 2);
        Relin_Part1_Part2_renewed_NEW << <numBlockss, 1024 >> > (evk.GPU_temp, temp1, GaloisKey + (power * (q_length * (q_length - 1) * 2 * n)), q_cons, mu_cons, q_bit_cons, n, q_length);
        //cudaDeviceSynchronize();
        //-------------------------------------------------------------------------------------------------------------------------------------------
        // Inverse NTT Part-------------------------------------------------------------------------------------------------------------------------

        Inverse_NTT_Inplace(evk.GPU_temp, q_INTT_cons, mu_INTT_cons, q_bit_INTT_cons, n, psi_inv_double, (q_length * 2), (q_length * 2), INTT_inv_double_q);


        //cudaDeviceSynchronize();

        //-------------------------------------------------------------------------------------------------------------------------------------------

        //Relin_Part_Last_renewed_rotation << < ((q_length - 1) * 2 * (n / 1024)), 1024 >> > (output.GPU_Location, temp0, evk.GPU_temp, half, half_mod, last_q_modinv, q_cons, mu_cons, q_bit_cons, n, q_length); //2.4
        //cudaDeviceSynchronize();

        dim3 numBlocks((n / 1024), q_length - 1, 2);
        Relin_Part_Last_renewed_NEW << < numBlocks, 1024 >> > (output.GPU_Location, temp0, evk.GPU_temp, half, half_mod, last_q_modinv, q_cons, mu_cons, q_bit_cons, n, q_length);  //2.4
        //------------------------------------------------------------------------------------------------------------------------------------------
    }

}