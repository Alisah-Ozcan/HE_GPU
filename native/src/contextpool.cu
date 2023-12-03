// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

#include "contextpool.cuh"



//int PrimePool::n = 12; // Default value of ring sizes
//PrimePool::security_level PrimePool::sec = PrimePool::security_level::HES_128; // Default value of security level
Data extendedGCD(Data a, Data b, Data &x, Data &y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return b;
    }

    Data x1, y1;
    Data gcd = extendedGCD(b % a, a, x1, y1);

    x = y1 - (b / a) * x1;
    y = x1;

    return gcd;
}

Data modInverse(Data a, Data m) {
    Data x, y;
    Data gcd = extendedGCD(a, m, x, y);

    if (gcd != 1) {
        // Modular inverse does not exist
        return 0;
    } else {
        // Ensure the result is positive
        Data result = (x % m + m) % m;
        return result;
    }
}

int PrimePool::n;
PrimePool::security_level PrimePool::sec;

PrimePool::PrimePool(int poly_degree, security_level sec_level){
    n = poly_degree;
    sec = sec_level;
};


int PrimePool::prime_count() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096: return 3;
        case 8192: return 4;
        case 16384: return 8;
        case 32768: return 15;
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


int PrimePool::total_primes_bits() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            static int bits_array = 109; // Bits
            return bits_array;
        }
        case 8192:
        {
            static int bits_array = 218; // Bits
            return bits_array;
        }
        case 16384:
        {
            static int bits_array = 438; // Bits
            return bits_array;
        }
        case 32768:
        {
            static int bits_array = 881; // Bits
            return bits_array;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Modulus> PrimePool::base_modulus() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            static unsigned long long prime_array[] = { 34359754753ULL, 34359771137ULL, 68719484929ULL };

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }

        case 8192:
        {
            static unsigned long long prime_array[] = { 9007199255019521ULL, 9007199255347201ULL, 18014398510645249ULL, 18014398510661633ULL };

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        case 16384:
        {
            static unsigned long long prime_array[] = { 18014398510661633ULL, 18014398511382529ULL, 18014398512136193ULL, 18014398512365569ULL,
                18014398514036737ULL, 18014398514200577ULL, 18014398514987009ULL, 18014398515511297ULL };

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        case 32768:
        {
            static unsigned long long prime_array[] = { 144115188078673921ULL, 144115188079656961ULL, 144115188081819649ULL, 144115188082409473ULL,
                288230376154267649ULL, 288230376155185153ULL, 288230376155250689ULL, 288230376156758017ULL, 288230376157413377ULL, 288230376158396417ULL,
                288230376160755713ULL, 288230376161280001ULL, 288230376161673217ULL, 288230376161738753ULL, 288230376162459649ULL };

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}




std::vector<Root> PrimePool::ntt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            static unsigned long long psi_array[] = { 6071469ULL, 18291550ULL, 28979647ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 8192:
        {
            static unsigned long long psi_array[] = { 1001816340395ULL, 97583054751ULL, 600254011689ULL, 305447760175ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = { 26448496311ULL, 32910392359ULL, 317176533655ULL, 1958804412711ULL, 186605226681ULL, 223342371418ULL, 4102367446149ULL, 117039768478ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 6076061706634ULL, 757812206199ULL, 14332630180726ULL, 4325862285354ULL, 3986778017537ULL,
                17957119137197ULL, 6510836882592ULL, 8505645339603ULL, 20417392538974ULL, 15790315796150ULL, 9174649664700ULL, 3037638144297ULL,
                1412431483320ULL, 11383777697068ULL, 4139725055370ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Root> PrimePool::intt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            static unsigned long long psi_array[] = { 6071469ULL, 18291550ULL, 28979647ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 8192:
        {
            static unsigned long long psi_array[] = { 1001816340395ULL, 97583054751ULL, 600254011689ULL, 305447760175ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = { 26448496311ULL, 32910392359ULL, 317176533655ULL, 1958804412711ULL, 186605226681ULL, 223342371418ULL, 4102367446149ULL, 117039768478ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 6076061706634ULL, 757812206199ULL, 14332630180726ULL, 4325862285354ULL, 3986778017537ULL,
                17957119137197ULL, 6510836882592ULL, 8505645339603ULL, 20417392538974ULL, 15790315796150ULL, 9174649664700ULL, 3037638144297ULL,
                1412431483320ULL, 11383777697068ULL, 4139725055370ULL };
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Ninverse> PrimePool::n_inverse() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {   
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 8192:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 16384:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 32768:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::last_q_modinv() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {   
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            std::vector<Data> last_q_modinv_;
            for (int i = 0; i < modulus_.size() - 1; i++)
            {
                last_q_modinv_.push_back(VALUE::modinv(modulus_[ modulus_.size() - 1].value, modulus_[i]));
            }
            
            return last_q_modinv_;
        }
        case 8192:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            std::vector<Data> last_q_modinv_;
            for (int i = 0; i < modulus_.size() - 1; i++)
            {
                last_q_modinv_.push_back(VALUE::modinv(modulus_[ modulus_.size() - 1].value, modulus_[i]));
            }
            
            return last_q_modinv_;
        }
        case 16384:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            std::vector<Data> last_q_modinv_;
            for (int i = 0; i < modulus_.size() - 1; i++)
            {
                last_q_modinv_.push_back(VALUE::modinv(modulus_[ modulus_.size() - 1].value, modulus_[i]));
            }
            
            return last_q_modinv_;
        }
        case 32768:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_modulus();
            std::vector<Data> last_q_modinv_;
            for (int i = 0; i < modulus_.size() - 1; i++)
            {
                last_q_modinv_.push_back(VALUE::modinv(modulus_[ modulus_.size() - 1].value, modulus_[i]));
            }
            
            return last_q_modinv_;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Modulus> PrimePool::base_Bsk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            //static unsigned long long base_Bsk_array[] = { 2305843009213554689ULL, 2305843009213489153ULL, 2305843009213317121ULL, 2305843009213243393ULL };
            static unsigned long long base_Bsk_array[] = { 2305843009213554689ULL, 2305843009213489153ULL, 2305843009213317121ULL };

            int size = sizeof(base_Bsk_array) / sizeof(base_Bsk_array[0]); 
            std::vector<Modulus> base_Bsk_vector;

            for(int i = 0; i < size; i++){
                base_Bsk_vector.push_back((Modulus)base_Bsk_array[i]);
            }
    
            return base_Bsk_vector;
        }

        case 8192:
        {
            static unsigned long long base_Bsk_array[] = { 2305843009213317121ULL, 2305843009213120513ULL, 2305843009212694529ULL, 2305843009212399617ULL, 2305843009211662337ULL };

            int size = sizeof(base_Bsk_array) / sizeof(base_Bsk_array[0]); 
            std::vector<Modulus> base_Bsk_vector;

            for(int i = 0; i < size; i++){
                base_Bsk_vector.push_back((Modulus)base_Bsk_array[i]);
            }
    
            return base_Bsk_vector;

        }
        case 16384:
        {
            static unsigned long long base_Bsk_array[] = { 2305843009211662337ULL, 2305843009211596801ULL, 2305843009211400193ULL, 2305843009210580993ULL, 2305843009210515457ULL, 2305843009210023937ULL, 2305843009208713217ULL, 2305843009208123393ULL };

            int size = sizeof(base_Bsk_array) / sizeof(base_Bsk_array[0]); 
            std::vector<Modulus> base_Bsk_vector;

            for(int i = 0; i < size; i++){
                base_Bsk_vector.push_back((Modulus)base_Bsk_array[i]);
            }
    
            return base_Bsk_vector;
        }
        case 32768:
        {
            static unsigned long long base_Bsk_array[] = { 2305843009211662337ULL, 2305843009211596801ULL, 2305843009211400193ULL, 2305843009210023937ULL, 2305843009208713217ULL, 2305843009208123393ULL, 2305843009207468033ULL, 2305843009202159617ULL, 2305843009201242113ULL, 2305843009200586753ULL, 2305843009197506561ULL, 2305843009196916737ULL, 2305843009195868161ULL, 2305843009195671553ULL, 2305843009195343873ULL };

            int size = sizeof(base_Bsk_array) / sizeof(base_Bsk_array[0]); 
            std::vector<Modulus> base_Bsk_vector;

            for(int i = 0; i < size; i++){
                base_Bsk_vector.push_back((Modulus)base_Bsk_array[i]);
            }
    
            return base_Bsk_vector;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Root> PrimePool::bsk_ntt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            //static unsigned long long psi_array[] = { 307554654119321ULL,168273734192536ULL,829315415491244ULL,32973993658837ULL };
            static unsigned long long psi_array[] = { 307554654119321ULL,168273734192536ULL,829315415491244ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 8192:
        {
            static unsigned long long psi_array[] = { 562556498301090074ULL, 2128477325179182330ULL, 972182959695038317ULL, 1993670270211764767ULL, 1428717407974281805ULL, 1359979752200406037ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = { 70072284713359ULL, 117297622845463ULL, 39472790483564ULL, 597089996664243ULL, 54890861537777ULL, 180991413543520ULL, 22745400076249ULL, 95920324194041ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 54086154900243ULL, 108184479186995ULL, 44627003565980ULL, 5466412987105ULL, 299243861837272ULL, 13621606365257ULL, 141711431679820ULL, 302433821420420ULL, 30890933577633ULL, 1211291919640ULL, 61972381836971ULL, 68991921477839ULL, 44611420972577ULL, 38109723409384ULL, 7854697884062ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Root> PrimePool::bsk_intt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            static unsigned long long psi_array[] = { 307554654119321ULL,168273734192536ULL,829315415491244ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_Bsk()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
            
        }
        case 8192:
        {
 
            static unsigned long long psi_array[] = { 562556498301090074ULL, 2128477325179182330ULL, 972182959695038317ULL, 1993670270211764767ULL, 1428717407974281805ULL, 1359979752200406037ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_Bsk()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = { 70072284713359ULL, 117297622845463ULL, 39472790483564ULL, 597089996664243ULL, 54890861537777ULL, 180991413543520ULL, 22745400076249ULL, 95920324194041ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_Bsk()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 54086154900243ULL, 108184479186995ULL, 44627003565980ULL, 5466412987105ULL, 299243861837272ULL, 13621606365257ULL, 141711431679820ULL, 302433821420420ULL, 30890933577633ULL, 1211291919640ULL, 61972381836971ULL, 68991921477839ULL, 44611420972577ULL, 38109723409384ULL, 7854697884062ULL };

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::base_Bsk()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::base_Bsk()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Ninverse> PrimePool::bsk_n_inverse() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {   
            std::vector<Modulus> modulus_ = PrimePool::base_Bsk();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 8192:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_Bsk();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 16384:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_Bsk();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 32768:
        {
            std::vector<Modulus> modulus_ = PrimePool::base_Bsk();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

Modulus PrimePool::m_tilde() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return (Modulus)4294967296ULL;
    case 8192: return (Modulus)4294967296ULL;
    case 16384: return (Modulus)4294967296ULL;
    case 32768: return (Modulus)4294967296ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Data> PrimePool::base_change_matrix_Bsk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> ibase = PrimePool::base_modulus();
            std::vector<Modulus> obase = PrimePool::base_Bsk();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
            /*
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                for(int k = 0; k < obase.size() - 1; k++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            */
            
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size(); k++){
                for(int i = 0; i < ibase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            


            return base_matrix;

        }
        case 8192:
        {
            std::vector<Modulus> ibase = PrimePool::base_modulus();
            std::vector<Modulus> obase = PrimePool::base_Bsk();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
           
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size(); k++){
                for(int i = 0; i < ibase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }

            return base_matrix;
        }
        case 16384:
        {
            std::vector<Modulus> ibase = PrimePool::base_modulus();
            std::vector<Modulus> obase = PrimePool::base_Bsk();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
            
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size(); k++){
                for(int i = 0; i < ibase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }

            return base_matrix;
        }
        case 32768:
        {
            std::vector<Modulus> ibase = PrimePool::base_modulus();
            std::vector<Modulus> obase = PrimePool::base_Bsk();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;

            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size(); k++){
                for(int i = 0; i < ibase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }

            return base_matrix;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}




std::vector<Data> PrimePool::inv_punctured_prod_mod_base_array() { // for q bases
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> rns_base = PrimePool::base_modulus();

            std::vector<Data> inv_punctured_prod_mod_base_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_base_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_base_array_;

        }
        case 8192:
        {

            std::vector<Modulus> rns_base = PrimePool::base_modulus();

            std::vector<Data> inv_punctured_prod_mod_base_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_base_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_base_array_;

        }
        case 16384:
        {

            std::vector<Modulus> rns_base = PrimePool::base_modulus();

            std::vector<Data> inv_punctured_prod_mod_base_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_base_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_base_array_;

        }
        case 32768:
        {

            std::vector<Modulus> rns_base = PrimePool::base_modulus();

            std::vector<Data> inv_punctured_prod_mod_base_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_base_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_base_array_;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::base_change_matrix_m_tilde() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> ibase = PrimePool::base_modulus();
            Modulus obase = PrimePool::m_tilde();
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 8192:
        {

            std::vector<Modulus> ibase = PrimePool::base_modulus();
            Modulus obase = PrimePool::m_tilde();
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 16384:
        {

            std::vector<Modulus> ibase = PrimePool::base_modulus();
            Modulus obase = PrimePool::m_tilde();
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 32768:
        {

            std::vector<Modulus> ibase = PrimePool::base_modulus();
            Modulus obase = PrimePool::m_tilde();
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::inv_prod_q_mod_m_tilde() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> ibase = PrimePool::base_modulus();
        Modulus obase = PrimePool::m_tilde();
        Data mult = 1;
        for(int i = 0; i < ibase.size() - 1; i++){
            mult = VALUE::mult(mult, ibase[i].value, obase);
        }

        return modInverse(mult, obase.value);

    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Data> PrimePool::inv_m_tilde_mod_Bsk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> Bsk = PrimePool::base_Bsk();
            Modulus m_tilde_ = PrimePool::m_tilde();
            std::vector<Data> inv_m_tilde_mod_Bsk_array_;

            for (int i = 0; i < Bsk.size(); i++){
                inv_m_tilde_mod_Bsk_array_.push_back(VALUE::modinv(m_tilde_.value, Bsk[i]));
            }
 
            return inv_m_tilde_mod_Bsk_array_;

        }
        case 8192:
        {

            std::vector<Modulus> Bsk = PrimePool::base_Bsk();
            Modulus m_tilde_ = PrimePool::m_tilde();
            std::vector<Data> inv_m_tilde_mod_Bsk_array_;

            for (int i = 0; i < Bsk.size(); i++){
                inv_m_tilde_mod_Bsk_array_.push_back(VALUE::modinv(m_tilde_.value, Bsk[i]));
            }
 
            return inv_m_tilde_mod_Bsk_array_;

        }
        case 16384:
        {

            std::vector<Modulus> Bsk = PrimePool::base_Bsk();
            Modulus m_tilde_ = PrimePool::m_tilde();
            std::vector<Data> inv_m_tilde_mod_Bsk_array_;

            for (int i = 0; i < Bsk.size(); i++){
                inv_m_tilde_mod_Bsk_array_.push_back(VALUE::modinv(m_tilde_.value, Bsk[i]));
            }
 
            return inv_m_tilde_mod_Bsk_array_;

        }
        case 32768:
        {

            std::vector<Modulus> Bsk = PrimePool::base_Bsk();
            Modulus m_tilde_ = PrimePool::m_tilde();
            std::vector<Data> inv_m_tilde_mod_Bsk_array_;

            for (int i = 0; i < Bsk.size(); i++){
                inv_m_tilde_mod_Bsk_array_.push_back(VALUE::modinv(m_tilde_.value, Bsk[i]));
            }
 
            return inv_m_tilde_mod_Bsk_array_;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::prod_q_mod_Bsk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
            case 4096:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;

            }
            case 8192:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;
            
            }
            case 16384:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;

            }
            case 32768:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;
            }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::inv_prod_q_mod_Bsk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
            case 4096:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(VALUE::modinv(temp, obase[i]));
                }

                return prod_q_mod_Bsk_array_;

            }
            case 8192:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(VALUE::modinv(temp, obase[i]));
                }

                return prod_q_mod_Bsk_array_;
            
            }
            case 16384:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(VALUE::modinv(temp, obase[i]));
                }

                return prod_q_mod_Bsk_array_;

            }
            case 32768:
            {

                std::vector<Modulus> ibase = PrimePool::base_modulus();
                std::vector<Modulus> obase = PrimePool::base_Bsk();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size(); i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(VALUE::modinv(temp, obase[i]));
                }

                return prod_q_mod_Bsk_array_;
            }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

Modulus PrimePool::plain_modulus() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return (Modulus)1032193ULL;
    case 8192: return (Modulus)1032193ULL;
    case 16384: return (Modulus)786433ULL;
    case 32768: return (Modulus)786433ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Modulus> PrimePool::plain_modulus2() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096:
    {   
        std::vector<Modulus> mod_;
        mod_.push_back((Modulus)1032193ULL);

        return mod_;
    }

    case 8192:
    { 
        std::vector<Modulus> mod_;
        mod_.push_back((Modulus)1032193ULL);
        
        return mod_;
    }
    case 16384:
    { 
        std::vector<Modulus> mod_;
        mod_.push_back((Modulus)786433ULL);
        
        return mod_;
    }
    case 32768:
    { 
        std::vector<Modulus> mod_;
        mod_.push_back((Modulus)786433ULL);
        
        return mod_;
    }
    }
}

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::plain_psi() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return 194ULL;
    case 8192: return 94ULL;
    case 16384: return 9ULL;
    case 32768: return 3ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Ninverse> PrimePool::n_plain_inverse() {
    if (sec == security_level::HES_128) {
        
        Data n_ = n;
        std::vector<Ninverse> n_inverse_;

        Modulus t = PrimePool::plain_modulus();
        
        n_inverse_.push_back(VALUE::modinv(n_, t));
        //n_inverse_.push_back(modInverse(n_, t.value));
        
        return n_inverse_;
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Root> PrimePool::plain_ntt_tables() {
    if (sec == security_level::HES_128) {
      
        int lg = log2(n);
        Data psi = PrimePool::plain_psi();
        Modulus modulus = PrimePool::plain_modulus();

        std::vector<Root> forward_table; // bit reverse order
        
        std::vector<Root> table;
        table.push_back(1);

        for (int j = 1; j < n; j++)
        {
            Data exp = VALUE::mult(table[(j - 1)], psi, modulus);
            table.push_back(exp);
        }

        for (int j = 0; j < n; j++) // take bit reverse order
        {
            forward_table.push_back(table[bitreverse(j, lg)]);
        }

        return forward_table;
    }
    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Root> PrimePool::plain_intt_tables() {
    if (sec == security_level::HES_128) {
      
        int lg = log2(n);
        Data psi = PrimePool::plain_psi();
        Modulus modulus = PrimePool::plain_modulus();

        Data inv_root = VALUE::modinv(psi, modulus);

        std::vector<Root> forward_table; // bit reverse order
        
        std::vector<Root> table;
        table.push_back(1);

        for (int j = 1; j < n; j++)
        {
            Data exp = VALUE::mult(table[(j - 1)], inv_root, modulus);
            table.push_back(exp);
        }

        for (int j = 0; j < n; j++) // take bit reverse order
        {
            forward_table.push_back(table[bitreverse(j, lg)]);
        }

        return forward_table;
    }
    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Data> PrimePool::base_change_matrix_q() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            std::vector<Modulus> obase = PrimePool::base_modulus();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
            /*
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                for(int k = 0; k < obase.size() - 1; k++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            */
            
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size() - 1; k++){
                for(int i = 0; i < ibase.size() - 1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            


            return base_matrix;

        }
        case 8192:
        {
            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            std::vector<Modulus> obase = PrimePool::base_modulus();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
           
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size() - 1; k++){
                for(int i = 0; i < ibase.size() - 1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            


            return base_matrix;

        }
        case 16384:
        {
            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            std::vector<Modulus> obase = PrimePool::base_modulus();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;
            
            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size() - 1; k++){
                for(int i = 0; i < ibase.size() - 1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            


            return base_matrix;

        }
        case 32768:
        {
            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            std::vector<Modulus> obase = PrimePool::base_modulus();

            std::vector<Data> punctured_prod_array;
            //std::vector<Data> inv_punctured_prod_mod_base_array;

            std::vector<Data> base_matrix;
            for(int k = 0; k < obase.size() - 1; k++){
                for(int i = 0; i < ibase.size() - 1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        if(i != j){
                            temp = VALUE::mult(temp, ibase[j].value, obase[k]);
                        }
                    }
                    base_matrix.push_back(temp);
                }
            }
            


            return base_matrix;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::base_change_matrix_msk() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            Modulus obase = ibase[ibase.size() - 1];
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 8192:
        {

            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            Modulus obase = ibase[ibase.size() - 1];
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 16384:
        {

            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            Modulus obase = ibase[ibase.size() - 1];
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        case 32768:
        {

            std::vector<Modulus> ibase = PrimePool::base_Bsk();
            Modulus obase = ibase[ibase.size() - 1];
            
            std::vector<Data> base_matrix;
            for(int i = 0; i < ibase.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < ibase.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, ibase[j].value, obase);
                    }
                }
                base_matrix.push_back(temp);
            }
            
            return base_matrix;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Data> PrimePool::inv_punctured_prod_mod_B_array() { // for B bases
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {

            std::vector<Modulus> rns_base = PrimePool::base_Bsk();

            std::vector<Data> inv_punctured_prod_mod_B_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_B_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_B_array_;

        }
        case 8192:
        {

            std::vector<Modulus> rns_base = PrimePool::base_Bsk();

            std::vector<Data> inv_punctured_prod_mod_B_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_B_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_B_array_;

        }
        case 16384:
        {

            std::vector<Modulus> rns_base = PrimePool::base_Bsk();

            std::vector<Data> inv_punctured_prod_mod_B_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_B_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_B_array_;

        }
        case 32768:
        {

           std::vector<Modulus> rns_base = PrimePool::base_Bsk();

            std::vector<Data> inv_punctured_prod_mod_B_array_;

            for(int i = 0; i < rns_base.size()-1; i++){
                Data temp = 1;
                for (int j = 0; j < rns_base.size()-1; j++){
                    if(i != j){
                        temp = VALUE::mult(temp, rns_base[j].value, rns_base[i]);
                    }
                }
                inv_punctured_prod_mod_B_array_.push_back(VALUE::modinv(temp, rns_base[i]));
            }
  
            return inv_punctured_prod_mod_B_array_;

        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::inv_prod_B_mod_m_sk() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> ibase = PrimePool::base_Bsk();
        Modulus obase = ibase[ibase.size() - 1];
        Data mult = 1;
        for(int i = 0; i < ibase.size() - 1; i++){
            mult = VALUE::mult(mult, ibase[i].value, obase);
        }

        //return modInverse(mult, obase.value);
        return VALUE::modinv(mult, obase);

    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Data> PrimePool::prod_B_mod_q() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
            case 4096:
            {

                std::vector<Modulus> ibase = PrimePool::base_Bsk();
                std::vector<Modulus> obase = PrimePool::base_modulus();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){
                        Data ibase_ = ibase[j].value % obase[i].value;
                        temp = VALUE::mult(temp, ibase_, obase[i]);
                        //VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;

            }
            case 8192:
            {

                std::vector<Modulus> ibase = PrimePool::base_Bsk();
                std::vector<Modulus> obase = PrimePool::base_modulus();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;

            }
            case 16384:
            {

                std::vector<Modulus> ibase = PrimePool::base_Bsk();
                std::vector<Modulus> obase = PrimePool::base_modulus();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;

            }
            case 32768:
            {

                std::vector<Modulus> ibase = PrimePool::base_Bsk();
                std::vector<Modulus> obase = PrimePool::base_modulus();
                Modulus m_tilde_ = PrimePool::m_tilde();
                std::vector<Data> prod_q_mod_Bsk_array_;

                for(int i = 0; i < obase.size()-1; i++){
                    Data temp = 1;
                    for (int j = 0; j < ibase.size()-1; j++){

                        temp = VALUE::mult(temp, ibase[j].value, obase[i]);
                        
                    }
                    prod_q_mod_Bsk_array_.push_back(temp);
                }

                return prod_q_mod_Bsk_array_;
            }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Modulus> PrimePool::q_Bsk_merge_modulus() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            //static unsigned long long prime_array[] = { 34359754753ULL, 34359771137ULL, // q (decomposition)
            // 2305843009213554689ULL, 2305843009213489153ULL, 2305843009213317121ULL, 2305843009213243393ULL }; // bsk
            static unsigned long long prime_array[] = { 34359754753ULL, 34359771137ULL, // q (decomposition)
             2305843009213554689ULL, 2305843009213489153ULL, 2305843009213317121ULL }; // bsk

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }

        case 8192:
        {
            static unsigned long long prime_array[] = { 9007199255019521ULL, 9007199255347201ULL, 18014398510645249ULL, // q (decomposition)
             2305843009213317121ULL, 2305843009213120513ULL, 2305843009212694529ULL, 2305843009212399617ULL, 2305843009211662337ULL }; // bsk

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        case 16384:
        {
            static unsigned long long prime_array[] = { 18014398510661633ULL, 18014398511382529ULL, 18014398512136193ULL, 18014398512365569ULL, 18014398514036737ULL, 18014398514200577ULL, 18014398514987009ULL, // q (decomposition)
            2305843009211662337ULL, 2305843009211596801ULL, 2305843009211400193ULL, 2305843009210580993ULL, 2305843009210515457ULL, 2305843009210023937ULL, 2305843009208713217ULL, 2305843009208123393ULL }; // bsk

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        case 32768:
        {
            static unsigned long long prime_array[] = { 144115188078673921ULL, 144115188079656961ULL, 144115188081819649ULL, 144115188082409473ULL,
                288230376154267649ULL, 288230376155185153ULL, 288230376155250689ULL, 288230376156758017ULL, 288230376157413377ULL, 288230376158396417ULL,
                288230376160755713ULL, 288230376161280001ULL, 288230376161673217ULL, 288230376161738753ULL, // q (decomposition)

                2305843009211662337ULL, 2305843009211596801ULL, 2305843009211400193ULL, 2305843009210023937ULL, 2305843009208713217ULL, 2305843009208123393ULL,
                 2305843009207468033ULL, 2305843009202159617ULL, 2305843009201242113ULL, 2305843009200586753ULL, 2305843009197506561ULL, 2305843009196916737ULL,
                  2305843009195868161ULL, 2305843009195671553ULL, 2305843009195343873ULL }; // bsk

            int size = sizeof(prime_array) / sizeof(prime_array[0]); 
            std::vector<Modulus> prime_vector;

            for(int i = 0; i < size; i++){
                prime_vector.push_back((Modulus)prime_array[i]);
            }
    
            return prime_vector;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}




std::vector<Root> PrimePool::q_Bsk_merge_ntt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            //static unsigned long long psi_array[] = {  6071469ULL, 18291550ULL, // q (decomp)
            //307554654119321ULL, 168273734192536ULL, 829315415491244ULL, 32973993658837ULL }; // bsk

            static unsigned long long psi_array[] = {  6071469ULL, 18291550ULL, // q (decomp)
            307554654119321ULL, 168273734192536ULL, 829315415491244ULL }; // bsk
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 8192:
        {
            static unsigned long long psi_array[] = { 1001816340395ULL, 97583054751ULL, 600254011689ULL, // q (decomp)
            562556498301090074ULL, 2128477325179182330ULL, 972182959695038317ULL, 1993670270211764767ULL, 1428717407974281805ULL }; // bsk
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = {26448496311ULL, 32910392359ULL, 317176533655ULL, 1958804412711ULL, 186605226681ULL, 223342371418ULL, 4102367446149ULL, // q (decomp)
            70072284713359ULL, 117297622845463ULL, 39472790483564ULL, 597089996664243ULL, 54890861537777ULL, 180991413543520ULL, 22745400076249ULL, 95920324194041ULL }; // bsk
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 6076061706634ULL, 757812206199ULL, 14332630180726ULL, 4325862285354ULL, 3986778017537ULL,
                17957119137197ULL, 6510836882592ULL, 8505645339603ULL, 20417392538974ULL, 15790315796150ULL, 9174649664700ULL, 3037638144297ULL,
                1412431483320ULL, 11383777697068ULL, // q (decomp)
             54086154900243ULL, 108184479186995ULL, 44627003565980ULL, 5466412987105ULL, 299243861837272ULL, 13621606365257ULL, 141711431679820ULL,
              302433821420420ULL, 30890933577633ULL, 1211291919640ULL, 61972381836971ULL, 68991921477839ULL, 44611420972577ULL, 38109723409384ULL, 7854697884062ULL}; // bsk
            
            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);

                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Root> PrimePool::q_Bsk_merge_intt_tables() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {
            //static unsigned long long psi_array[] = {  6071469ULL, 18291550ULL, // q (decomp)
            //307554654119321ULL, 168273734192536ULL, 829315415491244ULL, 32973993658837ULL }; // bsk

            static unsigned long long psi_array[] = {  6071469ULL, 18291550ULL, // q (decomp)
            307554654119321ULL, 168273734192536ULL, 829315415491244ULL }; // bsk

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        case 8192:
        {
 
            static unsigned long long psi_array[] = { 1001816340395ULL, 97583054751ULL, 600254011689ULL, // q (decomp)
            562556498301090074ULL, 2128477325179182330ULL, 972182959695038317ULL, 1993670270211764767ULL, 1428717407974281805ULL }; // bsk

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 16384:
        {
            static unsigned long long psi_array[] = {26448496311ULL, 32910392359ULL, 317176533655ULL, 1958804412711ULL, 186605226681ULL, 223342371418ULL, 4102367446149ULL, // q (decomp)
            70072284713359ULL, 117297622845463ULL, 39472790483564ULL, 597089996664243ULL, 54890861537777ULL, 180991413543520ULL, 22745400076249ULL, 95920324194041ULL }; // bsk

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;

        }
        case 32768:
        {
            static unsigned long long psi_array[] = { 6076061706634ULL, 757812206199ULL, 14332630180726ULL, 4325862285354ULL, 3986778017537ULL,
                17957119137197ULL, 6510836882592ULL, 8505645339603ULL, 20417392538974ULL, 15790315796150ULL, 9174649664700ULL, 3037638144297ULL,
                1412431483320ULL, 11383777697068ULL, // q (decomp)
             54086154900243ULL, 108184479186995ULL, 44627003565980ULL, 5466412987105ULL, 299243861837272ULL, 13621606365257ULL, 141711431679820ULL,
              302433821420420ULL, 30890933577633ULL, 1211291919640ULL, 61972381836971ULL, 68991921477839ULL, 44611420972577ULL, 38109723409384ULL, 7854697884062ULL}; // bsk

            int size = sizeof(psi_array) / sizeof(psi_array[0]); 
            int lg = log2(n);
            std::vector<Root> forward_table; // bit reverse order

            for(int i = 0; i < size; i++){
                std::vector<Root> table;
                table.push_back(1);
                Data inv_root = VALUE::modinv(psi_array[i], PrimePool::q_Bsk_merge_modulus()[i]);
                for (int j = 1; j < n; j++)
                {
                    Data exp = VALUE::mult(table[(j - 1)], inv_root, PrimePool::q_Bsk_merge_modulus()[i]);
                    table.push_back(exp);
                }

                for (int j = 0; j < n; j++) // take bit reverse order
                {
                    forward_table.push_back(table[bitreverse(j, lg)]);
                }

            }

            return forward_table;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}



std::vector<Ninverse> PrimePool::q_Bsk_n_inverse() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {   
            std::vector<Modulus> modulus_ = PrimePool::q_Bsk_merge_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 8192:
        {
            std::vector<Modulus> modulus_ = PrimePool::q_Bsk_merge_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 16384:
        {
            std::vector<Modulus> modulus_ = PrimePool::q_Bsk_merge_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        case 32768:
        {
            std::vector<Modulus> modulus_ = PrimePool::q_Bsk_merge_modulus();
            Data n_ = n;
            std::vector<Ninverse> n_inverse_;
            for (int i = 0; i < modulus_.size(); i++)
            {
                n_inverse_.push_back(VALUE::modinv(n_, modulus_[i]));
            }
            
            return n_inverse_;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::half() {

    std::vector<Modulus> modulus_ = PrimePool::base_modulus();
     
    return modulus_[modulus_.size() - 1].value >> 1;

}

std::vector<Data> PrimePool::half_mod(){

    std::vector<Modulus> modulus_ = PrimePool::base_modulus();
    Data half_ = modulus_[modulus_.size() - 1].value >> 1;

    std::vector<Data> half_mod_;
    for(int i = 0; i < modulus_.size() - 1; i++){
        half_mod_.push_back(half_ % modulus_[i].value);
    }
    
    return half_mod_;

}

std::vector<Data> PrimePool::factor(){

    std::vector<Modulus> modulus_ = PrimePool::base_modulus();
    Data back_value = modulus_[modulus_.size() - 1].value;

    std::vector<Data> factor_;
    for(int i = 0; i < modulus_.size() - 1; i++){
        factor_.push_back(back_value % modulus_[i].value);
    }
    
    return factor_;

}

Modulus PrimePool::gamma() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return (Modulus)2305843009213145089ULL;
    case 8192: return (Modulus)2305843009211400193ULL;
    case 16384: return (Modulus)2305843009211596801ULL;
    case 32768: return (Modulus)2305843009211596801ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::Q_mod_t() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return 238537ULL;
    case 8192: return 114198ULL;
    case 16384: return 151972ULL;
    case 32768: return 108147ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Data> PrimePool::coeeff_div_plainmod() {
    if (sec == security_level::HES_128) {
        switch (n)
        {
        case 4096:
        {   
            static unsigned long long coeeff_div_plainmod_array[] = { 4345862704, 3800472112 }; 

            int size = sizeof(coeeff_div_plainmod_array) / sizeof(coeeff_div_plainmod_array[0]); 
            int lg = log2(n);
            std::vector<Data> array_; 

            for(int i = 0; i < size; i++){
                array_.push_back(coeeff_div_plainmod_array[i]);
            }

            return array_;
        }
        case 8192:
        {
            static unsigned long long coeeff_div_plainmod_array[] = { 5541262720954987, 372437387405474, 3610984665704993 }; 

            int size = sizeof(coeeff_div_plainmod_array) / sizeof(coeeff_div_plainmod_array[0]); 
            int lg = log2(n);
            std::vector<Data> array_; 

            for(int i = 0; i < size; i++){
                array_.push_back(coeeff_div_plainmod_array[i]);
            }

            return array_;
        }
        case 16384:
        {
            static unsigned long long coeeff_div_plainmod_array[] = { 7183970776545138, 9538434495154420, 5472079162200964, 2414043429030407, 15048148978250887, 1958067374324805, 13459127632045453 }; 

            int size = sizeof(coeeff_div_plainmod_array) / sizeof(coeeff_div_plainmod_array[0]); 
            int lg = log2(n);
            std::vector<Data> array_; 

            for(int i = 0; i < size; i++){
                array_.push_back(coeeff_div_plainmod_array[i]);
            }

            return array_;
        }
        case 32768:
        {
            static unsigned long long coeeff_div_plainmod_array[] = { 130745510188956816, 50779047441896441, 128432873675295556, 44274894678919508, 245181985138183869, 51354457794706661, 58134037870986535, 72012422493919506, 160359168960562863, 80186181541933104, 127309357470951023, 81055894134438261, 163957865944788234, 144039138623314110 }; 

            int size = sizeof(coeeff_div_plainmod_array) / sizeof(coeeff_div_plainmod_array[0]); 
            int lg = log2(n);
            std::vector<Data> array_; 

            for(int i = 0; i < size; i++){
                array_.push_back(coeeff_div_plainmod_array[i]);
            }

            return array_;
        }
        }
    }

    throw std::runtime_error("Security Level is not supported!");
}


Data PrimePool::upper_threshold() {
if (sec == security_level::HES_128) {
    switch (n)
    {
    case 4096: return 516097ULL;
    case 8192: return 516097ULL;
    case 16384: return 393217ULL;
    case 32768: return 393217ULL;
    }
}

    throw std::runtime_error("Security Level is not supported!");
}




std::vector<Data> PrimePool::Qi_t() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();
        Modulus plain_mod_ = PrimePool::plain_modulus();
        std::vector<Data> Qi_t_;

        for(int i = 0; i < modulus_.size()-1; i++){
            Data temp = 1;
            for (int j = 0; j < modulus_.size()-1; j++){
                if(i != j){
                    Data mod_ = modulus_[j].value % plain_mod_.value;
                    temp = VALUE::mult(temp, mod_, plain_mod_);
                }                        
            }
            Qi_t_.push_back(temp);
        }

        return Qi_t_;
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Data> PrimePool::Qi_gamma() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();
        Modulus gamma_ = PrimePool::gamma();
        std::vector<Data> Qi_gamma_;

        for(int i = 0; i < modulus_.size()-1; i++){
            Data temp = 1;
            for (int j = 0; j < modulus_.size()-1; j++){
                if(i != j){
                    Data mod_ = modulus_[j].value % gamma_.value;
                    temp = VALUE::mult(temp, mod_, gamma_);
                }                        
            }
            Qi_gamma_.push_back(temp);
        }

        return Qi_gamma_;
    }

    throw std::runtime_error("Security Level is not supported!");
}

std::vector<Data> PrimePool::Qi_inverse() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();

        std::vector<Data> Qi_inverse_;

        for(int i = 0; i < modulus_.size()-1; i++){
            Data temp = 1;
            for (int j = 0; j < modulus_.size()-1; j++){
                if(i != j){
                    Data mod_ = modulus_[j].value % modulus_[i].value;
                    Data inv_ = VALUE::modinv(mod_, modulus_[i]);
                    temp = VALUE::mult(temp, inv_, modulus_[i]);
                }                        
            }
            Qi_inverse_.push_back(temp);
        }

        return Qi_inverse_;
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::mulq_inv_t() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();
        Modulus plain_mod_ = PrimePool::plain_modulus();

        Data mulq_inv_t = 1;

        for(int i = 0; i < modulus_.size()-1; i++){
        
            Data mod_ = modulus_[i].value % plain_mod_.value;
            Data inv_ = VALUE::modinv(mod_, plain_mod_);
            mulq_inv_t = VALUE::mult(mulq_inv_t, inv_, plain_mod_);
                                      
        }

        return plain_mod_.value - mulq_inv_t;
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::mulq_inv_gamma() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();
        Modulus gamma_ = PrimePool::gamma();

        Data mulq_inv_gamma = 1;

        for(int i = 0; i < modulus_.size()-1; i++){
        
            Data mod_ = modulus_[i].value % gamma_.value;
            Data inv_ = VALUE::modinv(mod_, gamma_);
            mulq_inv_gamma = VALUE::mult(mulq_inv_gamma, inv_, gamma_);
                                      
        }

        return gamma_.value - mulq_inv_gamma;
    }

    throw std::runtime_error("Security Level is not supported!");
}

Data PrimePool::inv_gamma() {
    if (sec == security_level::HES_128) {

        Modulus gamma_ = PrimePool::gamma();
        Modulus plain_mod_ = PrimePool::plain_modulus();

        Data mod_ = gamma_.value % plain_mod_.value;
        Data inv_gamma = VALUE::modinv(mod_, plain_mod_);
        
        return inv_gamma;
    }

    throw std::runtime_error("Security Level is not supported!");
}


std::vector<Data> PrimePool::upper_halfincrement() {
    if (sec == security_level::HES_128) {

        std::vector<Modulus> modulus_ = PrimePool::base_modulus();

        Modulus plain_modulus_ = PrimePool::plain_modulus();

        std::vector<Data> upper_halfincrement_;

        for(int i = 0; i < modulus_.size()-1; i++){
            upper_halfincrement_.push_back(modulus_[i].value - plain_modulus_.value);
        }

        return upper_halfincrement_;
    }

    throw std::runtime_error("Security Level is not supported!");
}