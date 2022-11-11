#pragma once

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

enum class security_level : int
{
    //128-bit classical security level according to HomomorphicEncryption.org standard.
    HES_128 = 128,

    //192-bit classical security level according to HomomorphicEncryption.org standard.
    HES_192 = 192,

    //256-bit classical security level according to HomomorphicEncryption.org standard.
    HES_256 = 256
};


class Prime_Pool {

public:
    int n;
    security_level sec;

    Prime_Pool(int poly_degree, security_level sec_level) {
        n = poly_degree;
        sec = sec_level;
    }

    int Prime_Count() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                return 3;
            }
            case 8192:
            {
                return 4;
            }
            case 16384:
            {
                return 8;
            }
            case 32768:
            {
                return 15;
            }
            }
        }
    }

    int Total_Primes_Bits() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                int bits_array = 109; // Bit
                return bits_array;
            }
            case 8192:
            {
                int bits_array = 218; // Bit
                return bits_array;
            }
            case 16384:
            {
                int bits_array = 438; // Bit
                return bits_array;
            }
            case 32768:
            {
                int bits_array = 881; // 881 Bit
                return bits_array;
            }
            }
        }
    }

    unsigned long long* Primes() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long prime_array[] = { 34359754753, 34359771137, 68719484929 };
                return prime_array;
            }

            case 8192:
            {
                unsigned long long prime_array[] = { 9007199255019521, 9007199255347201, 18014398510645249, 18014398510661633 };
                return prime_array;
            }
            case 16384:
            {
                //unsigned long long prime_array[] = { 9007199255560193, 9007199255658497, 18014398510661633, 18014398511382529,
                //    18014398512136193, 18014398512365569, 18014398514036737, 18014398514200577 };
                unsigned long long prime_array[] = { 18014398510661633, 18014398511382529, 18014398512136193, 18014398512365569,
                    18014398514036737, 18014398514200577, 18014398514987009, 18014398515511297 };
                return prime_array;
            }
            case 32768:
            {
                unsigned long long prime_array[] = { 144115188078673921, 144115188079656961, 144115188081819649, 144115188082409473,
                    288230376154267649, 288230376155185153, 288230376155250689, 288230376156758017, 288230376157413377, 288230376158396417,
                    288230376160755713, 288230376161280001, 288230376161673217, 288230376161738753, 288230376162459649 };
                return prime_array;
            }
            }
        }
    }

    unsigned long long* Primes_double() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Primes_double_array[] = { 34359754753, 34359754753, 34359771137, 34359771137, 68719484929, 68719484929 };
                return Primes_double_array;
            }
            case 8192:
            {
                unsigned long long Primes_double_array[] = { 9007199255019521,9007199255019521,9007199255347201,9007199255347201,18014398510645249,
                    18014398510645249,18014398510661633,18014398510661633 };
                return Primes_double_array;
            }
            case 16384:
            {
                //unsigned long long Primes_double_array[] = { 9007199255560193,9007199255560193,9007199255658497,9007199255658497,18014398510661633,18014398510661633,
                //    18014398511382529,18014398511382529,18014398512136193,18014398512136193,18014398512365569,18014398512365569,18014398514036737,18014398514036737,
                //    18014398514200577,18014398514200577 };
                unsigned long long Primes_double_array[] = { 18014398510661633, 18014398510661633, 18014398511382529, 18014398511382529, 18014398512136193, 18014398512136193,
                    18014398512365569, 18014398512365569, 18014398514036737, 18014398514036737, 18014398514200577, 18014398514200577, 18014398514987009, 18014398514987009,
                    18014398515511297, 18014398515511297 };
                return Primes_double_array;
            }
            case 32768:
            {
                unsigned long long Primes_double_array[] = { 144115188078673921,144115188078673921,144115188079656961,144115188079656961,144115188081819649,144115188081819649,
                    144115188082409473,144115188082409473,288230376154267649,288230376154267649,288230376155185153,288230376155185153,288230376155250689,288230376155250689,288230376156758017,
                    288230376156758017,288230376157413377,288230376157413377,288230376158396417,288230376158396417,288230376160755713,288230376160755713,288230376161280001,288230376161280001,
                    288230376161673217,288230376161673217,288230376161738753,288230376161738753,288230376162459649,288230376162459649 };
                return Primes_double_array;
            }
            }
        }
    }

    unsigned long long* Primes_Mu() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long mu_array[] = { 137438887932, 137438822396, 274877874172 };
                return mu_array;
            }
            case 8192:
            {
                unsigned long long mu_array[] = { 36028797017849852, 36028797016539132, 72057594033274876, 72057594033209340 };
                return mu_array;
            }
            case 16384:
            {
                //unsigned long long mu_array[] = { 36028797015687164,36028797015293948,72057594033209340,72057594030325756,
                //    72057594027311100,72057594026393596,72057594019708924,72057594019053564 };
                unsigned long long mu_array[] = { 72057594033209340, 72057594030325756, 72057594027311100, 72057594026393596, 72057594019708924, 72057594019053564,
                    72057594015907836, 72057594013810684 };
                return mu_array;
            }
            case 32768:
            {
                unsigned long long mu_array[] = { 576460752292151292,576460752288219132,576460752279568380,576460752277209084,
                    1152921504596623356,1152921504592953340,1152921504592691196,1152921504586661884,1152921504584040444,1152921504580108284,1152921504570671100,
                    1152921504568573948,1152921504567001084,1152921504566738940,1152921504563855356
                };
                return mu_array;
            }
            }
        }
    }

    unsigned long long* Primes_Mu_double() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Primes_Mu_double_array[] = { 137438887932, 137438887932, 137438822396, 137438822396, 274877874172, 274877874172 };
                return Primes_Mu_double_array;
            }
            case 8192:
            {
                unsigned long long Primes_Mu_double_array[] = { 36028797017849852,36028797017849852,36028797016539132,36028797016539132,72057594033274876,
                    72057594033274876,72057594033209340,72057594033209340 };
                return Primes_Mu_double_array;
            }
            case 16384:
            {
                //unsigned long long Primes_Mu_double_array[] = { 36028797015687164,36028797015687164,36028797015293948,36028797015293948,72057594033209340,72057594033209340,
                //    72057594030325756,72057594030325756,72057594027311100,72057594027311100,72057594026393596,72057594026393596,72057594019708924,72057594019708924,
                //    72057594019053564,72057594019053564 };
                unsigned long long Primes_Mu_double_array[] = { 72057594033209340, 72057594033209340, 72057594030325756, 72057594030325756, 72057594027311100, 72057594027311100,
                    72057594026393596, 72057594026393596, 72057594019708924, 72057594019708924, 72057594019053564, 72057594019053564, 72057594015907836, 72057594015907836,
                    72057594013810684, 72057594013810684 };
                return Primes_Mu_double_array;
            }
            case 32768:
            {
                unsigned long long Primes_Mu_double_array[] = { 576460752292151292,576460752292151292,576460752288219132,576460752288219132,576460752279568380,576460752279568380,
                    576460752277209084,576460752277209084,1152921504596623356,1152921504596623356,1152921504592953340,1152921504592953340,1152921504592691196,1152921504592691196,
                    1152921504586661884,1152921504586661884,1152921504584040444,1152921504584040444,1152921504580108284,1152921504580108284,1152921504570671100,1152921504570671100,
                    1152921504568573948,1152921504568573948,1152921504567001084,1152921504567001084,1152921504566738940,1152921504566738940,1152921504563855356,1152921504563855356 };
                return Primes_Mu_double_array;
            }
            }
        }
    }

    unsigned long long* Primes_Bits() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long bits_array[] = { 36, 36, 37 }; // 109 Bit
                return bits_array;
            }
            case 8192:
            {
                unsigned long long bits_array[] = { 54, 54, 55, 55 }; // 218 Bit
                return bits_array;
            }
            case 16384:
            {
                //unsigned long long bits_array[] = { 54, 54, 55, 55, 55, 55, 55, 55 }; // 438 Bit
                unsigned long long bits_array[] = { 55, 55, 55, 55, 55, 55, 55, 55 }; // 438 Bit
                return bits_array;
            }
            case 32768:
            {
                unsigned long long bits_array[] = { 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59 }; // 881 Bit
                return bits_array;
            }
            }
        }
    }

    unsigned long long* Primes_Bits_double() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Primes_Bits_double_array[] = { 36, 36, 36, 36, 37, 37 };
                return Primes_Bits_double_array;
            }
            case 8192:
            {
                unsigned long long Primes_Bits_double_array[] = { 54, 54, 54, 54, 55, 55, 55, 55 };
                return Primes_Bits_double_array;
            }
            case 16384:
            {
                //unsigned long long Primes_Bits_double_array[] = { 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55 };
                unsigned long long Primes_Bits_double_array[] = { 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55 };
                return Primes_Bits_double_array;
            }
            case 32768:
            {
                unsigned long long Primes_Bits_double_array[] = { 58,58,58,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59 };
                return Primes_Bits_double_array;
            }
            }
        }
    }

    unsigned long long* Primes_Psi() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long psi_array[] = { 6071469, 18291550, 28979647 };
                return psi_array;
            }
            case 8192:
            {
                unsigned long long psi_array[] = { 1001816340395, 97583054751, 600254011689, 305447760175 };
                return psi_array;
            }
            case 16384:
            {
                //unsigned long long psi_array[] = { 523433461118,785944929279,26448496311,32910392359,317176533655,1958804412711,
                //    186605226681, 223342371418 };
                unsigned long long psi_array[] = { 26448496311, 32910392359, 317176533655, 1958804412711, 186605226681, 223342371418, 4102367446149, 117039768478 };
                return psi_array;
            }
            case 32768:
            {
                unsigned long long psi_array[] = { 6076061706634, 757812206199, 14332630180726, 4325862285354, 3986778017537,
                    17957119137197, 6510836882592, 8505645339603, 20417392538974, 15790315796150, 9174649664700, 3037638144297,
                    1412431483320, 11383777697068, 4139725055370 };
                return psi_array;
            }
            }
        }
    }

    unsigned long long* Last_q_modinv() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Last_q_modinv_array[] = { 23551467981, 14762508349 };
                return Last_q_modinv_array;
            }
            case 8192:
            {
                unsigned long long Last_q_modinv_array[] = { 8744053871855373, 5121094001232070, 18013298999017402 };
                return Last_q_modinv_array;
            }
            case 16384:
            {

                //unsigned long long Last_q_modinv_array[] = { 7096753318393859, 5630138704232243, 6671994358061587, 9216662613314155, 16584675618223575,
                //    5146961186464913, 3602769751644542 };
                unsigned long long Last_q_modinv_array[] = { 6816255181359713, 6862623641194821, 7345671725766079, 4503593901468330, 8406707089754610, 7205745661784880,
                    9007164897755126 };
                return Last_q_modinv_array;
            }
            case 32768:
            {
                unsigned long long Last_q_modinv_array[] = { 86216811529655848, 7235888322811655, 56133563886076630, 137152533650358975, 73786941111120429,
                    257070295867719090, 188659842592104895, 39755863400397529, 269514060847886389, 92977469760022857, 88686100432289787, 192153339771602709,
                    288230009657797279, 157216168993083751 };
                return Last_q_modinv_array;
            }
            }
        }
    }

    unsigned long long* n_inverse() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                /*
                unsigned long long n_inverse_array[] = { 34355560447, 34355576829, 68711096320 };
                */
                unsigned long long n_inverse_array[] = { 34351366141, 34351382521, 68702707711 };
                return n_inverse_array;
            }
            case 8192:
            {
                unsigned long long n_inverse_array[] = { 9006099743391711, 9006099743719351, 18012199487389555, 18012199487405937 };
                return n_inverse_array;
            }
            case 16384:
            {
                /*
                unsigned long long n_inverse_array[] = { 9006099743932317, 9006099744030609, 18012199487405937, 18012199488126745, 18012199488880317,
                    18012199489109665, 18012199490780629, 18012199490944449 };

                unsigned long long n_inverse_array[] = { 9006649499746255, 9006649499844553, 18013298999033785, 18013298999754637, 18013299000508255,
                    18013299000737617, 18013299002408683, 18013299002572513 };
                */
                unsigned long long n_inverse_array[] = { 18013298999033785,18013298999754637,18013299000508255,18013299000737617,18013299002408683,18013299002572513,
                    18013299003358897,18013299003883153 };
                return n_inverse_array;
            }
            case 32768:
            {
                /*
                unsigned long long n_inverse_array[] = { 144097595892629161, 144097595893612081, 144097595895774505, 144097595896364257, 288195191782178505,
                    288195191783095897, 288195191783161425, 288195191784668569, 288195191785323849, 288195191786306769, 288195191788665777, 288195191789190001,
                    288195191789583169, 288195191789648697, 288195191790369505 };
                */
                unsigned long long n_inverse_array[] = { 144110790032162731, 144110790033145741, 144110790035308363, 144110790035898169, 288221580061245363,
                    288221580062162839, 288221580062228373, 288221580063735655, 288221580064390995, 288221580065374005, 288221580067733229, 288221580068257501,
                    288221580068650705, 288221580068716239, 288221580069437113 };
                return n_inverse_array;
            }
            }
        }
    }

    unsigned long long* n_inverse_double() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                //unsigned long long n_inverse_double_array[] = { 34355560447, 34355560447, 34355576829, 34355576829, 68711096320, 68711096320 };
                unsigned long long n_inverse_double_array[] = { 34351366141, 34351366141, 34351382521, 34351382521, 68702707711, 68702707711 };
                return n_inverse_double_array;
            }
            case 8192:
            {
                unsigned long long n_inverse_double_array[] = { 9006099743391711, 9006099743391711, 9006099743719351, 9006099743719351, 18012199487389555,
                18012199487389555, 18012199487405937, 18012199487405937 };
                return n_inverse_double_array;
            }
            case 16384:
            {
                //unsigned long long n_inverse_double_array[] = { 9006099743932317, 9006099743932317, 9006099744030609, 9006099744030609, 18012199487405937, 18012199487405937,
                //    18012199488126745, 18012199488126745, 18012199488880317, 18012199488880317, 18012199489109665, 18012199489109665, 18012199490780629, 18012199490780629,
                //    18012199490944449, 18012199490944449 };
                unsigned long long n_inverse_double_array[] = { 18013298999033785, 18013298999033785, 18013298999754637, 18013298999754637, 18013299000508255, 18013299000508255,
                    18013299000737617,  18013299000737617, 18013299002408683, 18013299002408683, 18013299002572513,  18013299002572513, 18013299003358897, 18013299003358897,
                    18013299003883153, 18013299003883153 };
                return n_inverse_double_array;
            }
            case 32768:
            {
                /*
                unsigned long long n_inverse_double_array[] = { 144097595892629161, 144097595892629161, 144097595893612081, 144097595893612081, 144097595895774505,
                    144097595895774505, 144097595896364257, 144097595896364257, 288195191782178505, 288195191782178505, 288195191783095897, 288195191783095897,
                    288195191783161425, 288195191783161425, 288195191784668569, 288195191784668569, 288195191785323849, 288195191785323849, 288195191786306769,
                    288195191786306769, 288195191788665777, 288195191788665777, 288195191789190001, 288195191789190001, 288195191789583169, 288195191789583169,
                    288195191789648697, 288195191789648697, 288195191790369505, 288195191790369505 };
                    */
                unsigned long long n_inverse_double_array[] = { 144110790032162731, 144110790032162731, 144110790033145741, 144110790033145741, 144110790035308363, 144110790035308363,
                    144110790035898169, 144110790035898169, 288221580061245363, 288221580061245363, 288221580062162839, 288221580062162839, 288221580062228373, 288221580062228373,
                    288221580063735655, 288221580063735655, 288221580064390995, 288221580064390995, 288221580065374005, 288221580065374005, 288221580067733229,  288221580067733229,
                    288221580068257501, 288221580068257501, 288221580068650705, 288221580068650705, 288221580068716239, 288221580068716239, 288221580069437113,  288221580069437113 };
                return n_inverse_double_array;
            }
            }
        }
    }

    unsigned long long* n_plain_inverse() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long n_plain_inverse_array[] = { 1031941 };
                return n_plain_inverse_array;
            }
            case 8192:
            {
                unsigned long long n_plain_inverse_array[] = { 1032067 };
                return n_plain_inverse_array;
            }
            case 16384:
            {
                unsigned long long n_plain_inverse_array[] = { 786385 };
                return n_plain_inverse_array;
            }
            case 32768:
            {
                unsigned long long n_plain_inverse_array[] = { 786409 };
                return n_plain_inverse_array;
            }
            }
        }
    }

    unsigned long long* Half() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Half_array[] = { 34359742464 };
                return Half_array;
            }
            case 8192:
            {
                unsigned long long Half_array[] = { 9007199255330816 };
                return Half_array;
            }
            case 16384:
            {
                //unsigned long long Half_array[] = { 9007199257100288 };
                unsigned long long Half_array[] = { 9007199257755648 };
                return Half_array;
            }
            case 32768:
            {
                unsigned long long Half_array[] = { 144115188081229824 };
                return Half_array;
            }
            }
        }
    }

    unsigned long long* Halfmod() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Halfmod_array[] = { 34359742464, 34359742464 };
                return Halfmod_array;
            }
            case 8192:
            {
                unsigned long long Halfmod_array[] = { 311295, 9007199255330816, 9007199255330816 };
                return Halfmod_array;
            }
            case 16384:
            {
                //unsigned long long Halfmod_array[] = { 1540095, 1441791, 9007199257100288, 9007199257100288, 9007199257100288, 9007199257100288, 9007199257100288 };
                unsigned long long Halfmod_array[] = { 9007199257755648, 9007199257755648, 9007199257755648, 9007199257755648, 9007199257755648, 9007199257755648, 9007199257755648 };
                return Halfmod_array;
            }
            case 32768:
            {
                unsigned long long Halfmod_array[] = { 2555903, 1572863, 144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824,
                    144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824, 144115188081229824 };
                return Halfmod_array;
            }
            }
        }
    }

    unsigned long long* Auxiliary_Bases() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Auxiliary_Bases_array[] = { 2305843009213317121, 2305843009213243393, 2305843009213554689 };
                return Auxiliary_Bases_array;
            }
            case 8192:
            {
                unsigned long long Auxiliary_Bases_array[] = { 2305843009212694529, 2305843009212399617, 2305843009211662337, 2305843009213317121 };
                return Auxiliary_Bases_array;
            }
            case 16384:
            {
                unsigned long long Auxiliary_Bases_array[] = { 2305843009211400193, 2305843009210580993, 2305843009210515457, 2305843009210023937,
                    2305843009208713217, 2305843009208123393, 2305843009207468033, 2305843009211662337 };

                return Auxiliary_Bases_array;
            }
            case 32768:
            {
                unsigned long long Auxiliary_Bases_array[] = { 2305843009211400193, 2305843009210023937, 2305843009208713217, 2305843009208123393,
                    2305843009207468033, 2305843009202159617, 2305843009201242113, 2305843009200586753, 2305843009197506561, 2305843009196916737,
                    2305843009195868161, 2305843009195671553, 2305843009195343873, 2305843009191936001, 2305843009211662337 };
                return Auxiliary_Bases_array;
            }
            }
        }
    }

    unsigned long long* Auxiliary_Bases_mu() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Auxiliary_Bases_mu_array[] = { 2305843009214070783,2305843009214144511,2305843009213833215 };
                return Auxiliary_Bases_mu_array;
            }
            case 8192:
            {
                unsigned long long Auxiliary_Bases_mu_array[] = { 2305843009214693375,2305843009214988287,2305843009215725567,2305843009214070783 };
                return Auxiliary_Bases_mu_array;
            }
            case 16384:
            {
                unsigned long long Auxiliary_Bases_mu_array[] = { 2305843009215987711, 2305843009216806911, 2305843009216872447, 2305843009217363967, 2305843009218674687,
                2305843009219264511, 2305843009219919871, 2305843009215725567 };
                return Auxiliary_Bases_mu_array;
            }
            case 32768:
            {
                unsigned long long Auxiliary_Bases_mu_array[] = { 2305843009215987711,2305843009217363967,2305843009218674687,2305843009219264511,2305843009219919871,
                2305843009225228287,2305843009226145791,2305843009226801151,2305843009229881343,2305843009230471167,2305843009231519743,2305843009231716351,2305843009232044031,
                2305843009235451903,2305843009215725567 };
                return Auxiliary_Bases_mu_array;
            }
            }
        }
    }

    unsigned long long* Auxiliary_Bases_bit() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Auxiliary_Bases_bit_array[] = { 61, 61, 61 };
                return Auxiliary_Bases_bit_array;
            }
            case 8192:
            {
                unsigned long long Auxiliary_Bases_bit_array[] = { 61, 61, 61, 61 };
                return Auxiliary_Bases_bit_array;
            }
            case 16384:
            {
                unsigned long long Auxiliary_Bases_bit_array[] = { 61, 61, 61, 61, 61, 61, 61, 61 };
                return Auxiliary_Bases_bit_array;
            }
            case 32768:
            {
                unsigned long long Auxiliary_Bases_bit_array[] = { 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61 };
                return Auxiliary_Bases_bit_array;
            }
            }
        }
    }

    unsigned long long* Auxiliary_Bases_inverse() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                //unsigned long long Auxiliary_Bases_inverse_array[] = { 2305561534236606511, 2305561534236532792, 2305561534236844050 };
                unsigned long long Auxiliary_Bases_inverse_array[] = { 2305280059259895901, 2305280059259822191, 2305280059260133411 };
                return Auxiliary_Bases_inverse_array;
            }
            case 8192:
            {
                unsigned long long Auxiliary_Bases_inverse_array[] = { 2305561534235983995, 2305561534235689119, 2305561534234951929, 2305561534236606511 };
                return Auxiliary_Bases_inverse_array;
            }
            case 16384:
            {
                //unsigned long long Auxiliary_Bases_inverse_array[] = { 2305561534234689817, 2305561534233870717, 2305561534233805189, 2305561534233313729,
                //    2305561534232003169, 2305561534231413417, 2305561534230758137, 2305561534234951929 };
                unsigned long long Auxiliary_Bases_inverse_array[] = { 2305702271723045005,2305702271722225855,2305702271722160323,2305702271721668833,2305702271720358193,
                    2305702271719768405,2305702271719113085,2305702271723307133 };
                return Auxiliary_Bases_inverse_array;
            }
            case 32768:
            {
                unsigned long long Auxiliary_Bases_inverse_array[] = { 2305772640467222599, 2305772640465846385, 2305772640464535705, 2305772640463945899, 2305772640463290559,
                    2305772640457982305, 2305772640457064829, 2305772640456409489, 2305772640453329391, 2305772640452739585, 2305772640451691041, 2305772640451494439,
                    2305772640451166769, 2305772640447759001, 2305772640467484735 };
                return Auxiliary_Bases_inverse_array;
            }
            }
        }
    }

    unsigned long long* Auxiliary_Bases_Psi() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Auxiliary_Bases_Psi_array[] = { 829315415491244, 32973993658837, 307554654119321 };
                return Auxiliary_Bases_Psi_array;
            }
            case 8192:
            {
                unsigned long long Auxiliary_Bases_Psi_array[] = { 153148382944507, 50542844763732, 171881840328130, 437651905986530 };
                return Auxiliary_Bases_Psi_array;
            }
            case 16384:
            {
                unsigned long long Auxiliary_Bases_Psi_array[] = { 39472790483564, 597089996664243, 54890861537777, 180991413543520, 22745400076249,
                    95920324194041, 179380723850, 70072284713359 };
                return Auxiliary_Bases_Psi_array;
            }
            case 32768:
            {
                unsigned long long Auxiliary_Bases_Psi_array[] = { 44627003565980, 5466412987105, 299243861837272, 13621606365257, 141711431679820,
                    302433821420420, 30890933577633, 1211291919640, 61972381836971, 68991921477839, 44611420972577, 38109723409384, 7854697884062,
                    17320401886454, 54086154900243 };
                return Auxiliary_Bases_Psi_array;
            }
            }
        }
    }

    unsigned long long* Plain_Modulus() { // Plain, Plain_Mu, Plain_Bit, Plain_Psi
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Plain_Modulus_array[] = { 1032193, 1065219, 20, 194 };
                return Plain_Modulus_array;
            }
            case 8192:
            {
                unsigned long long Plain_Modulus_array[] = { 1032193, 1065219, 20, 94 };
                return Plain_Modulus_array;
            }
            case 16384:
            {
                unsigned long long Plain_Modulus_array[] = { 786433, 1398099, 20, 9 };
                return Plain_Modulus_array;
            }
            case 32768:
            {
                unsigned long long Plain_Modulus_array[] = { 786433, 1398099, 20, 3 };
                return Plain_Modulus_array;
            }
            }
        }
    }

    unsigned long long* M_SK() { // M_SK, M_SK_Mu, M_SK_Bit
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long M_SK_array[] = { 2305843009213554689, 2305843009213833215, 61 };
                return M_SK_array;
            }
            case 8192:
            {
                unsigned long long M_SK_array[] = { 2305843009213317121, 2305843009214070783, 61 };
                return M_SK_array;
            }
            case 16384:
            {
                unsigned long long M_SK_array[] = { 2305843009211662337, 2305843009215725567, 61 };
                return M_SK_array;
            }
            case 32768:
            {
                unsigned long long M_SK_array[] = { 2305843009211662337, 2305843009215725567, 61 };
                return M_SK_array;
            }
            }
        }
    }

    unsigned long long* M_Tilde() { // M_Tilde, M_Tilde_Mu, M_Tilde_Bit
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long M_Tilde_array[] = { 4294967296, 17179869184, 33 };
                return M_Tilde_array;
            }
            case 8192:
            {
                unsigned long long M_Tilde_array[] = { 4294967296, 17179869184, 33 };
                return M_Tilde_array;
            }
            case 16384:
            {
                unsigned long long M_Tilde_array[] = { 4294967296, 17179869184, 33 };
                return M_Tilde_array;
            }
            case 32768:
            {
                unsigned long long M_Tilde_array[] = { 4294967296, 17179869184, 33 };
                return M_Tilde_array;
            }
            }
        }
    }

    unsigned long long* Gamma() { // Gamma, Gamma_Mu, Gamma_Bit
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Gamma_array[] = { 2305843009213489153, 2305843009213898751, 61 };
                return Gamma_array;
            }
            case 8192:
            {
                unsigned long long Gamma_array[] = { 2305843009213120513, 2305843009214267391, 61 };
                return Gamma_array;
            }
            case 16384:
            {
                unsigned long long Gamma_array[] = { 2305843009211596801, 2305843009215791103, 61 };
                return Gamma_array;
            }
            case 32768:
            {
                unsigned long long Gamma_array[] = { 2305843009211596801, 2305843009215791103, 61 };
                return Gamma_array;
            }
            }
        }
    }

    unsigned long long* Q_mod_t() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Q_mod_t_array[] = { 238537 };
                return Q_mod_t_array;
            }
            case 8192:
            {
                unsigned long long Q_mod_t_array[] = { 114198 };
                return Q_mod_t_array;
            }
            case 16384:
            {
                //unsigned long long Q_mod_t_array[] = { 119947 };
                unsigned long long Q_mod_t_array[] = { 151972 };
                return Q_mod_t_array;
            }
            case 32768:
            {
                unsigned long long Q_mod_t_array[] = { 108147 };
                return Q_mod_t_array;
            }
            }
        }
    }

    unsigned long long* Coeff_div_plain_modulus() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Coeff_div_plain_modulus_array[] = { 4345862704, unsigned(3800472112) };
                return Coeff_div_plain_modulus_array;
            }
            case 8192:
            {
                unsigned long long Coeff_div_plain_modulus_array[] = { 5541262720954987, 372437387405474, 3610984665704993 };
                return Coeff_div_plain_modulus_array;
            }
            case 16384:
            {
                //unsigned long long Coeff_div_plain_modulus_array[] = { 6274286406829919, 6993102672854152, 7307574051539359, 1093233858897379, 16322069016709854,
                //    12309498058923674, 11573009457701122 };
                unsigned long long Coeff_div_plain_modulus_array[] = { 7183970776545138, 9538434495154420, 5472079162200964, 2414043429030407, 15048148978250887,
                    1958067374324805, 13459127632045453 };
                return Coeff_div_plain_modulus_array;
            }
            case 32768:
            {
                unsigned long long Coeff_div_plain_modulus_array[] = { 130745510188956816, 50779047441896441, 128432873675295556, 44274894678919508, 245181985138183869,
                    51354457794706661, 58134037870986535, 72012422493919506, 160359168960562863, 80186181541933104, 127309357470951023, 81055894134438261, 163957865944788234,
                    144039138623314110 };
                return Coeff_div_plain_modulus_array;
            }
            }
        }
    }

    unsigned long long* Upper_half_increment() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Upper_half_increment_array[] = { 34358722560, 34358738944 };
                return Upper_half_increment_array;
            }
            case 8192:
            {
                unsigned long long Upper_half_increment_array[] = { 9007199253987328, 9007199254315008, 18014398509613056 };
                return Upper_half_increment_array;
            }
            case 16384:
            {
                //unsigned long long Upper_half_increment_array[] = { 9007199254773760, 9007199254872064, 18014398509875200, 18014398510596096, 18014398511349760, 18014398511579136,
                //    18014398513250304 };
                unsigned long long Upper_half_increment_array[] = { 18014398509875200, 18014398510596096, 18014398511349760, 18014398511579136, 18014398513250304, 18014398513414144,
                    18014398514200576 };
                return Upper_half_increment_array;
            }
            case 32768:
            {
                unsigned long long Upper_half_increment_array[] = { 144115188077887488, 144115188078870528, 144115188081033216, 144115188081623040, 288230376153481216, 288230376154398720,
                    288230376154464256, 288230376155971584, 288230376156626944, 288230376157609984, 288230376159969280, 288230376160493568, 288230376160886784, 288230376160952320 };
                return Upper_half_increment_array;
            }
            }
        }
    }

    unsigned long long* Upper_half_threshold() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Upper_half_threshold_array[] = { 516097 };
                return Upper_half_threshold_array;
            }
            case 8192:
            {
                unsigned long long Upper_half_threshold_array[] = { 516097 };
                return Upper_half_threshold_array;
            }
            case 16384:
            {
                //unsigned long long Upper_half_threshold_array[] = { 393217 };
                unsigned long long Upper_half_threshold_array[] = { 393217 };
                return Upper_half_threshold_array;
            }
            case 32768:
            {
                unsigned long long Upper_half_threshold_array[] = { 393217 };
                return Upper_half_threshold_array;
            }
            }
        }
    }

    unsigned long long* Qi_ready_t_gama() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Qi_ready_t_gama_array[] = { 130553, 114169, 34359771137, 34359754753 };
                return Qi_ready_t_gama_array;
            }
            case 8192:
            {
                unsigned long long Qi_ready_t_gama_array[] = { 635009, 42680, 206903, 1179873444139867056, 1179873061493188016, 1747361641880842873 };
                return Qi_ready_t_gama_array;
            }
            case 16384:
            {
                //unsigned long long Qi_ready_t_gama_array[] = { 547818, 610579, 319018, 47726, 712553, 537381, 505229,
                //    2094234071793249157, 1970542012203097601, 1063070540596922616, 1563907882537579777, 214137663864873087, 1740446396828053422, 1802021977617077536 };
                unsigned long long Qi_ready_t_gama_array[] = { 313622, 416408, 238888, 105387, 656939, 85481, 587569,
                    1069416679124636117, 482898016806734369, 1455258118579221662, 1782420035375715097, 805330537892708406, 817011862735702273, 2133318395226086626 };
                return Qi_ready_t_gama_array;
            }
            case 32768:
            {
                unsigned long long Qi_ready_t_gama_array[] = { 713475, 277100, 700855, 241607, 668976, 140120, 158618, 196485, 437538, 218787, 347362, 221160, 447357, 393009,
                    1759770962190465852, 2245587759343512871, 1230733864155984453, 2047983027837690302, 839305276536519100, 1233162520854083144, 367623398438830941, 1979578182230951548,
                    1503826594461116976, 654475390784834359, 395754255354762632, 1757591063663058207, 386617500734117514, 1266424566213269833 };
                return Qi_ready_t_gama_array;
            }
            }
        }
    }


    unsigned long long* Qi_inverse_ready_t_gama() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Qi_inverse_ready_t_gama_array[] = { 34357657600, 2097154 };
                return Qi_inverse_ready_t_gama_array;
            }
            case 8192:
            {
                unsigned long long Qi_inverse_ready_t_gama_array[] = { 3096931905595448, 5341731126598518, 1137072446078036 };
                return Qi_inverse_ready_t_gama_array;
            }
            case 16384:
            {
                //unsigned long long Qi_inverse_ready_t_gama_array[] = { 5195300481608768, 151496616239346, 11985883879843094, 8157440476148195, 10321218678272456, 16857698433783696, 14041758384575106 };
                unsigned long long Qi_inverse_ready_t_gama_array[] = { 9178625477083386, 15338556192680036, 11250666065966425, 3195642168021686, 6549444380720365, 1511289068638229, 7018972183921045 };
                return Qi_inverse_ready_t_gama_array;
            }
            case 32768:
            {
                unsigned long long Qi_inverse_ready_t_gama_array[] = { 20793216718985204, 70783256731351521, 2323391398761906, 62408705113629965, 184859875139454956, 253071862068424276, 86521487851322818,
                    265239111375312800, 228103514649382291, 102686963708822841, 115858211630177724, 287599451854204872, 199848230488154775, 269437160580146412 };
                return Qi_inverse_ready_t_gama_array;
            }
            }
        }
    }

    unsigned long long* Mul_inv_t() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Mul_inv_t_array[] = { 658662 };
                return Mul_inv_t_array;
            }
            case 8192:
            {
                unsigned long long Mul_inv_t_array[] = { 162867 };
                return Mul_inv_t_array;
            }
            case 16384:
            {
                //unsigned long long Mul_inv_t_array[] = { 741121 };
                unsigned long long Mul_inv_t_array[] = { 237531 };
                return Mul_inv_t_array;
            }
            case 32768:
            {
                unsigned long long Mul_inv_t_array[] = { 156971 };
                return Mul_inv_t_array;
            }
            }
        }
    }

    unsigned long long* Mul_inv_gama() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Mul_inv_gama_array[] = { 2168716010993759228 };
                return Mul_inv_gama_array;
            }
            case 8192:
            {
                unsigned long long Mul_inv_gama_array[] = { 1136145586738581869 };
                return Mul_inv_gama_array;
            }
            case 16384:
            {
                //unsigned long long Mul_inv_gama_array[] = { 576168193873809311 };
                unsigned long long Mul_inv_gama_array[] = { 487943148009086509 };
                return Mul_inv_gama_array;
            }
            case 32768:
            {
                unsigned long long Mul_inv_gama_array[] = { 2040502093557558336 };
                return Mul_inv_gama_array;
            }
            }
        }
    }

    unsigned long long* Mod_inv_gama() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long Mod_inv_gama_array[] = { 123782 };
                return Mod_inv_gama_array;
            }
            case 8192:
            {
                unsigned long long Mod_inv_gama_array[] = { 186314 };
                return Mod_inv_gama_array;
            }
            case 16384:
            {
                unsigned long long Mod_inv_gama_array[] = { 352538 };
                return Mod_inv_gama_array;
            }
            case 32768:
            {
                unsigned long long Mod_inv_gama_array[] = { 352538 };
                return Mod_inv_gama_array;
            }
            }
        }
    }



    // Multiplication Special Part
    unsigned long long* p() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long p_array[] = { 34357657600, 2097154 };
                return p_array;
            }
            case 8192:
            {
                unsigned long long p_array[] = { 3096931905595448, 5341731126598518, 1137072446078036 };
                return p_array;
            }
            case 16384:
            {
                unsigned long long p_array[] = { 9178625477083386, 15338556192680036, 11250666065966425, 3195642168021686, 6549444380720365, 1511289068638229, 7018972183921045 };
                return p_array;
            }
            case 32768:
            {
                unsigned long long p_array[] = { 20793216718985204, 70783256731351521, 2323391398761906, 62408705113629965, 184859875139454956, 253071862068424276, 86521487851322818,
                    265239111375312800, 228103514649382291, 102686963708822841, 115858211630177724, 287599451854204872, 199848230488154775, 269437160580146412 };
                return p_array;
            }
            }
        }
    }


    unsigned long long* inv_prod_q_mod_m_tilde() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long inv_prod_q_mod_m_tilde_array[] = { unsigned(1878999041) };
                return inv_prod_q_mod_m_tilde_array;
            }
            case 8192:
            {
                unsigned long long inv_prod_q_mod_m_tilde_array[] = { unsigned(2682306561) };
                return inv_prod_q_mod_m_tilde_array;
            }
            case 16384:
            {
                unsigned long long inv_prod_q_mod_m_tilde_array[] = { unsigned(1050345473) };
                return inv_prod_q_mod_m_tilde_array;
            }
            case 32768:
            {
                unsigned long long inv_prod_q_mod_m_tilde_array[] = { unsigned(4210229249) };
                return inv_prod_q_mod_m_tilde_array;
            }
            }
        }
    }


    unsigned long long* inv_prod_B_mod_m_sk() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long inv_prod_B_mod_m_sk_array[] = { 1215689658116253589 };
                return inv_prod_B_mod_m_sk_array;
            }
            case 8192:
            {
                unsigned long long inv_prod_B_mod_m_sk_array[] = { 577523214654633100 };
                return inv_prod_B_mod_m_sk_array;
            }
            case 16384:
            {
                unsigned long long inv_prod_B_mod_m_sk_array[] = { 681925659376296986 };
                return inv_prod_B_mod_m_sk_array;
            }
            case 32768:
            {
                unsigned long long inv_prod_B_mod_m_sk_array[] = { 29009664779288534 };
                return inv_prod_B_mod_m_sk_array;
            }
            }
        }
    }


    unsigned long long* prod_B_mod_q() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long prod_B_mod_q_array[] = { 4309214705, unsigned(2579007967) };
                return prod_B_mod_q_array;
            }
            case 8192:
            {
                unsigned long long prod_B_mod_q_array[] = { 3705515776219393, 6491133995069435, 7987694172793163 };
                return prod_B_mod_q_array;
            }
            case 16384:
            {
                unsigned long long prod_B_mod_q_array[] = { 8055510881989774, 841840110358728, 14056574160647014, 1865494121317683, 13022875128197394, 3968794384783381, 12661751944994914 };
                return prod_B_mod_q_array;
            }
            case 32768:
            {
                unsigned long long prod_B_mod_q_array[] = { 12487327142249814, 127322001572395861, 1303860627955317, 1701994054248316, 239292985036444648, 119647588517412918, 7434877967473758, 
                    17790047380700033, 41754712703539797, 194663863953227821, 143994086227610164, 244536037165506841, 256942975140977856, 75283712396476730 };
                return prod_B_mod_q_array;
            }
            }
        }
    }


    unsigned long long* prod_q_mod_Bsk() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long prod_q_mod_Bsk_array[] = { 1688919309598209, 1688919347346945, 1688919187963393 };
                return prod_q_mod_Bsk_array;
            }
            case 8192:
            {
                unsigned long long prod_q_mod_Bsk_array[] = { 841130162357760706, 1833527135188735916, 2008711005908971924, 27119524859437553 };
                return prod_q_mod_Bsk_array;
            }
            case 16384:
            {
                unsigned long long prod_q_mod_Bsk_array[] = { 1263002569192111762, 126714973380761410, 459654039802679993, 1550658080820427245, 476988625358900837, 1022271479857513667, 
                    966701715152045858, 2204319208495098393 };
                return prod_q_mod_Bsk_array;
            }
            case 32768:
            {
                unsigned long long prod_q_mod_Bsk_array[] = { 2231175638687838825, 1785134052923602096, 1461948584475525308, 980456140378681754, 1711112618864446048, 1600404500405992239,
                    232664188190152445, 1090614621079883479, 78647168978945936, 156545964079165649, 1102522510442444129, 842137545738185705, 1540588140854420709, 1850735047838579765, 728870369274829555 };
                return prod_q_mod_Bsk_array;
            }
            }
        }
    }


    unsigned long long* inv_prod_q_mod_Bsk() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long inv_prod_q_mod_Bsk_array[] = { 1143972115860745787, 7386690354891144, 1273496960839828507 };
                return inv_prod_q_mod_Bsk_array;
            }
            case 8192:
            {
                unsigned long long inv_prod_q_mod_Bsk_array[] = { 2289896425193762460, 409760811800688767, 517695741399353018, 1504151688773085839 };
                return inv_prod_q_mod_Bsk_array;
            }
            case 16384:
            {
                unsigned long long inv_prod_q_mod_Bsk_array[] = { 302553338083359340, 1869009471017596453, 116479471473768399, 78768693709152494, 352901150007318090, 1560309623769738648,
                    2292028507267470557, 526095235479569976 };
                return inv_prod_q_mod_Bsk_array;
            }
            case 32768:
            {
                unsigned long long inv_prod_q_mod_Bsk_array[] = { 1164456138296178239, 1433229682200113361, 1262357535223119932, 1856830556188143783, 1010364487596033954, 
                    652034595659760506, 379614499840686479, 1763690190788265629, 2114846385008225815, 1039016187987349351, 68202406691432234, 1401003722894028825,
                    1298832370622850555, 2149327070927844128, 1809361202486458342 };
                return inv_prod_q_mod_Bsk_array;
            }
            }
        }
    }


    unsigned long long* inv_m_tilde_mod_Bsk() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long inv_m_tilde_mod_Bsk_array[] = { 2161525510461103138, 1693111566795994032, 1116817940260255877 };
                return inv_m_tilde_mod_Bsk_array;
            }
            case 8192:
            {
                unsigned long long inv_m_tilde_mod_Bsk_array[] = { 1008269754319328489, 2161032929250999687, 2304752293140038594, 2161525510461103138 };
                return inv_m_tilde_mod_Bsk_array;
            }
            case 16384:
            {
                unsigned long long inv_m_tilde_mod_Bsk_array[] = { 2304611555651421386, 1727710998696847569, 1727675814324709681, 2303872683836181569, 2303168996393096849, 2302852337043708986,
                    2302500493322167106, 2304752293140038594 };
                return inv_m_tilde_mod_Bsk_array;
            }
            case 32768:
            {
                unsigned long long inv_m_tilde_mod_Bsk_array[] = { 2304611555651421386, 2303872683836181569, 2303168996393096849, 2302852337043708986, 2302500493322167106, 2299650559177685249, 
                    2299157977967529221, 2298806134245989441, 2297152468754755154, 2296835809405370369, 2296272859450908929, 2296167306334447466, 2295991384473678401, 2294161797121683089, 2304752293140038594 };
                return inv_m_tilde_mod_Bsk_array;
            }
            }
        }
    }


    unsigned long long* p1() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long p1_array[] = { 31274997412290, 2305811734215831104 };
                return p1_array;
            }
            case 8192:
            {
                unsigned long long p1_array[] = { 652704604022835395, 315999620114501675, 1337138785074718484 };
                return p1_array;
            }
            case 16384:
            {
                unsigned long long p1_array[] = { 1116395147185069986, 835372868861291064, 1370978531592177902, 1365571261771076682, 182248176306651602, 1959082078993952118, 87880962919197098 };
                return p1_array;
            }
            case 32768:
            {
                unsigned long long p1_array[] = { 1004511924229394752, 772256718107703939, 510770302424648854, 541037864853596492, 2075265835814268589, 356936869770670716, 589269113596939947,
                    323850301632043496, 474228824953464819, 439934108825704697, 1093496959747107507, 719355456031055490, 355754064325351269, 2272546701694587679 };
                return p1_array;
            }
            }
        }
    }


    unsigned long long* base_change_matrix1() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long base_change_matrix1_array[] = { 34359771137, 34359771137, 34359771137, 34359754753, 34359754753, 34359754753 };
                return base_change_matrix1_array;
            }
            case 8192:
            {
                unsigned long long base_change_matrix1_array[] = { 1179873448113232803, 1179873450870512538, 26951953181105027, 1179873442309741494, 1179873064376034723, 1179873066378339738,
                    26951566801495427, 1179873060166378934, 594440138756600434, 1747361644392264302, 17979390068421218, 1747361641198662268 };
                return base_change_matrix1_array;
            }
            case 16384:
            {
                unsigned long long base_change_matrix1_array[] = { 66943621874921667, 488293610689575072, 1329769140668683969, 1979844028981711555, 998806481376441901, 1961696723214790017, 
                    997361506584716604, 1737348928091918041, 578493156840193835, 1818172842894424059, 970347230202194057, 1988740573976983940, 741577031689813602, 2000401086602592857, 318798877509129984,
                    1803838564205116362, 604523641739450443, 2026117885468414753, 348863907506389258, 1502121887713337675, 1031678176176553660, 2268041737373239959, 466733665495219763, 1132745098794341631,
                    1945722740301066584, 603198711874978495, 1650831863108290306, 828932829377948247, 710769926004943276, 1617126851200479618, 1890237714307375469, 320915413284841266, 1237566117471640313,
                    894192107308753149, 930162954187367635, 1630371265412679604, 479585682053741822, 1738048487902119635, 1995510252536099305, 294081155183790779, 396510702554066049, 2237431322101114725,
                    2274549905397362595, 2044656996934206710, 1166188520497136344, 2174792140377956812, 816493106093377974, 450226338857023248, 1297297624116973313, 669585577288753685, 1547295784653432319,
                    2138326822561197546, 1014538453178507904, 2291516017101541691, 1608350385074311699, 189818920331088014 };
                return base_change_matrix1_array;
            }
            case 32768:
            {
                unsigned long long base_change_matrix1_array[] = { 2093643377899696358, 2283596107207619881, 1363687465262566078, 34831416700798137, 2271674764278593600, 959757911628444775, 415013450587480445,
                    1868686592725643318, 1955061849862409351, 1589780908607982356, 239237497525488577, 1742502465703014013, 1811461577964985879, 1140972802831463120, 2054767216987657098, 234312973252860347,
                    1350689627742490630, 1653601511042676817, 204199761710506787, 1748633043811523796, 2142295727270078689, 1781247263704565171, 773353669350108893, 1326879206126459802, 441615701683827774,
                    718972529111734689, 1070688428662532088, 512697447422444180, 36179799594917485, 349059011268168288, 1714535474345226187, 1143230535937403293, 1520179752801341940, 965389977757750711,
                    1561167899927016247, 2208810261562327138, 854268774787454280, 1108036685537759253, 245829281234961616, 2241267078126209624, 1112528408196385191, 1772173248658709404, 1759347976817444416,
                    1283470273609059503, 2110555034541860439, 578505571570956590, 837325525687191206, 1274621830363255073, 231666795553717796, 192059543722751117, 92679588422401315, 1514160863442881142,
                    1852735391639409875, 838345407654439571, 1537765457207391318, 461969614762525268, 980924708791798486, 1130024532200441501, 807283829636421661, 1045283930183255321, 1634715994336278977,
                    404884254007457628, 1217407328946195884, 1572706392212882110, 1726653323715141328, 2034852714029107475, 406011433673322943, 1811689517845884246, 2106801798533939044, 2184877831572420111,
                    1307842443216017395, 572340115936276580, 614676298396401330, 1004275911881339382, 2187349418219463508, 2121903427523837581, 618317570544279408, 168748580428547119, 1575069173749132600,
                    193535221452391665, 1113550436884878878, 988939455383414405, 689363550719826958, 1040681580777992305, 1963233094750104911, 591998445409649651, 1030522664328474206, 2037815317512915902,
                    611932815424473487, 632347578702510911, 1054794893049661281, 730392234271499166, 1333669097233578566, 705662888382110750, 1900083755874616120, 739471470585903577, 2028715318957066190,
                    815418494440832638, 585518188793660829, 2253479903341032261, 2054039632672613894, 36926362277105727, 2034191053045266346, 968340258138083667, 1645906880621176711, 1909252762953417710, 
                    973334251391224650, 1542154131895286258, 1141893384036124472, 1973538092823392691, 162571081995346515, 2127022644036915433, 934337555897151937, 1853156868074221555, 1869151103898020, 
                    1056104240758918657, 237952982501325158, 1959220727556439719, 266886289400749904, 629386619321857219, 868416633857777723, 1896004936355301545, 1670005165370031130, 1729182194883430527, 
                    1358699587163824499, 1317776853222108063, 743202650717461709, 408018301844951725, 2233653949183274085, 387533807957737700, 356504950239239770, 1667580086935557590, 3203772672025830, 
                    1249839514372155355, 1967647609000232915, 1297734666335238042, 1850269084721117942, 867555832078978544, 2118527463852730494, 2063640962713065266, 488491986092593777, 108483963195060089,
                    1912093419114416139, 1152175277335598974, 2026586564391297704, 1215467680657938656, 206655725327924316, 189399457926434719, 1534619311154415119, 731784024181755522, 692089716360698187, 
                    1611624540368632977, 192021259729111693, 2012210582971436607, 1090805465453761330, 181249928927552315, 1106950388384791829, 652967113430061450, 940071065262592835, 634985366177673040, 
                    1793015760589756274, 1154621081139599195, 649583522008222384, 1880543794472279078, 2209614768978643012, 2041041667917802220, 1687250209433932658, 1795883113491210475, 454203652445684791,
                    2287425759179601500, 1743760393982951237, 72888276489391731, 916439135294307552, 2198694402207903007, 1659225446826210562, 1259672037919671057, 19791583860225828, 745482590025840755, 
                    622721998134247193, 324190170266856097, 308765556428721275, 1838344705986169169, 1396665649342434477, 1937288186035911342, 1366545574497515521, 725937976723637033, 1005906977960456314, 
                    1988097411206904346, 2245194878691992785, 449070789649162966, 1342032864186688236, 562770449719242352, 2204857667618299998, 1539455987821907399, 1785616733665404086, 1509412417566106497, 
                    305144665642977907, 2221628844577732730, 577687438310857548, 1933888047034125111, 1577772640819439199, 1748799937019371528, 1440092826506473446, 641610903649307371, 2097143035318497369, 
                    590585302858267098, 721729460382246592, 79581741675943040, 2252372680892348585, 1316840490729973428 };
                return base_change_matrix1_array;
            }
            }
        }
    }


    unsigned long long* base_change_matrix2() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long base_change_matrix2_array[] = { 32769, 16385 };
                return base_change_matrix2_array;
            }
            case 8192:
            {
                unsigned long long base_change_matrix2_array[] = { 807075841, 1880489985, 1343062017 };
                return base_change_matrix2_array;
            }
            case 16384:
            {
                unsigned long long base_change_matrix2_array[] = { 3243442177, 3242721281, 2168225793, 3241738241, 2166325249, 3239903233, 3239116801 };
                return base_change_matrix2_array;
            }
            case 32768:
            {
                unsigned long long base_change_matrix2_array[] = { 81920001, 80936961, 78774273, 78184449, 82182145, 81264641, 81199105, 79691777, 79036417, 78053377, 75694081, 75169793, 74776577, 74711041 };
                return base_change_matrix2_array;
            }
            }
        }
    }


    unsigned long long* base_change_matrix3() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long base_change_matrix3_array[] = { 34292719650, 34294308930, 34292793378, 34294382658 };
                return base_change_matrix3_array;
            }
            case 8192:
            {
                unsigned long long base_change_matrix3_array[] = { 5323959487577601, 6588165762301439, 4654048697106176, 5302332104556033, 6541799367654911, 4609537766080256, 5249024661519873,
                    6426644395556351, 4499021453033216 };
                return base_change_matrix3_array;
            }
            case 16384:
            {
                unsigned long long base_change_matrix3_array[] = { 15080752030462379, 6580361163923341, 16506576365251010, 12394501539656713, 2489643801127173, 15335344409583099, 6994467768713295, 
                    567105625195307, 9057506708945362, 13645337105724390, 11751693678992313, 15219155687580395, 16084243280556319, 17881228288129674, 10452267625254690, 8114988969978420, 12020631797847642,
                    12787095547357600, 667612519232767, 12140035032843051, 11615103290007972, 12202929047294478, 6159179233137950, 773567268819912, 15282944586819165, 11049134468260431, 7251401288901291
                    , 9814100380695021, 17963054426163868, 4030344957901775, 6349229366744209, 13404273408942004, 7015243557072334, 4232118733931334, 7295495838965702, 4013682658028542, 3286235583980536,
                    6992908765493048, 10190254057425909, 10016590403713801, 16802542010597874, 7233665702569658, 7219715508386423, 6881623523829575, 2055639031587237, 7734126894176307, 5591475640751928,
                    10084867160142651, 7076893455377522 };
                return base_change_matrix3_array;
            }
            case 32768:
            {
                unsigned long long base_change_matrix3_array[] = { 6458875727395257, 38106008888209784, 33223378033302379, 64911111739325452, 152773996895037467, 201968593160481102, 172743596397482793,
                    95759245072655978, 91474413665558682, 138919084507394941, 22241152733294948, 275625916531982458, 187116314632976322, 48891956660645836, 77763139638737565, 118420070174442894,
                    66060171322716798, 18518395864784949, 223632188557618111, 234168918634350593, 90746838271043004, 22043947644835897, 82044200018542147, 180904982976189485, 38837497794525022,
                    15006141117942950, 104772838383597412, 162974526164004256, 73738664533591725, 81725561695008071, 2074562809706073, 11236241039649532, 92923455949804206, 191675838353838701, 
                    23610076380701856, 219938028524601882, 131955060330901643, 156360496052478144, 62384120501378908, 128869016992201052, 247724255773150114, 144459180376839167, 90920640429021184, 
                    88098641498401549, 1707663300862021, 50458202613960125, 170862079870653120, 230235686988251919, 104373745977965233, 136411734346308763, 185876753360299678, 1240289604069682, 
                    57857985228122492, 287291571098922794, 101064592558127449, 245503612262483806, 14121154057979829, 53606842113427793, 19448099242095131, 113077875743280564, 133200035577528408, 
                    114832001211009510, 87065605288016327, 136875871233756925, 84237279438153527, 236151526774893029, 93320154336551236, 280934303686243124, 68006145990655625, 31497551965383036, 
                    28044113022423791, 74094603469311718, 112794419343359765, 104411012337462160, 211479985681015739, 84581335902828902, 111689543190918780, 42708601710345045, 75602046394752981, 
                    282037476558159267, 95919402132602205, 203800230405867142, 755149926229463, 9578852647789919, 131520310494360358, 64458659520409625, 26422974792736960, 39147605258669625, 
                    85157842302114562, 264112414329692134, 261438746903380652, 229531173758162630, 104916095999942908, 79937823968325079, 255235848194596466, 230527006990888383, 199742991816280287,
                    119415288707772510, 135387081083996074, 62603740801127863, 60099795225318638, 101534800052367159, 19141790530272577, 245450152200946908, 226553191172535094, 284878773774090512,
                    22424217209968166, 116869577747299975, 129091415211708910, 102831447730776869, 172947721589791920, 35008100883246074, 123288523595063225, 51841298985011078, 141731739007105832,
                    119447344749291982, 69151079230612957, 58485498871154076, 104210672117681675, 61947083579761767, 228678505185378885, 265782187678758245, 234945424665830095, 204219669339827075,
                    43591212593726826, 155238974781713998, 18712963358731163, 69672944368047126, 94105185134051516, 50609215412565316, 81465791311085217, 155714238948876185, 210584551727801961,
                    63984981606340035, 120555767302161156, 135913619143299617, 227930707113850020, 108020421994157037, 23496051191416026, 276582979132292544, 11632850665972125, 44303026099412075, 
                    122893978900167307, 109536479607815895, 68972473569244155, 91919610323982846, 123773004556189262, 127015983690746965, 18965577816025068, 73344036967615743, 155891647092449800, 
                    78827266477782205, 54515059766873047, 7143243540438532, 97282867908365685, 24361394878554111, 26682426330415627, 75198378808761642, 104523783311100588, 159831161838030148, 
                    63571855235564218, 193132098847806423, 5300903979668088, 252800443858629345, 135587363799743669, 118713690657972930, 122693893808542998, 103531239006055210, 115482248966441779, 
                    82205029741706504, 54958109886216720, 41436385486504541, 251774664438710698, 65440876721348261, 190300417084835928, 116416674561307568, 33145368557571589, 261652186433617316,
                    152893894796299591, 126288909584698253, 12635748736679664, 163846771566605628, 49619900671385808, 104816424641879208, 35696928099615872, 18513524501340868, 53459210562234590,
                    56690204082145030, 200930759544944234, 145997191941832307, 12823045989692040, 90530033485939378, 285437174274078841, 200000365517703056, 248280155518571802, 52001935166069385 };
                return base_change_matrix3_array;
            }
            }
        }
    }


    unsigned long long* base_change_matrix4() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long base_change_matrix4_array[] = { 2305843009213243393, 2305843009213317121 };
                return base_change_matrix4_array;
            }
            case 8192:
            {
                unsigned long long base_change_matrix4_array[] = { 1518270939136, 1030255280128, 571230650368 };
                return base_change_matrix4_array;
            }
            case 16384:
            {
                unsigned long long base_change_matrix4_array[] = { 1417729862842541880, 1601424638440916882, 982864828424484526, 1610342583581804103, 894634768656557835, 1898450478486296031, 1241529621033490036 };
                return base_change_matrix4_array;
            }
            case 32768:
            {
                unsigned long long base_change_matrix4_array[] = { 1374239577439075205, 1603384137917249435, 378359407906991389, 2236835347598878105, 1671157042422960057, 117421885212514569, 368121682525940469, 
                    2283793223844263825, 2288591093808466279, 1459177687108395680, 13241142325911363, 627339798767633977, 818471715268912698, 194456304058553271 };
                return base_change_matrix4_array;
            }
            }
        }
    }



    unsigned long long* BackValue() {
        if (sec == security_level::HES_128) {
            switch (n)
            {
            case 4096:
            {
                unsigned long long BackValue_array[] = { 34359730176, 34359713792 };
                return BackValue_array;
            }

            case 8192:
            {
                unsigned long long BackValue_array[] = { 622591,9007199255314432,16384 };
                return BackValue_array;
            }
            case 16384:
            {
                unsigned long long BackValue_array[] = { 4849664,4128768,3375104,3145728,1474560,1310720,524288 };
                return BackValue_array;
            }
            case 32768:
            {
                unsigned long long BackValue_array[] = { 5111807,3145727,144115188080640000,144115188080050176,8192000,7274496,7208960,5701632,5046272,4063232,1703936,1179648,786432,720896 };
                return BackValue_array;
            }
            }
        }
    }


};