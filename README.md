This study is a part of Alişah Özcan's master's thesis and was done for educational purposes.
This library contains part of (https://eprint.iacr.org/2022/1222) work.
There may be shortcomings.

In this Library, BFV operations are performed with using GPU. Since the library uses multiple coefficient modulus with RNS,
it is very suitable for parallel operation. That's why the library has an NTT GPU implementation that can perform batch-NTT.

The library currently only performing for 128 bit security level and certain ring sizes(4096, 8192, 16384, 32768).
Since this library is still a prototype, it works with precomputed parameters for now. For example,
the user cannot use own plain modulus or own coeffmodulus, because all values are pre-calculated and written in Contect_Pool.
For this reason, the user should use the library with the currently existing parameters. In the future,
the library will be updated and made suitable for users to use the parameters they want within the rules.
Also, CKKS scheme and Bootstrapping for BFV will be added to the library in the future.
