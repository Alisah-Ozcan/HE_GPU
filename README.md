# Notice: This Repository is no longer maintained, can be accessed the restructured library here (https://github.com/Alisah-Ozcan/HEonGPU).


This study is a part of Alişah Özcan's master's thesis(https://research.sabanciuniv.edu/id/eprint/49756/1/10602991.%C3%96zcan.pdf) and was done for educational purposes.
This library contains part of (https://eprint.iacr.org/2022/1222) work.
There may be shortcomings.

In this Library, BFV operations are performed with using GPU. Since the library uses multiple coefficient modulus with RNS,
it is very suitable for parallel operation. That's why the library has an NTT GPU implementation that can perform batch-NTT.

The library currently only performing for 128 bit security level and certain ring sizes(4096, 8192, 16384, 32768).
Since this library is still a prototype, it works with precomputed parameters for now. For example,
the user cannot use own plain modulus or own coeffmodulus, because all values are pre-calculated and written in Contect_Pool.
For this reason, the user should use the library with the currently existing parameters.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)