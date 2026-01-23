Beginnings of a nueral network and matrix library. The matrix library is mostly done and about 90% avx512 accelerated (so it only works on modern CPU's). The nueral network library is good enough now for traditional dense feedfoward networks with back propagation, but the CNN isn't working yet and the architecture is a bit wonky (currently refactoring ... will be much simpler and a lot less code).

Feel free to check it out, appologies in advance for ugly hacks and messy code in places that aren't baked yet.

It's been a blast!

Some benchmark metrics, on a 9800x3d:
// * Summary *

BenchmarkDotNet v0.14.0, Windows 11 (10.0.26200.7623)
Unknown processor
.NET SDK 9.0.310
  [Host]     : .NET 8.0.23 (8.0.2325.60607), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI
  DefaultJob : .NET 8.0.23 (8.0.2325.60607), X64 RyuJIT AVX-512F+CD+BW+DQ+VL+VBMI


| Method                    | Mean           | Error       | StdDev      |
|-------------------------- |---------------:|------------:|------------:|
| NaiveConvolution          |    31,851.1 us |    84.17 us |    70.29 us |
| AvxConvolution4x4         |     3,426.8 us |    13.64 us |    12.76 us |
| AvxConvolution8x8         |     6,975.0 us |    20.83 us |    17.40 us |
| AvxConvolution7x7         |    38,002.0 us |   108.47 us |    96.15 us |
| AvxConvolution14x14       |   152,110.3 us |   586.66 us |   520.06 us |
| NaiveMatrixMultiply       | 1,972,985.7 us | 8,555.16 us | 8,002.50 us |
| IntrinsicMatrixMultiply   |   118,868.4 us |   437.43 us |   341.52 us |
| IntrinsicMatrixMult_Tiled |   228,512.0 us | 2,407.62 us | 2,252.09 us |
| NaiveAdd                  |     1,408.9 us |    10.82 us |    10.12 us |
| IntrinsicAdd              |       310.3 us |     3.28 us |     2.91 us |
| NaiveTranspose            |     2,021.4 us |    19.57 us |    18.30 us |
| AvxTranspose              |     1,561.2 us |    10.46 us |     9.79 us |

