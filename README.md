Beginnings of a nueral network and matrix library. The matrix library is mostly done and about 90% avx512 accelerated (so it only works on modern CPU's). The nueral network library is good enough now for traditional dense feedfoward networks with back propagation, but the CNN isn't working yet and the architecture is a bit wonky (currently refactoring ... will be much simpler and a lot less code).

Feel free to check it out, appologies in advance for ugly hacks and messy code in places that aren't baked yet.

It's been a blast!

Benchmarks (using the new GPU matrix library!) on an AMD 7600 and a GTX 3090

| Method   | Size | Mean            | 
|--------- |----- |----------------:|
| Software | 64   |       255.28 µs |
| AVX      | 64   |        35.01 µs |
| GPU      | 64   |        78.23 µs |
|          |      |                 |
| Software | 256  |    18,472.34 µs |
| AVX      | 256  |     2,085.79 µs | 
| GPU      | 256  |       105.87 µs | 
|          |      |                 |
| Software | 1024 | 1,925,101.06 µs |
| AVX      | 1024 |   128,724.05 µs |
| GPU      | 1024 |       947.98 µs |


