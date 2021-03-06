# CUDAMatrixMultiplication
 Implementation of non-squared matrix multiplication in CUDA and Strassen algorithm in CUDA

## Algorithms

1. Normal multiplication (sequential)
2. Parallel multiplication (OpenMP)
3. Normal Strassen multiplication
4. Parallel Strassen multiplication (OpenMP)
5. CUDA multiplication using shared memory
6. CUDA Strassen multiplication

## Results

```
dimension: 256, threshold: 128
---------------------------------------------------------------
|           Algoritmo          | Tiempo (s) | Error to normal |
---------------------------------------------------------------
| Normal                       |   0.262385 |               0 |
| Seq Strassen                 |   0.245866 |      0.00354004 |
| OMP multiplication           |  0.0979605 |               0 |
| Strassen OMP multiplication  |  0.0979133 |      0.00354004 |
| CUDA multiplication          |  0.0982905 |     0.000610352 |
| CUDA Strassen multiplication | 0.00692471 |      0.00354004 |
---------------------------------------------------------------
dimension: 512, threshold: 128
---------------------------------------------------------------
|           Algoritmo          | Tiempo (s) | Error to normal |
---------------------------------------------------------------
| Normal                       |     2.0809 |               0 |
| Seq Strassen                 |    1.67201 |       0.0129395 |
| OMP multiplication           |    0.40131 |               0 |
| Strassen OMP multiplication  |   0.420058 |       0.0129395 |
| CUDA multiplication          | 0.00200797 |     0.000976562 |
| CUDA Strassen multiplication |  0.0590284 |       0.0124512 |
---------------------------------------------------------------
dimension: 1024, threshold: 128
---------------------------------------------------------------
|           Algoritmo          | Tiempo (s) | Error to normal |
---------------------------------------------------------------
| Normal                       |    17.7252 |               0 |
| Seq Strassen                 |    11.7618 |       0.0546875 |
| OMP multiplication           |    2.85481 |               0 |
| Strassen OMP multiplication  |    3.15138 |       0.0546875 |
| CUDA multiplication          |  0.0121833 |      0.00244141 |
| CUDA Strassen multiplication |   0.465359 |       0.0556641 |
---------------------------------------------------------------
dimension: 2048, threshold: 128
---------------------------------------------------------------
|           Algoritmo          | Tiempo (s) | Error to normal |
---------------------------------------------------------------
| Normal                       |    167.779 |               0 |
| Seq Strassen                 |    80.7198 |            0.25 |
| OMP multiplication           |    27.6151 |               0 |
| Strassen OMP multiplication  |    19.6159 |            0.25 |
| CUDA multiplication          |  0.0564348 |      0.00488281 |
| CUDA Strassen multiplication |    3.40015 |        0.246094 |
---------------------------------------------------------------
dimension: 4096, threshold: 128
---------------------------------------------------------------
|           Algoritmo          | Tiempo (s) | Error to normal |
---------------------------------------------------------------
| Normal                       |    1402.97 |               0 |
| Seq Strassen                 |    569.205 |         1.20898 |
| OMP multiplication           |    253.326 |               0 |
| Strassen OMP multiplication  |    129.817 |         1.20898 |
| CUDA multiplication          |   0.321269 |      0.00976562 |
| CUDA Strassen multiplication |    24.8411 |         1.18945 |
---------------------------------------------------------------
```