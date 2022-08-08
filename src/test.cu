#include "typedef.h"
#include "cuda_utils.h"
#include <iostream>

__global__ void matmul_cuda(real_t* m1, real_t* m2, real_t* m3, size_t N) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < N && j < N) {
        m3[i*N+j] = 0;
        for (size_t k = 0; k < N; k++) {
            m3[i*N+j] += m1[i*N+k]*m2[k*N+j];
        }
    }
}

size_t ceilInt(const size_t num, const size_t den) {
    return num/den + (num % den != 0);
}

void matmul() {
    size_t N = 15;
    size_t total_size = N*N;

    real_t* m1 = new real_t[total_size];
    real_t* m2 = new real_t[total_size];
    real_t* m3 = new real_t[total_size];

    for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
        m1[i*N+j] = i;
        m2[i*N+j] = j;
    }}

    real_t* m1_d = newCUDA<real_t>(total_size);
    real_t* m2_d = newCUDA<real_t>(total_size);
    real_t* m3_d = newCUDA<real_t>(total_size);

    copyToDeviceCUDA(m1_d, m1, total_size);
    copyToDeviceCUDA(m2_d, m1, total_size);

    dim3 block_size {10,10,10};
    dim3 grid_size;
    grid_size.x = ceilInt(N, block_size.x);
    grid_size.y = ceilInt(N, block_size.y);
    grid_size.z = ceilInt(N, block_size.z);

    matmul_cuda _CK(grid_size, block_size) (m1_d, m2_d, m3_d, N);

    copyToHostCUDA(m3, m3_d, total_size);
    
    for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
        std::cout << m3[i*N+j] << ((j==N-1) ? "\n" : " ");
    }}
}
