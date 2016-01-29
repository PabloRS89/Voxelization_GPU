#include <stdio.h>
#include <cuda.h>
 
__global__ void cuda_sum_kernel(int *a, int *b, int *c, size_t size, float *pos)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    c[idx] = a[idx] + b[idx];
    pos[idx] = pos[idx] / 2;
} 

extern "C" {
void cuda_sum(int *a, int *b, int *c, size_t size, float *pos)
{
    int *d_a, *d_b, *d_c;    
    float *d_pos;

    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));
    cudaMalloc((void **)&d_pos, size * sizeof(float));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_sum_kernel <<< ceil(size / 256.0), 256 >>> (d_a, d_b, d_c, size, d_pos);

    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, d_pos, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_pos);
}
}
