
#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define MAX_THREADS_BLOCK 1024
 
__global__ void cuda_sum_kernel(size_t size, float *pos)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    pos[idx] = pos[idx] / 2;
} 

extern "C" {
void cuda_sum(size_t size, float *pos)
{    
    float *d_pos;
    clock_t t_ini,t_fin;

    dim3 BLOCK(ceil(size/MAX_THREADS_BLOCK));
    dim3 THREAD(MAX_THREADS_BLOCK);

    cudaMalloc((void **)&d_pos, size * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemcpy(d_pos, pos, size * sizeof(float), cudaMemcpyHostToDevice);

    t_ini=clock();
    cudaEventRecord(start);
    
    cuda_sum_kernel <<< BLOCK, MAX_THREADS_BLOCK >>> (size, d_pos);
    //cuda_sum_kernel <<< 2, 1024 >>> (size, d_pos);

    cudaEventRecord(stop);
        
    
    cudaDeviceSynchronize();
    t_fin=clock();
    printf("%f\n",(double)(t_fin-t_ini)/CLOCKS_PER_SEC);

    cudaMemcpy(pos, d_pos, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_pos);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n",double(milliseconds));
    
}
}
