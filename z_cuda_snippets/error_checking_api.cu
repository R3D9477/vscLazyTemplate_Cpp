#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x + blockDim.x + blockIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    long long SIZE = 1024LL * 1024 * 1024 * 20;
    long size = SIZE * sizeof(int);

    cudaError_t err;

    err = cudaMalloc((void **)&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        if (abort)
        {
            exit(err);
        }
    }

    err = cudaMalloc((void **)&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        if (abort)
        {
            exit(err);
        }
    }

    err = cudaMalloc((void **)&d_c, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        if (abort)
        {
            exit(err);
        }
    }

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    for (int i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = SIZE - i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 96;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, SIZE);

    cudaError_t err_kernel = cudaGetLastError();
    if (err_kernel != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        if (abort)
        {
            exit(err_kernel);
        }
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceSynchronize();

    return 0;
}
