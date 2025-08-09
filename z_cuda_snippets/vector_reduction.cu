#include <cuda_runtime.h>
#include <iostream>

__global__ void reduce_in_place(float *input, int n)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        if (index + stride < n) // if (tid % (2 * stride) == 0 && index + stride < n)
        {
            input[index] += input[index + stride];
        }
    }

    if (tid == 0)
    {
        input[blockIdx.x] = input[blockIdx.x + blockDim.x];
    }
}

float cpu_reduce(float *input, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        sum += input[i];
    }
    return sum;
}

int main(void)
{
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);

    float *h_input = new float[n];

    float *d_input;
    cudaMalloc((void **)&d_input, size);

    for (int i = 0; i < n; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    float sum = cpu_reduce(h_input, n);
    std::cout << "CPU sum: " << sum << std::endl;

    while (gridSize > 1)
    {
        reduce_in_place<<<gridSize, blockSize>>>(d_input, n);
        cudaDeviceSynchronize(); // ensure kernel execution completes

        // update n to reflect the reduced number f elelemtns
        n = gridSize; // 4096; 16
        // update gridSize for the next iteration
        gridSize = (n + blockSize - 1) / blockSize; // (4096+256-1)/256 = 16; (16+256-1)/256 = 1
    }

    // final reduction when gridSize is 1
    reduce_in_place<<<1, blockSize>>>(d_input, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_input[0], d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU sum: " << h_input[0] << std::endl;

    cudaFree(d_input);

    delete[] h_input;

    return 0;
}
