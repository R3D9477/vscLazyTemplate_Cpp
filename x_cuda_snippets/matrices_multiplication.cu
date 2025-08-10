#include <cuda_runtime.h>
#include <iostream>

#define matrix_t float

#define N 512
#define TILE_SIZE 16

__global__ void mm_tiled(const matrix_t *a, const matrix_t *b, matrix_t *c)
{
    __shared__ matrix_t sA[TILE_SIZE][TILE_SIZE];
    __shared__ matrix_t sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    matrix_t sum = 0.0f;

    int numTitles = N / TILE_SIZE;
    for (int t = 0; t < numTitles; ++t)
    {
        sA[threadIdx.y][threadIdx.x] = a[row * N + (t * TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    c[row * N + col] = sum;
}

void cpu_mm_naive(const matrix_t *a, const matrix_t *b, matrix_t *c)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            matrix_t sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main(void)
{
    matrix_t *h_a = new matrix_t[N * N];
    matrix_t *h_b = new matrix_t[N * N];
    matrix_t *h_c = new matrix_t[N * N];
    matrix_t *h_c_gpu = new matrix_t[N * N];

    matrix_t *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * N * sizeof(matrix_t));
    cudaMalloc((void **)&d_b, N * N * sizeof(matrix_t));
    cudaMalloc((void **)&d_c, N * N * sizeof(matrix_t));

    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = std::rand() % 100;
        h_b[i] = std::rand() % 100;
        h_c[i] = 0;
        h_c_gpu[i] = 0;
    }

    cudaMemcpy(d_a, h_a, N * N * sizeof(matrix_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(matrix_t), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cpu_mm_naive(h_a, h_b, h_c);

    mm_tiled<<<blocks, threads>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c_gpu, d_c, N * N * sizeof(matrix_t), cudaMemcpyDeviceToHost);

    long temp = 0;
    for (int i = 0; i < N * N; ++i)
    {
        temp += std::abs(h_c_gpu[i] - h_c[i]);
    }
    if (temp == 0)
    {
        std::cout << "result are correct" << std::endl;
    }
    else
    {
        std::cout << "wrong results" << std::endl;
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_gpu;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
