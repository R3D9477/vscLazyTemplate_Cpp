#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void gridInfo()
{
    int warp_id = threadIdx.x / 32;
    printf("The block ID is %d, The thread ID is %d, The warp ID is %d\n", blockIdx.x, threadIdx.x, warp_id);
}

int main(void)
{
    gridInfo<<<2, 64>>>(); // kernel_name <<< num_of_blocks, num_of_threads_per_block >>> ()

    cudaDeviceSynchronize();

    return 0;
}
