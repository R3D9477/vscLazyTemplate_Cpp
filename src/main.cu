#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

#include "cppinfo.hpp"
#include "project_version.hpp"

__global__ void gridInfo()
{
    int warp_id = threadIdx.x / 32;
    printf("Hello from block ID %d, thread ID %d, warp ID %d\n", blockIdx.x, threadIdx.x, warp_id);
}

int main(void)
{
    gridInfo<<<2, 64>>>(); // kernel_name <<< num_of_blocks, num_of_threads_per_block >>> ()
    cudaDeviceSynchronize();

    CppInfo cppInfo;
    std::cout << "C++ language version: " << cppInfo.GetLanguageVersion() << std::endl;
    std::cout << "C++ compiler version: " << cppInfo.GetCompilerVersion() << std::endl;

    std::cout << "Project version: " << project_version() << std::endl;

    return 0;
}
