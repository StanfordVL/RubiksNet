#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>  // CUDA supports native assert
#include <THC/THCAtomics.cuh>

#ifndef TORCHX_CUDA_H_
#define TORCHX_CUDA_H_

namespace torchx {

template <typename T, int N>
using PTA = torch::PackedTensorAccessor<T, N, torch::RestrictPtrTraits, uint32_t>;

#define GET_PTA(...)  OVERLOAD_MACRO(GET_PTA, __VA_ARGS__)
#define GET_PTA2(x, N)   x.packed_accessor<scalar_t,N,torch::RestrictPtrTraits,uint32_t>()
#define GET_PTA3(x, N, T)  x.packed_accessor<T,N,torch::RestrictPtrTraits,uint32_t>()

#define GRID_STRIDE_LOOP(var, N) \
    for (uint32_t var = blockIdx.x * blockDim.x + threadIdx.x; \
        var < N; \
        var += blockDim.x * gridDim.x)


// Defines macro and function for error checking
// Reference: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
            << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

// Populates block_count and thread_per_block with properties of CUDA Device
// Remark: the argument "count" is inferred as virtual_thread_count
inline void get_cuda_device_properties(
    const int count, int& block_count, int& thread_per_block) {

    int max_thread_per_block = 0;
    cudaErrorCheck( cudaDeviceGetAttribute(&max_thread_per_block, \
		cudaDevAttrMaxThreadsPerBlock, 0) );

    int max_thread_per_multiprocessor = 0;
    cudaErrorCheck( cudaDeviceGetAttribute(&max_thread_per_multiprocessor, \
		cudaDevAttrMaxThreadsPerMultiProcessor, 0) );

    int num_multiprocessor = 0;
    cudaErrorCheck( cudaDeviceGetAttribute(&num_multiprocessor, \
		cudaDevAttrMultiProcessorCount, 0) );

    const int virtual_thread_count = count;
    const int physical_thread_count = std::min(
        num_multiprocessor * max_thread_per_multiprocessor, virtual_thread_count
    );

    thread_per_block = std::min(1024, max_thread_per_block);

    block_count = std::min(
        (physical_thread_count + thread_per_block - 1) / thread_per_block,
        num_multiprocessor
    );

#ifdef _DEBUG_
    std::cout << "max_thread/block " << max_thread_per_block
              << "\n\tmax_thread/SM " << max_thread_per_multiprocessor
              << "\n\tnum_SM " << num_multiprocessor
              << "\n\tphysical_thread_count " << physical_thread_count
              << "\n\tnum_blocks " << block_count
              << "\n\tthread/block " << thread_per_block << std::endl;
#endif
}

}  // namespace torchx

#endif  // TORCHX_CUDA_H_
