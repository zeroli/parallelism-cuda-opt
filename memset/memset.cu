#include <helper_math.h>
#include <cstdio>
#include <cassert>
#include "timer.h"

template <typename T>
__global__
void memsetKernel(void* ptr, int n, T val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        *(reinterpret_cast<T*>(ptr) + idx) = val;
    }
}

#define BYTES_PER_THREAD sizeof(int)
__global__
void memsetKernel0(void* ptr, int nbytes, int8_t val)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tnumx = gridDim.x * blockDim.x;

    do {
        size_t index = tx * BYTES_PER_THREAD;
        if (index >= nbytes) {
            break;
        }
        size_t end = (index + BYTES_PER_THREAD >= nbytes) ?
                    nbytes : (index + BYTES_PER_THREAD);

        char* pc = reinterpret_cast<char*>(ptr) + index;
        int8_t tmp = val;
        while (index < end) {
            if (index + sizeof(int) <= end) {  // write 4 bytes once
                int val32 = (tmp << 24) | (tmp << 16) | (tmp << 8) | tmp;
                *reinterpret_cast<int*>(pc) = val32;
                pc += sizeof(int);
                index += sizeof(int);
            } else {
                while (index++ < end) {
                    *pc++ = tmp;
                }
            }
        }
        // will this thread handle next memory chunk?
        tx += tnumx;
    } while (1);
}

__global__
void memsetKernel1(int8_t* ptr, int n, int m, uint4 val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) {
        return;
    }

    if (idx == m - 1 && (idx + 1) * sizeof(uint4) > n) {  // the last thread
        #pragma unroll
        for (int k = idx * sizeof(uint4); k < n; k++) {
            ptr[k] = val.x & 0xFF;
        }
    } else {  // not the last thread
        *(reinterpret_cast<uint4*>(ptr) + idx) = val;
    }
}

#define CUDA_CHECK_ERROR(call) do { \
    auto err = call; \
    if (err != cudaSuccess) { \
        printf("error of call '%s': %s\n", #call, cudaGetErrorString(err)); \
        assert(0); \
    } \
} while (0)

int main(int argc, char** argv)
{
    int tail = 0;
    if (argc > 1) {
        tail = atoi(argv[1]);
    }
    printf("tail = %d\n", tail);

    char* d_ptr = nullptr;
    int n = (15U << 20) + tail;
    cudaMalloc(&d_ptr, sizeof(char) * n);

    constexpr int N = 10;

    {        
        if (n % sizeof(uint4) == 0) {
            int size = n / sizeof(uint4);
            dim3 blockSize(1024);
            dim3 gridSize((size + blockSize.x - 1) / blockSize.x);      
            double t0 = GetTime();      
            for (int i = 0; i < N; i++) {        
                memsetKernel<<<gridSize, blockSize>>>(d_ptr, size, make_uint4(0));
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(0));
            printf("memset kernel: %f ms\n", 1e3 * (GetTime() - t0) / N);
        }        
    }
    {       
        int size = (n + sizeof(uint4) - 1) / sizeof(uint4);
        dim3 blockSize(1024);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        double t0 = GetTime();
        for (int i = 0; i < N; i++) {        
            memsetKernel1<<<gridSize, blockSize>>>(reinterpret_cast<int8_t*>(d_ptr), n, size, make_uint4(0));
        }

        CUDA_CHECK_ERROR(cudaStreamSynchronize(0));
        printf("memset kernel#1: %f ms\n", 1e3 * (GetTime() - t0) / N);
    }
    {
        dim3 blockSize(1024);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        double t0 = GetTime();
        for (int i = 0; i < N; i++) {        
            memsetKernel0<<<gridSize, blockSize>>>(d_ptr, n, 0);
        }
        CUDA_CHECK_ERROR(cudaStreamSynchronize(0));
        printf("memset kernel#0: %f ms\n", 1e3 * (GetTime() - t0) / N);
    }
    {
        double t0 = GetTime();
        for (int i = 0; i < N; i++) {        
            CUDA_CHECK_ERROR(cudaMemsetAsync(d_ptr, 0, n * sizeof(char), 0));
        }
        CUDA_CHECK_ERROR(cudaStreamSynchronize(0));
        printf("cudaMemsetAsync kernel: %f ms\n", 1e3 * (GetTime() - t0) / N);
    }
}