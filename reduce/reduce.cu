#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <memory>


#define CHECK_CUDA_ERROR(call) do { \
    auto err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda call failed %s: %s\n", #call, cudaGetErrorString(err)); \
        assert(0); \
    } \
} while (0)


int main(int argc, char* argv[])
{
    size_t size = 10ull << 20;
    if (argc > 1) {
        size = atoi(argv[1]) << 20;
    }
    fprintf(stderr, "input size: %d MB\n", size >> 20);

    int* input_d = nullptr;
    CHECK_CUDA_ERROR(cudaMallocManaged(&input_d, sizeof(int) * size));
    int device = -1;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(x, size * sizeof(int), device, NULL));

    
    CHECK_CUDA_ERROR(cudaFree(input_d));
}