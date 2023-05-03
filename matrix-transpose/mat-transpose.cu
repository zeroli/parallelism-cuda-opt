#include <iostream>
#include <cstdio>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__
void copyKernel(float* output, const float* input)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[(y + j) * width + x] = input[(y + j) * width + x];
    }
}

__global__
void copyShmKernel(float* output, const float* input)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
    }
    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[(y + j) * width + x] = tile[threadIdx.y + j][threadIdx.x];
    }
}

__global__
void transponseMatNaive(float* output, const float* input)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[x * width + y + j] = input[(y + j) * width + x];
    }
}

__global__
void transponseMatCoalesing(float* output, const float* input)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__
void transponseMatCoalesingNoBankConflict(float* output, const float* input)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main()
{
    const int nx = 1024;
    const int ny = 1024;
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    size_t size = nx * ny;
    float* h_input = new float[size];
    float* h_output = new float[size];

    float* d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * size);
    cudaMalloc(&d_output, sizeof(float) * size);
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    {
        float ms = 0;
        copyKernel<<<dimGrid, dimBlock>>>(d_output, d_input);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i< 10; i++) {
            copyKernel<<<dimGrid, dimBlock>>>(d_output, d_input);
        }
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= 10;
        printf("copy kernel: %f ms, bandwidth: %f GB/s\n", ms, 2 * sizeof(float) * nx * ny * 1e-6 / ms);
    }
    {
        float ms = 0;
        copyShmKernel<<<dimGrid, dimBlock>>>(d_output, d_input);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i< 10; i++) {
            copyShmKernel<<<dimGrid, dimBlock>>>(d_output, d_input);
        }
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= 10;
        printf("copy shm kernel: %f ms, bandwidth: %f GB/s\n", ms, 2 * sizeof(float) * nx * ny * 1e-6 / ms);
    }
    {
        float ms = 0;
        transponseMatNaive<<<dimGrid, dimBlock>>>(d_output, d_input);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i< 10; i++) {
            transponseMatNaive<<<dimGrid, dimBlock>>>(d_output, d_input);
        }
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= 10;
        printf("transponseMatNaive kernel: %f ms, bandwidth: %f GB/s\n", ms, 2 * sizeof(float) * nx * ny * 1e-6 / ms);
    }
    {
        float ms = 0;
        transponseMatCoalesing<<<dimGrid, dimBlock>>>(d_output, d_input);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i< 10; i++) {
            transponseMatCoalesing<<<dimGrid, dimBlock>>>(d_output, d_input);
        }
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= 10;
        printf("transponseMatCoalesing kernel: %f ms, bandwidth: %f GB/s\n", ms, 2 * sizeof(float) * nx * ny * 1e-6 / ms);
    }
    {
        float ms = 0;
        transponseMatCoalesingNoBankConflict<<<dimGrid, dimBlock>>>(d_output, d_input);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i< 10; i++) {
            transponseMatCoalesingNoBankConflict<<<dimGrid, dimBlock>>>(d_output, d_input);
        }
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= 10;
        printf("transponseMatCoalesingNoBankConflict kernel: %f ms, bandwidth: %f GB/s\n", ms, 2 * sizeof(float) * nx * ny * 1e-6 / ms);
    }
    cudaFree(d_input);      
    cudaFree(d_output);
    delete [] h_input;
    delete [] h_output;
}