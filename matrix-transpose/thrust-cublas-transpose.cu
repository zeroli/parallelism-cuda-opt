#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <assert.h>

#include "timer.h"

/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
        assert(0); 
    }
}

// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
    size_t operator()(size_t linear_index)
    {
        size_t i = linear_index / n;
        size_t j = linear_index % n;

        return m * j + i;
    }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
    size_t n;

    __host__ __device__
    row_index(size_t _n) : n(_n) {}

    __host__ __device__

    size_t operator()(size_t i)
    {
        return i / n;
    }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
    thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
    src.begin(),dst.begin());
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            std::cout << std::setw(8) << h_data[i * n + j] << " ";
            std::cout << "\n";
    }
}

int main(void)
{
    size_t m = 1024; // number of rows
    size_t n = 1024; // number of columns

    // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
    thrust::device_vector<float> data(m * n, 1.);
    //data[1] = 2.;
    //data[3] = 3.;

    std::cout << "Initial array" << std::endl;
    //print(m, n, data);

    std::cout << "Transpose array - Thrust" << std::endl;
    thrust::device_vector<float> transposed_thrust(m * n);
    {
        transpose(m, n, data, transposed_thrust);
        double t0 = GetTime();
        for (int i = 0; i < 10; i++) {
            double t1 = GetTime();
            transpose(m, n, data, transposed_thrust);
            printf("thrust transpose runtime: %f ms\n", (GetTime() - t1) * 1e3 / 1);
        }
        printf(">>thrust transpose runtime: %f ms\n", (GetTime() - t0) * 1e3 / 10);
    }
    //print(n, m, transposed_thrust);

    std::cout << "Transpose array - cuBLAS" << std::endl;
    thrust::device_vector<float> transposed_cuBLAS(m * n);
    float* dv_ptr_in  = thrust::raw_pointer_cast(data.data());
    float* dv_ptr_out = thrust::raw_pointer_cast(transposed_cuBLAS.data());
    float alpha = 1.;
    float beta  = 0.;
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    {
        thrust::device_vector<float> tmp(10, 0);
        thrust::device_vector<float> tmp1(10, 0);
        cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, 10, 10, &alpha,
            thrust::raw_pointer_cast(tmp.data()), 10, &beta, thrust::raw_pointer_cast(tmp.data()), n, thrust::raw_pointer_cast(tmp1.data()), 10)); 
        //cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
        double t0 = GetTime();
        for (int i = 0; i < 10; i++) {
            double t1 = GetTime();
            cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
            cudaStreamSynchronize(0);
            printf("cublas transpose runtime: %f ms\n", (GetTime() - t1) * 1e3 / 1);
        }
        //cudaStreamSynchronize(0);
        printf(">>cublas transpose runtime: %f ms\n", (GetTime() - t0) * 1e3 / 10);
    }
    //print(n, m, transposed_cuBLAS);


    return 0;
}