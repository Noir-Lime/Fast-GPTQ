#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 700
__device__ __forceinline__ void atomicAdd(__half *address, c10::Half val);
#endif
#endif

__global__ void VecQuant4MatMulKernel(const half2 *__restrict__ vec,
                                      const int *__restrict__ mat,
                                      float *__restrict__ mul,
                                      const float *__restrict__ scales,
                                      const int *__restrict__ zeros, int batch,
                                      int vec_height, int height, int width,
                                      int zero_width, int groupsize);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const half2 *__restrict__ vec, const int *__restrict__ mat,
    scalar_t *__restrict__ mul, const scalar_t *__restrict__ scales,
    const int *__restrict__ zeros, const int *__restrict__ g_idx, int batch,
    int vec_height, int height, int width, int zero_width);

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline unsigned int as_unsigned(int i);
__device__ inline int as_int(int i);

void vecquant4matmul_cuda(half2 *vec, int *mat, float *mul, float *scales,
                          int *zeros, int groupsize, int vec_height,
                          int batch);

__global__ void VecQuant4MatMulKernel(const half2 *__restrict__ vec,
                                      const int *__restrict__ mat,
                                      float *__restrict__ mul,
                                      const float *__restrict__ scales,
                                      const int *__restrict__ zeros, int batch,
                                      int vec_height, int height, int width,
                                      int zero_width, int groupsize);