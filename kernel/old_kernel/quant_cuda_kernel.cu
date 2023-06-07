
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 700
// adapted from
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
__device__ __forceinline__ void atomicAdd(__half *address, c10::Half val) {
  unsigned int *address_as_ui =
      reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) -
                                       (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    unsigned short hsum =
        reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (hsum << 16)
                                                : (old & 0xffff0000) | hsum;
    old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
}
#endif
#endif

__global__ void VecQuant4MatMulKernel(const half2 *__restrict__ vec,
                                      const int *__restrict__ mat,
                                      float *__restrict__ mul,
                                      const float *__restrict__ scales,
                                      const int *__restrict__ zeros, int batch,
                                      int vec_height, int height, int width,
                                      int zero_width, int groupsize);

// https://github.com/iwalton3/GPTQ-for-LLaMa/commit/209d16b0187f149bf13318360925cc4f679cb2ea
template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const half2 *__restrict__ vec, const int *__restrict__ mat,
    scalar_t *__restrict__ mul, const scalar_t *__restrict__ scales,
    const int *__restrict__ zeros, const int *__restrict__ g_idx, int batch,
    int vec_height, int height, int width, int zero_width);

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int *>(&i);
}

__device__ inline int as_int(int i) { return *reinterpret_cast<int *>(&i); }

void vecquant4matmul_cuda(half2 *vec, int *mat, float *mul, float *scales,
                          int *zeros, int groupsize, int vec_height,
                          int batch) {
  int height = vec_height / 8;
  int width = vec_height;
  int zero_width = vec_height / 8;

  dim3 blocks((height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
              (width + BLOCKWIDTH - 1) / BLOCKWIDTH, batch);
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernel<<<blocks, threads>>>((half2 *)vec, mat, mul, scales,
                                             zeros, batch, vec_height, height,
                                             width, zero_width, groupsize);

  cudaDeviceSynchronize();
}

__global__ void VecQuant4MatMulKernel(const half2 *__restrict__ vec,
                                      const int *__restrict__ mat,
                                      float *__restrict__ mul,
                                      const float *__restrict__ scales,
                                      const int *__restrict__ zeros, int batch,
                                      int vec_height, int height, int width,
                                      int zero_width, int groupsize) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] =
        vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0xF), __int2half_rn(val >> 4));
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
    float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);

    uint32_t zero_int =
        ((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1;
    half2 zero = __float2half2_rn(-(scale_f * (float)zero_int));

    res2 = {};
    tmp = as_unsigned(mat[i]);

    res2 = __hfma2(__hfma2(deq2[(tmp >> 0) & 0xff][off], scale, zero),
                   blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 8) & 0xff][off], scale, zero),
                   blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero),
                   blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero),
                   blockvec[k + 3], res2);

    i += width;
    k += 4;
    res += __half2float(res2.x) + __half2float(res2.y);

    // Print out everything to make sure it's working
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   printf("row: %d | temp: %d | scale: %f | zero: %d | res: %f\n",
    //          (i / 5120) - 1, tmp, scale_f, zero_int, res);
    // }
  }

  atomicAdd(&mul[b * width + w], res);
}