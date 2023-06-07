
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <thrust/reduce.h>

// Aim for 128 threads per thread block
const int THREADS_X = 32; // 32 Threads per Warp
const int THREADS_Y = 4;  // 4 Warps per Thread Block, this is the fastest.

template <bool debug>
__global__ void matmul_int4_v2(
    const half *__restrict__ input_tensor,                // (original_dim)
    const uint32_t *__restrict__ quantized_weight_tensor, // (original_dim / 8,
                                                          // original_dim)
    const half *__restrict__ scales, // (original_dim / groupsize, original_dim)
    const uint32_t
        *__restrict__ zeros, // (original_dim / groupsize, original_dim / 8)
    half *output_tensor,     // (original_dim)
    const int width, const int height, const int groupsize) {

  const unsigned int quantized_width = width / 8;

  // Imagine we have a matrix of size (original_dim, original_dim).
  // Every thread block is responsible for computing 32 columns and a variable
  // number of rows. The number of rows is determined by the groupsize and the y
  // dimension of the thread block.
  // blockDim.y * gridDim.y = groupsize ALWAYS
  const unsigned int abs_col_index = blockIdx.x * blockDim.x + threadIdx.x;
  // From a (original_dim, original_dim) matrix perspective, this is the row
  // index that this specific thread is responsible for.
  const unsigned int abs_row_index_start =
      blockIdx.y * blockDim.y * groupsize + (threadIdx.y * groupsize);
  const unsigned int abs_row_index_end = abs_row_index_start + groupsize;

  const unsigned int group_row_index = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int zero_col_index = abs_col_index / 8;
  const unsigned int zero_col_offset = (abs_col_index % 8) * 4;

  // Increment by 8 because the quantized weight tensor is 8 times smaller
  // in height than original_dim

  // (height, width) = (THREADS_Y, THREADS_X)
  __shared__ float shared_acc[THREADS_X][THREADS_Y];
  float acc;

  if constexpr (THREADS_Y == 1) {
    acc = 0;
  } else {
    shared_acc[threadIdx.x][threadIdx.y] = 0;
  }

  for (int row_index = abs_row_index_start; row_index < abs_row_index_end;
       row_index += 8) {

    const unsigned int quantized_row_index = row_index / 8;
    const unsigned int quantized_weight = __ldg(
        &quantized_weight_tensor[quantized_row_index * width + abs_col_index]);

    const half2 scale =
        __half2half2(__ldg(&scales[group_row_index * width + abs_col_index]));

    const unsigned int zero =
        ((zeros[group_row_index * quantized_width + zero_col_index] >>
          zero_col_offset) &
         0x0f) +
        1;

    const half *input_ptr = &input_tensor[row_index];

    half val_0 = __int2half_rn(((quantized_weight >> 0) & 0x0f) - zero);
    half val_1 = __int2half_rn(((quantized_weight >> 4) & 0x0f) - zero);
    half val_2 = __int2half_rn(((quantized_weight >> 8) & 0x0f) - zero);
    half val_3 = __int2half_rn(((quantized_weight >> 12) & 0x0f) - zero);
    half val_4 = __int2half_rn(((quantized_weight >> 16) & 0x0f) - zero);
    half val_5 = __int2half_rn(((quantized_weight >> 20) & 0x0f) - zero);
    half val_6 = __int2half_rn(((quantized_weight >> 24) & 0x0f) - zero);
    half val_7 = __int2half_rn(((quantized_weight >> 28) & 0x0f) - zero);

    half2 val_01 = __halves2half2(val_0, val_1);
    half2 val_23 = __halves2half2(val_2, val_3);
    half2 val_45 = __halves2half2(val_4, val_5);
    half2 val_67 = __halves2half2(val_6, val_7);

    val_01 = __hmul2(val_01, scale);
    val_23 = __hmul2(val_23, scale);
    val_45 = __hmul2(val_45, scale);
    val_67 = __hmul2(val_67, scale);

    half2 input_01 = __ldg((const half2 *)(input_ptr + 0));
    half2 input_23 = __ldg((const half2 *)(input_ptr + 2));
    half2 input_45 = __ldg((const half2 *)(input_ptr + 4));
    half2 input_67 = __ldg((const half2 *)(input_ptr + 6));

    half2 result = __half2half2(0);

    result = __hfma2(input_01, val_01, result);
    result = __hfma2(input_23, val_23, result);
    result = __hfma2(input_45, val_45, result);
    result = __hfma2(input_67, val_67, result);

    if constexpr (THREADS_Y == 1) {
      acc += __half2float(__hadd_rn(result.x, result.y));
    } else {
      shared_acc[threadIdx.x][threadIdx.y] +=
          __half2float(__hadd_rn(result.x, result.y));
    }
  }

  if constexpr (THREADS_Y == 1) {
    // Atomic add to output tensor
    atomicAdd(&output_tensor[abs_col_index], __float2half(acc));
  } else {
    __syncthreads();
    // Alternative Reduce method using thrust
    float sum = thrust::reduce(thrust::seq, &shared_acc[threadIdx.x][0],
                               &shared_acc[threadIdx.x][THREADS_Y], 0.0f);

    // Atomic add to output tensor
    if (threadIdx.y == 0) {
      atomicAdd(&output_tensor[abs_col_index], __float2half(sum));
    }

    if (debug && threadIdx.x == 2 && threadIdx.y == 0 && blockIdx.x == 0 &&
        blockIdx.y == 0) {
      printf("sum: %f\n", __half2float(sum));
    }
  }
}

void execute_kernel(const half *input_tensor,
                    const uint32_t *quantized_weight_tensor,
                    half *output_tensor, const half *scales,
                    const uint32_t *zeros, const int width, const int height,
                    const int groupsize) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Set thread dimensions
  dim3 threads(THREADS_X, THREADS_Y, 1);
  dim3 blocks(width / threads.x, (height / groupsize) / threads.y, 1);

  // Print out thread and block dimensions
  printf("Threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
  printf("Blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);

  // Launching the kernel
  cudaEventRecord(start);

  // Run kernel x times to get a good average
  int run_times = 10000;
  for (int i = 0; i < run_times; i++) {
    matmul_int4_v2<false>
        <<<blocks, threads>>>(input_tensor, quantized_weight_tensor, scales,
                              zeros, output_tensor, width, height, groupsize);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken: %f\n", milliseconds / (double)run_times);

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
