//
// Created by 王子路 on 2023/10/17.
//
#include "common.h"
float __device__ projL1(float U, float lambda, bool pos) {
  if (pos)
	return __maxf(0, U - lambda);
  else
	return __maxf(0, U - lambda) + __minf(0, U + lambda);
}

void __global__ gKernel(float *p_v, float *p_sum, float lambda, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheId = threadIdx.x;
  float __shared__ cache[BLOCKDIM];
  float temp = 0.0f;
  while (idx < size) {
	temp += __absf(p_v[idx] * lambda);
	idx += gridDim.x * blockDim.x;
  }
  cache[cacheId] = temp;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i != 0) {
	if (chacheId < i)
	  cache[chacheId] += cache[chacheId + i];
	__syncthreads();
	i /= 2;
  }
  p_sum[blockIdx.x] = cache[0];
}

void __global__ normF2Kernel(float *p_v, float *p_result, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheId = threadIdx.x;
  float __shared__ cache[BLOCKDIM];
  float temp = 0.0f;
  while (idx < size) {
	temp += __powf(p_v[idx], 2);
	idx += gridDim.x * blockDim.x;
  }
  cache[cacheId] = temp;
  __syncthreads();
  int i = blockIdx.x / 2;
  while(i != 0) {
	if (cacheId < i)
	  cache[cacheId] += cache[cacheId + i];
	__syncthreads();
	i /= 2;
  }
  p_result[blockIdx.x] = cache[0];
}