//
// Created by 王子路 on 2023/10/17.
//
#include "cuda_runtime_api.h"
#include "options.h"
float __device__ projL1(float U, float lambda, bool pos) {
  if (pos)
	return fmaxf(0, U - lambda);
  else
	return fmaxf(0, U - lambda) + fminf(0, U + lambda);
}

void __global__ gKernel(float *p_v, float *p_sum, float lambda, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheId = threadIdx.x;
  float __shared__ cache[BLOCKDIM];
  float temp = 0.0f;
  while (idx < size) {
	temp += fabsf(p_v[idx] * lambda);
	idx += gridDim.x * blockDim.x;
  }
  cache[cacheId] = temp;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i != 0) {
	if (cacheId < i)
	  cache[cacheId] += cache[cacheId + i];
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

void __global__ dotVecKernel(float *p_v1, float *p_v2, float *p_result, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheId = threadIdx.x;
  float __shared__ cache[BLOCKDIM];
  float temp = 0.0f;
  while (idx < size) {
	temp += p_v1[idx] * p_v2[idx];
	idx += gridDim.x * blockDim.x;
  }
  cache[cacheId] = temp;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i != 0) {
	if (cacheId < i)
	  cache[cacheId] += cache[cacheId + i];
	__syncthreads();
	i /= 2;
  }
  p_result[blockIdx.x] = cache[0];
}

void __global__ subVecKernel(float *p_v1, float *p_v2, float *p_result, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  while(idx < size) {
	p_result[idx] = p_v1[idx] - p_v2[idx];
	idx += gridDim.x * blockDim.x;
  }
}

void __global__ addVecKernel(float *p_v1, float *p_v2, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  while(idx < size) {
	p_v1[idx] += p_v2[idx];
	idx += gridDim.x * blockDim.x;
  }
}

void subVec(float *p_v1, float *p_v2, float *p_result, int size) {
  subVecKernel<<<GRIDDIM, BLOCKDIM>>>(p_v1, p_v2, p_result, size);
}

void addVec(float *p_v1, float *p_v2, int size) {
  addVecKernel<<<GRIDDIM, BLOCKDIM>>>(p_v1, p_v2, size);
}

float g(float *d_v, float *d_sum, float lambda, int size) {
  gKernel<<<GRIDDIM, BLOCKDIM>>>(d_v, d_sum, lambda, size);
  float *h_sum = (float *)malloc(sizeof(float) * GRIDDIM);
  cudaMemcpy(h_sum, d_sum, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
  float temp = 0.0f;
  for (int i = 0; i < GRIDDIM; i++) {
	temp += h_sum[i];
  }
  free(h_sum);
  return temp;
}

float normF2(float *d_v, float *d_sum, int size) {
  normF2Kernel<<<GRIDDIM, BLOCKDIM>>>(d_v, d_sum, size);
  float *h_sum = (float *)malloc(sizeof(float) * GRIDDIM);
  cudaMemcpy(h_sum, d_sum, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
  float temp = 0.0f;
  for (int i = 0; i < GRIDDIM; i++) {
	temp += h_sum[i];
  }
  free(h_sum);
  return temp;
}