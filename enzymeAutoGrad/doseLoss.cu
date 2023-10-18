//////////////////////////////////////////////////////////////////////////////////////////////////////
//created by Wang Zilu on Oct. 1sr 2023
//
//this is a shared library used for minDose and maxDose loss calculation
//sign:1 max dose loss
//sign:-1 min dose loss
//////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdio.h"
#include "stdlib.h"

#define BLOCKDIM 256
#define GRIDDIM  128


float __device__ loss(float *dose, float d_value, int sign) {
	float temp = 0.0f;
	if (sign * (dose[0] - d_value) >= 0) {
		temp = 1.0f;
	}
	return temp *= (sign * (d_value - dose[0])) * (sign * (d_value - dose[0]));
}

typedef float (*f_ptr)(float*, float, int);
extern void __device__ __enzyme_autodiff(f_ptr, int, float*, float*, int, float, int);

int __device__ enzyme_dup;
int __device__ enzyme_out;
int __device__ enzyme_const;

void __global__ backwardKernel(float *p_dose, float *p_dose_grad, float *p_loss, float d_value, int size, int sign) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheIndex = threadIdx.x;
	__shared__ float cache[BLOCKDIM];
	float temp = 0.0f;
	while (id < size) {
		temp += loss(p_dose + id, d_value, sign) / size;
		float d_dose = 0.0f;
		__enzyme_autodiff(loss, enzyme_dup, p_dose + id, &d_dose, enzyme_const, d_value, sign);
		p_dose_grad[id] = d_dose / size;
		id += gridDim.x * blockDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	p_loss[blockIdx.x] = cache[0];
}

float calDoseLoss(float *d_dose, float *d_dose_grad, float *d_loss, float dose_value, int size, int sign) {
	float loss = 0.0f;
	if (sign != 1 && sign != -1) {
		printf("sign should be 1 or -1.\n");
		return 0;
	}
	backwardKernel<<<GRIDDIM, BLOCKDIM>>>(d_dose, d_dose_grad, d_loss, dose_value, size, sign);
	float *h_loss = (float *)malloc(sizeof(float) * GRIDDIM);
	memset(h_loss, 0, sizeof(float) * GRIDDIM);
	cudaMemcpy(h_loss, d_loss, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
	for (int i = 0; i < GRIDDIM; i++) {
		loss += h_loss[i];
	}
	free(h_loss);
	return loss;
}

int main() {
	printf("_____%d\n",0 | -1);
	int size = 300;
	int sign = -1;
	float target = 100.f;
	float *h_dose = (float *)malloc(size * sizeof(float));
	float *h_dose_grad = (float *)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		h_dose[i] = i;
	}
	float *d_dose;
	float *d_dose_grad;
	float *d_loss;
	cudaMalloc((void **)&d_loss, sizeof(float) * GRIDDIM);
	cudaMalloc((void **)&d_dose, sizeof(float) * size);
	cudaMalloc((void **)&d_dose_grad, sizeof(float) * size);
	cudaMemcpy(d_dose, h_dose, sizeof(float) * size, cudaMemcpyHostToDevice);
	float loss = calDoseLoss(d_dose, d_dose_grad, d_loss, target, size, sign);
	cudaMemcpy(h_dose_grad, d_dose_grad, sizeof(float) * size, cudaMemcpyDeviceToHost);
	printf("loss:%f\n", loss);
	for (int i = 0; i < size; i++) {
		printf("dose grad %d:%f\n", i, h_dose_grad[i]);
	}
	cudaFree(d_dose);
	cudaFree(d_dose_grad);
	cudaFree(d_loss);
	free(h_dose);
	free(h_dose_grad);
}
