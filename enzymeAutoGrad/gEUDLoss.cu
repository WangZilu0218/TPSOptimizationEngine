/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//create by Wang Zilu on Oct. 1st 2023
//
//this is a shared lib used for gEUD loss calculation
//sign:1   upper gEUD loss
//sign:-1  lower gEUD loss
//sign:0   target gEUD loss
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include"stdio.h"
#include"stdlib.h"
#include<cmath>


#define BLOCKDIM 256
#define GRIDDIM  128

void __global__ sumKernel(float *d_dose, float *d_sum, float a, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheId = threadIdx.x;
	float __shared__ cache[BLOCKDIM];
	float temp = 0.0f;
	while (idx < size) {
		temp += __powf(d_dose[idx], a);
		idx += blockDim.x * gridDim.x;
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
	d_sum[blockIdx.x] = cache[0];
}

float sumPower(float *d_dose, float *d_sum, float a, int size) {
	float *h_sum = (float *)malloc(sizeof(float) * size);
	sumKernel<<<GRIDDIM, BLOCKDIM>>>(d_dose, d_sum, a, size);
	cudaMemcpy(h_sum, d_sum, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
	float sum = 0.0f;
	for (int i = 0; i < GRIDDIM; i++) {
		sum += h_sum[i];
	}
	free(h_sum);
	return sum;
}

void __global__ backwardUpperKernel(float *d_dose, float *d_dose_grad, float sum_dose_pow, float target, float a, int size, int sign) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (idx < size) {
		int temp = 0;
		if (sign * (__powf(sum_dose_pow / size, 1/a) - target) >= 0)
			temp = 1;
		d_dose_grad[idx] = temp*(__powf(sum_dose_pow / size, 1/a)-target)*(__powf(sum_dose_pow/size, 1/a-1)*__powf(d_dose[idx], a-1)/size);
		idx += gridDim.x * blockDim.x;
	}
}

float calgEUDLoss(float *d_dose, float *d_dose_grad, float target, float a, int size, int sign) {
	float *d_sum;
	cudaMalloc((void **)&d_sum, sizeof(float) * GRIDDIM);
	float sum_dose_pow = sumPower(d_dose, d_sum, a, size);
	printf("sum dose pow a:%f gEUD:%f\n", sum_dose_pow, std::pow(sum_dose_pow/size, 1/a));
	backwardUpperKernel<<<GRIDDIM, BLOCKDIM>>>(d_dose, d_dose_grad, sum_dose_pow, target, a, size, sign);
	cudaFree(d_sum);
	int temp = 0;
	if (sign * (std::pow(sum_dose_pow/size,1/a) - target) >= 0)
		temp = 1;
	return temp * 0.5 * std::pow(std::pow(sum_dose_pow/size,1/a)-target,2);
}

int main(int argc, char *args[]) {
	int size = 300;
	int sign = 1;
	float a = 2;
	float target = 100.0f;
	float *h_dose      = (float *)malloc(sizeof(float) * size);
	float *h_dose_grad = (float *)malloc(sizeof(float) * size);
	for (int i = 0; i < size; i++) {
		h_dose[i] = i;
	}
	float *d_dose;
	float *d_dose_grad;
	cudaMalloc((void **)&d_dose, sizeof(float) * size);
	cudaMalloc((void **)&d_dose_grad, sizeof(float) * size);

	cudaMemcpy(d_dose, h_dose, sizeof(float) * size, cudaMemcpyHostToDevice);
	float loss = calgEUDLoss(d_dose, d_dose_grad, target, a, size, sign);
	cudaMemcpy(h_dose_grad, d_dose_grad, sizeof(float) * size, cudaMemcpyDeviceToHost);
	printf("loss:%f\n", loss);
	for (int i = 0; i < size; i++) {
	 printf("the %dth grad:%f\n", i, h_dose_grad[i]);
	}
	free(h_dose);
	free(h_dose_grad);
	cudaFree(d_dose);
	cudaFree(d_dose_grad);
	return 0;
}
