//////////////////////////////////////////////////////////////////////////////////////////
//created by Wang Zilu on Oct 1st 2023
//
//this is a shared library used for minDVH loss and maxDVH loss calculation
//sign:1   max DVH loss
//sign:-1  min DVH loss
//////////////////////////////////////////////////////////////////////////////////////////
#include"stdlib.h"
#include"stdio.h"
#include<algorithm>
#include<exception>
#ifdef USE_THRUST
#include"thrust/sort.h"
#endif

#define BLOCKDIM 256
#define GRIDDIM  128


float __device__ loss(float *p_dose, float d1, float d2, int sign){
	float temp1 = 0.0f;
	float temp2 = 0.0f;
	if (sign * (p_dose[0] - d1) >= 0.0f)
		temp1 = 1;
	if (sign * (d2 - p_dose[0]) >= 0.0f)
		temp2 = 1;
	return temp1 * temp2 * (sign * (p_dose[0] - d1)) * (sign * (p_dose[0] - d1));
}

typedef float (*f_ptr)(float*, float, float, int);
extern void __device__ __enzyme_autodiff(f_ptr, int, float*, float*, int, float, float, int);

int __device__ enzyme_dup;
int __device__ enzyme_out;
int __device__ enzyme_const;

void __global__ backwardKernel(float *d_dose, float *d_dose_grad, float *d_loss, float d1, float d2, int size, int sign) {
	float __shared__ cache[BLOCKDIM];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheId = threadIdx.x;
	float temp = 0.0f;
	while (idx < size) {
		float grad = 0.0f;
		__enzyme_autodiff(loss, enzyme_dup, d_dose + idx, &grad, enzyme_const, d1, d2, sign);
		d_dose_grad[idx] = grad / size;
		temp += loss(d_dose + idx, d1, d2, sign) / size;
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
	d_loss[blockIdx.x] = cache[0];
}

float calDVHLoss(float *d_dose, float *d_dose_grad, float *d_loss, float d1, int size, float v1, int sign){
	if (sign != 1 && sign != -1) {
		printf("sign should be 1 or -1.\n");
		return -1;
	}
	if (v1 < 0 || v1 > 1) {
                printf("v1 should between 0 and 1.\n");
		return -1;
        }

	if (size <= 0) {
		printf("size should greater than 0.\n");
		return -1;
	}

	if (d1 < 0) {
		printf("d1 should greater equal than 0.\n");
		return -1;
	}
	float loss = 0.0f;
	float *h_loss = (float *)malloc(sizeof(float) * GRIDDIM);
	float *h_dose = (float *)malloc(sizeof(float) * size);
	cudaMemcpy(h_dose, d_dose, sizeof(float) * size, cudaMemcpyDeviceToHost);
#ifdef USE_THRUST
	thrust::sort(h_dose, h_dose + size);
#else
	std::sort(h_dose, h_dose + size);
#endif
	float temp = v1 * size;
	int id1 = (int)temp;
	int id2 = id1 + 1;
	float d2 = 0.0f;
	if (id2 >= size)
		d2 = h_dose[size -1];
	else
		d2 = h_dose[id1] + (temp - id1) * (h_dose[id2] - h_dose[id1]);
	backwardKernel<<<GRIDDIM, BLOCKDIM>>>(d_dose, d_dose_grad, d_loss, d1, d2, size, sign);
	cudaMemcpy(h_loss, d_loss, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
	for (int i = 0; i < GRIDDIM; i++) {
		loss += h_loss[i];
	}

	free(h_loss);
	free(h_dose);
	return loss;
}

int main(int argc, char *args[]){
	int size = 300;
	int sign = -1;
	float v = 0.3;
	float d1 = 150;
        float *h_dose = (float *)malloc(size * sizeof(float));
        float *h_dose_grad = (float *)malloc(size * sizeof(float));
        for (int i = 0; i < size; i++) {
                h_dose[i] = i;
        }
        float *d_dose;
        float *d_dose_grad;
        float *d_loss;
        cudaMalloc((void **)&d_dose, sizeof(float) * size);
        cudaMalloc((void **)&d_dose_grad, sizeof(float) * size);
        cudaMalloc((void **)&d_loss, sizeof(float) * GRIDDIM);
        cudaMemcpy(d_dose, h_dose, sizeof(float) * size, cudaMemcpyHostToDevice);
	float loss = calDVHLoss(d_dose, d_dose_grad, d_loss, d1, size, v, sign);

	cudaMemcpy(h_dose_grad, d_dose_grad, sizeof(float) * size, cudaMemcpyDeviceToHost);

	printf("loss:%f\n", loss);

	for (int i = 0; i < size; i++) {
		printf("the %d th grad:%f\n", i, h_dose_grad[i]);
	}
	free(h_dose);
	free(h_dose_grad);
	cudaFree(d_dose);
	cudaFree(d_dose_grad);
	cudaFree(d_loss);
	return 0;
}
