#include"stdio.h"
#include"stdlib.h"

#define BLOCKDIM 256
#define GRIDDIM  32

float __device__ loss(float *dose, float d_value) {
        return (d_value - dose[0]) * (d_value - dose[0]);
}


typedef float (*f_ptr)(float*, float);
extern void __device__ __enzyme_autodiff(f_ptr, int, float*, float*, int, float);

int __device__ enzyme_dup;
int __device__ enzyme_out;
int __device__ enzyme_const;


void __global__ backwardKernel(float *p_dose, float *p_dose_grad, float *p_loss, float *d_value, int size) {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        int cacheIndex = threadIdx.x;
        __shared__ float cache[BLOCKDIM];
        float temp = 0.0f;
        while (id < size) {
                temp += loss(p_dose + id, d_value[id]);
                float d_dose = 0.0f;
                __enzyme_autodiff(loss, enzyme_dup, p_dose + id, &d_dose, enzyme_const, d_value[id]);
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

float calUniformDoseLoss(float *d_dose, float *d_dose_grad, float *d_loss, float *d_value, int size) {
        float loss = 0.0f;
        backwardKernel<<<GRIDDIM, BLOCKDIM>>>(d_dose, d_dose_grad, d_loss, d_value, size);
        float *h_loss = (float *)malloc(sizeof(float) * GRIDDIM);
        memset(h_loss, 0, sizeof(float) * GRIDDIM);
        cudaMemcpy(h_loss, d_loss, sizeof(float) * GRIDDIM, cudaMemcpyDeviceToHost);
        for (int i = 0; i < GRIDDIM; i++) {
                loss += h_loss[i];
        }
        free(h_loss);
        return loss;
}

int main(int argc, char *args[]) {
	return 0;
}
