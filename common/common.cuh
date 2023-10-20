#ifndef CUDACMC__COMMON_H_
#define CUDACMC__COMMON_H_
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "vector_types.h"
#include "iostream"
#include "math.h"
#include "cstring"

// #define NDEBUG
#include <assert.h>

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif



typedef struct __align__(32) {
float x, y, z, w, o, k, m, n;
} float8;

inline __host__ __device__ float8 make_float8(float x, float y, float z, float w, float o, float k, float m, float n) {
  float8 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  t.o = o;
  t.k = k;
  t.m = m;
  t.n = n;
  return t;
}

inline __host__ __device__ float8 operator+(float8 a, float8 b) {
return make_float8(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w, a.o + b.o, a.k + b.k, a.m + b.m, a.n + b.n);
}

inline __host__ __device__ float8 operator-(float8 a, float8 b) {
return make_float8(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w, a.o - b.o, a.k - b.k, a.m - b.m, a.n - b.n);
}

inline __host__ __device__ float8 operator/(float8 a, float b) {
return make_float8(a.x / b, a.y / b, a.z / b, a.w / b, a.o / b, a.k / b, a.m / b, a.n / b);
}

inline __host__ __device__ void operator+=(float8 &a, float8 b) {
a.x += b.x;
a.y += b.y;
a.z += b.z;
a.w += b.w;
a.o += b.o;
a.k += b.k;
a.m += b.m;
a.n += b.n;
}

inline __host__ __device__ void operator/=(float8 &a, float b) {
a.x /= b;
a.y /= b;
a.z /= b;
a.w /= b;
a.o /= b;
a.k /= b;
a.m /= b;
a.n /= b;
}

inline __host__ __device__ float8 operator*(float8 a, float8 b) {
return make_float8(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w, a.o * b.o, a.k * b.k, a.m * b.m, a.n * b.n);
}

inline __host__ __device__ float8 operator*(float8 a, float b) {
return make_float8(a.x * b, a.y * b, a.z * b, a.w * b, a.o * b, a.k * b, a.m * b, a.n * b);
}

// inline __device__ void transform(vec3f *dir, float theta, float phi) {
//
//   float temp = 1.0 - 1e-7;
//   if ((*dir).z * (*dir).z >= temp) {
// 	if ((*dir).z > 0) {
// 	  (*dir).x = sinf(theta) * cosf(phi);
// 	  (*dir).y = sinf(theta) * sinf(phi);
// 	  (*dir).z = cosf(theta);
// 	} else {
// 	  (*dir).x = -sinf(theta) * cosf(phi);
// 	  (*dir).y = -sinf(theta) * sinf(phi);
// 	  (*dir).z = -cosf(theta);
// 	}
//   } else {
// 	float u, v, w;
// 	u = (*dir).x * cosf(theta)
// 		+ sinf(theta) * ((*dir).x * (*dir).z * cosf(phi) - (*dir).y * sinf(phi)) / sqrtf(1.0 - (*dir).z * (*dir).z);
// 	v = (*dir).y * cosf(theta)
// 		+ sinf(theta) * ((*dir).y * (*dir).z * cosf(phi) + (*dir).x * sinf(phi)) / sqrtf(1.0 - (*dir).z * (*dir).z);
// 	w = (*dir).z * cosf(theta) - sqrtf(1.0f - (*dir).z * (*dir).z) * sinf(theta) * cosf(phi);
//
// 	(*dir).x = u;
// 	(*dir).y = v;
// 	(*dir).z = w;
//   }
//   *dir = normalize(*dir);
// }

enum sampler { LINEAR, POINTS };
enum channels { R, RG, RGB, RGBA };

template<typename T>
__host__ void queryDimGrid(T *f, dim3 &dimGrid) {
  int numBlocksPerSm = 0;
  int numThreads = BLOCKDIM;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, f, numThreads, 0);
  dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
  dimGrid.y = 1;
  dimGrid.z = 1;
}

template<typename T>
__host__ void create3DTexture(T *hp,
							  cudaArray_t *cu_array_t,
							  cudaTextureObject_t *cuObj,
							  sampler s,
							  channels c,
							  size_t w,
							  size_t h,
							  size_t d,
							  int gpuId) {
  cudaSetDevice(gpuId);
  cudaChannelFormatDesc channelDesc;
  if (sizeof(T) == 4 && c == R) {
	channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 8 && c == RG) {
	channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 12 && c == RGB) {
	channelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 16 && c == RGBA) {
	channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  } else {
	printf("not supported texture format!\n");
	throw -1;
  }

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  switch (s) {
	case LINEAR: texDesc.filterMode = cudaFilterModeLinear;
	  break;
	case POINTS: texDesc.filterMode = cudaFilterModePoint;
  }
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  checkCudaErrors(cudaMalloc3DArray(cu_array_t, &channelDesc, make_cudaExtent(w, h, d)));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr = make_cudaPitchedPtr(hp, w * sizeof(T), w, h);
  myparms.dstArray = *cu_array_t;
  myparms.extent = make_cudaExtent(w, h, d);
  myparms.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&myparms));
  resDesc.res.array.array = *cu_array_t;
  checkCudaErrors(cudaCreateTextureObject(cuObj, &resDesc, &texDesc, NULL));
}

template<typename T>
__host__ void create2DTexture(T *hP,
							  cudaArray_t *cu_array_t,
							  cudaTextureObject_t *cuObj,
							  sampler s,
							  channels c,
							  size_t w,
							  size_t h,
							  int gpuId) {
  cudaSetDevice(gpuId);
  cudaChannelFormatDesc cuChannelFormatDesc;
  if (sizeof(T) == 4 && c == R) {
	cuChannelFormatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 8 && c == RG) {
	cuChannelFormatDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 12 && c == RGB) {
	cuChannelFormatDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
  } else if (sizeof(T) == 16 && c == RGBA) {
	cuChannelFormatDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  } else {
	printf("not supported texture format!\n");
	throw -1;
  }
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  const int spitch = w * sizeof(T);
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  switch (s) {
	case LINEAR: texDesc.filterMode = cudaFilterModeLinear;
	  break;
	case POINTS: texDesc.filterMode = cudaFilterModePoint;
  }
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  checkCudaErrors(cudaMallocArray(cu_array_t, &cuChannelFormatDesc, w, h));
  checkCudaErrors(cudaMemcpy2DToArray(*cu_array_t,
									  0,
									  0,
									  hP,
									  spitch,
									  w * sizeof(T),
									  h,
									  cudaMemcpyHostToDevice));
  resDesc.res.array.array = *cu_array_t;
  checkCudaErrors(cudaCreateTextureObject(cuObj, &resDesc, &texDesc, NULL));
}

template<typename T>
__host__ int queryBatchSize(T *f, int gpuId) {
  cudaSetDevice(gpuId);
  int numBlocksPerSm = 0;
  int numThreads = BLOCKDIM;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, f, numThreads, 0);
  return numThreads * deviceProp.multiProcessorCount * numBlocksPerSm;
}

template<typename T>
void launchCudaKernel(T *f, size_t batchSize, void *args[]) {
  int numThreads = BLOCKDIM;
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid((int)ceil((float)batchSize / (float)numThreads), 1, 1);
//  cudaLaunchKernel((void *)f, dimGrid, dimBlock, args);
  cudaLaunchCooperativeKernel((void *)f, dimGrid, dimBlock, args, 0, 0);
  cudaDeviceSynchronize();
}

inline size_t cuMemQueryRemain(int gid) {
  checkCudaErrors(cudaSetDevice(gid));
  size_t free, total;
  //get gpu memory info
  cudaError_t error = cudaMemGetInfo(&free, &total);
  if (error != cudaSuccess) {
	return 0;
  }
  return free;
}

inline int cuMemTestAlloc(unsigned long long size, int gid) {
  float *gpu_dosemap;
  checkCudaErrors(cudaSetDevice(gid));
  //allocate gpu memory
  cudaError_t error = cudaMalloc((void **)&gpu_dosemap, size * sizeof(char));
  if (error != cudaSuccess) {
	return -1;
  }
  // printf("gpu memory alloc done\n");
  checkCudaErrors(cudaFree(gpu_dosemap));
  return 1;
}

#endif //CUDACMC__COMMON_H_
