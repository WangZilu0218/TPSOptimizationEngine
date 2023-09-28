#ifndef CUDACMC_SRC_UTILS_CUDABUFFER_CUH_
#define CUDACMC_SRC_UTILS_CUDABUFFER_CUH_
#include "assert.h"
#include "cuda_runtime_api.h"
#include "helper_cuda.h"
#include "memory"
using namespace std;
template<typename T>
class CUDABuffer {
 private:
  shared_ptr<T> d_ptr;
  size_t length;
  size_t sizeInBytes;
  int gpuId;
 public:
  __host__ explicit CUDABuffer(const size_t length, int gpuId) : gpuId(gpuId), length(length), d_ptr([length, gpuId]() {
	cudaSetDevice(gpuId);
	T *ptr = nullptr;
	size_t free, total;
	checkCudaErrors(cudaMemGetInfo(&free, &total));
	if (sizeof(T) * length >= free) {
	  printf("not available gpu mem to alloc!\n");
	  throw -1;
	}
	if (length > 0) {
	  checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * length));
	  shared_ptr<T> t(ptr, [](T *ptr) { checkCudaErrors(cudaFree(ptr)); });
	  return t;
	} else {
	  shared_ptr<T> t(ptr);
	  return t;
	}
  }()) {
	if (length > 0) {
	  sizeInBytes = length * sizeof(T);
	  this->clear();
	} else {
	  sizeInBytes = 0;
	}
  }

  __host__ ~CUDABuffer() {}

  CUDABuffer(const CUDABuffer &cudaBuffer) = delete;
  CUDABuffer() = delete;
  CUDABuffer &operator=(const CUDABuffer &) = delete;

  __host__ T *addr(const int offset = 0) {
	return d_ptr.get() + offset;
  }

  __host__ shared_ptr<T> getSharedPtr() { return d_ptr; }

  __host__ void clear() {
	cudaSetDevice(gpuId);
	checkCudaErrors(cudaMemset(this->addr(), 0, sizeInBytes));
  }

  __host__ void reSize(size_t length) {
	d_ptr.reset();
	T *ptr;
	checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * length));
	shared_ptr<T> t(ptr, [](T *ptr) { checkCudaErrors(cudaFree(ptr)); });

	d_ptr = t;
	this->length = length;
	this->sizeInBytes = sizeof(T) * length;
  }

  __host__ size_t size() { return length; }

  __host__ void upload(T *p, size_t length) {
	cudaSetDevice(gpuId);
	assert(d_ptr.get() != nullptr);
	assert(sizeInBytes == length * sizeof(T));
	checkCudaErrors(cudaMemcpy(d_ptr.get(), (void *)p,
									sizeInBytes, cudaMemcpyHostToDevice));
  }

  __host__ void download(T *p, size_t length) {
	cudaSetDevice(gpuId);
	assert(d_ptr.get() != nullptr);
	assert(sizeInBytes == length * sizeof(T));
	checkCudaErrors(cudaMemcpy((void *)p, d_ptr.get(),
									sizeInBytes, cudaMemcpyDeviceToHost));
  }

  __host__ void copy(T *dp, size_t length, size_t offset) {
	cudaSetDevice(gpuId);
	assert(d_ptr.get() != nullptr);
	assert(dp != nullptr);
//	assert(sizeInBytes >= count * sizeof(T) && count * sizeof(T) >= 0);
	checkCudaErrors(cudaMemcpy(d_ptr.get(), dp + offset, sizeof(T) * length, cudaMemcpyDeviceToDevice));
  }
};

#endif //CUDACMC_SRC_UTILS_CUDABUFFER_CUH_
