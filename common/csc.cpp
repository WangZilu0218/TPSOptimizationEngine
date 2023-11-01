#include "csc.h"
#include "cuda_runtime_api.h"
#include "helper_cuda.h"
#include "assert.h"
#define CUSPARSE_SAFE_CALL(call)                                                     \
{                                                                                  \
  const cusparseStatus_t stat = call;                                                \
  if (stat != CUSPARSE_STATUS_SUCCESS) {                                             \
    std::cout << "cuSparse Error: " << __FILE__ << ":" << __LINE__ << std::endl;     \
    std::cout << "  Code: " << stat << std::endl;                                  \
    exit(1);                                                                       \
  }                                                                                \
}

CSC::CSC() {
  m = 0;
  n = 0;
  nnz = 0;
  nzdata = nullptr;
  indices = nullptr;
  indptr = nullptr;

  copyGPUFlag = false;
  nnzDataFlag = false;

  //init cusparse context
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUSPARSE_SAFE_CALL(cusparseCreate(&handle));
  CUSPARSE_SAFE_CALL(cusparseSetStream(handle, stream));
  CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&descrC));

  CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_SAFE_CALL(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
}

void CSC::initFromFile(std::string nzdataFile, std::string indicesFile, std::string indptrFile, int nrow, int ncol) {
  m = nrow;
  n = ncol;

  assert(m > 0);
  assert(n > 0);

  indptr = new int[ncol + 1];
  std::ifstream fp(indptrFile.c_str(), std::ios::binary);
  fp.read((char *)indptr, (ncol + 1) * sizeof(int));
  fp.close();

  fp.open(nzdataFile.c_str(), std::ios::binary);
  fp.seekg(0, std::ios::end);
  nnz = fp.tellg() / sizeof(float);
  nzdata = new float[nnz];
  indices = new int[nnz];
  fp.seekg(0, std::ios::beg);
  fp.read((char *)nzdata, nnz * sizeof(float));
  fp.close();

  fp.open(indicesFile.c_str(), std::ios::binary);
  fp.read((char *)indices, nnz * sizeof(float));
  fp.close();

  assert(m * n >= nnz);
  nnzDataFlag = true;
}

void CSC::initFromMemory(float *nzdata1, int *indices1, int *indptr1, int nrow, int ncol, int nz) {
  // std::cout << "Initialize CSC sparse matrix " << nrow << " x " << ncol << std::endl;
  m = nrow;
  n = ncol;
  nnz = nz;

  assert(m > 0);
  assert(n > 0);
  assert(m * n >= nnz);

  nzdata = new float[nnz];
  indices = new int[nnz];
  indptr = new int[n + 1];

  memcpy(nzdata, nzdata1, sizeof(float) * nnz);
  memcpy(indices, indices1, sizeof(int) * nnz);
  memcpy(indptr, indptr1, sizeof(int) * (n + 1));

  nnzDataFlag = true;
}

void CSC::copyToGPU() {
  checkCudaErrors(cudaMalloc((void **)&nzdata_d, sizeof(float) * nnz));
  checkCudaErrors(cudaMalloc((void **)&indices_d, sizeof(int) * nnz));
  checkCudaErrors(cudaMalloc((void **)&indptr_d, sizeof(int) * (n + 1)));

  checkCudaErrors(cudaMalloc((void **)&y_forward_d, sizeof(float) * m));
  checkCudaErrors(cudaMalloc((void **)&y_backward_d, sizeof(float) * n));

  checkCudaErrors(cudaMemcpy(nzdata_d, nzdata, sizeof(float) * nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(indices_d, indices, sizeof(int) * nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(indptr_d, indptr, sizeof(float) * (n + 1), cudaMemcpyHostToDevice));

  copyGPUFlag = true;
}

void CSC::freeGPU() {
  if (copyGPUFlag) {
	checkCudaErrors(cudaFree(bsrRowPtrC_d));
	checkCudaErrors(cudaFree(bsrColIndC_d));
	checkCudaErrors(cudaFree(bsrValC_d));

	checkCudaErrors(cudaFree(bsrRowPtrC_d_));
	checkCudaErrors(cudaFree(bsrColIndC_d_));
	checkCudaErrors(cudaFree(bsrValC_d_));

	checkCudaErrors(cudaFree(y_forward_d));
	checkCudaErrors(cudaFree(y_backward_d));
//	checkCudaErrors(cudaFree(indices_d));
//	checkCudaErrors(cudaFree(indptr_d));
//	checkCudaErrors(cudaFree(nzdata_d));

//	checkCudaErrors(cudaFree(indices_d_));
//	checkCudaErrors(cudaFree(indptr_d_));
//	checkCudaErrors(cudaFree(nzdata_d_));

	copyGPUFlag = false;
	// std::cout << "Free GPU CSC matrix success!\n";
  }
}

CSC::~CSC() {
  if (nnzDataFlag) {
	delete[] nzdata;
	delete[] indptr;
	delete[] indices;

	nnzDataFlag = false;
	// std::cout << "Free CPU CSC matrix success!\n";
  }
  cusparseDestroy(handle);
  cusparseDestroyMatDescr(descrC);
  checkCudaErrors(cudaStreamDestroy(stream));
}

void CSC::csc2csr2bsr() {
  checkCudaErrors(cudaMalloc((void **)&nzdata_d_,  sizeof(float) * nnz));
  checkCudaErrors(cudaMalloc((void **)&indices_d_, sizeof(int)   * nnz));
  checkCudaErrors(cudaMalloc((void **)&indptr_d_,  sizeof(int)   * (m + 1)));

  size_t lworkInBytes = 0;
  char *d_work = NULL;

  CUSPARSE_SAFE_CALL(cusparseCsr2cscEx2_bufferSize(handle,
												   n,
												   m,
												   nnz,
												   nzdata_d,
												   indptr_d,
												   indices_d,
												   nzdata_d_,
												   indptr_d_,
												   indices_d_,
												   CUDA_R_32F,
												   CUSPARSE_ACTION_NUMERIC,
												   CUSPARSE_INDEX_BASE_ZERO,
												   CUSPARSE_CSR2CSC_ALG1,
												   &lworkInBytes));

  printf("lworkInBytes (csr2csc) = %lld \n", (long long)lworkInBytes);
  checkCudaErrors(cudaMalloc((void **)&d_work, lworkInBytes));

  CUSPARSE_SAFE_CALL(cusparseCsr2cscEx2(handle,
										n,
										m,
										nnz,
										nzdata_d,
										indptr_d,
										indices_d,
										nzdata_d_,
										indptr_d_,
										indices_d_,
										CUDA_R_32F,
										CUSPARSE_ACTION_NUMERIC,
										CUSPARSE_INDEX_BASE_ZERO,
										CUSPARSE_CSR2CSC_ALG1,
										d_work));

  checkCudaErrors(cudaFree(d_work));

  //csr2bsr for forward
  int mb = (m + BLOCKDIM - 1) / BLOCKDIM;
  int nb = (n + BLOCKDIM - 1) / BLOCKDIM;
  checkCudaErrors(cudaMalloc((void **)&bsrRowPtrC_d_, sizeof(int) * (mb + 1)));
  CUSPARSE_SAFE_CALL(cusparseXcsr2bsrNnz(handle,
										 CUSPARSE_DIRECTION_COLUMN,
										 m,
										 n,
										 descrC,
										 indptr_d_,
										 indices_d_,
										 BLOCKDIM,
										 descrC,
										 bsrRowPtrC_d_,
										 &nnb);)

  //csr2bsr for backward
  checkCudaErrors(cudaMalloc((void **)&bsrRowPtrC_d, sizeof(int) * (nb + 1)));
  CUSPARSE_SAFE_CALL(cusparseXcsr2bsrNnz(handle,
										 CUSPARSE_DIRECTION_COLUMN,
										 n,
										 m,
										 descrC,
										 indptr_d,
										 indices_d,
										 BLOCKDIM,
										 descrC,
										 bsrRowPtrC_d,
										 &nnb));

//free csr data
  checkCudaErrors(cudaFree(indices_d));
  checkCudaErrors(cudaFree(indptr_d));
  checkCudaErrors(cudaFree(nzdata_d));

  checkCudaErrors(cudaFree(indices_d_));
  checkCudaErrors(cudaFree(indptr_d_));
  checkCudaErrors(cudaFree(nzdata_d_));
}

void CSC::forward(float *weights_d) {
  float alpha = 1;
  float beta  = 0;
  int   mb    = (m + BLOCKDIM - 1) / BLOCKDIM;
  int   nb    = (n + BLOCKDIM - 1) / BLOCKDIM;
  CUSPARSE_SAFE_CALL(cusparseSbsrmv(handle,
									CUSPARSE_DIRECTION_COLUMN,
									CUSPARSE_OPERATION_NON_TRANSPOSE,
									mb,
									nb,
									nnb,
									&alpha,
									descrC,
									bsrValC_d_,
									bsrRowPtrC_d_,
									bsrColIndC_d_,
									BLOCKDIM,
									weights_d,
									&beta,
									y_forward_d));
}

void CSC::backward(float *dose_d) {
  float alpha = 1;
  float beta  = 0;
  int   mb    = (m + BLOCKDIM - 1) / BLOCKDIM;
  int   nb    = (n + BLOCKDIM - 1) / BLOCKDIM;

  CUSPARSE_SAFE_CALL(cusparseSbsrmv(handle,
									CUSPARSE_DIRECTION_COLUMN,
									CUSPARSE_OPERATION_NON_TRANSPOSE,
									nb,
									mb,
									nnb,
									&alpha,
									descrC,
									bsrValC_d,
									bsrRowPtrC_d,
									bsrColIndC_d,
									BLOCKDIM,
									dose_d,
									&beta,
									y_backward_d));
}