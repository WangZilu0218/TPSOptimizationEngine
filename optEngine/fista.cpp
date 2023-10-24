//
// Created by 王子路 on 2023/10/18.
//

#include "fista.h"
#include "helper_cuda.h"
#include "cusparse.h"
#include "enzymeAutoGrad/loss.h"
fista::fista(const opts &op,
			 const CSC &csc,
			 float *pWeights,
			 float *nzdata1,
			 int *indices1,
			 int *indptr1,
			 int nrow,
			 int ncol,
			 int nz)
	: op(op), csc(csc) {

  weights.resize(csc.n);
  memcpy(weights.data(), pWeights, sizeof(float) * csc.n);
  this->csc.initFromMemory(nzdata1, indices1, indptr1, nrow, ncol, nz);

  checkCudaErrors(cudaMalloc((void **)&dXOld, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dXNew, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dYOld, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dYNew, sizeof(float) * csc.n));

  checkCudaErrors(cudaMalloc((void **)&dDose, sizeof(float) * csc.m));
  checkCudaErrors(cudaMalloc((void **)&dDoseGrad, sizeof(float) * csc.m));
  checkCudaErrors(cudaMalloc((void **)&dLoss, sizeof(float) * GRIDDIM));

  checkCudaErrors(cudaMemcpy(dXOld, pWeights, sizeof(float) * csc.n, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dYOld, pWeights, sizeof(float) * csc.n, cudaMemcpyHostToDevice));
}

float fista::calc_F(float *pX) {
  calDoseLoss(dDose, dDoseGrad, dLoss, 100, csc.m, 1);
  return 0.0f;
}

void fista::step() {

}

void fista::forwardMV() {
  cusparseHandle_t handle = NULL;
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrC = NULL;
  int nnzb;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
  int mb = (csc.m + BLOCKDIM - 1) / BLOCKDIM;
  int nb = (csc.n + BLOCKDIM - 1) / BLOCKDIM;

  cudaMalloc((void **)&bsrRowPtrC, sizeof(int) * (mb + 1));
  cusparseXcsr2bsrNnz(handle, dirA, csc.m, csc.n,
					  descrA, csc.indptr_d, csc.indices_d, BLOCKDIM,
					  descrC, bsrRowPtrC, &nnzb);
  cudaMalloc((void **)&bsrColIndC, sizeof(int) * nnzb);
  cudaMalloc((void **)&bsrValC, sizeof(float) * (BLOCKDIM * BLOCKDIM) * nnzb);
  cusparseScsr2bsr(handle, dirA, csc.m, csc.n,
				   descrA, csc.nzdata, csrRowPtrA, csrColIndA, BLOCKDIM,
				   descrC, bsrValC, bsrRowPtrC, bsrColIndC);

}

void fista::backwardMV() {
  cusparseHandle_t handle = NULL;
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrC = NULL;
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
}

fista::~fista() {
  checkCudaErrors(cudaFree(dXOld));
  checkCudaErrors(cudaFree(dXNew));
  checkCudaErrors(cudaFree(dYOld));
  checkCudaErrors(cudaFree(dYNew));
}