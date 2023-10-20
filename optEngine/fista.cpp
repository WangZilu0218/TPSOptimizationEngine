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
}

void fista::step() {
  cusparseHandle_t    handle = NULL;
  cudaStream_t        stream = NULL;
  cusparseMatDescr_t  descrC = NULL;
  cusparseStatus_t    status = CUSPARSE_STATUS_SUCCESS;
  cusparseDirection_t dirA   = CUSPARSE_DIRECTION_COLUMN;
}

fista::~fista() {
  checkCudaErrors(cudaFree(dXOld));
  checkCudaErrors(cudaFree(dXNew));
  checkCudaErrors(cudaFree(dYOld));
  checkCudaErrors(cudaFree(dYNew));
}