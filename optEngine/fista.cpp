//
// Created by 王子路 on 2023/10/18.
//

#include "fista.h"
#include "helper_cuda.h"
#include "enzymeAutoGrad/loss.h"
#include "cublas_api.h"
#define CUBLAS_SAFE_CALL(call)                                                     \
{                                                                                  \
  const cublasStatus_t stat = call;                                                \
  if (stat != CUBLAS_STATUS_SUCCESS) {                                             \
    std::cout << "cuBlas Error: " << __FILE__ << ":" << __LINE__ << std::endl;     \
    std::cout << "  Code: " << stat << std::endl;                                  \
    exit(1);                                                                       \
  }                                                                                \
}

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
  CUBLAS_SAFE_CALL(cublasCreate(&handle_));

}

float fista::calc_F(float *pX) {
  calDoseLoss(dDose, dDoseGrad, dLoss, 100, csc.m, 1);
  return 0.0f;
}

float fista::cal_loss(float *dose) {
  float *tempDoseGradBuffer;
  checkCudaErrors(cudaMalloc((void **)&tempDoseGradBuffer, sizeof(float) * csc.m));
  float result = 0.0f;
  float alpha = 1.0f;
  for (auto iter: op.lossName) {
	if (iter.compare("minDoseLoss") == 0) {
	  result += calDoseLoss(dose, tempDoseGradBuffer, dLoss, minDoseValue, csc.m, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("maxDoseLoss") == 0) {
	  result += calDoseLoss(dose, tempDoseGradBuffer, dLoss, minDoseValue, csc.m, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("minDVHLoss") == 0) {
	  result += calDVHLoss(dose, tempDoseGradBuffer, dLoss, d1, csc.m, v1, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("maxDVHLoss") == 0) {
	  result += calDVHLoss(dose, tempDoseGradBuffer, dLoss, d1, csc.m, v1, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("lowerGEUDLoss") == 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, lowerGEUDTarget, a, csc.m, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("targetGEUDLoss") == 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, GEUDTarget, a, csc.m, 0);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("upperGEUDLoss") == 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, upperGEUDTarget, a, csc.m, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	}
  }
  checkCudaErrors(cudaFree(tempDoseGradBuffer));
  return result;
}

float fista::calculateQ(float *x, float *y, float L) {
  float temp = 0.0f;
  float *tempBuffer;
  float alpha = -1.0f;
  checkCudaErrors(cudaMalloc((void **)&tempBuffer, sizeof(float) * csc.n));
  checkCudaErrors(cudaMemcpy(tempBuffer, y, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, x, 1, tempBuffer, 1));
//  subVec(x, y, tempBuffer, csc.n);
  csc.forward(y);
  temp += cal_loss(csc.y_forward_d) + g(y, dLoss, op.lambda, csc.n);
  csc.backward(y);
  float nrm, dt;
  CUBLAS_SAFE_CALL(cublasSnrm2(handle_, csc.n, tempBuffer, 1, &nrm));
  CUBLAS_SAFE_CALL(cublasSdot_v2(handle_, csc.n, tempBuffer, 1, csc.y_backward_d, 1, &dt));
  temp += dt + L / 2 * nrm + g(x, dLoss, op.lambda, csc.n);
  checkCudaErrors(cudaFree(tempBuffer));
  return temp;
}

void fista::step() {
  iter ++;
  float Lbar = L;
  float *zk;
  checkCudaErrors(cudaMalloc((void **)&zk, sizeof(float) * csc.n));
  while (true) {
	op0.lambda = op.lambda / Lbar;
  }
  checkCudaErrors(cudaFree(zk));
}

fista::~fista() {
  checkCudaErrors(cudaFree(dXOld));
  checkCudaErrors(cudaFree(dXNew));
  checkCudaErrors(cudaFree(dYOld));
  checkCudaErrors(cudaFree(dYNew));
  checkCudaErrors(cudaFree(dLoss));
}