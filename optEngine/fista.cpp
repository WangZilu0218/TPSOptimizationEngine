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
			 int   *indices1,
			 int   *indptr1,
			 int   nrow,
			 int   ncol,
			 int   nz)
	: op(op), csc(csc) {

  weights.resize(csc.n);
  memcpy(weights.data(), pWeights, sizeof(float) * csc.n);
  this->csc.initFromMemory(nzdata1, indices1, indptr1, nrow, ncol, nz);

  this->csc.csc2csr2bsr();

  checkCudaErrors(cudaMalloc((void **)&dXOld, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dXNew, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dYOld, sizeof(float) * csc.n));
  checkCudaErrors(cudaMalloc((void **)&dYNew, sizeof(float) * csc.n));

  checkCudaErrors(cudaMalloc((void **)&dDose, sizeof(float) * csc.m));
  checkCudaErrors(cudaMalloc((void **)&dDoseGrad, sizeof(float) * csc.m));
  checkCudaErrors(cudaMalloc((void **)&dLoss, sizeof(float) * GRIDDIM));

  checkCudaErrors(cudaMemcpy(dXOld, pWeights, sizeof(float) * csc.n, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dYOld, pWeights, sizeof(float) * csc.n, cudaMemcpyHostToDevice));
  CUBLAS_SAFE_CALL(cublasCreate_v2(&handle_));

}

float fista::calc_F(float *pX) {
  calDoseLoss(dDose, dDoseGrad, dLoss, 100, csc.m, 1);
  return 0.0f;
}

float fista::cal_loss(float *dose) {
  float *tempDoseGradBuffer;
  checkCudaErrors(cudaMalloc((void **)&tempDoseGradBuffer, sizeof(float) * csc.m));
  //set dose gradients to zeros
  checkCudaErrors(cudaMemset(dDoseGrad, 0, sizeof(float) * csc.m));
  float result = 0.0f;
  float alpha = 1.0f;
  for (auto iter: losp.lossName) {
	if (iter.compare("minDoseLoss") != 0) {
	  result += calDoseLoss(dose, tempDoseGradBuffer, dLoss, minDoseValue, csc.m, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("maxDoseLoss") != 0) {
	  result += calDoseLoss(dose, tempDoseGradBuffer, dLoss, minDoseValue, csc.m, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("minDVHLoss") != 0) {
	  result += calDVHLoss(dose, tempDoseGradBuffer, dLoss, d1, csc.m, v1, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("maxDVHLoss") != 0) {
	  result += calDVHLoss(dose, tempDoseGradBuffer, dLoss, d1, csc.m, v1, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("lowerGEUDLoss") != 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, lowerGEUDTarget, a, csc.m, -1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("targetGEUDLoss") != 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, GEUDTarget, a, csc.m, 0);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("upperGEUDLoss") != 0) {
	  result += calgEUDLoss(dose, tempDoseGradBuffer, upperGEUDTarget, a, csc.m, 1);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
//	  addVec(dDoseGrad, tempDoseGradBuffer, csc.m);
	} else if (iter.compare("uniformLoss") != 0) {
	  result += calUniformDoseLoss(dose, tempDoseGradBuffer, dLoss, d_value, csc.m);
	  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.m, &alpha, tempDoseGradBuffer, 1, dDoseGrad, 1));
	}
  }
  checkCudaErrors(cudaFree(tempDoseGradBuffer));
  return result;
}

float fista::calculateQ(float *x, float *y, float L, float loss) {
  float temp = 0.0f;
  float alpha = -1.0f;
  float nrm, dt;
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, y, 1, x, 1));
  CUBLAS_SAFE_CALL(cublasSnrm2_v2(handle_, csc.n, x, 1, &nrm));
  CUBLAS_SAFE_CALL(cublasSdot_v2(handle_, csc.n, x, 1, csc.y_backward_d, 1, &dt));
  temp += loss + dt + L / 2 * nrm + g(x, dLoss, op.lambda, csc.n);
  return temp;
}

bool fista::step() {
  bool ifBreak = false;
  iter ++;
  float Lbar = L;
  float *tempBuffer;
  checkCudaErrors(cudaMalloc((void **)&tempBuffer, sizeof(float) * csc.n));

  csc.forward(dYOld);
  loss = cal_loss(csc.y_forward_d);
  //here we get gradients of y_old
  csc.backward(dDoseGrad);

  while (true) {
	op0.lambda = op.lambda / Lbar;
	float alpha = -1 / Lbar;
	checkCudaErrors(cudaMemcpy(tempBuffer, dYOld, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
	CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, csc.y_backward_d, 1, tempBuffer, 1));
	projection(tempBuffer, op0.lambda, op0.pos, csc.n);
	csc.forward(tempBuffer);
	float F = cal_loss(csc.y_forward_d) + g(tempBuffer, dLoss, op.lambda, csc.n);
	float Q = calculateQ(tempBuffer, dYOld, Lbar, loss);
	if (F <= Q)
	  break;
	Lbar *= op.eta;
	L = Lbar;
  }
  checkCudaErrors(cudaMemcpy(dXNew, dYOld, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  float alpha = -1 / L;
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, csc.y_backward_d, 1, dXNew, 1));
  projection(dXNew, op.lambda / L, op.pos, csc.n);
  checkCudaErrors(cudaMemcpy(dYNew, dXNew, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  float t_new = 0.5 * (1 + sqrt(1 + 4 * pow(t_old, 2)));

  alpha = -1.0f;
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, dXOld, 1, dYNew, 1));
  alpha = (t_old - 1) / t_new;
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, dXNew, 1, dYNew, 1));
  //cal e
  checkCudaErrors(cudaMemcpy(tempBuffer, dXNew, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  alpha = -1.0f;
  CUBLAS_SAFE_CALL(cublasSaxpy_v2(handle_, csc.n, &alpha, dXOld, 1, tempBuffer, 1));
  float nrm1 = 0.0f;
  absVec(tempBuffer, csc.n);
  CUBLAS_SAFE_CALL(cublasSasum_v2(handle_, csc.n, tempBuffer, 1, &nrm1));
  nrm1 /= csc.n;
  if (nrm1 < op.tol)
	ifBreak = true;
  checkCudaErrors(cudaFree(tempBuffer));

  //update weights here
  checkCudaErrors(cudaMemcpy(dXOld, dXNew, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(dYOld, dYNew, sizeof(float) * csc.n, cudaMemcpyDeviceToDevice));
  t_old = t_new;
  return ifBreak;
}

fista::~fista() {
  checkCudaErrors(cudaFree(dXOld));
  checkCudaErrors(cudaFree(dXNew));
  checkCudaErrors(cudaFree(dYOld));
  checkCudaErrors(cudaFree(dYNew));
  checkCudaErrors(cudaFree(dLoss));
  CUBLAS_SAFE_CALL(cublasDestroy_v2(handle_));
}