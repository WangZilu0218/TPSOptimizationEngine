//
// Created by 王子路 on 2023/10/18.
//

#include "fista.h"
#include "cusparse.h"

fista::fista(const opts &op, int *pCscColInd, int *pRowInd, float *pValue, float *pWeights, int nnZ, int spotsNum)
	: op(op), nnz(nnZ), numSpots(spotsNum) {
  cudaMalloc((void **)&dCscColInd, sizeof(int)   * (numSpots + 1));
  cudaMalloc((void **)&dRowInd,    sizeof(int)   * nnZ);
  cudaMalloc((void **)&dValues,    sizeof(float) * nnZ);

  cudaMemcpy(dCscColInd, pCscColInd, sizeof(int) * (numSpots + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(dRowInd,    pRowInd,    sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(dValues,    pValue,     sizeof(float) * nnz, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&dXOld, sizeof(float) * spotsNum);
  cudaMalloc((void **)&dXNew, sizeof(float) * spotsNum);
  cudaMalloc((void **)&dYOld, sizeof(float) * spotsNum);
  cudaMalloc((void **)&dYNew, sizeof(float) * spotsNum);

  cudaMemcpy(dXOld, pWeights, sizeof(float) * spotsNum, cudaMemcpyHostToDevice);
  cudaMemcpy(dYOld, pWeights, sizeof(float) * spotsNum, cudaMemcpyHostToDevice);
}

fista::~fista() {
  cudaFree(dCscColInd);
  cudaFree(dRowInd);
  cudaFree(dValues);
  cudaFree(dXOld);
  cudaFree(dXNew);
  cudaFree(dYOld);
  cudaFree(dYNew);
}