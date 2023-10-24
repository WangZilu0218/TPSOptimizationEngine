#ifndef CSC_H
#define CSC_H

#include <iostream>
#include <fstream>

#include "common.cuh"
#include "cusparse.h"
#define BLOCKDIM 256
#define GRIDDIM  128
class CSC {
 private:
  /* data */
  bool copyGPUFlag, nnzDataFlag;

  cusparseHandle_t   handle = NULL;
  cudaStream_t       stream = NULL;
  cusparseStatus_t   status = CUSPARSE_STATUS_SUCCESS;
  cusparseMatDescr_t descrC = NULL;

 public:
  int   m;         // matrix size m rows by n columns
  int   n;         // m:voxels n:spots num
  int   nnz;       // number of non zero elements
  int   nnb;

  float *dose;
  float *dose_d;
  float *weights;
  float *weights_d;

  float *y_forward_d;
  float *y_backward_d;

  float *nzdata;   // non zero value
  int   *indices;  // row indices for CSC format   size:nnz
  int   *indptr;   // number of elements befor column j for CSC format size: n+1

  float *nzdata_d_;  // csc2csr
  int   *indices_d_; // csc2csr
  int   *indptr_d_;  // csc2csr

  int   *bsrRowPtrC_d;
  int   *bsrColIndC_d;
  float *bsrValC_d;

  int   *bsrRowPtrC_d_;
  int   *bsrColIndC_d_;
  float *bsrValC_d_;

  int   *m_d;      // _d means variable in GPU
  int   *n_d;
  float *nzdata_d;
  int   *indices_d;
  int   *indptr_d;

  CSC();
  ~CSC();

  void csc2csr2bsr();

  void forward();
  void backward();

  void initFromFile(std::string nzdataFile, std::string indiceFile, std::string indptrFile, int nrow, int ncol);
  void initFromMemory(float *nzdata1, int *indices1, int *indptr1, int nrow, int ncol, int nz);
  void copyToGPU();
  void freeGPU();
};

#endif