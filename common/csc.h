#ifndef CSC_H
#define CSC_H

#include <iostream>
#include <fstream>

//#include "common.cuh"
#include "cusparse_v2.h"
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
  int   m;               // matrix size m rows by n columns
  int   n;               // m:voxels n:number of spots
  int   nnz;             // number of non zero elements
  int   nnb;

  float *y_forward_d;    //final dose, a m length vector
  float *y_backward_d;   //gradient of weights, a n length vector

  float *nzdata;         // non zero value
  int   *indices;        // row indices for CSC format   size:nnz
  int   *indptr;         // number of elements befor column j for CSC format size: n+1

  float *nzdata_d_;      // influence map csr
  int   *indices_d_;     //
  int   *indptr_d_;      //

  int   *bsrRowPtrC_d;   //transpose influence map bsr
  int   *bsrColIndC_d;
  float *bsrValC_d;

  int   *bsrRowPtrC_d_;  //influence map bsr
  int   *bsrColIndC_d_;
  float *bsrValC_d_;

  int   *m_d;            // _d means variable in GPU
  int   *n_d;
  float *nzdata_d;       //transpose influence map csr
  int   *indices_d;
  int   *indptr_d;

  CSC();
  ~CSC();

  void csc2csr2bsr();

  void forward(float *);  //mv influence map multiply weights
  void backward(float *); //mv transpose of influence map multiply gradients of final dose

  void initFromFile(std::string nzdataFile, std::string indiceFile, std::string indptrFile, int nrow, int ncol);
  void initFromMemory(float *nzdata1, int *indices1, int *indptr1, int nrow, int ncol, int nz);
  void copyToGPU();
  void freeGPU();
};

#endif