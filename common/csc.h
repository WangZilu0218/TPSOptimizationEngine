#ifndef CSC_H
#define CSC_H

#include <iostream>
#include <fstream>

#include "common.cuh"

class CSC
{
private:
    /* data */
    bool copyGPUFlag, nnzDataFlag;
public:
    int m; // matrix size m rows by n columns
    int n;
    int nnz; // number of non zero elements
    float *nzdata; // non zero value
    int *indices; // row indices for CSC format
    int *indptr; // number of elements befor column j for CSC format
    
    int *m_d; // _d means variable in GPU
    int *n_d;
    float *nzdata_d;
    int *indices_d;
    int *indptr_d;

    CSC();
    ~CSC();

    void initFromFile(std::string nzdataFile, std::string indiceFile, std::string indptrFile, int nrow, int ncol);
    void initFromMemory(float *nzdata1, int* indices1, int *indptr1, int nrow, int ncol, int nz);
    void copyToGPU();
    void freeGPU();
};

#endif