#ifndef CSC_H
#define CSC_H

#include "global.h"
#include <iostream>
#include <fstream>

class CSC
{
private:
    /* data */
    bool copyGPUFlag, nnzDataFlag;
public:
    int m;
    int n;
    int nnz;
    float *nzdata; // non zero value
    int *indices; // row indices for CSC format
    int *indptr; // number of elements befor column j for CSC format
    
    int *m_d;
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