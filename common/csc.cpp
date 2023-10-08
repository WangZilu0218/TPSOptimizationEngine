#include "csc.h"

CSC::CSC()
{
    m = 0;
    n = 0;
    nnz = 0;
    nzdata = nullptr;
    indices = nullptr;
    indptr = nullptr;

    copyGPUFlag = false;
    nnzDataFlag = false;
}

void CSC::initFromFile(std::string nzdataFile, std::string indicesFile, std::string indptrFile, int nrow, int ncol)
{
    m = nrow;
    n = ncol;

    assert(m>0);
    assert(n>0);

    indptr = new int[ncol+1];
    std::ifstream fp(indptrFile.c_str(),std::ios::binary);
    fp.read((char*)indptr,(ncol+1)*sizeof(int));
    fp.close();

    fp.open(nzdataFile.c_str(),std::ios::binary);
	fp.seekg(0,std::ios::end);
    nnz = fp.tellg()/sizeof(float);
    nzdata = new float[nnz];
    indices = new int[nnz];
    fp.seekg(0,std::ios::beg);
    fp.read((char*)nzdata,nnz*sizeof(float));
    fp.close();

    fp.open(indicesFile.c_str(),std::ios::binary);
    fp.read((char*)indices,nnz*sizeof(float));
    fp.close();

    assert(m*n>=nnz);

    nnzDataFlag = true; 
}

void CSC::initFromMemory(float *nzdata1, int* indices1, int *indptr1, int nrow, int ncol, int nz)
{
    // std::cout << "Initialize CSC sparse matrix " << nrow << " x " << ncol << std::endl;
    m = nrow;
    n = ncol;
    nnz = nz;

    assert(m>0);
    assert(n>0);
    assert(m*n>=nnz);

    nzdata = new float[nnz];
    indices = new int[nnz];
    indptr = new int[n+1];

    memcpy(nzdata, nzdata1, sizeof(float)*nnz);
    memcpy(indices, indices1, sizeof(int)*nnz);
    memcpy(indptr, indptr1, sizeof(int)*(n+1));

    nnzDataFlag = true; 
}

void CSC::copyToGPU()
{
    checkCudaErrors(cudaMalloc((void **) &nzdata_d, sizeof(float)*nnz));
    checkCudaErrors(cudaMalloc((void **) &indices_d, sizeof(int)*nnz));
    checkCudaErrors(cudaMalloc((void **) &indptr_d, sizeof(int)*(n+1)));
    
    checkCudaErrors(cudaMemcpy(nzdata_d, nzdata, sizeof(float)*nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(indices_d, indices, sizeof(int)*nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(indptr_d, indptr, sizeof(float)*(n+1), cudaMemcpyHostToDevice));

    copyGPUFlag = true;
}

void CSC::freeGPU()
{
    if(copyGPUFlag)
    {
        checkCudaErrors(cudaFree(indices_d));
        checkCudaErrors(cudaFree(indptr_d));
        checkCudaErrors(cudaFree(nzdata_d));

        copyGPUFlag = false;
        // std::cout << "Free GPU CSC matrix success!\n";
    }
}

CSC::~CSC()
{
    if(nnzDataFlag)
    {
        delete[] nzdata;
        delete[] indptr;
        delete[] indices;

        nnzDataFlag = false;
        // std::cout << "Free CPU CSC matrix success!\n";
    }
}