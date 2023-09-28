#include "csc.h"

CSC::CSC()
{
    m = 0;
    n = 0;
    nnz = 0;
    nzdata = nullptr;
    indices = nullptr;
    indptr = nullptr;
}

void CSC::initFromFile(std::string nzdataFile, std::string indicesFile, std::string indptrFile, int nrow, int ncol)
{
    // std::cout << "Initialize CSC sparse matrix " << nrow << " x " << ncol << std::endl;
    m = nrow;
    n = ncol;

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
}

void CSC::initFromMemory(float *nzdata1, int* indices1, int *indptr1, int nrow, int ncol, int nz)
{
    // std::cout << "Initialize CSC sparse matrix " << nrow << " x " << ncol << std::endl;
    m = nrow;
    n = ncol;
    nnz = nz;

    nzdata = new float[nnz];
    indices = new int[nnz];
    indptr = new int[n+1];

    memcpy(nzdata, nzdata1, sizeof(float)*nnz);
    memcpy(indices, indices1, sizeof(int)*nnz);
    memcpy(indptr, indptr1, sizeof(int)*(n+1));
}

void CSC::copyToGPU()
{
    CUDA_CALL(cudaMalloc((void **) &nzdata_d, sizeof(float)*nnz));
    CUDA_CALL(cudaMalloc((void **) &indices_d, sizeof(int)*nnz));
    CUDA_CALL(cudaMalloc((void **) &indptr_d, sizeof(int)*(n+1)));
    
    CUDA_CALL(cudaMemcpy(nzdata_d, nzdata, sizeof(float)*nnz, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(indices_d, indices, sizeof(int)*nnz, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(indptr_d, indptr, sizeof(float)*(n+1), cudaMemcpyHostToDevice));
}

void CSC::freeGPU()
{
    if(nnz>0)
    {
        CUDA_CALL(cudaFree(indices_d));
        CUDA_CALL(cudaFree(indptr_d));
        CUDA_CALL(cudaFree(nzdata_d));

        // std::cout << "Free GPU CSC matrix success!\n";
    }
}

CSC::~CSC()
{
    if(nnz>0)
    {
        delete[] nzdata;
        delete[] indptr;
        delete[] indices;

        // std::cout << "Free CPU CSC matrix success!\n";
    }
}