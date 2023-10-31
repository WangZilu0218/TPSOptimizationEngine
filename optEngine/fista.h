//
// Created by 王子路 on 2023/10/18.
//

#ifndef OPTIMIZE_OPTENGINE_FISTA_H_
#define OPTIMIZE_OPTENGINE_FISTA_H_
#include "vector"
#include "options.h"
#include "common/csc.h"
#include "cublas.h"
#include "cublas_v2.h"
using namespace std;

//void subVec(float *p_v1, float *p_v2, float *p_result, int size);
//void addVec(float *p_v1, float *p_v2, int size);
float g(float *d_v, float *d_sum, float lambda, int size);
//float dotVec(float *p_v1, float *p_v2, float *p_result, int size);
//float normF2(float *d_v, float *d_sum, int size);
void projection(float *p_v, float lambda, bool pos, int size);
void absVec(float *p_v, int size);

class fista {
 public:
  fista();
  fista(const opts &, const CSC &, float *, float *, int *, int *, int, int, int);
  ~fista();
 public:
  void setMinDoseValue(float minDoseValue) {this->minDoseValue = minDoseValue;}
  void setMaxDoseValue(float maxDoseValue) {this->maxDoseValue = maxDoseValue;}
 private:
  float calculateQ(float *, float *, float, float);
  float calc_F(float *);
  float cal_loss(float *);
  void optimize();
  bool step();

 private:
  const opts op;
  opts op0;
  lossParams losp;
  CSC csc;

 private:
  float *dWeights;
  float *dXOld;
  float *dXNew;
  float *dYOld;
  float *dYNew;

 private:
  float *dDose;
  float *dDoseGrad;
  float *dLoss;

 private:
  float minDoseValue;
  float maxDoseValue;
  float d1;
  float v1;
  float d2;
  float v2;
  float upperGEUDTarget;
  float lowerGEUDTarget;
  float GEUDTarget;
  float a;
  float *d_value;

  float L;
  float t_old;
  int iter;

 public:
  vector<float> weights;
  float loss;
 private:
  cublasHandle_t handle_;
  cublasStatus_t stat;
};

#endif //OPTIMIZE_OPTENGINE_FISTA_H_
