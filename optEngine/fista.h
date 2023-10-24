//
// Created by 王子路 on 2023/10/18.
//

#ifndef OPTIMIZE_OPTENGINE_FISTA_H_
#define OPTIMIZE_OPTENGINE_FISTA_H_
#include "vector"
#include "options.h"
#include "common/csc.h"
using namespace std;

void subVec(float *p_v1, float *p_v2, float *p_result, int size);
void addVec(float *p_v1, float *p_v2, int size);
float g(float *d_v, float *d_sum, float lambda, int size);
float dotVec(float *p_v1, float *p_v2, float *p_result, int size);
float normF2(float *d_v, float *d_sum, int size);

class fista {
 public:
  fista();
  fista(const opts &, const CSC &, float *, float *, int *, int *, int, int, int);
  ~fista();
 public:
  void setMinDoseValue(float minDoseValue) {this->minDoseValue = minDoseValue;}
  void setMaxDoseValue(float maxDoseValue) {this->maxDoseValue = maxDoseValue;}
 private:
  float calculateQ(float *, float *, float);
  float calc_F(float *);
  float cal_loss(float *);
  void optimize();
  void step();

 private:
  const opts op;
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

 public:
  vector<float> weights;
  float loss;
};

#endif //OPTIMIZE_OPTENGINE_FISTA_H_
