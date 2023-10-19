//
// Created by 王子路 on 2023/10/18.
//

#ifndef OPTIMIZE_OPTENGINE_FISTA_H_
#define OPTIMIZE_OPTENGINE_FISTA_H_
#include "vector"
#include "options.h"
#include "common/csc.h"
using namespace std;
#define BLOCKDIM 256
#define GRIDDIM  128
class fista {
 public:
  fista();
  fista(const opts &, const CSC &, float *, float *, int *, int *, int, int, int);
  ~fista();
 private:
  float calculateQ();
  float calc_F(float *);
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
 public:
  vector<float> weights;

};

#endif //OPTIMIZE_OPTENGINE_FISTA_H_
