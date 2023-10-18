//
// Created by 王子路 on 2023/10/18.
//

#ifndef OPTIMIZE_OPTENGINE_FISTA_H_
#define OPTIMIZE_OPTENGINE_FISTA_H_
#include "vector"
#include "common.h"
using namespace std;
class fista {
 public:
  fista();
  fista(const opts &, int *, int *, float *, float *, int , int);
  ~fista();
 private:
  float calculateQ();
  void optimize()
  void calculateQ();
  void step();
 private:
  const opts op;

 private:
  int   *dCscColInd;
  int   *dRowInd;
  float *dValues;
  float *dWeights;

  float *dXOld;
  float *dXNew;
  float *dYOld;
  float *dYNew;
  int   nnz;
  int   numSpots;
};

#endif //OPTIMIZE_OPTENGINE_FISTA_H_
