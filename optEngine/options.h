//
// Created by 王子路 on 2023/10/17.
//

#ifndef OPTIMIZE_OPTENGINE_OPTIONS_H_
#define OPTIMIZE_OPTENGINE_OPTIONS_H_
#include <string>
#include "vector"
#define BLOCKDIM 256
#define GRIDDIM  128

typedef struct {
  int    max_iter = 500;
  float  tol = 1e-8;
  float  L0 = 1;
  float  eta = 2;
  bool   pos = false;
  float  lambda = 0.01;
} opts;

typedef struct {
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
  std::vector<std::string> lossName;
  float *p_dose;
} lossParams;
#endif //OPTIMIZE_OPTENGINE_OPTIONS_H_
