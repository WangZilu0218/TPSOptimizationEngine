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
  int    L0 = 1;
  int    eta = 2;
  bool   pos = false;
  float  lambda = 0.01;
  std::vector<std::string> lossName; //define loss name here
} opts;
#endif //OPTIMIZE_OPTENGINE_OPTIONS_H_
