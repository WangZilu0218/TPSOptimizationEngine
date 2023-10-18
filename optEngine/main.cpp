#include "common.cuh"
#include "optEngine.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/stl.h"

void cudaOptEngine(pybind11::array_t<float> dvh) // target on no-overlapping region and multiple objectives/constraints first as of 20231008
{

}

PYBIND11_MODULE(cudaOptEngine, m) {
  m.def("cuOptEngine", cudaOptEngine);
}