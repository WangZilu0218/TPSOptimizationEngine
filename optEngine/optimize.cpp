//
// Created by 王子路 on 2023/10/31.
//
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/stl.h"
#include "helper_cuda.h"
#include "options.h"

void optimizeStep(pybind11::array nzData,
				  pybind11::array indices,
				  pybind11::array indptr,
				  pybind11::array weights,
				  pybind11::list  lossName,
				  pybind11::dict  optparams,
				  pybind11::dict  lossparams) {
  auto h_nzData  = pybind11::cast<pybind11::array_t<float>>(nzData).request();
  auto h_weights = pybind11::cast<pybind11::array_t<float>>(weights).request();
  auto h_indices = pybind11::cast<pybind11::array_t<int>>(indices).request();
  auto h_indptr  = pybind11::cast<pybind11::array_t<int>>(indptr).request();
  lossParams lp;
  opts op;
  for (std::pair<pybind11::handle, pybind11::handle> item : lossparams) {
	auto key = item.first.cast<std::string>();
	auto value = item.second.cast<float>();
	if (key.compare("min dose") != 0) {
	  lp.minDoseValue = value;
	  break;
	}
	if (key.compare("max dose") != 0) {
	  lp.maxDoseValue = value;
	  break;
	}
	if (key.compare("d1") != 0) {
	  lp.d1 = value;
	  break;
	}
	if (key.compare("v1") != 0) {
	  lp.v1 = value;
	  break;
	}
	if (key.compare("d2") != 0) {
	  lp.d2 = value;
	  break;
	}
	if (key.compare("v2") != 0) {
	  lp.v2 = value;
	  break;
	}
	if (key.compare("upper g EUD") != 0) {
	  lp.upperGEUDTarget = value;
	  break;
	}
	if (key.compare("lower g EUD") != 0) {
	  lp.lowerGEUDTarget = value;
	  break;
	}
	if (key.compare("g EUD target") != 0) {
	  lp.GEUDTarget = value;
	  break;
	}
	if (key.compare("a") != 0) {
	  lp.a = value;
	  break;
	}
  }
  for (std::pair<pybind11::handle, pybind11::handle> item : optparams) {
	auto key   = item.first.cast<std::string>();
	auto value = item.second.cast<float>();
	if (key.compare("tolerance") != 0) {
	  op.tol = value;
	  break;
	}
	if (key.compare("L0") != 0) {
	  op.L0 = value;
	  break;
	}
	if (key.compare("eta") != 0) {
	  op.eta = value;
	  break;
	}
  }
  lp.lossName = lossName.cast<std::vector<std::string>>();
}