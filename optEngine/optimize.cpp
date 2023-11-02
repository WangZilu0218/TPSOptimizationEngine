//
// Created by 王子路 on 2023/10/31.
//
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/stl.h"
#include "options.h"
#include "fista.h"
#include "cuda_runtime_api.h"
//#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "googletest/googletest/include/gtest/gtest.h"
float optimizeStep(pybind11::array nzData,
				  pybind11::array indices,
				  pybind11::array indptr,
				  pybind11::array Xold,
				  pybind11::array Yold,
				  pybind11::list  lossName,
				  pybind11::dict  optparams,
				  pybind11::dict  lossparams,
				  int m,
				  int n) {
  auto h_nzData  = pybind11::cast<pybind11::array_t<float>>(nzData).request();
  auto h_x_old   = pybind11::cast<pybind11::array_t<float>>(Xold).request();
  auto h_y_old   = pybind11::cast<pybind11::array_t<float>>(Yold).request();
  auto h_indices = pybind11::cast<pybind11::array_t<int>>(indices).request();
  auto h_indptr  = pybind11::cast<pybind11::array_t<int>>(indptr).request();
  lossParams lp;
  opts op;
  lp.minDoseValue    = pybind11::cast<float>(lossparams["min dose"]);
  lp.maxDoseValue    = pybind11::cast<float>(lossparams["max dose"]);
  lp.d1              = pybind11::cast<float>(lossparams["d1"]);
  lp.d2              = pybind11::cast<float>(lossparams["d2"]);
  lp.v1              = pybind11::cast<float>(lossparams["v1"]);
  lp.v2              = pybind11::cast<float>(lossparams["v2"]);
  lp.upperGEUDTarget = pybind11::cast<float>(lossparams["upper gEUD"]);
  lp.lowerGEUDTarget = pybind11::cast<float>(lossparams["lower gEUD"]);
  lp.GEUDTarget      = pybind11::cast<float>(lossparams["gEUD target"]);
  lp.a               = pybind11::cast<float>(lossparams["a"]);
  auto h_dose_value  = pybind11::cast<pybind11::array_t<float>>(lossparams["dose value"]);
  lp.p_dose          = (float *)h_dose_value.ptr();
  lp.lossName        = lossName.cast<std::vector<std::string>>();

  op.tol             = pybind11::cast<float>(optparams["tolerance"]);
  op.L0              = pybind11::cast<float>(optparams["L0"]);
  op.lambda          = pybind11::cast<float>(optparams["lambda"]);
  op.eta             = pybind11::cast<float>(optparams["eta"]);
  op.pos             = pybind11::cast<bool>(optparams["pos"]);
//  for (std::pair<pybind11::handle, pybind11::handle> item: lossparams) {
//	auto key = item.first.cast<std::string>();
//	auto value = item.second.cast<float>();
//	if (key.compare("min dose") != 0) {
//	  lp.minDoseValue = value;
//	  break;
//	}
//	if (key.compare("max dose") != 0) {
//	  lp.maxDoseValue = value;
//	  break;
//	}
//	if (key.compare("d1") != 0) {
//	  lp.d1 = value;
//	  break;
//	}
//	if (key.compare("v1") != 0) {
//	  lp.v1 = value;
//	  break;
//	}
//	if (key.compare("d2") != 0) {
//	  lp.d2 = value;
//	  break;
//	}
//	if (key.compare("v2") != 0) {
//	  lp.v2 = value;
//	  break;
//	}
//	if (key.compare("upper g EUD") != 0) {
//	  lp.upperGEUDTarget = value;
//	  break;
//	}
//	if (key.compare("lower g EUD") != 0) {
//	  lp.lowerGEUDTarget = value;
//	  break;
//	}
//	if (key.compare("g EUD target") != 0) {
//	  lp.GEUDTarget = value;
//	  break;
//	}
//	if (key.compare("a") != 0) {
//	  lp.a = value;
//	  break;
//	}
//  }
//  for (std::pair<pybind11::handle, pybind11::handle> item: optparams) {
//	auto key = item.first.cast<std::string>();
//	auto value = item.second.cast<float>();
//	if (key.compare("tolerance") != 0) {
//	  op.tol = value;
//	  break;
//	}
//	if (key.compare("L0") != 0) {
//	  op.L0 = value;
//	  break;
//	}
//	if (key.compare("eta") != 0) {
//	  op.eta = value;
//	  break;
//	}
//  }

  fista fis(op,
			lp,
			(float *)h_x_old.ptr,
			(float *)h_y_old.ptr,
			(float *)h_nzData.ptr,
			(int *)h_indices.ptr,
			(int *)h_indptr.ptr,
			m,
			n,
			h_nzData.size);
  float loss = fis.step();
  checkCudaErrors(cudaMemcpy(h_x_old.ptr, fis.dXOld, sizeof(float) * n, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_old.ptr, fis.dYOld, sizeof(float) * n, cudaMemcpyDeviceToHost));
  optparams["L0"] = fis.L;
  return loss;
}

PYBIND11_MODULE(opti, m) {
  m.def("optimizeStep", optimizeStep);
}