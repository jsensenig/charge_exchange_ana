#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "TSpline.h"

//class TSPline3;
namespace py = pybind11;

class BetheBloch {

 public:

  BetheBloch(double mass, int charge);

  py::array_t<double> ke_along_track(double init_ke, const py::array_t<double> &track_cumlen);

  double meandEdx(double KE);

  double IntegratedEdx(double KE0, double KE1, int n = 10000);

  double KEAtLength(double KE0, double tracklength);

  void CreateSplineAtKE(int iKE);


 private:

  double _mass;
  int _charge;

  TSpline3 *sp_KE_range;
  TSpline3 *sp_range_KE;

  std::map<int, TSpline3*> spmap;

  double densityEffect(double beta, double gamma);

};
