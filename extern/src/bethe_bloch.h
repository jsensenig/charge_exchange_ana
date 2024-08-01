#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "TSpline.h"

//class TSPline3;
namespace py = pybind11;

class TSpline3;

class BetheBloch {

 public:

  BetheBloch(double mass, int charge);

  double range_from_ke(double ke);

  py::array_t<double> ke_along_track(double init_ke, const py::array_t<double> &track_cumlen);

  py::array_t<double> ke_at_point(const py::array_t<double> &init_ke, const py::array_t<double> &track_cumlen); 

  double meandEdx(double KE);

  double IntegratedEdx(double KE0, double KE1, int n = 10000);

  double KEAtLength(double KE0, double tracklength);

  double ke_from_range_spline(double range);

  void CreateSplineAtKE(int iKE);

  void create_splines(int np, double minke, double maxke); // (1e3, 0.01, 2e5)

 protected:

  TSpline3 *sp_KE_range;
  TSpline3 *sp_range_KE;


 private:

  double _mass;
  int _charge;

  std::map<int, TSpline3*> spmap;

  double densityEffect(double beta, double gamma);

};
