#include <iostream>
#include <cmath>
#include <algorithm>
#include "bethe_bloch.h"

// Credit Kang Yang

namespace py = pybind11;

BetheBloch::BetheBloch(double mass, int charge) : _mass(mass), _charge(charge) {}

py::array_t<double> BetheBloch::ke_along_track(double init_ke, const py::array_t<double> &track_cumlen) {

  py::buffer_info buf_track_cumlen = track_cumlen.request();
  double *ptr_track_cumlen = static_cast<double *>(buf_track_cumlen.ptr);

  // Allocate result numpy buffer
  auto track_ke = py::array_t<double>(buf_track_cumlen.size);
  py::buffer_info buf_track_ke = track_ke.request();
  double *ptr_track_ke = static_cast<double *>(buf_track_ke.ptr);

  for (size_t i = 0; i < buf_track_cumlen.shape[0]; ++i) {
    ptr_track_ke[i] = KEAtLength(init_ke, ptr_track_cumlen[i]);
  }

  return track_ke;
}


double BetheBloch::KEAtLength(double KE0, double tracklength) {

  int iKE = int(KE0);

  if (spmap.find(iKE)==spmap.end()){
    CreateSplineAtKE(iKE);
  }

  double deltaE = spmap[iKE]->Eval(tracklength);

  if (deltaE < 0) return 0;
  if (KE0 - deltaE < 0) return 0;
  
  return KE0 - deltaE;

}

void BetheBloch::CreateSplineAtKE(int iKE){

  double KE0 = iKE;

  // Sample every 10 MeV
  int np = int(KE0 / 10);
  double *deltaE;
  double *trklength;
  if ( np > 1 ){
    deltaE = new double[np];
    trklength = new double[np];
    for( int i = 0; i<np; ++i ) {
      double KE = KE0 - i*10;
      deltaE[i] = KE0 - KE;
      trklength[i] = IntegratedEdx(KE, KE0);
    }
  }
  else{
    np = 2;
    deltaE = new double[np];
    trklength = new double[np];
    deltaE[0] = 0;
    trklength[0] = 0;
    deltaE[1] = KE0;
    trklength[1] = IntegratedEdx(0, KE0);
  }

  spmap[iKE] = new TSpline3(Form("KE %d",iKE), trklength, deltaE, np, "b2e2", 0, 0);
  delete[] trklength;
  delete[] deltaE;

}

double BetheBloch::IntegratedEdx(double KE0, double KE1, int n){

  if (KE0>KE1) std::swap(KE0, KE1);

  double step = (KE1-KE0)/n;

  double area = 0;

  for (int i = 0; i<n; ++i){
    double dEdx = meandEdx(KE0 + (i+0.5)*step);
    if (dEdx)
      area += 1/dEdx*step;
  }
  return area;
}

double BetheBloch::meandEdx(double KE){

  //KE is kinetic energy in MeV

  double K = 0.307;
  double rho = 1.396;
  double Z = 18;
  double A = 39.948;
  double I = pow(10,-6)*10.5*18; //MeV
  double me = 0.511; //MeV me*c^2

  double gamma = (KE + _mass) / _mass;
  double beta = sqrt( 1 - 1/pow(gamma,2));

  double wmax = 2*me*pow(beta,2)*pow(gamma,2)/(1+2*gamma*me/_mass + pow(me,2)/pow(_mass,2));

  double dEdX = (rho*K*Z*pow(_charge,2))/(A*pow(beta,2))*(0.5*log(2*me*pow(gamma,2)*pow(beta,2)*wmax/pow(I,2)) - pow(beta,2) - densityEffect( beta, gamma )/2 );

  return dEdX;
}

double BetheBloch::densityEffect(double beta, double gamma){

  double lar_C = 5.215, lar_x0 = 0.201, lar_x1 = 3, lar_a = 0.196, lar_k = 3;
  double x = log10(beta * gamma);

  if( x >= lar_x1 ){
    return 2*log(10)*x - lar_C;
  }
  else if ( lar_x0 <= x && x < lar_x1){
    return 2*log(10)*x - lar_C + lar_a * pow(( lar_x1 - x ) , lar_k );
  }
  else{
    return 0; //if x < lar_x0
  }
}

