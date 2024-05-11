#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;



py::array_t<double> make_true_incident_energies(const py::array_t<double> &true_beam_traj_Z,
                                                const py::array_t<double> &true_beam_traj_KE) {

  py::buffer_info buf_true_beam_traj_Z = true_beam_traj_Z.request();
  double *ptr_true_beam_traj_Z = static_cast<double *>(buf_true_beam_traj_Z.ptr);

  py::buffer_info buf_true_beam_traj_KE = true_beam_traj_KE.request();
  double *ptr_true_beam_traj_KE = static_cast<double *>(buf_true_beam_traj_KE.ptr);

  std::vector<double> vec_true_beam_new_incidentEnergies;

  // Only include trajectory points starting in the active volume
  double fTrajZStart = -0.49375;
  // Only include trajectory points less than slice 464 (the end of APA3)
  int fSliceCut = 464;
  // ProtoDUNE TPC wire pitch [cm]
  double fPitch = 0.4794; 

  double next_slice_z = fTrajZStart;
  int next_slice_num = 0;

  for (size_t j = 1; j < buf_true_beam_traj_Z.shape[0] - 1; ++j) {
    double z = ptr_true_beam_traj_Z[j];
    double ke = ptr_true_beam_traj_KE[j];

    if (z < fTrajZStart) continue;

    if (z >= next_slice_z) {
      double temp_z = ptr_true_beam_traj_Z[j-1];
      double temp_e = ptr_true_beam_traj_KE[j-1];

      while (next_slice_z < z && next_slice_num < fSliceCut) {
        double sub_z = next_slice_z - temp_z;
        double delta_e = ptr_true_beam_traj_KE[j-1] - ke;
        double delta_z = z - ptr_true_beam_traj_Z[j-1];
        temp_e -= (sub_z/delta_z)*delta_e;
        vec_true_beam_new_incidentEnergies.push_back(temp_e);
        temp_z = next_slice_z;
        next_slice_z += fPitch;
        ++next_slice_num;
      }
    }
  }

  // If the trajectory does not reach the end of the fiducial slices it must have interacted.
  // The interacting energy will be the last incident energy.
  if( next_slice_num >= fSliceCut || vec_true_beam_new_incidentEnergies.size() < 1 ) {
    vec_true_beam_new_incidentEnergies.push_back(-999.);
  }

  // No pointer is passed, so NumPy will allocate the buffer
  auto inc_energy = py::array_t<double>(vec_true_beam_new_incidentEnergies.size());
                                                              
  py::buffer_info buf_return = inc_energy.request();
  double *ptr_return = static_cast<double *>(buf_return.ptr);
                                                                           
  // Now fill the output buffer with the selected data
  for(size_t i = 0; i < vec_true_beam_new_incidentEnergies.size(); i++) ptr_return[i] = vec_true_beam_new_incidentEnergies[i];

  return inc_energy;
}


/////////////////////////////////////////////


PYBIND11_MODULE(cross_section_utils, m) {
    m.doc() = R"pbdoc(
        Pybind11 cross section utilities plugin
        -----------------------
        .. currentmodule:: cross_section_utils
        .. autosummary::
           :toctree: _generate
           make_true_incident_energies
    )pbdoc";


    /// Create the true incident energy of the beam particle from the true trajectory points
    m.def("make_true_incident_energies", &make_true_incident_energies, R"pbdoc(
       Create the true incident energy of the beam particle from the true trajectory points input=(<1D numpy.ndarray true_beam_traj_Z>, <1D numpy.ndarray true_beam_traj_KE>)
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
