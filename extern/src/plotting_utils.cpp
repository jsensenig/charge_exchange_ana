#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


// Function to select daughters accordign to pdg
// Reads/writes array data directly from numpy buffer to save time and memory
// Must receive a flattened array
py::array_t<double> daughter_by_pdg(py::array_t<double> x, py::array_t<int> pdg, int select_pdg, std::vector<int> &pdg_vec) {
    
    py::buffer_info buf_x = x.request();
    // Here we need access to the actual values so we can make the selection
    auto pdg_arr = pdg.unchecked<1>();

    if (buf_x.size != pdg_arr.size()) throw std::runtime_error("Input shapes must match");

    double *ptr_x = static_cast<double *>(buf_x.ptr);

    // Use a vector since we don't know a priori the size of the array to be returned
    std::vector<double> xptr_vec;

    // select_pdg = 0 = "Other"
    if( select_pdg == 0 ) {
      for (size_t idx = 0; idx < buf_x.shape[0]; idx++) {
        auto it = std::find( pdg_vec.begin(), pdg_vec.end(), pdg_arr(idx) );
        if( it != pdg_vec.end() ) continue;
        xptr_vec.emplace_back(ptr_x[idx]);
      }
    } else {
      for (size_t idx = 0; idx < buf_x.shape[0]; idx++) {
        if( pdg_arr(idx) != select_pdg ) continue;
        xptr_vec.emplace_back(ptr_x[idx]);
      }
    }

    // No pointer is passed, so NumPy will allocate the buffer
    auto pdg_selected = py::array_t<double>(xptr_vec.size());
                                                                
    py::buffer_info buf_return = pdg_selected.request();
    double *ptr_return = static_cast<double *>(buf_return.ptr);

    // Now fill the output buffer with the selected data
    for(size_t i = 0; i < xptr_vec.size(); i++) ptr_return[i] = xptr_vec[i];

    return pdg_selected;
}


/////////////////////////////////////////////


PYBIND11_MODULE(plotting_utils, m) {
    m.doc() = R"pbdoc(
        Pybind11 plot utilities plugin
        -----------------------
        .. currentmodule:: plotting_utils
        .. autosummary::
           :toctree: _generate
           daughter_by_pdg
           test_fill_hist
    )pbdoc";


    /// Read a Numpy array directly from memory and select daughter by PDG
    m.def("daughter_by_pdg", &daughter_by_pdg, R"pbdoc(
       Read a Numpy array directly from memory and select daughter by PDG input=(1D numpy.ndarray>, <1D numpy.ndarray PDG>)
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
