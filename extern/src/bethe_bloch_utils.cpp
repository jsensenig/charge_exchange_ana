#include <pybind11/pybind11.h>
#include "bethe_bloch.h"

namespace py = pybind11;


PYBIND11_MODULE(bethe_bloch_utils, m) {
    py::class_<BetheBloch>(m, "BetheBloch")
        .def(py::init<double, int>(), py::arg("mass"), py::arg("charge"))
        .def("ke_along_track", &BetheBloch::ke_along_track)
        .def("ke_at_point", &BetheBloch::ke_at_point)
        .def("meandEdx", &BetheBloch::meandEdx);
}

