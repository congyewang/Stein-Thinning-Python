#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "kernel/kernel.h"
#include "kernel/thinning.h"

namespace py = pybind11;

py::array_t<double> thinning(int &m, arma::mat &smp, arma::mat &scr)
{
    arma::uvec idx = thin(smp, scr, m);

    // return carma::row_to_arr(idx);
    return carma::row_to_arr(arma::conv_to<arma::Row<unsigned long long>>::from(idx));
}

PYBIND11_MODULE(pysteinthin, m)
{
    m.def("thinning", &thinning, "Performs thinning operation",
        py::arg("m"), py::arg("smp"), py::arg("scr"));
}
