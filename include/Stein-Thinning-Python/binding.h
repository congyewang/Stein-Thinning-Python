#ifndef BINDING_H
#define BINDING_H

#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "kernel.h"
#include "thinning.h"

namespace py = pybind11;
namespace st = stein_thinning;

typedef std::vector<long long unsigned> idxvec;
typedef py::array_t<double> nparray;
typedef py::array_t<long long unsigned> idxnparray;

namespace pysteinthin
{
    idxnparray thin(nparray smp, nparray scr, int m);
}

#endif // BINDING_H
