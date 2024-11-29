#include "kernel.h"
#include "thinning.h"
#include "binding.h"

idxnparray pysteinthin::thin(nparray smp, nparray scr, int m)
{
    idxnparray result_np;

    // Convert numpy arrays to armadillo matrices
    arma::mat smp_arma = carma::arr_to_mat<double>(smp);
    arma::mat scr_arma = carma::arr_to_mat<double>(scr);

    // Perform thinning operation
    arma::uvec result_arma = st::thin(smp_arma, scr_arma, m);

    result_np = carma::col_to_arr(result_arma);

    return result_np;
}

PYBIND11_MODULE(pysteinthin, m)
{
    m.def("thin", &pysteinthin::thin, "Performs thinning operation", py::arg("smp"), py::arg("scr"), py::arg("m"));
}
