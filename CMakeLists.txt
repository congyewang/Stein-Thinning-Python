project(pysteinthin CXX)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)

# Find the necessary packages
find_package(BLAS REQUIRED)

set(ARMADILLO_INCLUDE_DIR include/armadillo/include)
include_directories(${ARMADILLO_INCLUDE_DIR})

set(pybind11_DIR ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)

find_package(carma CONFIG REQUIRED)

set(stein_thinning_cpp_libs ${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Cpp/src)
include_directories(${stein_thinning_cpp_libs})

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################

pybind11_add_module(${PROJECT_NAME}
    MODULE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/stein_thinning_python/main.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC carma::carma ${BLAS_LIBRARIES})

target_compile_options(${PROJECT_NAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# ##############################################################################
#                                   INSTALL                                    #
# ##############################################################################
install(TARGETS ${PROJECT_NAME} DESTINATION examples)
file(GLOB PY_EXAMPLE_FILES "${PROJECT_SOURCE_DIR}/examples/*.py")
install(FILES ${PY_EXAMPLE_FILES} DESTINATION examples)

# ##############################################################################
#                                  EXAMPLES                                    #
# ##############################################################################
add_test(NAME example
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${PYTHON_EXECUTABLE} carma_examples.py)
set_property(TEST example
        PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
