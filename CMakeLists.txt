project(pysteinthin CXX)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)

# Find the necessary packages
find_package(BLAS REQUIRED)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")

# Execute python -m pybind11 --cmakedir and capture its output
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add the result to CMAKE_PREFIX_PATH
if(PYBIND11_CMAKEDIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKEDIR})
    message(STATUS "pybind11 cmake directory found: ${PYBIND11_CMAKEDIR}")
else()
    message(FATAL_ERROR "pybind11 cmake directory not found. Make sure pybind11 is installed.")
endif()

find_package(pybind11 REQUIRED)
find_package(carma CONFIG REQUIRED)

set(stein_thinning_cpp_libs ${CMAKE_SOURCE_DIR}/external/Stein-Thinning-Cpp/include/Stein-Thinning-Cpp)
include_directories(${stein_thinning_cpp_libs})

add_library(stein_thinning_cpp STATIC
    ${CMAKE_SOURCE_DIR}/external/Stein-Thinning-Cpp/src/kernel.cpp
    ${CMAKE_SOURCE_DIR}/external/Stein-Thinning-Cpp/src/thinning.cpp
    ${CMAKE_SOURCE_DIR}/external/Stein-Thinning-Cpp/src/utils.cpp
)
set_target_properties(stein_thinning_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################

include_directories(${CMAKE_SOURCE_DIR}/include/Stein-Thinning-Python)

pybind11_add_module(${PROJECT_NAME}
    MODULE
        ${CMAKE_SOURCE_DIR}/src/main.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC carma::carma ${BLAS_LIBRARIES} ${ARMADILLO_LIBRARIES} ${stein_thinning_cpp_libs} stein_thinning_cpp)

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
