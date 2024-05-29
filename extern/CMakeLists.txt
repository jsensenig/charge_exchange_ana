cmake_minimum_required(VERSION 3.4...3.18)
project(pybind_utils LANGUAGES CXX)

find_package( ROOT 6.22...6.30 CONFIG REQUIRED )

add_subdirectory(pybind11)


add_library(plotting_utils MODULE src/plotting_utils.cpp)


target_link_libraries(plotting_utils
                      PRIVATE pybind11::module
                              pybind11::lto
                              pybind11::windows_extras
                              ROOT::Core
                              ROOT::Physics
                              ROOT::Hist)

pybind11_extension(plotting_utils)
pybind11_strip(plotting_utils)

set_target_properties(plotting_utils PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                     CUDA_VISIBILITY_PRESET "hidden")



add_library(cross_section_utils MODULE src/cross_section_utils.cpp)

target_link_libraries(cross_section_utils
                      PRIVATE pybind11::module
                              pybind11::lto
                              pybind11::windows_extras
                              ROOT::Core
                              ROOT::Physics
                              ROOT::Hist)

pybind11_extension(cross_section_utils)
pybind11_strip(cross_section_utils)

set_target_properties(cross_section_utils PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                     CUDA_VISIBILITY_PRESET "hidden")
include_directories(src/)
add_library(bethe_bloch_utils MODULE src/bethe_bloch_utils.cpp src/bethe_bloch.cpp)

target_link_libraries(bethe_bloch_utils
                      PRIVATE pybind11::module
                              pybind11::lto
                              pybind11::windows_extras
                              ROOT::Core
                              ROOT::Physics
                              ROOT::Hist)

pybind11_extension(bethe_bloch_utils)
pybind11_strip(bethe_bloch_utils)

set_target_properties(bethe_bloch_utils PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                     CUDA_VISIBILITY_PRESET "hidden")



