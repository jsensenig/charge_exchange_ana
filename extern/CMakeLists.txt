cmake_minimum_required(VERSION 3.4...3.18)
project(plotting_utils LANGUAGES CXX)

find_package( ROOT 6.22 CONFIG REQUIRED )

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
