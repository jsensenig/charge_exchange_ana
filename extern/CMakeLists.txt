cmake_minimum_required(VERSION 3.4...3.18)
project(pybind_utils LANGUAGES CXX)

add_subdirectory(pybind11)

pybind11_add_module(plotting_utils MODULE src/plotting_utils.cpp)
pybind11_add_module(cross_section_utils MODULE src/cross_section_utils.cpp)

