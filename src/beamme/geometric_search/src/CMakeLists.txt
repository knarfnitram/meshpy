# The MIT License (MIT)
#
# Copyright (c) 2018-2026 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.18)

# Project name
project(arborxpy LANGUAGES CXX)

# Compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find python packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Add pybind11 package
add_subdirectory(../../../pybind11 pybind11-build)

# Get ArborX
include(FetchContent)
FetchContent_Declare(
  arborx
  GIT_REPOSITORY https://github.com/arborx/ArborX.git
  GIT_TAG 4c015c70f3927b55e730515ce8db235e68c13480
)
FetchContent_MakeAvailable(arborx)

# Build the python extension module
set(lib_name "arborx_lib")
pybind11_add_module(${lib_name}
  geometric_search.cpp
  find_close_points.cpp
)

# Set compile options
target_compile_options(${lib_name} PRIVATE -Wall -Wextra -O3 -fvisibility=hidden)

target_link_libraries(${lib_name} PRIVATE ArborX::ArborX)

# Copy the module where the import are expected
add_custom_command(
  TARGET ${lib_name}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib_name}> ${CMAKE_SOURCE_DIR}/../
)
