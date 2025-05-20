# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
"""Boundary condition with node tracking files."""

import numpy as np

from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.function_utility import create_linear_interpolation_function


class Node_Position_Tracker:
    """This object keeps track of the positions (or displacement) of nodes.

    It captures the positions of a node at certain times and can return
    the position sorted by ascending time.
    """

    def __init__(self):
        # list containing nodes
        self.nodes = []

        # list containing the times
        self.times = []

        # list containing the positions
        self.positions = []

    def add(self, node, time, position):
        """Add or update the position and time of a node."""
        if node in self.nodes:
            index = self.nodes.index(node)
            self.times[index] = np.append(self.times[index], time)
            self.positions[index] = np.vstack((self.positions[index], position))
        else:
            self.nodes.append(node)
            index = len(self.nodes) - 1
            self.times.append(time)
            self.positions.append(position)

        if np.linalg.norm(node.coordinates - self.nodes[index].coordinates) > 1e-8:
            raise ValueError("You are trying to add values to the wrong node.")

        if int(time.shape[0]) - int(position.shape[0]) != 0:
            print(time.shape[0], position.shape[0])
            raise ValueError(
                "You are trying to add times and positions with different lengths"
            )

        if int(position.shape[1]) != 3:
            raise ValueError("Positions shape must always be 3.")

    def get_sorted_by_time(self, node):
        """Returns an array with ascending time and positions."""
        if node in self.nodes:
            index = self.nodes.index(node)
            sorted_indices = np.argsort(self.times[index])
            return self.times[index][sorted_indices], self.positions[index][
                sorted_indices
            ]
        else:
            raise ValueError("Couldn't find node with coordinates:", node.coordinates)


def create_dbc_with_functions_from_position_tracker(
    mesh, position_tracker, n_dof_per_node=3
):
    """Adds a dirichlet boundary conditions with appropriate functions based on
    the displacement(position) values from the position tracker.

    Args
    ----
    mesh: Mesh or Inputfile
        applies the positions to it
    position_tracker:
        object that has the position and displacement values
    n_dof_per_node: [int]
        number of zeros for the condition
    :return:
    """

    # check if node position tracker was initialized
    if position_tracker is not None:
        for i_node, node in enumerate(position_tracker.nodes):
            time_values, displacement_values = position_tracker.get_sorted_by_time(node)

            # Create the functions that describe the deformation
            fun_pos = [
                create_linear_interpolation_function(
                    time_values, displacement_values[:, i_dir]
                )
                for i_dir in range(3)
            ]
            for fun in fun_pos:
                mesh.add(fun)
            additional_dof = "0 " * (n_dof_per_node - 3)
            mesh.add(
                BoundaryCondition(
                    GeometrySet(node),
                    "NUMDOF {4} ONOFF 1 1 1 {3}VAL 1.0 1.0 1.0 {3}FUNCT {0} {1} {2} {3}TAG monitor_reaction",
                    format_replacement=fun_pos + [additional_dof, n_dof_per_node],
                    bc_type=mpy.bc.dirichlet,
                )
            )

    # Print a status update every for every 10% of done work
    if i_node % round(len(position_tracker.nodes) / 10) == 0 or i_node + 1 == len(
        position_tracker.nodes
    ):
        print(f"Done {i_node + 1}/{len(position_tracker.nodes)}")
