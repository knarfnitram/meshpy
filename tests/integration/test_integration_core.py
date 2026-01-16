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
"""This script is used to test general core functionality of BeamMe."""

import autograd.numpy as npAD
import numpy as np

from beamme.core.geometry_set import GeometryName, GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)


def test_integration_core_close_beam(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circle with different methods.
    - Create the mesh manually by creating the nodes and connecting them to
        the elements.
    - Create one full circle and connect it to its beginning.
    - Create two half circle and connect their start / end nodes.
    All of those methods should give the exact same mesh.
    Both variants are also tried with different rotations at the beginning.
    """

    # Parameters for this test case.
    n_el = 3
    R = 1.235
    additional_rotation = Rotation([0, 1, 0], 0.5)

    # Define material.
    mat = get_default_test_beam_material(material_type="reissner")

    def create_mesh_manually(start_rotation):
        """Create the full circle manually."""
        mesh = Mesh()
        mesh.add(mat)

        # Add nodes.
        for i in range(4 * n_el):
            basis = start_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            r = [R, 0, 0]
            node = NodeCosserat(r, basis)
            rotation = Rotation([0, 0, 1], 0.5 * i * np.pi / n_el)
            node.rotate(rotation, origin=[0, 0, 0])
            mesh.nodes.append(node)

        # Add elements.
        for i in range(2 * n_el):
            node_index = [2 * i, 2 * i + 1, 2 * i + 2]
            nodes = []
            for index in node_index:
                if index == len(mesh.nodes):
                    nodes.append(mesh.nodes[0])
                else:
                    nodes.append(mesh.nodes[index])
            element = Beam3rHerm2Line3(mat, nodes)
            mesh.add(element)

        # Add sets.
        geom_set = GeometryName()
        geom_set["start"] = GeometrySet(mesh.nodes[0])
        geom_set["end"] = GeometrySet(mesh.nodes[0])
        geom_set["line"] = GeometrySet(mesh.elements)
        mesh.add(geom_set)
        return mesh

    def one_full_circle_closed(function, argument_list, additional_rotation=None):
        """Create one full circle and connect it to itself."""

        mesh = Mesh()

        if additional_rotation is not None:
            start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            mesh.add(NodeCosserat([R, 0, 0], start_rotation))
            beam_sets = function(
                mesh,
                start_node=mesh.nodes[0],
                close_beam=True,
                **(argument_list),
            )
        else:
            beam_sets = function(mesh, close_beam=True, **(argument_list))
        mesh.add(beam_sets)
        return mesh

    def two_half_circles_closed(function, argument_list, additional_rotation=None):
        """Create two half circles and close them, by reusing the connecting
        nodes."""

        mesh = Mesh()

        if additional_rotation is not None:
            start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            mesh.add(NodeCosserat([R, 0, 0], start_rotation))
            set_1 = function(mesh, start_node=mesh.nodes[0], **(argument_list[0]))
        else:
            set_1 = function(mesh, **(argument_list[0]))

        set_2 = function(
            mesh,
            start_node=set_1["end"],
            end_node=set_1["start"],
            **(argument_list[1]),
        )

        # Add sets.
        geom_set = GeometryName()
        geom_set["start"] = GeometrySet(set_1["start"])
        geom_set["end"] = GeometrySet(set_2["end"])
        geom_set["line"] = GeometrySet([set_1["line"], set_2["line"]])
        mesh.add(geom_set)

        return mesh

    def get_arguments_arc_segment(circle_type):
        """Return the arguments for the arc segment function."""
        if circle_type == 0:
            # Full circle.
            arg_rot_angle = np.pi / 2
            arg_angle = 2 * np.pi
            arg_n_el = 2 * n_el
        elif circle_type == 1:
            # First half circle.
            arg_rot_angle = np.pi / 2
            arg_angle = np.pi
            arg_n_el = n_el
        elif circle_type == 2:
            # Second half circle.
            arg_rot_angle = 3 * np.pi / 2
            arg_angle = np.pi
            arg_n_el = n_el
        return {
            "beam_class": Beam3rHerm2Line3,
            "material": mat,
            "center": [0, 0, 0],
            "axis_rotation": Rotation([0, 0, 1], arg_rot_angle),
            "radius": R,
            "angle": arg_angle,
            "n_el": arg_n_el,
        }

    def circle_function(t):
        """Function for the circle."""
        return R * npAD.array([npAD.cos(t), npAD.sin(t)])

    def get_arguments_curve(circle_type):
        """Return the arguments for the curve function."""
        if circle_type == 0:
            # Full circle.
            arg_interval = [0, 2 * np.pi]
            arg_n_el = 2 * n_el
        elif circle_type == 1:
            # First half circle.
            arg_interval = [0, np.pi]
            arg_n_el = n_el
        elif circle_type == 2:
            # Second half circle.
            arg_interval = [np.pi, 2 * np.pi]
            arg_n_el = n_el
        return {
            "beam_class": Beam3rHerm2Line3,
            "material": mat,
            "function": circle_function,
            "interval": arg_interval,
            "n_el": arg_n_el,
        }

    # Check the meshes without additional rotation.
    assert_results_close(
        get_corresponding_reference_file_path(), create_mesh_manually(Rotation())
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        one_full_circle_closed(
            create_beam_mesh_arc_segment_via_rotation, get_arguments_arc_segment(0)
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        two_half_circles_closed(
            create_beam_mesh_arc_segment_via_rotation,
            [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        one_full_circle_closed(
            create_beam_mesh_parametric_curve, get_arguments_curve(0)
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        two_half_circles_closed(
            create_beam_mesh_parametric_curve,
            [get_arguments_curve(1), get_arguments_curve(2)],
        ),
    )

    # Check the meshes with additional rotation.
    additional_identifier = "rotation"
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        create_mesh_manually(additional_rotation),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        one_full_circle_closed(
            create_beam_mesh_arc_segment_via_rotation,
            get_arguments_arc_segment(0),
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        two_half_circles_closed(
            create_beam_mesh_arc_segment_via_rotation,
            [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        one_full_circle_closed(
            create_beam_mesh_parametric_curve,
            get_arguments_curve(0),
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        two_half_circles_closed(
            create_beam_mesh_parametric_curve,
            [get_arguments_curve(1), get_arguments_curve(2)],
            additional_rotation=additional_rotation,
        ),
    )
