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
"""This script is used to test the mesh creation functions."""

import re

import autograd.numpy as npAD
import numpy as np
import pytest
import splinepy

from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3, get_four_c_reissner_beam
from beamme.mesh_creation_functions.applications.beam_fibers_in_rectangle import (
    create_fibers_in_rectangle,
)
from beamme.mesh_creation_functions.applications.beam_stent import (
    create_beam_mesh_stent,
)
from beamme.mesh_creation_functions.applications.beam_wire import create_wire_fibers
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_arc_segment_via_axis,
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_generic import create_beam_mesh_generic
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.mesh_creation_functions.beam_node_continuation import (
    create_beam_mesh_arc_at_node,
    create_beam_mesh_line_at_node,
)
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)
from beamme.mesh_creation_functions.beam_splinepy import (
    create_beam_mesh_from_splinepy,
    get_curve_function_and_jacobian_for_integration,
)


def create_helix_function(
    radius, incline, *, transformation_factor=None, number_of_turns=None
):
    """Create and return a parametric function that represents a helix shape.
    The parameter coordinate can optionally be stretched to make the curve arc-
    length along the parameter coordinated non-constant and create a more
    complex curve for testing purposes.

    Args
    ----
    radius: float
        Radius of the helix
    incline: float
        Incline of the helix
    transformation_factor: float
        Factor to control the coordinate stretching (no direct physical interpretation)
    number_of_turns: float
        Number of turns the helix will have to get approximate boundaries for the transformation.
        This is only used for the transformation, not the actual geometry, as we return the
        function to create the geometry and not the geometry itself.
    """

    if transformation_factor is None and number_of_turns is None:

        def transformation(t):
            """Return identity transformation."""
            return 1.0

    elif transformation_factor is not None and number_of_turns is not None:

        def transformation(t):
            """Transform the parameter coordinate to make the function more
            complex."""
            return (
                npAD.exp(transformation_factor * t / (2.0 * np.pi * number_of_turns))
                * t
                / npAD.exp(transformation_factor)
            )

    else:
        raise ValueError(
            "You have to set none or both optional parameters: "
            "transformation_factor and number_of_turns"
        )

    def helix(t):
        """Parametric function to describe a helix."""
        return npAD.array(
            [
                radius * npAD.cos(transformation(t)),
                radius * npAD.sin(transformation(t)),
                transformation(t) * incline / (2 * np.pi),
            ]
        )

    return helix


def create_testing_bezier_curve():
    """Create a Bezier curve used for testing."""

    control_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 0.0, 0.0],
        ]
    )
    return splinepy.Bezier(degrees=[3], control_points=control_points)


def create_testing_nurbs_curve():
    """Create a NURBS curve used for testing."""

    return splinepy.NURBS(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
        control_points=[[0, 0, 0], [1, 2, -1], [2, 0, 0]],
        weights=[[1.0], [1.0], [1.0]],
    )


def test_mesh_creation_functions_arc_segment_via_axis(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segment via the axis method and compare it with the
    reference file."""

    # Create mesh
    mesh = Mesh()
    radius = 2.0
    beam_set = create_beam_mesh_arc_segment_via_axis(
        mesh,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        [0, 0, 1],
        [0, radius, 0],
        [0, 0, 0],
        1.0,
        n_el=3,
    )
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_arc_segment_start_end_node(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Check that if start end nodes with non-matching positions or tangents
    are provided we get an error."""

    angle = 1.0
    radius = 2.0
    start_node_pos = [0, 0, 0]
    start_node_rot = Rotation()
    end_node_pos = radius * np.array([np.sin(angle), 1.0 - np.cos(angle), 0])
    end_node_rot = Rotation([0, 0, 1], angle)

    def create_beam(*, start_node=None, end_node=None):
        """This is the base function we use to generate the beam in this test
        case."""
        mesh = Mesh()
        mat = get_default_test_beam_material(material_type="reissner")
        if start_node is not None:
            mesh.add(start_node)
        if end_node is not None:
            mesh.add(end_node)
        create_beam_mesh_arc_segment_via_axis(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 0, 1],
            [0, radius, 0],
            [0, 0, 0],
            angle,
            n_el=3,
            start_node=start_node,
            end_node=end_node,
        )
        return mesh

    # This should work as expected, as all the values match.
    start_node = NodeCosserat(start_node_pos, start_node_rot)
    end_node = NodeCosserat(end_node_pos, end_node_rot)
    mesh = create_beam(start_node=start_node, end_node=end_node)
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Create with start node where the position does not match.
    with pytest.raises(
        ValueError,
        match="start_node position does not match with function!",
    ):
        start_node = NodeCosserat([0, 1, 0], start_node_rot)
        create_beam(start_node=start_node)

    # Create with start node where the rotation does not match.
    with pytest.raises(
        ValueError,
        match="The tangent of the start node does not match with the given function!",
    ):
        start_node = NodeCosserat(start_node_pos, Rotation([1, 2, 3], np.pi / 3))
        create_beam(start_node=start_node)

    # Create with end node where the position does not match.
    with pytest.raises(
        ValueError,
        match="end_node position does not match with function!",
    ):
        end_node = NodeCosserat([0, 0, 0], end_node_rot)
        create_beam(end_node=end_node)

    # Create with end node where the rotation does not match.
    with pytest.raises(
        ValueError,
        match="The tangent of the end node does not match with the given function!",
    ):
        end_node = NodeCosserat(end_node_pos, Rotation([1, 2, 3], np.pi / 3))
        create_beam(end_node=end_node)


def test_mesh_creation_functions_arc_segment_via_rotation(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segment via the rotation method and compare it with
    the reference file."""

    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    beam_set = create_beam_mesh_arc_segment_via_rotation(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [3, 6, 9.2],
        Rotation([4.5, 7, 10], np.pi / 5),
        10,
        np.pi / 2.3,
        n_el=5,
    )

    # Add boundary conditions.
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_arc_segment_2d(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segments in 2D."""

    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    beam_set_1 = create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0],
        1.5,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        n_el=5,
    )
    beam_set_2 = create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0] - 2.0 * 0.5 * np.array([1, np.sqrt(3), 0]),
        0.5,
        np.pi / 3.0,
        -np.pi,
        n_el=3,
        start_node=mesh.nodes[-1],
    )

    # Add geometry sets
    mesh.add(beam_set_1, beam_set_2)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_node_positions_of_elements_option(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Creates a line, a circular segments in 2D and a helix by setting the
    node_positions_of_elements."""

    # Create a mesh
    mesh = Mesh()

    # Create and add material to mesh.
    material = get_default_test_beam_material(material_type="reissner")
    mesh.add(material)

    # Create a beam line with specified node_positions_of_elements.
    create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        material,
        [-1, -1, 0],
        [-1, -1, 3],
        node_positions_of_elements=[0, 0.1, 0.5, 0.9, 1.0],
    )

    # Create an arc segment similar to the equally spaced one,
    # but based on the provided node_positions_of_elements.
    create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        material,
        [1.0, 2.0, 0.0],
        1.5,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        node_positions_of_elements=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    # Create a 2d segments with different element sizes.
    create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        material,
        [1.0, 2.0, 0.0],
        2.0,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        node_positions_of_elements=[0, 1.0 / 3.0, 1.0],
    )

    # Create a helix with different node positions.
    create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        material,
        [0.0, 0.0, 1.0],
        [4.0, 4.0, 0.0],
        [5.0, 5.0, 0.0],
        height_helix=10.0,
        turns=2.5 / np.pi,
        node_positions_of_elements=[0, 1.0 / 3.0, 1],
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_stent(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the stent creation function."""

    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    create_beam_mesh_stent(
        mesh,
        Beam3rHerm2Line3,
        mat,
        0.11,
        0.02,
        5,
        8,
        fac_bottom=0.6,
        fac_neck=0.52,
        fac_radius=0.36,
        alpha=0.47 * np.pi,
        n_el=2,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_fibers_in_rectangle(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_fibers_in_rectangle function."""

    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        0,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        90,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        -90,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        235,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        1,
        4,
        30,
        0.45,
        5,
        fiber_element_length_min=0.2,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        30,
        0.45,
        0.9,
    )
    mesh.translate([0, 0, 1])

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_fibers_in_rectangle_reference_point(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_fibers_in_rectangle function with using the
    reference_point option."""

    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    reference_point = 0.5 * np.array([4.0, 1.0]) + 0.1 * np.array(
        [-1.0, 1.0]
    ) / np.sqrt(2.0)
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
        reference_point=reference_point,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_fibers_in_rectangle_return_set(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the set returned by the create_fibers_in_rectangle function."""

    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_wire(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_wire_fibers function."""

    # Create mesh
    mesh = Mesh()

    # Create two wires with different parameters.
    mat = get_default_test_beam_material(material_type="reissner")
    mesh_1 = Mesh()
    set_1 = create_wire_fibers(
        mesh_1,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        3.0,
        layers=2,
        n_el=2,
        radius=0.05,
    )
    mesh_2 = Mesh()
    set_2 = create_wire_fibers(
        mesh_2,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        3.0,
        layers=2,
        n_el=2,
        radius=0.1,
    )
    mesh_2.translate([0.0, 1.5, 0.0])
    mesh.add(mesh_1, mesh_2, set_1, set_2)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize(
    ("name", "curve_creation_function", "ref_length"),
    [
        ("bezier", create_testing_bezier_curve, 5.064502358928783),
        ("nurbs", create_testing_nurbs_curve, 3.140204411551537),
    ],
)
def test_mesh_creation_functions_splinepy(
    name,
    curve_creation_function,
    ref_length,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_beam_mesh_from_splinepy function with different splinepy
    curves."""

    curve = curve_creation_function()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()
    _, length = create_beam_mesh_from_splinepy(
        mesh, Beam3rHerm2Line3, mat, curve, n_el=3, output_length=True
    )
    assert_results_close(ref_length, length)

    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=name), mesh
    )


def test_mesh_creation_functions_splinepy_unit(assert_results_close):
    """Unittest the function and jacobian creation in the
    create_beam_mesh_from_splinepy function."""

    curve = create_testing_nurbs_curve()
    r, dr, _, _ = get_curve_function_and_jacobian_for_integration(curve, tol=10)

    t_values = [5.0 / 7.0, -0.3, 1.2]
    results_r = [
        [1.4285714285714286, 0.8163265306122449, -0.4081632653061225],
        [-0.6, -1.2, 0.6],
        [2.4, -0.8, 0.4],
    ]
    results_dr = [
        [2.0, -1.7142857142857144, 0.8571428571428572],
        [2.0, 4.0, -2.0],
        [2.0, -4.0, 2.0],
    ]

    for t, result_r, result_dr in zip(t_values, results_r, results_dr):
        assert_results_close(r(t), result_r)
        assert_results_close(dr(t), result_dr)


def test_mesh_creation_functions_node_continuation(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the node continuation function work as expected."""

    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    start_node = NodeCosserat([1, 2, 3], Rotation())
    mesh.add(start_node)
    beam_set = create_beam_mesh_line_at_node(
        mesh, Beam3rHerm2Line3, mat, start_node, 1.2, n_el=1
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 1, 0],
        1.0,
        np.pi * 0.5,
        n_el=1,
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 1, 0],
        1.0,
        -np.pi * 0.5,
        n_el=2,
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 0, 1],
        1.0,
        -np.pi * 3.0 / 4.0,
        n_el=1,
    )
    create_beam_mesh_line_at_node(
        mesh, Beam3rHerm2Line3, mat, beam_set["end"], 2.3, n_el=3
    )

    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_node_continuation_accumulated(
    get_default_test_beam_material, assert_results_close
):
    """Test that the arc node continuation function can be applied multiple
    times in a row.

    This function can lead to accumulated errors in the rotations if not
    implemented carefully.
    """

    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    n_segments = 100

    rotation_ref = Rotation([1, 0, 0], 0.5 * np.pi) * Rotation([0, 0, 1], 0.5 * np.pi)
    start_node = NodeCosserat([1, 2, 3], rotation_ref)
    mesh.add(start_node)
    beam_set = {"end": start_node}
    angle = np.pi
    angle_increment = angle / n_segments
    axis = [1, 0, 0]
    for i in range(n_segments):
        beam_set = create_beam_mesh_arc_at_node(
            mesh,
            Beam3rHerm2Line3,
            mat,
            beam_set["end"],
            axis,
            1.0,
            angle_increment,
            n_el=2,
        )

    rotation_actual = beam_set["end"].get_points()[0].rotation

    # Calculate the solution and get the "analytical" solution
    rotation_expected = Rotation(axis, angle) * rotation_ref
    quaternion_expected = np.array([-0.5, 0.5, -0.5, -0.5])
    assert rotation_actual == rotation_expected
    assert_results_close(rotation_actual.q, quaternion_expected)


def test_mesh_creation_functions_element_length_option(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the element length can be specified in the beam creation
    functions."""

    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    l_el = 1.5

    mesh_line = Mesh()
    create_beam_mesh_line(
        mesh_line,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 6.0],
        l_el=l_el,
    )

    mesh_line_long = Mesh()
    create_beam_mesh_line(
        mesh_line_long,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 2.0],
        [3.0, 4.0, 8.0],
        l_el=100,
    )

    mesh_arc = Mesh()
    create_beam_mesh_arc_segment_via_rotation(
        mesh_arc,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 3.0],
        Rotation([1, 3, 4], np.pi / 3.0),
        2.0,
        np.pi * 2.0 / 3.0,
        l_el=l_el,
    )

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 0.5  # number of turns
    helix = create_helix_function(R, tz, transformation_factor=2.0, number_of_turns=n)

    mesh_curve = Mesh()
    create_beam_mesh_parametric_curve(
        mesh_curve,
        Beam3rHerm2Line3,
        mat,
        helix,
        [0.0, 2.0 * np.pi * n],
        l_el=l_el,
    )

    # Check the output
    mesh.add(mesh_line, mesh_line_long, mesh_arc, mesh_curve)
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_mesh_creation_functions_argument_checks(
    get_default_test_beam_material,
):
    """Test that wrong input values leads to failure."""

    dummy_arg = "dummy"

    # Check error messages for input parameters
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error since we dont allow `n_el` and `l_el`
        # to be set at the same time.
        create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            get_default_test_beam_material(material_type="reissner"),
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 6.0],
            n_el=1,
            l_el=1.5,
        )
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with l_el.

        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            l_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with n_el.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            n_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError, match='The parameter "l_el" requires "interval_length" to be set.'
    ):
        mesh = Mesh()
        # This should raise an error because we set `l_el` but don't provide
        # `interval_length`.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=[0, 1],
            l_el=2.0,
        )

    with pytest.raises(
        ValueError,
        match="First entry of node_positions_of_elements must be 0, got -1.0",
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            node_positions_of_elements=[-1.0, 0.0, 1.0],
        )

    with pytest.raises(
        ValueError, match="Last entry of node_positions_of_elements must be 1, got 2.0"
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            node_positions_of_elements=[0.0, 1.0, 2.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The given node_positions_of_elements must be in ascending order. Got [0.0, 0.2, 0.1, 1.0]"
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            node_positions_of_elements=[0.0, 0.2, 0.1, 1.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            'The arguments "close_beam" and "end_node" are mutually exclusive'
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            function_generator=dummy_arg,
            interval=dummy_arg,
            n_el=1,
            close_beam=True,
            end_node=dummy_arg,
        )
