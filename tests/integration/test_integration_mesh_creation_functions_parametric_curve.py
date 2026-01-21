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
"""This script is used to test the create parametric curve mesh creation
functions."""

from typing import Callable

import autograd.numpy as npAD
import numpy as np
from autograd import jacobian

from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)
from beamme.utils.nodes import get_nodal_coordinates


def create_helix_function(
    radius: float,
    incline: float,
    *,
    transformation_factor: float | None = None,
    number_of_turns: float | None = None,
) -> Callable:
    """Create and return a parametric function that represents a helix shape.
    The parameter coordinate can optionally be stretched to make the curve arc-
    length along the parameter coordinated non-constant and create a more
    complex curve for testing purposes.

    Args:
        radius: Radius of the helix
        incline: Incline of the helix
        transformation_factor: Factor to control the coordinate stretching (no direct physical interpretation)
        number_of_turns: Number of turns the helix will have to get approximate boundaries for the transformation.
            This is only used for the transformation, not the actual geometry, as we return the
            function to create the geometry and not the geometry itself.

    Returns:
        A function that describes a helix in 3D space.
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


def test_integration_mesh_creation_functions_parametric_curve_3d_helix(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix from a parametric curve where the parameter is
    transformed so the arc length along the beam is not proportional to the
    parameter."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 1  # number of turns
    n_el = 5
    helix = create_helix_function(R, tz, transformation_factor=2.0, number_of_turns=n)

    helix_set = create_beam_mesh_parametric_curve(
        mesh, Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n], n_el=n_el
    )
    mesh.add(helix_set)

    # Compare the coordinates with the ones from Mathematica.
    coordinates_mathematica = np.loadtxt(
        get_corresponding_reference_file_path(
            additional_identifier="mathematica", extension="csv"
        ),
        delimiter=",",
    )
    assert_results_close(coordinates_mathematica, get_nodal_coordinates(mesh.nodes))

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_parametric_curve_3d_helix_length(
    get_default_test_beam_material, assert_results_close
):
    """Create a helix from a parametric curve where and check that the correct
    length is returned."""

    mesh_1 = Mesh()
    mesh_2 = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 1  # number of turns
    n_el = 3
    helix = create_helix_function(R, tz, transformation_factor=2.0, number_of_turns=n)

    args = [Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n]]
    kwargs = {"n_el": n_el}

    helix_set_1 = create_beam_mesh_parametric_curve(mesh_1, *args, **kwargs)
    mesh_1.add(helix_set_1)

    helix_set_2, length = create_beam_mesh_parametric_curve(
        mesh_2, *args, output_length=True, **kwargs
    )
    mesh_2.add(helix_set_2)

    # Check the computed length
    assert_results_close(length, 13.18763323790246)

    # Check that both meshes are equal
    assert_results_close(mesh_1, mesh_2)


def test_integration_mesh_creation_functions_parametric_curve_2d_sin(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a sin from a parametric curve."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Set parameters for the sin.
    n_el = 8

    def sin(t):
        """Parametric curve as a sin."""
        return npAD.array([t, npAD.sin(t)])

    sin_set = create_beam_mesh_parametric_curve(
        mesh, Beam3rHerm2Line3, mat, sin, [0.0, 2.0 * np.pi], n_el=n_el
    )
    mesh.add(sin_set)

    # Compare the coordinates with the ones from Mathematica.
    coordinates_mathematica = np.loadtxt(
        get_corresponding_reference_file_path(
            additional_identifier="mathematica", extension="csv"
        ),
        delimiter=",",
    )
    assert_results_close(coordinates_mathematica, get_nodal_coordinates(mesh.nodes))

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_parametric_curve_3d_rotation(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a line from a parametric curve and prescribe the rotation."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Set parameters for the line.
    L = 1.1
    n_el = 4

    def curve(t):
        """Parametric representation of the centerline curve."""
        return npAD.array([L * t, t * t * L * L, 0.0])

    def rotation(t):
        """Function that defines the triad along the centerline curve."""
        rp2 = jacobian(curve)(t)
        rp = [rp2[0], rp2[1], 0]
        R1 = Rotation([1, 0, 0], t * 2 * np.pi)
        R2 = Rotation.from_basis(rp, [0, 0, 1])
        return R2 * R1

    sin_set = create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        curve,
        [0.0, 1.0],
        n_el=n_el,
        function_rotation=rotation,
    )
    mesh.add(sin_set)

    # extend test case with different meshing strategy
    create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        curve,
        [1.0, 2.5],
        node_positions_of_elements=[0, 1.0 / 1.3, 1.0],
        function_rotation=rotation,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_parametric_curve_3d_line(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a line from a parametric curve.

    Once the interval is in ascending order, once in descending. This
    tests checks that the elements are created with the correct tangent
    vectors.
    """

    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    def line(t):
        """Create a line with a parametric curve (and a transformed
        parameter)."""
        factor = 2
        t_trans = npAD.exp(factor * t / (2.0 * np.pi)) * t / npAD.exp(factor)
        return npAD.array([t_trans, 0, 0])

    # Create mesh.
    set_1 = create_beam_mesh_parametric_curve(
        mesh, Beam3rHerm2Line3, mat, line, [0.0, 5.0], n_el=3
    )
    mesh.translate([0, 1, 0])
    set_2 = create_beam_mesh_parametric_curve(
        mesh, Beam3rHerm2Line3, mat, line, [5.0, 0.0], n_el=3
    )
    mesh.add(set_1, set_2)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)
