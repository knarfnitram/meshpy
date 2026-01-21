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
"""Unit tests for the generic beam mesh creation function utils functions."""

import re

import pytest

from beamme.core.element_beam import Beam3
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.mesh_creation_functions.beam_generic import create_beam_mesh_generic
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_beamme_mesh_creation_functions_beam_generic_start_end_node_error(
    get_default_test_beam_material,
):
    """Check that an error is raised if wrong start and end nodes are given to
    a mesh creation function."""

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="base")
    mesh.add(mat)

    # Try to create a line with a starting node that is not in the mesh.
    node = NodeCosserat([0, 0, 0], Rotation())
    args = [mesh, Beam3, mat, [0, 0, 0], [1, 0, 0]]
    kwargs = {"start_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)
    node.coordinates = [1, 0, 0]
    kwargs = {"end_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)


def test_beamme_mesh_creation_functions_beam_generic_arc_length_argument_checks(
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
            Beam3,
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
