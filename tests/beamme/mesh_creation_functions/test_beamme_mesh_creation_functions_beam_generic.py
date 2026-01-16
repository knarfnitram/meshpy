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

import pytest

from beamme.core.element_beam import Beam3
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
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
