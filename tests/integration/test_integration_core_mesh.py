# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""This script is used to test general functionality of the core mesh class
with end-to-end integration tests."""

import copy

from beamme.core.conf import bme
from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_core_mesh_deep_copy_with_geometry_sets(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that deep-copying a mesh together with geometry sets referencing it
    works such that the copied geometry sets also reference the copied mesh."""

    mesh = Mesh()
    beam_set = create_beam_mesh_line(
        mesh=mesh,
        beam_class=Beam3rHerm2Line3,
        material=get_default_test_beam_material(material_type="reissner"),
        start_point=[0, 0, 0],
        end_point=[1, 0, 0],
    )

    # Deep-copy both mesh and beam_set to keep node/element references consistent
    mesh_copy, beam_set_copy = copy.deepcopy((mesh, beam_set))

    mesh.add(mesh_copy)
    mesh.add(beam_set)
    mesh.add(beam_set_copy)
    bme.check_overlapping_elements = False
    assert_results_close(get_corresponding_reference_file_path(), mesh)
