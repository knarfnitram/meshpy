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
"""This script is used to test the functionality of the core modules."""

import numpy as np
import pytest

from beamme.core.conf import bme
from beamme.core.coupling import Coupling
from beamme.core.mesh import Mesh
from beamme.core.node import Node
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


@pytest.mark.parametrize(
    "coupling_type",
    [
        ["exact", bme.bc.point_coupling, bme.coupling_dof.fix],
        [
            "penalty",
            bme.bc.point_coupling_penalty,
            {
                "POSITIONAL_PENALTY_PARAMETER": 10000,
                "ROTATIONAL_PENALTY_PARAMETER": 0,
            },
        ],
    ],
)
def test_point_couplings(
    get_default_test_beam_material,
    coupling_type,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the input file for the test_point_couplings method."""

    # Create material and mesh
    material = get_default_test_beam_material(
        material_type="reissner", interaction_radius=2.0
    )
    mesh = Mesh()

    # Create a 2x2 grid of beams.
    for i in range(3):
        for j in range(2):
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, material, [j, i, 0.0], [j + 1, i, 0.0]
            )
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, material, [i, j, 0.0], [i, j + 1, 0.0]
            )

    # Couple the beams.
    mesh.couple_nodes(
        reuse_matching_nodes=True,
        coupling_type=coupling_type[1],
        coupling_dof_type=coupling_type[2],
    )

    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=coupling_type[0]),
        mesh,
    )


def test_point_couplings_check():
    """Test that the check for points at the same spatial position works for
    point couplings."""

    def get_nodes(scale_factor):
        """Return a list with nodes to be added to a coupling condition.

        The coordinates are modified such that they are close to each
        other within a radius of bme.eps_pos * scale_factor
        """
        coordinates = np.zeros((10, 3))
        ref_point = [1, 2, 3]
        coordinates[0] = ref_point
        for i in range(1, 10):
            coordinates[i] = ref_point
            for i_dir in range(3):
                factor = 2 * ((i + i_dir % 3) % 2) - 1
                # Multiply with 0.5 here, because we add the tolerance in + and - direction
                coordinates[i, i_dir] += factor * bme.eps_pos * scale_factor
        return [Node(coord) for coord in coordinates]

    # This should work, as the points are within the global tolerance of each
    # other
    Coupling(get_nodes(0.5), None, None)

    with pytest.raises(ValueError):
        # This should fail, as the points are not within the global tolerance
        # of each other
        Coupling(get_nodes(1.0), None, None)

    # This should work, as the points are not within the global tolerance of
    # each other but we dont perform the check
    Coupling(get_nodes(1.0), None, None, check_overlapping_nodes=False)
