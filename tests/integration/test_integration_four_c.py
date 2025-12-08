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
"""This script is used to test general functionality of the 4C module with end-
to-end integration tests."""

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import (
    Beam3rHerm2Line3,
    get_four_c_reissner_beam,
)
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_four_c_point_coupling_indirect(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that indirect point coupling works as expected."""

    material = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()

    beam_set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [-1, 0, 0], [1, 0, 0]
    )
    beam_set_2 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, -1, 0.1], [0, 1, 0.1], n_el=3
    )
    for i_set, beam_set in enumerate([beam_set_1, beam_set_2]):
        data = {"COUPLING_ID": 3}
        if i_set == 0:
            data["PARAMETERS"] = {
                "POSITIONAL_PENALTY_PARAMETER": 1.1,
                "ROTATIONAL_PENALTY_PARAMETER": 1.2,
                "PROJECTION_VALID_FACTOR": 1.3,
            }
        mesh.add(
            BoundaryCondition(
                beam_set["line"],
                data=data,
                bc_type=bme.bc.point_coupling_indirect,
            )
        )

    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_four_c_fluid_element_section(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Add beam elements to an input file containing fluid elements."""

    input_file, _ = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="import"
        )
    )

    beam_mesh = Mesh()
    material = get_default_test_beam_material(material_type="reissner")
    beam_mesh.add(material)

    create_beam_mesh_line(
        beam_mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        material,
        [0, -0.5, 0],
        [0, 0.2, 0],
        n_el=5,
    )

    input_file.add(beam_mesh)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_surface_to_surface_contact_import(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that surface-to-surface contact problems can be imported as
    expected."""

    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="solid_mesh"
        ),
        convert_input_to_mesh=True,
    )

    input_file.add(mesh)

    # Compare with the reference file.
    assert_results_close(get_corresponding_reference_file_path(), input_file)
