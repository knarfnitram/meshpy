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

import warnings

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.coupling import Coupling
from beamme.core.element_beam import Beam
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.node import Node, NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.model_importer import import_cubitpy_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from tests.create_test_models import (
    create_beam_to_solid_conditions_model,
    create_tube_cubit,
)


# TODO: Standardize test parameterization for (full_import, additional_identifier).
# Currently, different tests use inconsistent patterns for parametrize:
#   - (False, "dict_import"), (True, "full_import")
#   - (False, None), (True, "full")
#   - Only "full_import" as a boolean param
# Consider unifying these under a shared fixture or helper to reduce redundancy
# and improve readability across tests. Also adjust reference file names.
@pytest.mark.parametrize(
    ("full_import", "additional_identifier"),
    [(False, None), (True, "full")],
)
def test_beam_to_solid_conditions(
    full_import,
    additional_identifier,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the input file for the beam-to-solid input conditions tests."""

    # Get the input file.
    input_file, mesh = create_beam_to_solid_conditions_model(
        get_default_test_beam_material,
        get_corresponding_reference_file_path,
        full_import=full_import,
    )
    input_file.add(mesh)

    # Check results
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        input_file,
    )


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


@pytest.mark.parametrize("full_import", [False, True])
@pytest.mark.cubitpy
def test_cubitpy_import(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that an import from a cubitpy object works as expected."""

    cubit = create_tube_cubit()
    input_file_cubit, mesh = import_cubitpy_model(
        cubit, convert_input_to_mesh=full_import
    )
    if full_import:
        input_file_cubit.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_tube"
        ),
        input_file_cubit,
    )


def test_deep_copy(get_bc_data, get_default_test_beam_material, assert_results_close):
    """This test checks that the deep copy function on a mesh does not copy the
    materials or functions."""

    # Create material and function object.
    mat = get_default_test_beam_material(material_type="reissner")
    fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")

    def create_mesh(mesh):
        """Add material and function to the mesh and create a beam."""
        mesh.add(fun, mat)
        set1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        set2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0])
        mesh.add(
            BoundaryCondition(
                set1["line"], get_bc_data(identifier=1), bc_type=bme.bc.dirichlet
            )
        )
        mesh.add(
            BoundaryCondition(
                set2["line"], get_bc_data(identifier=2), bc_type=bme.bc.neumann
            )
        )
        mesh.couple_nodes()

    # The second mesh will be translated and rotated with those vales.
    translate = [1.0, 2.34535435, 3.345353]
    rotation = Rotation([1, 0.2342342423, -2.234234], np.pi / 15 * 27)

    # First create the mesh twice, move one and get the input file.
    mesh_ref_1 = Mesh()
    mesh_ref_2 = Mesh()
    create_mesh(mesh_ref_1)
    create_mesh(mesh_ref_2)
    mesh_ref_2.rotate(rotation)
    mesh_ref_2.translate(translate)

    mesh = Mesh()
    mesh.add(mesh_ref_1, mesh_ref_2)

    # Now copy the first mesh and add them together in the input file.
    mesh_copy_1 = Mesh()
    create_mesh(mesh_copy_1)
    mesh_copy_2 = mesh_copy_1.copy()
    mesh_copy_2.rotate(rotation)
    mesh_copy_2.translate(translate)

    mesh_copy = Mesh()
    mesh_copy.add(mesh_copy_1, mesh_copy_2)

    # Check that the input files are the same.
    # TODO: add reference file check here as well
    assert_results_close(mesh, mesh_copy)


def test_mesh_add_checks():
    """This test checks that Mesh raises an error when double objects are added
    to the mesh."""

    # Mesh instance for this test.
    mesh = Mesh()

    # Create basic objects that will be added to the mesh.
    node = Node([0, 1.0, 2.0])
    element = Beam()
    mesh.add(node)
    mesh.add(element)

    # Create objects based on basic mesh items.
    coupling = Coupling(mesh.nodes, bme.bc.point_coupling, bme.coupling_dof.fix)
    coupling_penalty = Coupling(
        mesh.nodes, bme.bc.point_coupling_penalty, bme.coupling_dof.fix
    )
    geometry_set = GeometrySet(mesh.elements)
    mesh.add(coupling)
    mesh.add(coupling_penalty)
    mesh.add(geometry_set)

    # Add the objects again and check for errors.
    # TODO catch and test error messages
    with pytest.raises(ValueError):
        mesh.add(node)
    with pytest.raises(ValueError):
        mesh.add(element)
    with pytest.raises(ValueError):
        mesh.add(coupling)
    with pytest.raises(ValueError):
        mesh.add(coupling_penalty)
    with pytest.raises(ValueError):
        mesh.add(geometry_set)


def test_check_two_couplings(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """The current implementation can handle more than one coupling on a node
    correctly, therefore we check this here."""

    # Create mesh object
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0])

    # Call coupling twice -> this will create two coupling objects for the
    # corner node
    mesh.couple_nodes()
    mesh.couple_nodes()

    # Create the input file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize("reuse_nodes", [[None, False], ["reuse", True]])
def test_check_multiple_node_penalty_coupling(
    get_default_test_beam_material,
    reuse_nodes,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """For point penalty coupling constraints, we add multiple coupling
    conditions.

    This is checked in this test case. This method creates the flag
    reuse_nodes decides if equal nodes are unified to a single node.
    """

    # Create mesh object
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add three beams that have one common point
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -1, 0])

    mesh.couple_nodes(
        reuse_matching_nodes=reuse_nodes[1],
        coupling_type=bme.bc.point_coupling_penalty,
        coupling_dof_type={
            "POSITIONAL_PENALTY_PARAMETER": 10000,
            "ROTATIONAL_PENALTY_PARAMETER": 0,
        },
    )
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=reuse_nodes[0]),
        mesh,
    )


def test_check_double_elements(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Check if there are overlapping elements in a mesh."""

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection.
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])

    # Rotate the mesh with an arbitrary rotation.
    mesh.rotate(Rotation([1, 2, 3.24313], 2.2323423), [1, 3, -2.23232323])

    # The elements in the created mesh are overlapping, check that an error
    # is thrown.
    with pytest.raises(ValueError):
        mesh.check_overlapping_elements()

    # Check if the overlapping elements are written to the vtk output.
    warnings.filterwarnings("ignore")
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="beam", extension="vtu"
    )
    vtk_file = tmp_path / ref_file.name
    mesh.write_vtk(
        output_name="test_check_double_elements",
        output_directory=tmp_path,
        binary=False,
        overlapping_elements=True,
    )

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


@pytest.mark.parametrize("check", [True, False])
def test_check_overlapping_coupling_nodes(get_default_test_beam_material, check):
    """Per default, we check that coupling nodes are at the same physical
    position.

    This check can be deactivated with the keyword
    check_overlapping_nodes when creating a Coupling.
    """

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection.
    set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [2, 0, 0], [3, 0, 0])

    # Couple two nodes that are not at the same position.

    # Create the input file. This will cause an error, as there are two
    # couplings for one node.
    args = [
        [set_1["start"], set_2["end"]],
        bme.bc.point_coupling,
        "coupling_type_string",
    ]
    if check:
        with pytest.raises(ValueError):
            Coupling(*args)
    else:
        Coupling(*args, check_overlapping_nodes=False)


def test_check_start_end_node_error(
    get_default_test_beam_material,
):
    """Check that an error is raised if wrong start and end nodes are given to
    a mesh creation function."""

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Try to create a line with a starting node that is not in the mesh.
    node = NodeCosserat([0, 0, 0], Rotation())
    args = [mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]]
    kwargs = {"start_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)
    node.coordinates = [1, 0, 0]
    kwargs = {"end_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)
