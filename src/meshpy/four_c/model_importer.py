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
"""This module contains functions to load and parse existing 4C input files."""

from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import fourcipp as _fourcipp
import fourcipp.fourc_input as _fourc_input
import yaml as _yaml

import meshpy.core.conf as _conf
from meshpy.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from meshpy.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from meshpy.core.conf import mpy as _mpy
from meshpy.core.coupling import Coupling as _Coupling
from meshpy.core.element_volume import (
    VolumeElement as _VolumeElement,
)
from meshpy.core.element_volume import (
    element_type_to_four_c_string as _element_type_to_four_c_string,
)
from meshpy.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import Node as _Node
from meshpy.four_c.element_volume import SolidRigidSphere as _SolidRigidSphere
from meshpy.four_c.input_file import InputFile as _InputFile
from meshpy.four_c.input_file import (
    get_geometry_set_indices_from_section as _get_geometry_set_indices_from_section,
)
from meshpy.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from meshpy.utils.environment import cubitpy_is_available as _cubitpy_is_available
from meshpy.utils.environment import fourcipp_is_available as _fourcipp_is_available


def import_cubitpy_model(cubit, convert_input_to_mesh: bool = False):
    """Imports an existing mesh with boundary conditions from cubit and
    optionally converts it into a MeshPy mesh and fourcipp input file object.

    Args:
        cubit: wrapper object of cubit containing the mesh and boundary conditions.
        convert_input_to_mesh: If True, the input file will be converted to a
            MeshPy mesh.
    Returns:
        A tuple with the input file and the mesh. If convert_input_to_mesh is
        False, the mesh will be empty. Note that the input sections which are
        converted to a MeshPy mesh are removed from the input file object.
    """

    if not _cubitpy_is_available():
        raise ImportError(
            "Import failed since cubitpy is not available. Please install cubitpy first."
        )
    from cubitpy.cubit_to_fourc_input import (
        get_input_file_with_mesh as _get_input_file_with_mesh,
    )

    input_file = _get_input_file_with_mesh(cubit)
    input_file.dump("test.4C.yaml")
    return import_four_c_model(
        _Path("test.4C.yaml"), convert_input_to_mesh=convert_input_to_mesh
    )


def import_four_c_model(
    input_file_path: _Path, convert_input_to_mesh: bool = False
) -> _Tuple[_InputFile, _Mesh]:
    """Import an existing 4C input file and optionally convert it into a MeshPy
    mesh.

    Args:
        input_file_path: A file path to an existing 4C input file that will be
            imported.
        convert_input_to_mesh: If True, the input file will be converted to a
            MeshPy mesh.

    Returns:
        A tuple with the input file and the mesh. If convert_input_to_mesh is
        False, the mesh will be empty. Note that the input sections which are
        converted to a MeshPy mesh are removed from the input file object.
    """

    input_file = _InputFile().from_4C_yaml(input_file_path=input_file_path)

    if convert_input_to_mesh:
        return _extract_mesh_sections(input_file)
    else:
        return input_file, _Mesh()


def _element_from_dict(nodes: _List[_Node], element: dict):
    """Create a solid element from a dictionary from a 4C input file.

    Args:
        nodes: A list of nodes that are part of the element.
        element: A dictionary with the element data.
    Returns:
        A solid element object.
    """

    # Depending on the number of nodes chose which solid element to return.
    # Add additional rigid sphere to strings.
    element_type_to_four_c_string_extended: _Dict[_Type[_VolumeElement], str] = {
        **_element_type_to_four_c_string,
        _SolidRigidSphere: "POINT1",
    }
    # invert key value pair
    element_type: _Dict[str, _Type[_VolumeElement]] = {
        v: k for k, v in element_type_to_four_c_string_extended.items()
    }

    if element["cell"]["type"] not in element_type:
        raise TypeError(
            f"Could not create a MeshPy element for {element['data']['type']} {element['cell']['type']}!"
        )

    return element_type[element["cell"]["type"]](nodes=nodes, data=element["data"])


def _boundary_condition_from_dict(
    geometry_set: _GeometrySetNodes,
    bc_key: _Union[_conf.BoundaryCondition, str],
    data: _Dict,
) -> _BoundaryConditionBase:
    """This function acts as a factory and creates the correct boundary
    condition object from a dictionary parsed from an input file."""

    del data["E"]

    if bc_key in (
        _mpy.bc.dirichlet,
        _mpy.bc.neumann,
        _mpy.bc.locsys,
        _mpy.bc.beam_to_solid_surface_meshtying,
        _mpy.bc.beam_to_solid_surface_contact,
        _mpy.bc.beam_to_solid_volume_meshtying,
    ) or isinstance(bc_key, str):
        return _BoundaryCondition(geometry_set, data, bc_type=bc_key)
    elif bc_key is _mpy.bc.point_coupling:
        return _Coupling(geometry_set, bc_key, data, check_overlapping_nodes=False)
    else:
        raise ValueError("Got unexpected boundary condition!")


def _get_yaml_geometry_sets(
    nodes: _List[_Node], geometry_key: _conf.Geometry, section_list: _List
) -> _Dict[int, _GeometrySetNodes]:
    """Add sets of points, lines, surfaces or volumes to the object."""

    # Create the individual geometry sets
    geometry_set_dict = _get_geometry_set_indices_from_section(section_list)
    geometry_sets_in_this_section = {}
    for geometry_set_id, node_ids in geometry_set_dict.items():
        geometry_sets_in_this_section[geometry_set_id] = _GeometrySetNodes(
            geometry_key, nodes=[nodes[node_id] for node_id in node_ids]
        )
    return geometry_sets_in_this_section


def _extract_mesh_sections(input_file: _InputFile) -> _Tuple[_InputFile, _Mesh]:
    """Convert an existing input file to a MeshPy mesh with mesh items, e.g.,
    nodes, elements, element sets, node sets, boundary conditions, materials.

    Args:
        input_file: The input file object that contains the sections to be
            converted to a MeshPy mesh.
    Returns:
        A tuple with the input file and the mesh. The input file will be
            modified to remove the sections that have been converted to a
            MeshPy mesh.
    """

    def _get_section_items(section_name):
        """Return the items in a given section.

        Since we will add the created MeshPy objects to the mesh, we
        delete them from the plain data storage to avoid having
        duplicate entries.
        """

        if section_name in input_file:
            return input_file.pop(section_name)
        else:
            return []

    # Go through all sections that have to be converted to full MeshPy objects
    mesh = _Mesh()

    # Add nodes
    if "NODE COORDS" in input_file:
        mesh.nodes = [_Node(node["COORD"]) for node in input_file.pop("NODE COORDS")]

    # Add elements
    if "STRUCTURE ELEMENTS" in input_file:
        for element in input_file.pop("STRUCTURE ELEMENTS"):
            nodes = [
                mesh.nodes[node_id - 1] for node_id in element["cell"]["connectivity"]
            ]
            mesh.elements.append(_element_from_dict(nodes=nodes, element=element))

    # Add geometry sets
    geometry_sets_in_sections: dict[str, dict[int, _GeometrySetNodes]] = {
        key: {} for key in _mpy.geo
    }
    for section_name in input_file.sections.keys():
        if section_name.endswith("TOPOLOGY"):
            section_items = _get_section_items(section_name)
            if len(section_items) > 0:
                # Get the geometry key for this set
                for key, value in _INPUT_FILE_MAPPINGS["geometry_sets"].items():
                    if value == section_name:
                        geometry_key = key
                        break
                else:
                    raise ValueError(f"Could not find the set {section_name}")
                geometry_sets_in_section = _get_yaml_geometry_sets(
                    mesh.nodes, geometry_key, section_items
                )
                geometry_sets_in_sections[geometry_key] = geometry_sets_in_section
                mesh.geometry_sets[geometry_key] = list(
                    geometry_sets_in_section.values()
                )

    # Add boundary conditions
    for (
        bc_key,
        geometry_key,
    ), section_name in _INPUT_FILE_MAPPINGS["boundary_conditions"].items():
        for item in _get_section_items(section_name):
            geometry_set_id = item["E"]
            geometry_set = geometry_sets_in_sections[geometry_key][geometry_set_id]
            mesh.boundary_conditions.append(
                (bc_key, geometry_key),
                _boundary_condition_from_dict(geometry_set, bc_key, item),
            )

    return input_file, mesh
