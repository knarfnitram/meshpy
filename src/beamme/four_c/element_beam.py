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
"""This file implements beam elements for 4C."""

import warnings as _warnings

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.element_beam import Beam as _Beam
from beamme.core.element_beam import Beam2 as _Beam2
from beamme.core.element_beam import Beam3 as _Beam3
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialEulerBernoulli as _MaterialEulerBernoulli
from beamme.four_c.material import MaterialKirchhoff as _MaterialKirchhoff
from beamme.four_c.material import MaterialReissner as _MaterialReissner
from beamme.four_c.material import (
    MaterialReissnerElastoplastic as _MaterialReissnerElastoplastic,
)


def dump_four_c_beam_to_list(self) -> dict:
    """Return the dictionary representing this beam element in 4C.

    Args:
        self: The beam element to be dumped.
    """

    # Check the material.
    self._check_material()

    # Gather the element data for the input file.
    element_data = self.data.copy()
    element_data["type"] = _INPUT_FILE_MAPPINGS["beam_types"][
        type(self).four_c_beam_type
    ]
    element_data["MAT"] = self.material
    if type(self).four_c_triads:
        element_data["TRIADS"] = [
            item
            for i in _INPUT_FILE_MAPPINGS["n_nodes_to_node_ordering"][len(self.nodes)]
            for item in self.nodes[i].rotation.get_rotation_vector()
        ]
    if type(self).four_c_is_hermite_centerline:
        element_data["HERMITE_CENTERLINE"] = True
    return {
        "id": self.i_global + 1,
        "cell": {
            "type": _INPUT_FILE_MAPPINGS["n_nodes_to_cell_type"][len(self.nodes)],
            "connectivity": [
                self.nodes[i]
                for i in _INPUT_FILE_MAPPINGS["n_nodes_to_node_ordering"][
                    len(self.nodes)
                ]
            ],
        },
        "data": element_data,
    }


class Beam3rHerm2Line3(_Beam3):
    """Represents a Simo-Reissner beam element with third order Hermitian
    interpolation of the centerline and second order Lagrangian interpolation
    of the rotations."""

    four_c_beam_type: _BeamType = _BeamType.reissner
    four_c_is_hermite_centerline = True
    four_c_triads = True

    valid_material = [_MaterialReissner, _MaterialReissnerElastoplastic]

    coupling_fix_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0]}
    coupling_joint_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0]}

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""
        return dump_four_c_beam_to_list(self)


class Beam3rLine2Line2(_Beam2):
    """Represents a Reissner beam with linear shape functions in the rotations
    as well as the displacements."""

    four_c_beam_type = _BeamType.reissner
    four_c_is_hermite_centerline = False
    four_c_triads = True

    valid_material = [_MaterialReissner]

    coupling_fix_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 1, 1, 1]}
    coupling_joint_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 0, 0, 0]}

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""
        return dump_four_c_beam_to_list(self)


class Beam3kClass(_Beam3):
    """Represents a Kirchhoff beam element."""

    four_c_beam_type = _BeamType.kirchhoff
    four_c_is_hermite_centerline = False  # In 4C, the centerline is Hermitian but we don't require this information in the input file.
    four_c_triads = True

    valid_material = [_MaterialKirchhoff]

    coupling_fix_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 1, 1, 1, 0]}
    coupling_joint_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 0, 0, 0, 0]}

    def __init__(self, *, weak=True, rotvec=True, is_fad=True, **kwargs):
        _Beam.__init__(self, **kwargs)

        # Set the parameters for this beam.
        self.data["WK"] = weak
        self.data["ROTVEC"] = 1 if rotvec else 0
        if is_fad:
            self.data["USE_FAD"] = True

        # Show warning when not using rotvec.
        if not rotvec:
            _warnings.warn(
                "Use rotvec=False with caution, especially when applying the boundary conditions "
                "and couplings."
            )

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""
        return dump_four_c_beam_to_list(self)


def Beam3k(**kwargs_class):
    """This factory returns a function that creates a new Beam3kClass object
    with certain attributes defined.

    The returned function behaves like a call to the object.
    """

    def create_class(**kwargs):
        """The function that will be returned.

        This function should behave like the call to the __init__
        function of the class.
        """
        return Beam3kClass(**kwargs_class, **kwargs)

    return create_class


class Beam3eb(_Beam2):
    """Represents a Euler Bernoulli beam element."""

    four_c_beam_type = _BeamType.euler_bernoulli
    four_c_is_hermite_centerline = False
    four_c_triads = False

    valid_material = [_MaterialEulerBernoulli]

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same, i.e. the beam "
                "has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if _np.linalg.norm(direction / _np.linalg.norm(direction) - t1) >= _bme.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )

        return dump_four_c_beam_to_list(self)
