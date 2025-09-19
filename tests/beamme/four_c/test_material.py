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
"""This file tests materials for 4C."""

from beamme.four_c.material import MaterialReissner


def test_material_reissner(assert_results_close):
    """Test Reissner material."""

    mat = MaterialReissner(
        radius=0.5,
        youngs_modulus=1234.56,
        nu=0.33,
        density=0.123,
        interaction_radius=2.4,
        shear_correction=17.15,
    )

    # set manually
    mat.i_global = 0

    mat_expected = {
        "MAT": 1,
        "MAT_BeamReissnerElastHyper": {
            "YOUNG": 1234.56,
            "POISSONRATIO": 0.33,
            "DENS": 0.123,
            "CROSSAREA": 0.7853981633974483,
            "SHEARCORR": 17.15,
            "MOMINPOL": 0.09817477042468103,
            "MOMIN2": 0.04908738521234052,
            "MOMIN3": 0.04908738521234052,
            "INTERACTIONRADIUS": 2.4,
        },
    }

    assert_results_close(mat.dump_to_list(), mat_expected)


def test_material_reissner_by_modes(assert_results_close):
    """Test Reissner material by modes with scaling factors."""

    mat = MaterialReissner(
        radius=0.5,
        youngs_modulus=1234.56,
        nu=0.33,
        density=0.123,
        interaction_radius=2.4,
        shear_correction=17.15,
        by_modes=True,
        scale_axial_rigidity=1.1,
        scale_shear_rigidity=1.2,
        scale_torsional_rigidity=1.3,
        scale_bending_rigidity=1.4,
    )

    # set manually
    mat.i_global = 0

    mat_expected = {
        "MAT": 1,
        "MAT_BeamReissnerElastHyper_ByModes": {
            "EA": 1066.5832722643493,
            "GA2": 7501.8057905674295,
            "GA3": 7501.8057905674295,
            "GI_T": 59.234375168474614,
            "EI2": 84.84185120284594,
            "EI3": 84.84185120284594,
            "RhoA": 0.09660397409788614,
            "MASSMOMINPOL": 0.012075496762235767,
            "MASSMOMIN2": 0.006037748381117884,
            "MASSMOMIN3": 0.006037748381117884,
            "INTERACTIONRADIUS": 2.4,
        },
    }

    assert_results_close(mat.dump_to_list(), mat_expected)
