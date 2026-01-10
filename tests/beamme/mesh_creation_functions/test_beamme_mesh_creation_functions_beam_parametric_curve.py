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
"""Unit tests for the parametric beam mesh creation functions."""

import re

import pytest

from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)


def test_beamme_mesh_creation_functions_beam_parametric_curve_interval():
    """Check that an error is raised if wrong intervals are given."""

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Interval must be a 1D sequence of exactly two values, got array with shape (1, 2)."
        ),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[[0, 1]])

    with pytest.raises(
        ValueError,
        match=re.escape("Interval must contain exactly two values, got 3."),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[0, 1, 2])

    with pytest.raises(
        ValueError,
        match=re.escape("Interval must be in ascending order, got [1. 0.]."),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[1, 0])
