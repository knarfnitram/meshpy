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
"""Unit tests for the dbc monitor."""

from beamme.four_c.dbc_monitor import read_dbc_monitor_file


def test_beamme_four_c_dbc_monitor_read_dbc_monitor_file(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that a dbc monitor file can be read correctly."""

    nodes, time, force, moment = read_dbc_monitor_file(
        get_corresponding_reference_file_path(extension="yaml")
    )
    assert_results_close(nodes, [2, 4, 5, 9, 10])
    assert_results_close(time, [0.0, 0.1, 0.2])
    assert_results_close(force, [[0, 0, 0], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    assert_results_close(moment, [[0, 0, 0], [0.4, 0.5, 0.6], [1.0, 1.1, 1.2]])
