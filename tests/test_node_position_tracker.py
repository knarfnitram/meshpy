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
# -----------------------------------------------------------------------------
"""Test node position tracker of MeshPy."""

# Python imports
import unittest

import numpy as np
import pytest

# Meshpy imports
from meshpy.core.mesh import Node
from meshpy.four_c.boundary_condition_utility import Node_Position_Tracker


def test_add_same_node():
    """Test if node on plane function works properly."""

    # create a node
    node = Node([1.0, 1.0, 1.0])
    n_steps = 2

    # define two arrays with positions
    time = np.linspace(0, 1, n_steps)
    position = np.zeros([n_steps, 3])
    position[1] = position[1] + 1

    # create a tracking object
    tracker = Node_Position_Tracker()

    # we test here also if the sorting is working
    tracker.add(node, time + 2.0, position + 2.0)
    tracker.add(node, time, position)

    # get sorted time
    sorted_time, sorted_positions = tracker.get_sorted_by_time(node)

    # check if time is sorted
    assert sorted_time.tolist() == np.linspace(0, 3, n_steps * 2).tolist()

    # check if positions are still valid
    assert sorted_positions[1].tolist() == (np.ones([1, 3])[0]).tolist()
    assert sorted_positions[2].tolist() == (2 * np.ones([1, 3])[0]).tolist()

    node2 = Node([1.0, 1.0, 0.0])
    tracker.add(node2, np.linspace(1, 0, n_steps), position)

    # check if we do not access node2
    sorted_time, sorted_positions = tracker.get_sorted_by_time(node)
    assert sorted_positions[3].tolist() == (3 * np.ones([1, 3])[0]).tolist()

    # check if results of node2 are correct
    sorted_time, sorted_positions = tracker.get_sorted_by_time(node2)
    assert sorted_time.tolist() == np.linspace(0, 1, n_steps).tolist()
    assert sorted_positions[0].tolist() == (np.ones([1, 3])[0]).tolist()
