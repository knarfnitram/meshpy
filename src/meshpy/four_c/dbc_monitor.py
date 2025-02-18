# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This function converts the DBC monitor log files to Neumann input
sections."""

from typing import Optional

import numpy as np

from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.function import Function
from meshpy.four_c.function_utility import (
    create_linear_interpolation_function,
)
from meshpy.four_c.input_file import InputFile


def linear_time_transformation(
    time, values, time_span, *, flip=False, valid_start_and_end_point=False
):
    """Performs a transformation of the time to a new interval with an
    appropriate value vector.

    Args
    ----
    time: np.array
        array with time values
    values: np.array
        corresponding values to time
    time_span: [list] with 2 or 3 entries:
        time_span[0:2] defines the time interval to which the initial time interval should be scaled.
        time_span[3] optional timepoint to repeat last value
    flip: Bool
        Flag if the values should be reversed
    valid_start_and_end_point: Bool
        optionally adds a valid starting point at t=0 and timespan[3] if provided
    """

    # flip values if desired and adjust time
    if flip is True:
        values = np.flipud(values)
        time = np.flip(-time) + time[-1]

    # transform time to interval
    min_t = np.min(time)
    max_t = np.max(time)

    # scaling/transforming the time into the user defined time
    time = time_span[0] + (time - min_t) * (time_span[1] - time_span[0]) / (
        max_t - min_t
    )

    # ensure that start time is valid
    if valid_start_and_end_point and time[0] > 0.0:
        # add starting time 0
        time = np.append(0.0, time)

        # add first coordinate again at the beginning of the array
        if len(values.shape) == 1:
            values = np.append(values[0], values)
        else:
            values = np.append(values[0], values).reshape(
                values.shape[0] + 1, values.shape[1]
            )

    # repeat last value at provided time point
    if valid_start_and_end_point and len(time_span) > 2:
        if time_span[2] > time_span[1]:
            time = np.append(time, time_span[2])
            values = np.append(values, values[-1]).reshape(
                values.shape[0] + 1, values.shape[1]
            )
    if not valid_start_and_end_point and len(time_span) > 2:
        raise Warning("You specified unnecessarily a third component of time_span.")

    return time, values


def read_dbc_monitor_file(file_path):
    """Load the Dirichlet boundary condition monitor log and return the data as
    well as the nodes of this boundary condition.

    Args
    ----
    file_path: str
        Path to the Dirichlet boundary condition monitor log.

    Return
    ----
    [node_ids], [time], [force], [moment]
    """

    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Extract the nodes for this condition.
    condition_nodes_prefix = "Nodes of this condition:"
    if not lines[1].startswith(condition_nodes_prefix):
        raise ValueError(
            f'The second line in the monitor file is supposed to start with "{condition_nodes_prefix}" but the line reads "{lines[1]}"'
        )
    node_line = lines[1].split(":")[1]
    node_ids_str = node_line.split()
    nodes = []
    for node_id_str in node_ids_str:
        node_id = int(node_id_str)
        nodes.append(node_id)

    # Find the start of the data lines.
    for i, line in enumerate(lines):
        if line.split(" ")[0] == "step":
            break
    else:
        raise ValueError('Could not find "step" in file!')
    start_line = i + 1

    # Get the monitor data.
    data = []
    for line in lines[start_line:]:
        data.append(np.fromstring(line, dtype=float, sep=" "))
    data = np.array(data)

    return nodes, data[:, 1], data[:, 4:7], data[:, 7:]


def add_point_neuman_condition_to_input_file(
    input_file: InputFile,
    nodes: list[int],
    function_array: list[Function],
    force: np.ndarray,
    *,
    n_dof: int = 3,
):
    """Adds a Neumann boundary condition to the input file for the given
    node_ids with the function_array and force values by creating a new
    geometry set.

    Args
    ----
    input_file: InputFile
        InputFile where the boundary conditions are added to
    nodes: [node_id]
        list containing the ids of the nodes for the condition
    function_array: [function]
        list with functions
    force: [np.ndarray]
        values to scale the function array with
    n_dof: int
        Number of DOFs per node.
    """

    # check if the dimensions of force and functions match
    if force.size != 3:
        raise ValueError(
            f"The forces vector must have dimensions [3x1] not [{force.size}x1]"
        )

    # repeat function automatically if it is only provided once
    if len(function_array) == 1:
        function_array.append(function_array[0])
        function_array.append(function_array[0])

    if len(function_array) != 3:
        raise ValueError(
            f"The function array must have length 3 not {len(function_array)}."
        )

    # Add the function to the input file, if they are not previously added.
    for function in function_array:
        input_file.add(function)

    # Create GeometrySet with nodes.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mesh_nodes)

    # Create the Boundary Condition.
    extra_dof_zero = " 0" * (n_dof - 3)
    bc = BoundaryCondition(
        geo,
        (
            "NUMDOF {n_dof} ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}"
            "{edz} FUNCT {{}} {{}} {{}}{edz}"
        ).format(n_dof=n_dof, data=force, edz=extra_dof_zero),
        bc_type=mpy.bc.neumann,
        format_replacement=function_array,
    )
    input_file.add(bc)


def dbc_monitor_to_input_all_values(
    input_file: InputFile,
    file_path: str,
    *,
    steps: list[int] = [],
    time_span: list[int] = [0, 1, 2],
    type: Optional[str] = "linear",
    flip_time_values: bool = False,
    functions: list[Function] = [],
    **kwargs,
):
    """Extracts all the force values of the monitored Dirichlet boundary
    condition and converts them into a Function with a Neumann boundary
    condition for the input_file. The monitor log force values must be obtained
    from a previous simulation with constant step size. The discretization of
    the previous simulation must be identical to the one within the input_file.
    The extracted force values are passed to a linear interpolation
    4C-function. It is advisable to only call this function once all nodes have
    been added to the input file.

    Args
    ----
    input_file: InputFile
        The input file where the created Neumann boundary condition is added
        to. The nodes(e.g., discretization) referred to in the log file must match with the ones
        in input_file.
    file_path: str
        Path to the Dirichlet boundary condition log file.
    steps: [int,int]
        Index range of which the force values are extracted. Default 0 and -1 extracts every point from the array.
    time_span: [t1, t2, t3] in float
        Transforms the given time array into this specific format.
        The time array always starts at 0 and ends at t3 to ensure a valid simulation.
    type: str or linear
        two types are available:
            1) "linear": not specified simply extract all values and apply them between time interval t1 and t2.
            2) "hat": puts the values first until last value is reached and then decays them back to first value.
            Interpolation starting from t1 going to the last value at (t1+t2)/2 and going back to the value at time t2.
    flip_time_values: bool
        indicates, if the extracted forces should be flipped or rearranged wrt. to the time
        For flip_time_values=true, the forces at the final time are applied at t_start.
    functions: [Function, Function, Function]
        Array consisting of 3 custom functions(x,y,z). The value for boundary condition is selected from the last steps.
    """

    nodes, time, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0

    # if special index range is provided use it
    if steps:
        time = time[steps[0] : steps[1] + 1]
        force = force[steps[0] : steps[1] + 1, :]
    else:
        # otherwise define steps from start to end
        steps = [0, -1]

    # apply transformations to time and forces according to the schema
    if type == "linear":
        if not len(time_span) == 2:
            raise ValueError(
                f"Please provide a time_span with size 1x2 not {len(time_span)}"
            )

        time, force = linear_time_transformation(
            time, force, time_span, flip=flip_time_values
        )
        if len(functions) != 3:
            print("Please provide a list with three valid Functions.")

    elif type == "hat":
        if not len(time_span) == 3:
            raise ValueError(
                f"Please provide a time_span with size 1x3 not {len(time_span)}"
            )

        if functions:
            print(
                "You selected type",
                type,
                ", however the provided functions ",
                functions,
                " are overwritten.",
            )
        functions = []

        # create the two intervals
        time1, force1 = linear_time_transformation(
            time, force, time_span[0:2], flip=flip_time_values
        )
        time2, force2 = linear_time_transformation(
            time, force, time_span[1:3], flip=(not flip_time_values)
        )

        # remove first element since it is duplicated zero
        np.delete(time2, 0)
        np.delete(force2, 0)

        # add the respective force
        time = np.concatenate((time1, time2[1:]))
        force = np.concatenate((force1, force2[1:]), axis=0)

    else:
        raise ValueError(
            "The selected type: "
            + str(type)
            + " is currently not supported. Feel free to add it here."
        )

    # overwrite the function, if one is provided since for the specific types the function is generated
    if not type == "linear":
        for dim in range(force.shape[1]):
            # create a linear function with the force values per dimension
            fun = create_linear_interpolation_function(
                time, force[:, dim], function_type="SYMBOLIC_FUNCTION_OF_TIME"
            )

            # add the function to the input array
            input_file.add(fun)

            # store function
            functions.append(fun)

        # now set forces to 1 since the force values are already extracted in the function's values
        force = np.zeros_like(force) + 1.0

    elif len(functions) != 3:
        raise ValueError("Please provide functions with ")

    # Create condition in input file.
    add_point_neuman_condition_to_input_file(
        input_file, nodes, functions, force[steps[1]], **kwargs
    )


def dbc_monitor_to_input(
    input_file: InputFile,
    file_path: str,
    *,
    step: int = -1,
    function: Function,
    **kwargs,
):
    """Converts the last value of a Dirichlet boundary condition monitor log to
    a Neumann boundary condition input section.

    Args
    ----
    input_file: InputFile
        The input file where the created Neumann boundary condition is added
        to. The nodes referred to in the log file have to match with the ones
        in the input section. It is advisable to only call this function once
        all nodes have been added to the input file.
    file_path: str
        Path to the Dirichlet boundary condition log file.
    step: int
        Step values to be used. Default is -1, i.e. the last step.
    function: Function
        Function for the Neumann boundary condition.
    """

    # read the force
    nodes, _, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0

    # Create condition in input file.
    add_point_neuman_condition_to_input_file(
        input_file, nodes, [function] * 3, force[step], **kwargs
    )
