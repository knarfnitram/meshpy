# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
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
# -----------------------------------------------------------------------------
"""
This script is used to simulate 4C input files created with MeshPy.
"""

# Python imports.
import os
import unittest
import numpy as np
import shutil

# Testing imports.
from utils import (
    compare_test_result,
    skip_fail_four_c,
    testing_temp,
    testing_input,
)

# Meshpy imports.
from meshpy import (
    mpy,
    Rotation,
    InputFile,
    InputSection,
    MaterialReissner,
    Function,
    Beam3rHerm2Line3,
    BoundaryCondition,
    Mesh,
    set_header_static,
    set_runtime_output,
    GeometrySet,
)
from meshpy.four_c import (
    all_dbc_monitor_values_to_input,
    run_four_c,
)
from meshpy.mesh_creation_functions.beam_basic_geometry import create_beam_mesh_line
from meshpy.mesh_creation_functions.beam_honeycomb import create_beam_mesh_honeycomb
from meshpy.utility import check_node_by_coordinate
from meshes_for_testing import create_cantilver_model


class TestFullFourC(unittest.TestCase):
    """
    Create and run input files in 4C. They are test files and 4C should
    return 0.
    """

    def run_four_c_test(self, name, mesh, n_proc=2, restart=[None, None], **kwargs):
        """Run 4C with a input file and check the output

        Args
        ----
        name: str
            Name of the test case
        mesh: InputFile
            The InputFile object that contains the simulation
        n_proc: int
            Number of processors to run 4C on
        restart: [restart_step, restart_from]
            If the simulation should be a restart
        """

        skip_fail_four_c(self)

        # Check if temp directory exists.
        testing_dir = os.path.join(testing_temp, name)
        os.makedirs(testing_dir, exist_ok=True)

        # Create input file.
        input_file = os.path.join(testing_dir, name + ".dat")
        mesh.write_input_file(input_file, add_script_to_header=False, **kwargs)

        return_code = run_four_c(
            input_file,
            testing_dir,
            output_name=name,
            n_proc=n_proc,
            restart_step=restart[0],
            restart_from=restart[1],
        )
        self.assertEqual(0, return_code, msg="Test {} failed!".format(name))

    def test_four_c_simulation_honeycomb_sphere_as_input(self):
        """
        Test the honeycomb sphere model with different types of mesh import.
        """

        mpy.set_default_values()
        mpy.import_mesh_full = True
        self.create_honeycomb_sphere_as_input(
            "honeycomb_sphere", compare_created_input_file=True
        )

        mpy.set_default_values()
        mpy.import_mesh_full = False
        self.create_honeycomb_sphere_as_input("honeycomb_sphere_full_input")

    def create_honeycomb_sphere_as_input(
        self, name, *, compare_created_input_file=False
    ):
        """
        Create the same honeycomb mesh as defined in
        /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_\
        honeycomb_stretch_r01_circ10.dat
        The honeycomb beam is in contact with a rigid sphere, the sphere is
        moved compared to the original test file, since there are some problems
        with the contact convergence. The sphere is imported as an existing
        mesh.
        """

        # Read input file with information of the sphere and simulation.
        input_file = InputFile(
            description="honeycomb beam in contact with sphere",
            dat_file=os.path.join(testing_input, "4C_input_honeycomb_sphere.dat"),
        )

        # Modify the time step options.
        input_file.add(
            InputSection(
                "STRUCTURAL DYNAMIC",
                "NUMSTEP 5",
                "TIMESTEP 0.2",
                option_overwrite=True,
            )
        )

        # Delete the results given in the input file.
        input_file.delete_section("RESULT DESCRIPTION")
        input_file.add("-----RESULT DESCRIPTION")

        # Add result checks.
        displacement = [0.0, -8.09347204109101170, 2.89298005937795688]

        nodes = [268, 188, 182]
        for node in nodes:
            for i, direction in enumerate(["x", "y", "z"]):
                input_file.add(
                    InputSection(
                        "RESULT DESCRIPTION",
                        "STRUCTURE DIS structure NODE {} QUANTITY disp{} VALUE {} TOLERANCE 1e-10".format(
                            node, direction, displacement[i]
                        ),
                    )
                )

        # Material for the beam.
        material = MaterialReissner(
            youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1
        )

        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = create_beam_mesh_honeycomb(
            mesh_honeycomb,
            Beam3rHerm2Line3,
            material,
            50.0,
            10,
            4,
            n_el=1,
            closed_top=False,
            add_sets=True,
        )
        mesh_honeycomb.rotate(Rotation([0, 0, 1], 0.5 * np.pi))

        # Functions for the boundary conditions
        ft = Function(
            "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
            "VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0"
        )
        mesh_honeycomb.add(ft)

        # Change the sets to lines, only for purpose of matching the test file
        honeycomb_set["bottom"].geo_type = mpy.geo.line
        honeycomb_set["top"].geo_type = mpy.geo.line
        mesh_honeycomb.add(
            BoundaryCondition(
                honeycomb_set["bottom"],
                "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        mesh_honeycomb.add(
            BoundaryCondition(
                honeycomb_set["top"],
                "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0",
                format_replacement=[ft],
                bc_type=mpy.bc.dirichlet,
            )
        )

        # Add the mesh to the imported solid mesh.
        input_file.add(mesh_honeycomb)

        # Check the created input file
        if compare_created_input_file:
            compare_test_result(
                self, input_file.get_string(check_nox=False, header=False)
            )

        # Run the input file in 4C.
        self.run_four_c_test(name, input_file)

    def test_four_c_simulation_beam_and_solid_tube(self):
        """
        Test the beam and solid tube model with different types of mesh import.
        """

        mpy.set_default_values()
        mpy.import_mesh_full = True
        self.create_beam_and_solid_tube(
            "beam_and_solid_tube", compare_created_input_file=True
        )

        mpy.set_default_values()
        mpy.import_mesh_full = False
        self.create_beam_and_solid_tube("beam_and_solid_tube")

    def create_beam_and_solid_tube(self, name, *, compare_created_input_file=False):
        """Merge a solid tube with a beam tube and simulate them together."""

        # Create the input file and read solid mesh data.
        input_file = InputFile(description="Solid tube with beam tube")
        input_file.read_dat(os.path.join(testing_input, "4C_input_solid_tube.dat"))

        # Add options for beam_output.
        input_file.add(
            InputSection(
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                """
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            """,
            )
        )

        # Add functions for boundary conditions and material.
        sin = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME sin(t*2*pi)")
        cos = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME cos(t*2*pi)")
        material = MaterialReissner(
            youngs_modulus=1e9, radius=0.25, shear_correction=0.75
        )
        input_file.add(sin, cos, material)

        # Add a straight beam.
        input_file.add(material)
        cantilever_set = create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, material, [2, 0, -5], [2, 0, 5], n_el=3
        )

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                cantilever_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        input_file.add(
            BoundaryCondition(
                cantilever_set["end"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 FUNCT {} {} 0 0 0 0 0 0 0",
                format_replacement=[cos, sin],
                bc_type=mpy.bc.dirichlet,
            )
        )

        # Add result checks.
        displacement = [
            [1.50796091342925, 1.31453288915877e-8, 0.0439008100184687],
            [0.921450108160878, 1.41113401669104e-15, 0.0178350143764099],
        ]

        nodes = [32, 69]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(["x", "y", "z"]):
                input_file.add(
                    InputSection(
                        "RESULT DESCRIPTION",
                        "STRUCTURE DIS structure NODE {} QUANTITY disp{} VALUE {} TOLERANCE 1e-10".format(
                            node, direction, displacement[j][i]
                        ),
                    )
                )

        # Call get_unique_geometry_sets to check that this does not affect the
        # mesh creation.
        input_file.get_unique_geometry_sets(link_to_nodes="all_nodes")

        # Check the created input file
        if compare_created_input_file:
            compare_test_result(
                self, input_file.get_string(check_nox=False, header=False)
            )

        # Run the input file in 4C.
        self.run_four_c_test(name, input_file)

    def test_four_c_simulation_honeycomb_variants(self):
        """
        Create a few different honeycomb structures.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(description="Varieties of honeycomb")

        # Set options with different syntaxes.
        input_file.add(InputSection("PROBLEM SIZE", "DIM 3"))
        input_file.add(
            """
        -----------------------------------PROBLEM TYPE
        PROBLEMTYPE                           Structure
        RESTART                               0
        --------------------------------------IO
        OUTPUT_BIN                            No
        STRUCT_DISP                           No
        FILESTEPS                             1000
        VERBOSITY                             Standard
        """
        )
        input_file.add(
            InputSection(
                "IO/RUNTIME VTK OUTPUT",
                """
            OUTPUT_DATA_FORMAT                    binary
            INTERVAL_STEPS                        1
            EVERY_ITERATION                       No
            """,
            )
        )
        input_file.add(
            """
            ------------------------------------STRUCTURAL DYNAMIC
            LINEAR_SOLVER                         1
            INT_STRATEGY                          Standard
            DYNAMICTYPE                           Statics
            RESULTSEVRY                           1
            NLNSOL                                fullnewton
            PREDICT                               TangDis
            TIMESTEP                              1.
            NUMSTEP                               666
            MAXTIME                               10.0
            TOLRES                                1.0E-4
            TOLDISP                               1.0E-11
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               20
            """
        )
        input_file.add(
            InputSection("STRUCTURAL DYNAMIC", "NUMSTEP 1", option_overwrite=True)
        )
        input_file.add(
            InputSection(
                "SOLVER 1",
                """
            NAME                                  Structure_Solver
            SOLVER                                UMFPACK
            """,
            )
        )
        input_file.add(
            InputSection(
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                """
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            """,
            )
        )

        # Create four meshes with different types of honeycomb structure.
        mesh = Mesh()
        material = MaterialReissner(
            youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1
        )
        ft = []
        ft.append(Function("SYMBOLIC_FUNCTION_OF_TIME t"))
        ft.append(Function("SYMBOLIC_FUNCTION_OF_TIME t"))
        ft.append(Function("SYMBOLIC_FUNCTION_OF_TIME t"))
        ft.append(Function("SYMBOLIC_FUNCTION_OF_TIME t"))
        mesh.add(ft)

        counter = 0
        for vertical in [False, True]:
            for closed_top in [False, True]:
                mesh.translate(17 * np.array([1, 0, 0]))
                honeycomb_set = create_beam_mesh_honeycomb(
                    mesh,
                    Beam3rHerm2Line3,
                    material,
                    10,
                    6,
                    3,
                    n_el=2,
                    vertical=vertical,
                    closed_top=closed_top,
                )
                mesh.add(
                    BoundaryCondition(
                        honeycomb_set["bottom"],
                        "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                        bc_type=mpy.bc.dirichlet,
                    )
                )
                mesh.add(
                    BoundaryCondition(
                        honeycomb_set["top"],
                        "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL {1} {1} {1} 0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0",
                        format_replacement=[ft[counter], 0.0001],
                        bc_type=mpy.bc.neumann,
                        double_nodes=mpy.double_nodes.remove,
                    )
                )
                counter += 1

        # Add mesh to input file.
        input_file.add(mesh)

        # Add result checks.
        displacement = [
            [1.31917210027397980e-01, 1.99334884558314690e-01, 6.92209310957152130e-02],
            [1.32982726482608615e-01, 2.00555145810952351e-01, 6.97003431426771458e-02],
            [7.69274209663553116e-02, 1.24993734710951515e-01, 5.86799180712692867e-02],
            [6.98802675783889299e-02, 1.09892533095288236e-01, 4.83525527530398319e-02],
        ]

        nodes = [190, 470, 711, 1071]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(["x", "y", "z"]):
                input_file.add(
                    InputSection(
                        "RESULT DESCRIPTION",
                        "STRUCTURE DIS structure NODE {} QUANTITY disp{} VALUE {} TOLERANCE 1e-10".format(
                            node, direction, displacement[j][i]
                        ),
                    )
                )

        # Check the created input file
        compare_test_result(
            self, input_file.get_string(check_nox=False, header=False), rtol=1e-10
        )

        # Run the input file in 4C.
        self.run_four_c_test("honeycomb_variants", input_file)

    def test_four_c_simulation_rotated_beam_axis(self):
        """
        Create three beams that consist of two connected lines.
        - The first case uses the same nodes for the connection of the lines,
          and the nodes are equal in this case.
        - The second case uses the same nodes for the connection of the lines,
          but the nodes have a different rotation along the basis vector 1.
        - The third case uses two nodes at the connection between the lines,
          and couples them with a coupling.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(description="Rotation of beam along axis")

        # Set header
        set_header_static(input_file, time_step=0.05, n_steps=20)

        # Define linear function over time.
        ft = Function("SYMBOLIC_FUNCTION_OF_TIME t")
        input_file.add(ft)

        # Set beam material.
        mat = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

        # Direction of the lines and the rotation between the beams.
        direction = np.array([0.5, 1, 2])
        alpha = np.pi / 27 * 7
        force_fac = 0.01

        # Create mesh.
        for i in range(3):
            mesh = Mesh()

            # Create the first line.
            set_1 = create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [0, 0, 0], 1.0 * direction, n_el=3
            )

            if not i == 0:
                # In the second case rotate the line, so the triads do not
                # match any more.
                mesh.rotate(Rotation(direction, alpha))

            if i == 2:
                # The third line is with couplings.
                start_node = None
            else:
                start_node = set_1["end"]

            # Add the second line.
            set_2 = create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                mat,
                1.0 * direction,
                2.0 * direction,
                n_el=3,
                start_node=start_node,
            )

            # Add boundary conditions.
            mesh.add(
                BoundaryCondition(
                    set_1["start"],
                    "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                    bc_type=mpy.bc.dirichlet,
                )
            )
            mesh.add(
                BoundaryCondition(
                    set_2["end"],
                    "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL {1} {1} {1} {1} {1} {1} 0 0 0 FUNCT {0} {0} {0} {0} {0} {0} 0 0 0",
                    bc_type=mpy.bc.neumann,
                    format_replacement=[ft, force_fac],
                )
            )

            if i == 2:
                # In the third case add a coupling.
                mesh.couple_nodes()

            # Add the mesh to the input file.
            input_file.add(mesh)

            # Each time move the whole mesh.
            input_file.translate([1, 0, 0])

        # Add result checks.
        displacement = [1.5015284845, 0.35139255451, -1.0126517891]
        nodes = [13, 26, 40]
        for node in nodes:
            for i, direction in enumerate(["x", "y", "z"]):
                input_file.add(
                    InputSection(
                        "RESULT DESCRIPTION",
                        "STRUCTURE DIS structure NODE {} QUANTITY disp{} VALUE {} TOLERANCE 1e-10".format(
                            node, direction, displacement[i]
                        ),
                    )
                )

        # Check the created input file
        compare_test_result(self, input_file.get_string(check_nox=False, header=False))

        # Run the input file in 4C.
        self.run_four_c_test("rotated_beam_axis", input_file)
        self.run_four_c_test("rotated_beam_axis", input_file, nox_xml_file="xml_name")

    def test_four_c_simulation_dirichlet_boundary_to_neumann_boundary(self):
        """
        First simulate a cantilever beam with Dirichlet boundary conditions and
        then apply those as Neumann boundaries.
        """

        # Create and run the initial simulation.
        initial_simulation, beam_set = create_cantilver_model(n_steps=2)
        initial_simulation.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        initial_simulation.add(
            BoundaryCondition(
                beam_set["end"],
                "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL -0.2 1.5 1 0 0 0 0 0 0 FUNCT 1 1 1 0 0 0 0 0 0 TAG monitor_reaction",
                bc_type=mpy.bc.dirichlet,
            )
        )
        initial_simulation.add(
            """
            --IO/MONITOR STRUCTURE DBC
            PRECISION_FILE         10
            PRECISION_SCREEN       5
            FILE_TYPE              csv
            WRITE_HEADER           yes
            INTERVAL_STEPS         1
            """
        )

        # Check the input file
        compare_test_result(
            self,
            initial_simulation.get_string(check_nox=False, header=False),
            additional_identifier="initial",
        )

        # Run the simulation in 4C
        self.run_four_c_test(
            "dbc_to_nbc_initial", initial_simulation, delete_files=False
        )

        # Create and run the second simulation.
        restart_simulation, beam_set = create_cantilver_model(n_steps=21)
        restart_simulation.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        function_nbc = Function(
            """SYMBOLIC_FUNCTION_OF_TIME nbc_value
            VARIABLE 0 NAME nbc_value TYPE linearinterpolation NUMPOINTS 2 """
            "TIMES 1.0 11.0 VALUES 1.0 0.0"
        )
        restart_simulation.add(function_nbc)
        all_dbc_monitor_values_to_input(
            restart_simulation,
            os.path.join(
                testing_temp,
                "xxx_dbc_to_nbc_initial_monitor_dbc",
                "xxx_dbc_to_nbc_initial_102_monitor_dbc.csv",
            ),
            n_dof=9,
            time_span=[2 * 0.5, 4 * 0.5],
            fun_array=[function_nbc, function_nbc, function_nbc],
        )
        restart_simulation.add(
            """--RESULT DESCRIPTION
            STRUCTURE DIS structure NODE 21 QUANTITY dispx VALUE -4.09988307566066690e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispy VALUE  9.93075098427816383e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispz VALUE  6.62050065618549843e-01 TOLERANCE 1e-10
            """
        )

        # Check the input file of the restart simulation
        compare_test_result(
            self,
            restart_simulation.get_string(check_nox=False, header=False),
            additional_identifier="restart",
        )

        # Run the restart simulation
        self.run_four_c_test(
            "dbc_to_nbc_restart",
            restart_simulation,
            restart=[2, "xxx_dbc_to_nbc_initial"],
            delete_files=False,
        )

        # Delete all files from this test.
        items = []
        items.extend(glob.glob(testing_temp + "/xxx_dbc_to_nbc_*"))
        items.extend(glob.glob(testing_temp + "/dbc_to_nbc_*"))
        for item in items:
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)

    def test_four_c_simulation_dirichlet_boundary_to_neumann_boundary(self):
        """
        First simulate a cantilever beam with Dirichlet boundary conditions and
        then apply those as Neumann boundaries.
        For the application of the boundary conditions, all linear values for the force are used.
        """

        # Create and run the initial simulation.
        initial_simulation, beam_set = create_cantilver_model(2)
        initial_simulation.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        initial_simulation.add(
            BoundaryCondition(
                beam_set["end"],
                "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL -0.2 1.5 1 0 0 0 0 0 0 FUNCT 1 1 1 0 0 0 0 0 0 TAG monitor_reaction",
                bc_type=mpy.bc.dirichlet,
            )
        )
        initial_simulation.add(
            """
            --IO/MONITOR STRUCTURE DBC
            PRECISION_FILE         10
            PRECISION_SCREEN       5
            FILE_TYPE              csv
            WRITE_HEADER           yes
            INTERVAL_STEPS         1
            """
        )

        # Check the input file
        compare_test_result(
            self,
            initial_simulation.get_string(check_nox=False, header=False),
            additional_identifier="initial",
        )

        # Run the simulation in 4C
        initial_run_name = "dbc_to_nbc_initial"
        self.run_four_c_test(initial_run_name, initial_simulation)

        # Create and run the second simulation.
        restart_simulation, beam_set = create_cantilver_model(n_steps=21)
        restart_simulation.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        function_nbc = Function(
            """SYMBOLIC_FUNCTION_OF_TIME nbc_value
            VARIABLE 0 NAME nbc_value TYPE linearinterpolation NUMPOINTS 2 """
            "TIMES 1.0 11.0 VALUES 1.0 0.0"
        )
        restart_simulation.add(function_nbc)
        all_dbc_monitor_values_to_input(
            restart_simulation,
            os.path.join(
                testing_temp,
                initial_run_name,
                f"{initial_run_name}_monitor_dbc",
                f"{initial_run_name}_102_monitor_dbc.csv",
            ),
            n_dof=9,
            time_span=[10 * 0.5, 21 * 0.5],
            fun_array=[function_nbc, function_nbc, function_nbc],
        )
        restart_simulation.add(
            """--RESULT DESCRIPTION
            STRUCTURE DIS structure NODE 21 QUANTITY dispx VALUE -4.09988307566066690e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispy VALUE  9.93075098427816383e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispz VALUE  6.62050065618549843e-01 TOLERANCE 1e-10
            """
        )

        # Check the input file of the restart simulation
        compare_test_result(
            self,
            restart_simulation.get_string(check_nox=False, header=False),
            additional_identifier="restart",
        )

        # Run the restart simulation
        self.run_four_c_test(
            "dbc_to_nbc_restart",
            restart_simulation,
            restart=[2, f"../{initial_run_name}/{initial_run_name}"],
        )

    def test_four_c_simulation_dirichlet_boundary_to_neumann_boundary_with_all_values(
        self,
    ):
        """
        First simulate a cantilever beam with Dirichlet boundary conditions and
        then apply those as Neumann boundaries.
        For the application of the boundary conditions, all values of the force are used.
        """

        # number of simulation steps
        n_steps = 5
        dt = 0.1  # time step size from create_cantilver_model

        # Create and run the initial simulation.
        initial_simulation, beam_set = create_cantilver_model(n_steps, dt)

        # add function with
        initial_simulation.add(
            Function(
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME a \nVARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 TIMES 0 {} {} 9999999999.0 VALUES 0.0 1.0 0.0 0.0".format(
                    dt * n_steps, 2 * dt * n_steps
                )
            )
        )

        # apply displacment to all nodes
        for i, node in enumerate(beam_set["line"].get_all_nodes()):

            # do not constraint middle nodes
            if not node.is_middle_node:

                # fix beam at initial point
                if check_node_by_coordinate(node, 0, 0):
                    initial_simulation.add(
                        BoundaryCondition(
                            GeometrySet(node),
                            "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 1 1 1 0 0 0 0 0 0 ",
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                else:
                    # add small deformation at tip
                    initial_simulation.add(
                        BoundaryCondition(
                            GeometrySet(node),
                            "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 {} 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0 TAG monitor_reaction ".format(
                                0.25 * np.sin(node.coordinates[0] * np.pi),
                                "2",
                            ),
                            bc_type=mpy.bc.dirichlet,
                        )
                    )

        initial_simulation.add(
            """
            --IO/MONITOR STRUCTURE DBC
            PRECISION_FILE         10
            PRECISION_SCREEN       5
            FILE_TYPE              csv
            WRITE_HEADER           yes
            INTERVAL_STEPS         1
            """
        )

        # Check the input file
        compare_test_result(
            self,
            initial_simulation.get_string(check_nox=False, header=False),
            additional_identifier="dirichlet",
        )

        # Run the simulation in 4C
        initial_run_name = "all_dbc_to_nbc_initial"
        self.run_four_c_test(initial_run_name, initial_simulation)

        # Create and run the second simulation.
        force_simulation, beam_set = create_cantilver_model(2 * n_steps, dt)
        force_simulation.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )

        # set up path to monitor
        monitor_db_path = os.path.join(
            testing_temp, initial_run_name + "/", initial_run_name + "_monitor_dbc"
        )

        # convert the dirichlet conditions into neuman conditions
        for root, dirs, file_names in os.walk(monitor_db_path):
            for file_name in sorted(file_names):
                if "_monitor_dbc" in file_name:
                    all_dbc_monitor_values_to_input(
                        force_simulation,
                        os.path.join(monitor_db_path, file_name),
                        n_dof=9,
                        steps=[0, n_steps + 1],
                        time_span=[0, n_steps * dt, 2 * n_steps * dt],
                        type="hat",
                    )

        force_simulation.add(
            """--RESULT DESCRIPTION
            STRUCTURE DIS structure NODE 21 QUANTITY dispx VALUE 0.0 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispy VALUE 0.0 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 21 QUANTITY dispz VALUE 0.0 TOLERANCE 1e-10
            """
        )

        # Check the input file of the restart simulation
        compare_test_result(
            self,
            force_simulation.get_string(check_nox=False, header=False),
            additional_identifier="neumann",
            atol=1e-6,
        )

        set_runtime_output(force_simulation)
        initial_run_name = "all_dbc_to_nbc_initial_3"
        self.run_four_c_test(initial_run_name, force_simulation)

    def test_four_c_simulation_cantilever_convergence(self):
        """Create multiple simulations of a cantilever beam. This is a legacy test that used to test
        the simulation manager."""

        def create_and_run_cantilever(n_el, *, n_proc=1):
            """Create a cantilever beam for a convergence analysis."""

            input_file = InputFile()
            set_header_static(input_file, time_step=0.25, n_steps=4)
            set_runtime_output(input_file, output_energy=True)
            mat = MaterialReissner(radius=0.1, youngs_modulus=10000.0)
            beam_set = create_beam_mesh_line(
                input_file, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0], n_el=n_el
            )
            input_file.add(
                BoundaryCondition(
                    beam_set["start"],
                    (
                        "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0"
                    ),
                    bc_type=mpy.bc.dirichlet,
                )
            )
            fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
            input_file.add(
                fun,
                BoundaryCondition(
                    beam_set["end"],
                    (
                        "NUMDOF 9 ONOFF 0 0 1 0 0 0 0 0 0 VAL 0 0 -{} 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0"
                    ),
                    format_replacement=[0.5, fun],
                    bc_type=mpy.bc.dirichlet,
                ),
            )
            output_name = f"cantilever_convergence_{n_el}"
            self.run_four_c_test(output_name, input_file, n_proc=n_proc)
            testing_dir = os.path.join(testing_temp, output_name)
            my_data = np.genfromtxt(
                testing_dir + f"/{output_name}_energy.csv", delimiter=","
            )
            return my_data[-1, 2]

        results = {}
        for n_el in range(1, 7, 2):
            results[str(n_el)] = create_and_run_cantilever(n_el)
        results["ref"] = create_and_run_cantilever(40, n_proc=4)

        results_ref = {
            "5": 0.335081498526998,
            "3": 0.335055487040675,
            "1": 0.33453718896204,
            "ref": 0.335085590674607,
        }
        for key in results_ref.keys():
            self.assertTrue(abs(results[key] - results_ref[key]) < 1e-12)


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
