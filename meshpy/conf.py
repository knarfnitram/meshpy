# -*- coding: utf-8 -*-
"""
This module defines a global object that manages all kind of stuff regarding
meshpy.
"""


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """
    
    def __init__(self):

        # Version information
        self.version = '0.0.2'
        self.git_sha = None
        self.git_date = None
        
        # Precision for floats in output
        self.dat_precision = '{:.12g}'

        # Set the epsilons for comparison of different types of values.
        self.eps_quaternion = 1e-10
        self.eps_pos = 1e-10
        
        # Values for the formating of the input file
        self.dat_len_section = 80
        
        # Geometry types
        self.point = 'geometry_point'
        self.line = 'geometry_line'
        self.surface = 'geometry_surface'
        self.volume = 'geometry_volume'
        self.geometry = [self.point, self.line, self.surface, self.volume]
        
        # Boundary conditions types
        self.dirichlet = 'boundary_condition_dirichlet'
        self.neumann = 'boundary_condition_neumann'
        self.boundary_condition = [self.dirichlet, self.neumann]
        
        # Coupling types
        self.coupling_fix = 'coupling_fix'
        self.coupling_joint = 'coupling_joint'
        
        # Handling of multiple nodes in neuman bcs.
        self.double_nodes_remove = 'double_nodes_remove'
        self.double_nodes_keep = 'double_nodes_keep'


mpy = MeshPy()