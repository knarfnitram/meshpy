# -*- coding: utf-8 -*-
"""
This module implements a class to handle boundary conditions in the input file.
"""

# Python modules.
import warnings

# Meshpy modules.
from . import mpy, BaseMeshItem, get_close_nodes


class BoundaryCondition(BaseMeshItem):
    """This object represents one boundary condition in the input file."""
    
    def __init__(self, geometry_set, bc_string, format_replacement=None,
            bc_type=None, **kwargs):
        """
        Initialize the object.
        
        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_string: str
            Text that will be displayed in the input file for this boundary
            condition.
        format_replacement: str, list
            Replacement with the str.format() function for bc_string.
        bc_type: mpy.boundary
            Type of the boundary condition (dirichlet or neumann).
        """
        
        BaseMeshItem.__init__(self, is_dat=False)
        self.bc_string = bc_string
        self.bc_type = bc_type
        self.format_replacement = format_replacement  
        self.geometry_set = geometry_set
        
        # Check the parameters for this object.
        self._check_multiple_nodes(**kwargs)

    def _get_dat(self, **kwargs):
        """
        Add the content of this object to the list of lines.
        
        Args:
        ----
        lines: list(str)
            The contents of this object will be added to the end of lines.
        """
        
        if self.format_replacement:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string
        
        return 'E {} - {}'.format(
            self.geometry_set.n_global,
            dat_string
            )


    def _check_multiple_nodes(self, double_nodes=None):
        """
        Check for point Neumann boundaries that there is not a double
        Node in the set
        """
        
        if double_nodes is mpy.double_nodes_keep:
            return
        
        if (self.bc_type == mpy.neumann
                and self.geometry_set.geometry_type == mpy.point):
            partners = get_close_nodes(self.geometry_set.nodes)
            # Create a list with nodes that will not be kept in the set.
            double_node_list = []
            for node_list in partners:
                for i, node in enumerate(node_list):
                    if i > 0:
                        double_node_list.append(node)
            if (len(double_node_list) > 0 and
                double_nodes is mpy.double_nodes_remove):
                # Create the nodes again for the set.
                self.geometry_set.nodes = [node for node in
                    self.geometry_set.nodes if (not node in double_node_list)]
            elif len(double_node_list) > 0:
                warnings.warn('There are overlapping nodes in this point ' + \
                    'Neumann boundary, and it is not specified on how to ' + \
                    'handle them!')
                