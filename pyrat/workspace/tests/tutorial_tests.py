#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program contains all the unit tests for the functions developed in the program "tutorial.py".
    Let's consider the following maze for our tests:
    #############################################################
    # (0)       # (1)      # (2)       ⵗ (3)       # (4)        #
    #           #          #           ⵗ           #            #
    #           #          #           ⵗ           #            #
    #⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅############⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅##########################
    # (5)       ⵗ (6)       ⵗ (7)       ⵘ (8)       ⵘ (9)       #
    #           ⵗ           ⵗ           6           9            #
    #           ⵗ           ⵗ           ⵘ           ⵘ           #
    #⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#ⴾⴾⴾⴾⴾⴾ8ⴾⴾⴾⴾⴾⴾ############⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#############
    # (10)      ⵗ (11)      # (12)      # (13)      # (14)      #
    #           ⵗ           #           #           #           #
    #           ⵗ           #           #           #           #
    #ⴾⴾⴾⴾⴾⴾ9ⴾⴾⴾⴾⴾⴾ#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#############ⴾⴾⴾⴾⴾⴾ6ⴾⴾⴾⴾⴾⴾ#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#
    # (15)      ⵘ (16)      ⵗ (17)      ⵘ (18)      ⵗ (19)      #
    #           4           ⵗ           5           ⵗ            #
    #           ⵘ           ⵗ           ⵘ           ⵗ           #
    #⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅#
    # (20)      # (21)      ⵗ (22)      # (23)      # (24)      #
    #           #           ⵗ           #           #           #
    #           #           ⵗ           #           #           #
    #############################################################
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports
import unittest
import numpy
import sys
import os

# Previously developed functions
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "programs"))
from tutorial import *

#####################################################################################################################################################
############################################################### UNIT TESTS DEFINITION ###############################################################
#####################################################################################################################################################

class TestsTutorial (unittest.TestCase):

    """
        Here we choose to use the unittest module to perform unit tests.
        This module is very simple to use and allows to easily check if the code is working as expected.
    """

    #############################################################################################################################################
    #                                                                CONSTRUCTOR                                                                #
    #############################################################################################################################################

    def __init__ ( self:     Self,
                   *args:    Any,
                   **kwargs: Any,
                 ) ->        Self:

        """
            This function is the constructor of the class.
            In:
                * self:   Reference to the current object.
                * args:   Arguments of the parent constructor.
                * kwargs: Keyword arguments of the parent constructor.
            Out:
                * self: Reference to the current object.
        """

        # Inherit from parent class
        super(TestsTutorial, self).__init__(*args, **kwargs)

        # We need to store the width of the maze
        self.maze_width = 5

        # We define the graph structures that will be used for the tests
        self.graph_dictionary = {0: {5: 1},
                                 2: {3: 1, 7: 1},
                                 3: {2: 1},
                                 5: {0: 1, 6: 1, 10: 1},
                                 6: {5: 1, 7: 1, 11: 8},
                                 7: {2: 1, 3: 1, 6: 1, 8: 6},
                                 8: {7: 6, 9: 9, 13: 1},
                                 9: {8: 9},
                                 10: {5: 1, 11: 1, 15: 9},
                                 11: {6: 8, 10: 1, 16: 1},
                                 13: {8: 1, 18: 6},
                                 14: {19: 1},
                                 15: {10: 9, 16: 4, 20: 1},
                                 16: {11: 1, 15: 4, 17: 1, 21: 1},
                                 17: {16: 1, 18: 5, 22: 1},
                                 18: {13: 6, 17: 5, 19: 1, 23: 1},
                                 19: {14: 1, 18: 1, 24: 1},
                                 20: {15: 1},
                                 21: {16: 1, 22: 1},
                                 22: {17: 1, 21: 1},
                                 23: {18: 1},
                                 24: {19: 1}}
        
        # Here is the same graph represented as an adjacency matrix
        self.graph_matrix = numpy.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 6, 0, 9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 5, 0, 1, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        
    #############################################################################################################################################
    #                                                               PUBLIC METHODS                                                              #
    #############################################################################################################################################

    def test_get_vertices ( self: Self
                          ) ->    None:

        """
            This function tests the function "get_vertices" of the file "tutorial.py".
            It checks that the function returns the correct list of vertices for both graph structures.
            In:
                * self: Reference to the current object.
            Out:
                * None.
        """

        # We test the function for both graph structures
        for graph in [self.graph_dictionary, self.graph_matrix]:

            # We check that the function returns the correct list of vertices
            self.assertEqual(get_vertices(graph), [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    #############################################################################################################################################

    def test_get_neighbors ( self: Self
                           ) ->    None:

        """
            This function tests the function "get_neighbors" of the file "tutorial.py".
            It checks that the function returns the correct list of neighbors for both graph structures.
            In:
                * self: Reference to the current object.
            Out:
                * None.
        """

        # We test the function for both graph structures
        for graph in [self.graph_dictionary, self.graph_matrix]:

            # We check that the function returns the correct list of neighbors for standard cases
            self.assertEqual(get_neighbors(9, graph), [8])
            self.assertEqual(get_neighbors(2, graph), [3, 7])
            self.assertEqual(get_neighbors(8, graph), [7, 9, 13])
            self.assertEqual(get_neighbors(16, graph), [11, 15, 17, 21])

            # The function should raise an exception if the vertex is not in the graph
            self.assertRaises(Exception, get_neighbors, 25, graph)
        
        # Note the different behavior between structures when not using the function correctly (cf. comments regarding assertions in function definition)
        self.assertRaises(Exception, get_neighbors, 1, self.graph_dictionary)
        self.assertEqual(get_neighbors(1, self.graph_matrix), [])

    #############################################################################################################################################

    def test_get_weight ( self: Self
                        ) ->    None:

        """
            This function tests the function "get_weight" of the file "tutorial.py".
            It checks that the function returns the correct weight for both graph structures.
            In:
                * self: Reference to the current object.
            Out:
                * None.
        """

        # We test the function for both graph structures
        for graph in [self.graph_dictionary, self.graph_matrix]:

            # We check that the function returns the correct weight for standard cases
            self.assertEqual(get_weight(9, 8, graph), 9)
            self.assertEqual(get_weight(17, 22, graph), 1)

            # The function should raise an exception if the vertex is not in the graph
            self.assertRaises(Exception, get_weight, 24, 25, graph)
        
        # Note the different behavior between structures when not using the function correctly (cf. comments regarding assertions in function definition)
        self.assertRaises(Exception, get_weight, 0, 0, self.graph_dictionary)
        self.assertRaises(Exception, get_weight, 0, 1, self.graph_dictionary)
        self.assertEqual(get_weight(0, 0, self.graph_matrix), 0)
        self.assertEqual(get_weight(0, 1, self.graph_matrix), 0)

    #############################################################################################################################################

    def test_locations_to_action ( self: Self
                                 ) ->    None:

        """
            This function tests the function "locations_to_action" of the file "tutorial.py".
            It checks that the function returns the correct action.
            In:
                * self: Reference to the current object.
            Out:
                * None.
        """

        # We check that the function returns the correct action for standard cases
        self.assertEqual(locations_to_action(16, 11, self.maze_width), "north")
        self.assertEqual(locations_to_action(16, 15, self.maze_width), "west")
        self.assertEqual(locations_to_action(16, 17, self.maze_width), "east")
        self.assertEqual(locations_to_action(16, 21, self.maze_width), "south")
        self.assertEqual(locations_to_action(16, 16, self.maze_width), "nothing")

        # The function should raise an exception if the locations are not adjacent
        self.assertRaises(Exception, locations_to_action, 16, 20, self.maze_width)

#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Run all unit tests
    unittest.main(verbosity=2)

#####################################################################################################################################################
#####################################################################################################################################################