#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is an improvement of "random_4".
    There are some places in the maze that are not worth exploring.
    In particular, dead ends that do not contain any piece of cheese are a loss of time.
    Here, we exploit the available time in the "preprocessing" function to compute a reduced version of the maze.
    To remove the dead ends, we proceed iteratively by disconnecting them (checking they contain no cheese).
    Their neighbors are then tested, and if they have become dead ends, they are removed, etc.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Standard imports
import random
import numpy

# Import PyRat
from pyrat import *

# Import previously developed functions
from tutorial import get_vertices, get_neighbors, locations_to_action

#####################################################################################################################################################
############################################################### VARIABLES & CONSTANTS ###############################################################
#####################################################################################################################################################

"""
    Global variable to store the already visited cells.
"""

visited_cells = []

#####################################################################################################################################################

"""
    Global variable to store the path taken from the origin to the current cell (except backtrack moves).
"""

trajectory = []

#####################################################################################################################################################

"""
    Global variable to store the cells where we don't want to go.
"""

forbidden_cells = []

#####################################################################################################################################################
##################################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ####################################################
#####################################################################################################################################################

def preprocessing (maze, maze_width, maze_height, name, teams, player_locations, cheese, possible_actions) :

    """
        This function is called once at the beginning of the game.
        It is typically given more time than the turn function, to perform complex computations.
        Store the results of these computations in global variables to reuse them later during turns.
        
        In:
            * maze ............... numpy.ndarray [or] dict : int -> (dict : int -> int) ... Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width ......... int .................................................... Width of the maze in number of cells.
            * maze_height ........ int .................................................... Height of the maze in number of cells.
            * name ............... str .................................................... Name of the player controlled by this function.
            * teams .............. dict : str -> list (str) ............................... Recap of the teams of players.
            * player_locations ... dict : str -> int ...................................... Locations for all players in the game.
            * cheese ............. list (int) ............................................. List of available pieces of cheese in the maze.
            * possible_actions ... list (str) ............................................. List of possible actions.
            
        Out:
            * None.
    """

    # Global variables used
    global forbidden_cells
    global trajectory
    
    # Initialize trajectory with starting cell
    trajectory.append(player_locations[name])
    
    # We iteratively forbid cells that have only one neighbor (i.e., dead ends) and do not contain a cheese or the initial location
    while True :
        allowed_cells = [cell for cell in get_vertices(maze) if cell not in forbidden_cells and cell not in cheese and cell != player_locations[name]]
        allowed_cells_one_neighbor = [cell for cell in allowed_cells if len([neighbor for neighbor in get_neighbors(cell, maze) if neighbor not in forbidden_cells]) == 1]
        if len(allowed_cells_one_neighbor) == 0 :
            break
        forbidden_cells += allowed_cells_one_neighbor
    
#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn (maze, maze_width, maze_height, name, teams, player_locations, player_scores, player_muds, cheese, possible_actions) :

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        
        In:
            * maze ............... numpy.ndarray [or] dict : int -> (dict : int -> int) ... Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width ......... int .................................................... Width of the maze in number of cells.
            * maze_height ........ int .................................................... Height of the maze in number of cells.
            * name ............... str .................................................... Name of the player controlled by this function.
            * teams .............. dict : str -> list (str) ............................... Recap of the teams of players.
            * player_locations ... dict : str -> int ...................................... Locations for all players in the game.
            * player_scores ...... dict : str -> float .................................... Scores for all players in the game.
            * player_muds ........ dict : str -> (dict : str -> int) ...................... Indicates which player is currently crossing mud.
            * cheese ............. list (int) ............................................. List of available pieces of cheese in the maze.
            * possible_actions ... list (str) ............................................. List of possible actions.
            
        Out:
            * action ... list (str) ... One of the possible actions, as given in possible_actions.
    """

    # Global variables used
    global visited_cells
    global forbidden_cells
    global trajectory

    # Mark current cell as visited
    if player_locations[name] not in visited_cells :
        visited_cells.append(player_locations[name])
    trajectory.append(player_locations[name])
    
    # Go to an unvisited non-forbidden neighbor in priority
    neighbors = get_neighbors(player_locations[name], maze)
    allowed_neighbors = [neighbor for neighbor in neighbors if neighbor not in forbidden_cells]
    unvisited_allowed_neighbors = [neighbor for neighbor in allowed_neighbors if neighbor not in visited_cells]
    if len(unvisited_allowed_neighbors) > 0 :
        neighbor = random.choice(unvisited_allowed_neighbors)
        visited_cells.append(neighbor)

    # If there is no unvisited neighbor, backtrack the trajectory
    else :
        _ = trajectory.pop(-1)
        neighbor = trajectory.pop(-1)

    # Retrieve the corresponding action
    action = locations_to_action(player_locations[name], neighbor, maze_width)
    return action

#####################################################################################################################################################
######################################################################## GO ! #######################################################################
#####################################################################################################################################################

if __name__ == "__main__" :

    # Map the functions to the character
    players = [{"name" : "rat", "preprocessing_function" : preprocessing, "turn_function" : turn}]

    # Customize the game elements
    config = {"maze_width" : 15,
              "maze_height" : 11,
              "mud_percentage" : 0.0,
              "nb_cheese" : 1}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################
