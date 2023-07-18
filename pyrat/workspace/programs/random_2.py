#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is an improvement of "random_1".
    Contrary to that previous version, here we take into account the maze structure.
    More precisely, we select at each turn a random move among those that don't hit a wall.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Standard imports
import random

# Import PyRat
from pyrat import *

# Import previously developed functions
from tutorial import get_neighbors, locations_to_action

#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn (maze, maze_width, maze_height, name, teams, player_locations, player_scores, player_muds, cheese, possible_actions, memory) :

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        You can access the memory you stored during the preprocessing function by doing memory.my_key.
        You can also update the existing memory with new information, or create new entries as memory.my_key = my_value.
        
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
            * memory ............. threading.local ........................................ Local memory to share information between preprocessing, turn and postprocessing.
            
        Out:
            * action ... list (str) ... One of the possible actions, as given in possible_actions
    """

    # Choose a random neighbor
    neighbors = get_neighbors(player_locations[name], maze)
    neighbor = random.choice(neighbors)
    
    # Retrieve the corresponding action
    action = locations_to_action(player_locations[name], neighbor, maze_width)
    return action

#####################################################################################################################################################
######################################################################## GO ! #######################################################################
#####################################################################################################################################################

if __name__ == "__main__" :

    # Map the function to the character
    players = [{"name" : "rat", "turn_function" : turn}]

    # Customize the game elements
    config = {"maze_width" : 15,
              "maze_height" : 11,
              "mud_percentage" : 0.0,
              "nb_cheese" : 1,
              "trace_length": 1000}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################
