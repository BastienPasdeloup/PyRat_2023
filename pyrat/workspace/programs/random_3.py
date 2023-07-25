#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is an improvement of "random_2".
    Here, we add elements that help us explore better the maze.
    More precisely, we keep a list (in a global variable to be updated at each turn) of cells that have already been visited in the game.
    Then, at each turn, we choose in priority a random move among those that lead us to an unvisited cell.
    If no such move exists, we move randomly using the method in "random_2".
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports 
import random

# Previously developed functions
from tutorial import get_neighbors, locations_to_action

#####################################################################################################################################################
##################################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ####################################################
#####################################################################################################################################################

def preprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                    maze_width:       int,
                    maze_height:      int,
                    name:             str,
                    teams:            Dict[str, List[str]],
                    player_locations: Dict[str, int],
                    cheese:           List[int],
                    possible_actions: List[str],
                    memory:           threading.local
                  ) ->                None:

    """
        This function is called once at the beginning of the game.
        It is typically given more time than the turn function, to perform complex computations.
        Store the results of these computations in the provided memory to reuse them later during turns.
        To do so, you can crete entries in the memory dictionary as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * None.
    """

    # To store the already visited cells
    memory.visited_cells = []

#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
           maze_width:       int,
           maze_height:      int,
           name:             str,
           teams:            Dict[str, List[str]],
           player_locations: Dict[str, int],
           player_scores:    Dict[str, float],
           player_muds:      Dict[str, Dict[str, Union[None, int]]],
           cheese:           List[int],
           possible_actions: List[str],
           memory:           threading.local
         ) ->                str:

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        You can access the memory you stored during the preprocessing function by doing memory.my_key.
        You can also update the existing memory with new information, or create new entries as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * player_muds:      Indicates which player is currently crossing mud.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * action: One of the possible actions, as given in possible_actions.
    """

    # Mark current cell as visited
    if player_locations[name] not in memory.visited_cells:
        memory.visited_cells.append(player_locations[name])

    # Go to an unvisited neighbor in priority
    neighbors = get_neighbors(player_locations[name], maze)
    unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in memory.visited_cells]
    if len(unvisited_neighbors) > 0:
        neighbor = random.choice(unvisited_neighbors)
        memory.visited_cells.append(neighbor)
        
    # If there is no unvisited neighbor, choose one randomly
    else:
        neighbor = random.choice(neighbors)
    
    # Retrieve the corresponding action
    action = locations_to_action(player_locations[name], neighbor, maze_width)
    return action

#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Map the function to the character
    players = [{"name": "Random 3", "skin": "default", "preprocessing_function": preprocessing, "turn_function": turn}]

    # Customize the game elements
    config = {"maze_width": 15,
              "maze_height": 11,
              "mud_percentage": 0.0,
              "nb_cheese": 1,
              "trace_length": 1000}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################