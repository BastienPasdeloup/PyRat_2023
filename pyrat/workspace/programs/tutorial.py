#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is a simple tutorial on the various elements in a PyRat game.
    It will not really do anything concretely, but proposes elementary functions to manipulate the maze that will be useful across all courses.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_vertices ( graph: Union[numpy.ndarray, Dict[int, Dict[int, int]]]
                 ) ->     List[int]:

    """
        Fuction to return the list of all vertices in a graph, except those with no neighbors.
        Here we propose an implementation for all types handled by the PyRat game.
        In:
            * graph: Graph on which to get the list of vertices.
        Out:
            * vertices: List of vertices in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        vertices = list(graph.keys())
    
    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        vertices = list(graph.sum(axis=0).nonzero()[0])
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return vertices

#####################################################################################################################################################

def get_neighbors ( vertex: int,
                    graph:  Union[numpy.ndarray, Dict[int, Dict[int, int]]]
                  ) ->      List[int]:

    """
        Fuction to return the list of neighbors of a given vertex.
        Here we propose an implementation for all types handled by the PyRat game.
        The function assumes that the vertex exists in the maze.
        It can be checked using for instance `assert vertex in get_vertices(graph)` but this takes time.
        In:
            * vertex: Vertex for which to compute the neighborhood.
            * graph:  Graph on which to get the neighborhood of the vertex.
        Out:
            * neighbors: List of vertices that are adjacent to the vertex in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        neighbors = list(graph[vertex].keys())

    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        neighbors = graph[vertex].nonzero()[0].tolist()
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return neighbors
    
#####################################################################################################################################################

def get_weight ( source: int,
                 target: int,
                 graph:  Union[numpy.ndarray, Dict[int, Dict[int, int]]]
               ) ->      List[int]:

    """
        Fuction to return the weight of the edge in the graph from the source to the target.
        Here we propose an implementation for all types handled by the PyRat game.
        The function assumes that both vertices exists in the maze and the target is a neighbor of the source.
        As above, it can be verified using `assert source in get_vertices(graph)` and `assert target in get_neighbors(source, graph)` but at some cost.
        In:
            * source: Source vertex in the graph.
            * target: Target vertex, assumed to be a neighbor of the source vertex in the graph.
            * graph:  Graph on which to get the weight from the source vertex to the target vertex.
        Out:
            * weight: Weight of the corresponding edge in the graph.
    """
    
    # If "maze_representation" option is set to "dictionary"
    if isinstance(graph, dict):
        weight = graph[source][target]
    
    # If "maze_representation" option is set to "matrix"
    elif isinstance(graph, numpy.ndarray):
        weight = graph[source, target]
    
    # Unhandled data type
    else:
        raise Exception("Unhandled graph type", type(graph))
    
    # Done
    return weight

#####################################################################################################################################################

def locations_to_action ( source:     int,
                          target:     int,
                          maze_width: int
                        ) ->          str: 

    """
        Function to transform two locations into an action to reach target from the source.
        In:
            * source:     Vertex on which the player is.
            * target:     Vertex where the character wants to go.
            * maze_width: Width of the maze in number of cells.
        Out:
            * action: Name of the action to go from the source to the target.
    """

    # Convert indices in row, col pairs
    source_row = source // maze_width
    source_col = source % maze_width
    target_row = target // maze_width
    target_col = target % maze_width
    
    # Check difference to get direction
    difference = (target_row - source_row, target_col - source_col)
    if difference == (0, 0):
        action = "nothing"
    elif difference == (0, -1):
        action = "west"
    elif difference == (0, 1):
        action = "east"
    elif difference == (1, 0):
        action = "south"
    elif difference == (-1, 0):
        action = "north"
    else:
        raise Exception("Impossible move from", source, "to", target)
    return action

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

    # Let's have a look at the various elements that the PyRat software sends us
    print("maze", type(maze), maze)
    print("maze_width", type(maze_width), maze_width)
    print("maze_height", type(maze_height), maze_height)
    print("name", type(name), name)
    print("teams", type(teams), teams)
    print("player_locations", type(player_locations), player_locations)
    print("cheese", type(cheese), cheese)
    print("possible_actions", type(possible_actions), possible_actions)
    
    # Let's check what are the useful vertices in the graph
    vertices = get_vertices(maze)
    print("vertices", vertices)
    
    # Let's find the neighbors of the initial location
    neighbors = get_neighbors(player_locations[name], maze)
    print("neighbors", neighbors)
    
    # Let's see how many actions it takes to reach all these neighbors
    for neighbor in neighbors:
        weight = get_weight(player_locations[name], neighbor, maze)
        print("Going from", player_locations[name], "to", neighbor, "takes", weight, "move(s)")
    
    # So, how do we reach these neighbors?
    for neighbor in neighbors:
        action = locations_to_action(player_locations[name], neighbor, maze_width)
        print("Going from", player_locations[name], "to", neighbor, "requires action", "'" + action + "'")
    
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

    # Let's return the action that does nothing for now
    action = possible_actions[0]
    return action

#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Map the functions to the character
    players = [{"name": "Tutorial", "preprocessing_function": preprocessing, "turn_function": turn}]

    # Customize the game elements
    config = {"maze_width": 15,
              "maze_height": 11,
              "mud_percentage": 30.0,
              "nb_cheese": 5}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################