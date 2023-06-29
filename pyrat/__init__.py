#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This is the main file of the PyRat library, used as a support software for the course with the same name at IMT Atlantique.
    It can be used either as a Gym environment (use the classical Gym functions "step", etc.) or as a standalone (use the "start" function).
    The Gym aspect of the game hasn't been super-checked, and is rather a future functionality for later.
    For the standalone game usage, please check the examples on the PyRat website: https://formations.imt-atlantique.fr/pyrat/.
    The games arguments and their descriptions can be accessed using the "python3 pyrat.py -h" command.
    They can be set either using the terminal or as arguments to the constructor of the PyRat environment.
    For any question regarding the code or course contents, please contact Bastien Pasdeloup (bastien.pasdeloup@imt-atlantique.fr).
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Imports
import gym
import numpy
import numpy.random as nprandom
import re
import colored
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import multiprocessing
import time
import traceback
import argparse
import ast
import sys
import os
import datetime
import glob
import pygame
import pygame.locals as pglocals
import queue
import shutil
import distinctipy
import playsound
import dill

#####################################################################################################################################################
##################################################################### ARGUMENTS #####################################################################
#####################################################################################################################################################

# Initialize parser
parser = argparse.ArgumentParser()
list_type = lambda x : ast.literal_eval(x) if isinstance(ast.literal_eval(x), list) else exec("raise argparse.ArgumentTypeError(\"Should be a valid interval [2, ...]\")")

# Arguments
parser.add_argument("--random_seed",         type=int,                                                   default=None,     help="Global random seed for all elements")
parser.add_argument("--random_seed_maze",    type=int,                                                   default=None,     help="Random seed for the maze generation")
parser.add_argument("--random_seed_cheese",  type=int,                                                   default=None,     help="Random seed for the pieces of cheese distribution")
parser.add_argument("--random_seed_players", type=int,                                                   default=None,     help="Random seed for the initial location of players")
parser.add_argument("--maze_width",          type=int,                                                   default=15,       help="Width of the maze in number of cells")
parser.add_argument("--maze_height",         type=int,                                                   default=13,       help="Height of the maze in number of cells")
parser.add_argument("--cell_percentage",     type=float,                                                 default=80.0,     help="Percentage of cells that can be accessed in the maze, 0%% being a useless maze, and 100%% being a full rectangular maze")
parser.add_argument("--wall_percentage",     type=float,                                                 default=60.0,     help="Percentage of walls in the maze, 0%% being an empty maze, and 100%% being the maximum number of walls that keep the maze connected")
parser.add_argument("--mud_percentage",      type=float,                                                 default=20.0,     help="Percentage of pairs of adjacent cells that are separated by mud in the maze")
parser.add_argument("--mud_range",           type=list_type,                                             default=[4, 9],   help="Interval of moves needed to cross mud")
parser.add_argument("--maze_representation", type=str, choices=["dictionary", "matrix"],                 default="matrix", help="Representation of the maze in memory as given to players")
parser.add_argument("--fixed_maze",          type=str,                                                   default=None,     help="Fixed maze in any PyRat accepted representation (takes priority over any maze description and will automatically set maze_height and maze_width)")
parser.add_argument("--nb_cheese",           type=int,                                                   default=21,       help="Number of pieces of cheese in the maze")
parser.add_argument("--fixed_cheese",        type=str,                                                   default=None,     help="Fixed list of cheese (takes priority over random number of cheese)")
parser.add_argument("--save_path",           type=str,                                                   default=".",      help="Path where games are saved")
parser.add_argument("--save_game",           action="store_true",                                        default=False,    help="Indicates if the game should be saved")
parser.add_argument("--preprocessing_time",  type=float,                                                 default=3.0,      help="Time given to the players before the game starts")
parser.add_argument("--turn_time",           type=float,                                                 default=0.1,      help="Time after which players will move in the maze, or miss a turn")
parser.add_argument("--synchronous",         action="store_true",                                        default=False,    help="If set, waits for all players to return a move before moving, even if turn_time is exceeded",)
parser.add_argument("--continue_on_error",   action="store_true",                                        default=False,    help="If a player crashes, continues the game anyway")
parser.add_argument("--render_mode",         type=str, choices=["ascii", "ansi", "gui", "no_rendering"], default="gui",    help="Method to display the game, or None to play without rendering")
parser.add_argument("--render_details",      action="store_true",                                        default=False,    help="If the maze is rendered, adds some elements that are not essential")
parser.add_argument("--trace_length",        type=int,                                                   default=0,        help="Maximum length of the trace to display when players are moving (GUI rendering only)")
                    
# Parse the arguments into a global variable
args = parser.parse_args()

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def setup_workspace () :

    """
        Creates all the directories for a clean student workspace.
        Also creates a few default programs to start with.

        In:
            * None.

        Out:
            * None.
    """

    # Copy the template workspace into the current directory if not already exixting
    shutil.copytree(os.path.join(os.path.dirname(os.path.realpath(__file__)), "workspace"), "pyrat_workspace")

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class DillProcess (multiprocessing.Process) :

    """
        This class is a small hack to allow compatibility with Windows.
        It allows to define process functions embedded within other functions easily.
        Check here for more info: https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing.
    """

    def __init__ (self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        try :
            multiprocessing.set_start_method("fork")
            print("aaaaaaa")
        except ValueError :
            self._target = dill.dumps(self._target)
        except :
            pass

    def run (self) :
        if self._target :
            try :
                multiprocessing.set_start_method("fork")
                print("aaaaaaa")
            except ValueError :
                self._target = dill.loads(self._target)
            except :
                pass
            self._target(*self._args, **self._kwargs)

#####################################################################################################################################################

class PyRat (gym.Env) :

    """
        This is the main class of the PyRat environment.
        It is inheriting from Gym's Env class, and is therefore compatible with Gym's API.
        The easiest way however is to use the "start" method, which will take care of everything.
    """

    #############################################################################################################################################
    #                                                                GYM METHODS                                                                #
    #############################################################################################################################################

    def __init__ (self, players, random_seed=args.random_seed, random_seed_maze=args.random_seed_maze, random_seed_cheese=args.random_seed_cheese, random_seed_players=args.random_seed_players, maze_width=args.maze_width, maze_height=args.maze_height, cell_percentage=args.cell_percentage, wall_percentage=args.wall_percentage, mud_percentage=args.mud_percentage, mud_range=args.mud_range, maze_representation=args.maze_representation, fixed_maze=args.fixed_maze, nb_cheese=args.nb_cheese, fixed_cheese=args.fixed_cheese, render_mode=args.render_mode, render_details=args.render_details, trace_length=args.trace_length, save_path=args.save_path, save_game=args.save_game, preprocessing_time=args.preprocessing_time, turn_time=args.turn_time, synchronous=args.synchronous, continue_on_error=args.continue_on_error) :

        """
            Main class of the Gym environment.
            
            In:
                * players ............... list (dict : str -> Any) ........................................ List of players to register to the game, given as dictionaries with keys as defined in _register_player.
                * random_seed ........... int ............................................................. Global random seed for all elements.
                * random_seed_maze ...... int ............................................................. Random seed for the maze generation.
                * random_seed_cheese .... int ............................................................. Random seed for the pieces of cheese distribution.
                * random_seed_players ... int ............................................................. Random seed for the initial location of players.
                * maze_width ............ int ............................................................. Width of the maze in number of cells.
                * maze_height ........... int ............................................................. Height of the maze in number of cells.
                * cell_percentage ....... float ........................................................... Percentage of cells that can be accessed in the maze, 0%% being a useless maze, and 100%% being a full rectangular maze.
                * wall_percentage ....... float ........................................................... Percentage of walls in the maze, 0%% being an empty maze, and 100%% being the maximum number of walls that keep the maze connected.
                * mud_percentage ........ float ........................................................... Percentage of pairs of adjacent cells that are separated by mud in the maze.
                * mud_range ............. list (int) ...................................................... Interval of moves needed to cross mud.
                * maze_representation ... str ............................................................. Representation of the maze in memory as given to players.
                * fixed_maze ............ str [or] numpy.ndarray [or] dict : int -> (dict : int -> int) ... Fixed maze in any PyRat accepted representation (takes priority over any maze description and will automatically set maze_height and maze_width).
                * nb_cheese ............. int ............................................................. Number of pieces of cheese in the maze.
                * fixed_cheese .......... str [or] list (int) ............................................. Fixed list of cheese (takes priority over random number of cheese).
                * render_mode ........... str ............................................................. Method to display the game, or None to play without rendering.
                * render_details ........ bool ............................................................ If the maze is rendered, adds some elements that are not essential.
                * trace_length .......... int ............................................................. Maximum length of the trace to display when players are moving (GUI rendering only).
                * save_path ............. str ............................................................. Path where games are saved.
                * save_game ............. bool ............................................................ Indicates if the game should be saved.
                * preprocessing_time .... float ........................................................... Time given to the players before the game starts.
                * turn_time ............. float ........................................................... Time after which players will move in the maze, or miss a turn.
                * synchronous ........... bool ............................................................ If set, waits for all players to return a move before moving, even if turn_time is exceeded.
                * continue_on_error ..... bool ............................................................ If a player crashes, continues the game anyway.
                
            Out:
                * None.
        """

        # Inherit from parent class
        super(PyRat, self).__init__()
        
        # Store arguments
        self.players = players
        self.random_seed = random_seed
        self.random_seed_maze = random_seed_maze
        self.random_seed_cheese = random_seed_cheese
        self.random_seed_players = random_seed_players
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.cell_percentage = cell_percentage
        self.wall_percentage = wall_percentage
        self.mud_percentage = mud_percentage
        self.mud_range = mud_range
        self.maze_representation = maze_representation
        self.fixed_maze = fixed_maze
        self.nb_cheese = nb_cheese
        self.fixed_cheese = fixed_cheese
        self.render_mode = render_mode
        self.render_details = render_details
        self.trace_length = trace_length
        self.save_path = save_path
        self.save_game = save_game
        self.preprocessing_time = preprocessing_time
        self.turn_time = turn_time
        self.synchronous = synchronous
        self.continue_on_error = continue_on_error

        # Check arguments are correct
        assert 0 < maze_width
        assert 0 < maze_height
        assert 0.0 <= cell_percentage <= 100.0
        assert 0.0 <= wall_percentage <= 100.0
        assert 0.0 <= mud_percentage <= 100.0
        assert 1 < mud_range[0] <= mud_range[1]
        assert 0 < nb_cheese
        assert 0.0 <= preprocessing_time
        assert 0.0 <= turn_time
        assert 0 < len(self.players)
        
        # Game elements
        self.game_random_seed_maze = None
        self.game_random_seed_cheese = None
        self.game_random_seed_players = None
        self.maze = None
        self.maze_public = None
        self.initial_cheese = None
        self.cheese = None
        self.teams = None
        self.player_functions = None
        self.player_locations = None
        self.player_initial_locations = None
        self.player_scores = None
        self.player_muds = None
        self.player_traces = None
        self.stats = None
        self.moves_history = None
        self.turn = None
        self.done = None
        self.gui_process = None
        self.gui_process_queue = None
        self.reset()
        
    #############################################################################################################################################

    def reset (self, seed=None, options=None) :
    
        """
            Resets the game.
            
            In:
                * seed ...... int ................. Global random seed for all elements.
                * options ... dict : str -> int ... Options for the reset, here used for individual seeds for elements.
                
            Out:
                * state ... Any ... Initial game state, as returned by the get_state function.
        """
    
        # Get arguments if none are provided here
        seed = self.random_seed if seed is None else seed
        seed_maze = self.random_seed_maze if options is None or "seed_maze" not in options else options["seed_maze"]
        seed_cheese = self.random_seed_cheese if options is None or "seed_cheese" not in options else options["seed_cheese"]
        seed_players = self.random_seed_players if options is None or "seed_players" not in options else options["seed_players"]
        
        # Check random seeds
        if seed is not None :
            self.game_random_seed_maze = seed
            self.game_random_seed_cheese = seed
            self.game_random_seed_players = seed
            print("Starting game with --random_seed=%d" % (seed), file=sys.stderr)
        else :
            max_rand_value = 2**32
            self.game_random_seed_maze = nprandom.randint(max_rand_value, dtype=numpy.int64) if seed_maze is None else seed_maze
            self.game_random_seed_cheese = nprandom.randint(max_rand_value, dtype=numpy.int64) if seed_cheese is None else seed_cheese
            self.game_random_seed_players = nprandom.randint(max_rand_value, dtype=numpy.int64) if seed_players is None else seed_players
            print("Starting game with --random_seed_maze=%d --random_seed_cheese=%d --random_seed_players=%d" % (self.game_random_seed_maze, self.game_random_seed_cheese, self.game_random_seed_players), file=sys.stderr)
    
        # Set game elements
        self.teams = {}
        self.player_functions = {}
        self.player_locations = {}
        self.player_initial_locations = {}
        self.player_scores = {}
        self.player_muds = {}
        self.player_traces = {}
        self.moves_history = {}
        self.stats = {"players" : {},
                        "turns" : -1}
        self.turn = 0
        self.done = False
        self.gui_process = None
        self.gui_process_queue = multiprocessing.Manager().Queue()

        # Initialize the maze
        self.maze, self.maze_public, self.maze_width, self.maze_height = self._create_maze()
        
        # Register players
        for player in self.players :
            self._register_player(**player)
    
        # Add the cheese
        self.cheese = self._distribute_cheese()
        self.initial_cheese = self.cheese.copy()

        # Return state of the game
        state = self.get_state("gym")
        return state

    #############################################################################################################################################

    def get_action_meanings (self) :
    
        """
            Returns a dictionary indicating the meanings of the possible actions.
            
            In:
                * None.
                
            Out:
                * descriptions ... dict : int -> str ... Description of the possible actions in a human-friendly format.
        """
        
        # Action space is one of the cardinal directions
        descriptions = {-1 : "nothing", 0 : "north", 1 : "east", 2 : "south", 3 : "west"}
        return descriptions

    #############################################################################################################################################

    @property
    def action_space (self) :
    
        """
            Returns the action space for the Gym environment.
            It has a @property decorator to act as a property and not a function, to be compliant with the API.
            
            In:
                * None.
                
            Out:
                * space ... gym.spaces.Dict ... Space of possible actions.
        """
        
        # Action space is one of the cardinal directions, plus a -1 value if no action is performed
        space = gym.spaces.Dict({player : gym.spaces.Discrete(5, start=-1) for player in self.player_locations})
        return space

    #############################################################################################################################################

    @property
    def observation_space (self) :
    
        """
            Returns the observation space for the Gym environment.
            It has a @property decorator to act as a property and not a function, to be compliant with the API.
            
            In:
                * None.
                
            Out:
                * space ... gym.spaces.Dict ... Space of observations.
        """
        
        # Observation space of the game is cheese availability, and info on the players
        max_weight = max([self.maze[vertex][neighbor] for vertex in self.maze for neighbor in self.maze[vertex]])
        space = gym.spaces.Dict({"cheese" : gym.spaces.MultiBinary(self.nb_cheese),
                                    "players" : gym.spaces.Dict({player : gym.spaces.Dict({"location" : gym.spaces.Discrete(self.maze_width * self.maze_height),
                                                                                        "mud_target" : gym.spaces.Discrete(self.maze_width * self.maze_height + 1, start=-1),
                                                                                        "mud_count" : gym.spaces.Discrete(max_weight),
                                                                                        "score" : gym.spaces.Box(0, self.nb_cheese, shape=())})
                                                                for player in self.player_locations})})
        return space
        
    #############################################################################################################################################

    def get_state (self,
                    mode = "human") :
        
        """
            Returns the current state of the environment, for elements that can vary.
            
            In:
                * mode ... str ... Indicates if we return the state in a human-friendly format or in gym format.
            
            Out (if mode == "human"):
                * player_locations ... dict : str -> int ................... Locations for all players in the game.
                * player_scores ...... dict : str -> float ................. Scores for all players in the game.
                * player_muds ........ dict : str -> (dict : str -> int) ... Indicates which player is currently crossing mud.
                * cheese ............. list (int) .......................... List of available pieces of cheese in the maze.
            
            Out (if mode == "gym"):
                * state ... dict : str -> Any ... State of the game.
        """
        
        # Return elements describing game state
        if mode == "human" :
            return self.player_locations.copy(), self.player_scores.copy(), self.player_muds.copy(), self.cheese.copy()
        
        # compile game state into a dictionary
        elif mode == "gym" :
            state = {"cheese" : numpy.array([c in self.cheese for c in self.initial_cheese]),
                        "players" : {player : {"location" : self.player_locations[player],
                                            "mud_target" : self.player_muds[player]["target"] if self.player_muds[player]["target"] is not None else -1,
                                            "mud_count" : self.player_muds[player]["count"],
                                            "score" : self.player_scores[player]}
                                    for player in self.player_locations}}
            return state.copy()
        
        # Invalid
        else :
            raise Exception("Invalid get_state mode %s chosen" % (mode))

    #############################################################################################################################################

    def step (self, actions, reward_for=None) :
        
        """
            Renders the current state of the environment.
            
            In:
                * actions ...... dict : str -> int ... Action performed per player.
                * reward_for ... str ................. In Gym, we can only return a reward for one player, choose the one you want here (if set to None, chooses the first one).
                
            Out:
                * state .... Any ..... Forward of the get_state function.
                * reward ... float ... Rewards for the chosen player after performing the action in that state.
                * done ..... bool .... Indicates if the game is over.
                * info ..... dict .... Additional information to return, as required by the Gym API.
        """

        # The turn is done
        self.turn += 1
        
        # Move all players accordingly
        action_meanings = self.get_action_meanings()
        for player in actions :
            try :
                row, col = self._i_to_rc(self.player_locations[player])
                target = None
                if action_meanings[actions[player]] == "north" and row > 0 :
                    target = self._rc_to_i(row - 1, col)
                elif action_meanings[actions[player]] == "south" and row < self.maze_height - 1 :
                    target = self._rc_to_i(row + 1, col)
                elif action_meanings[actions[player]] == "west" and col > 0 :
                    target = self._rc_to_i(row, col - 1)
                elif action_meanings[actions[player]] == "east" and col < self.maze_width - 1 :
                    target = self._rc_to_i(row, col + 1)
                if target is not None and target in self.maze[self.player_locations[player]] :
                    weight = self.maze[self.player_locations[player]][target]
                    if weight == 1 :
                        self.player_locations[player] = target
                    elif weight > 1 :
                        self.player_muds[player]["target"] = target
                        self.player_muds[player]["count"] = weight
            except :
                print("Warning: Invalid action %s for player %s" % (str(actions[player]), player), file=sys.stderr)

        # All players in mud advance a bit
        for player in self.player_muds :
            if self._is_in_mud(player) :
                self.player_muds[player]["count"] -= 1
                if self.player_muds[player]["count"] == 0 :
                    self.player_locations[player] = self.player_muds[player]["target"]
                    self.player_muds[player]["target"] = None

        # Update cheese and scores
        scores_before = self.player_scores.copy()
        remaining_cheese = []
        for c in self.cheese :
            players_on_cheese = [player for player in self.player_locations if c == self.player_locations[player]]
            for player in players_on_cheese :
                self.player_scores[player] += 1.0 / len(players_on_cheese)
            if len(players_on_cheese) == 0 :
                remaining_cheese.append(c)
        self.cheese = remaining_cheese
        
        # Store trace for GUI
        for player in self.player_locations :
            self.player_traces[player].append(self.player_locations[player])
            self.player_traces[player] = self.player_traces[player][-self.trace_length:]
        
        # Compute reward
        # Here, reward is the gain in score that turn
        # Could be improved
        if reward_for is None :
            reward_for = list(self.player_locations.keys())[0]
        reward = self.player_scores[reward_for] - scores_before[reward_for]
        
        # Check the state of the game
        team_scores = self._score_per_team()
        max_team = max(team_scores, key=team_scores.get)
        max_team_score = team_scores[max_team]
        del team_scores[max_team]
        second_max_team_score = max(team_scores.values()) if len(team_scores) > 0 else float("inf")
        self.done = second_max_team_score + len(self.cheese) < max_team_score or len(self.cheese) == 0
        
        # Add info
        info = {}
        
        # Return the state of the game
        state = self.get_state("gym")
        return state, reward, self.done, info

    #############################################################################################################################################

    def close (self) :
        
        """
            Actions to do at the end of the game if needed.
            
            In:
                * None.
                
            Out:
                * None.
        """
        
        # We save the game if asked
        if self.save_game :
            
            # Generic config stuff we want to save
            config = {"synchronous" : True,
                        "continue_on_error" : False}
            
            # We save the maze as it was given (fixed or random)
            if self.fixed_maze is not None :
                config["fixed_maze"] = self.fixed_maze
            else :
                config["maze_width"] = self.maze_width
                config["maze_height"] = self.maze_height
                config["random_seed_maze"] = self.random_seed_maze
                config["cell_percentage"] = self.cell_percentage
                config["wall_percentage"] = self.wall_percentage
                config["mud_percentage"] = self.mud_percentage
                config["mud_range"] = self.mud_range
                config["random_seed_maze"] = self.game_random_seed_maze
            
            # Same for the cheese
            if self.fixed_cheese is not None :
                config["fixed_cheese"] = self.fixed_cheese
            else :
                config["nb_cheese"] = self.nb_cheese
                config["random_seed_cheese"] = self.game_random_seed_cheese
            
            # Create the players' file, forcing players to their initial locations
            output_file_name = os.path.join(self.save_path, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.py"))
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "save_template.py"), "r") as save_template_file :
                save_template = save_template_file.read()
                save_template = save_template.replace("{ACTIONS}", str(self.moves_history).replace("], '", "],\n                      '"))
                save_template = save_template.replace("{PLAYERS}", str([{"\"name\" : \"%s\", \"team\" : \"%s\", \"turn_function\" : turn, \"location\" : %d" % (player, [team for team in self.teams if player in self.teams[team]][0], self.player_initial_locations[player])} for player in self.player_locations]).replace("'", "").replace("},", "},\n              "))
                save_template = save_template.replace("{CONFIG}", str(config).replace(", '", ",\n              '"))
                with open(output_file_name, "w") as output_file :
                    print(save_template, file=output_file)

        # Wait for GUI to be exited to quit if there is one
        if self.gui_process is not None and self.done :
            self.gui_process.join()
        for process in multiprocessing.active_children() :
            process.terminate()
            process.join()
        
    #############################################################################################################################################
    #                                                             RENDERING METHODS                                                             #
    #############################################################################################################################################

    def render (self) :
        
        """
            Renders the current state of the environment.
            
            In:
                * None.
                
            Out:
                * None.
        """

        # Render using the corresponding mode
        if self.render_mode == "no_rendering" :
            pass
        elif self.render_mode == "ascii" :
            self._render_ascii(False)
        elif self.render_mode == "ansi" :
            self._render_ascii(True)
        elif self.render_mode == "gui" :
            self._render_gui()
        else :
            raise Exception("Invalid rendering mode %s chosen" % (self.render_mode))
        
    #############################################################################################################################################

    def _render_ascii (self, use_colors) :
        
        """
            Renders the current state of the environment in ascii/ansi.
            
            In:
                * use_colors ... bool ... Indicates if we use colors for rendering.
                
            Out:
                * None.
        """

        # Dimensions
        max_weight = max([self.maze[vertex][neighbor] for vertex in self.maze for neighbor in self.maze[vertex]])
        max_weight_len = len(str(max_weight))
        max_player_name_len = max([len(player) for player in self.player_locations]) + (max_weight_len + 5 if max_weight > 1 else 0)
        max_cell_number_len = len(str(self.maze_width * self.maze_height - 1))
        cell_width = max(max_player_name_len, max_weight_len, max_cell_number_len + 1) + 2
        
        # Function to colorize text
        def colorize (text, colorization, alternate_text=None) :
            if not use_colors :
                if alternate_text is None :
                    return str(text)
                else :
                    return str(alternate_text)
            return colorization + str(text) + colored.attr(0)

        # Function to return the true len of a color-formated string
        def colored_len (text) :
            return len(re.sub(r"[\u001B\u009B][\[\]()#;?]*((([a-zA-Z\d]*(;[-a-zA-Z\d\/#&.:=?%@~_]*)*)?\u0007)|((\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-ntqry=><~]))", "", text))

        # Game elements
        wall = colorize(" ", colored.bg("light_gray"), "#")
        ground = colorize(" ", colored.bg("grey_23"))
        cheese = colorize("▲", colored.bg("grey_23") + colored.fg("yellow_1"))
        mud_horizontal = colorize("ⴾ", colored.bg("grey_23") + colored.fg("orange_4b"))
        mud_vertical = colorize("ⵘ", colored.bg("grey_23") + colored.fg("orange_4b"))
        mud_value = lambda number : colorize(number, colored.bg("grey_23") + colored.fg("orange_4b"))
        path_horizontal = colorize("⋅", colored.bg("grey_23") + colored.fg("orange_4b"))
        path_vertical = colorize("ⵗ", colored.bg("grey_23") + colored.fg("orange_4b"))
        cell_number = lambda number : colorize(number, colored.bg("grey_23") + colored.fg("magenta"))
        score_cheese = colorize("▲ ", colored.fg("yellow_1"))
        score_half_cheese = colorize("△ ", colored.fg("yellow_1"))
        
        # Player/team elements
        teams = {team : colorize(team, colored.fg(9 + list(self.teams.keys()).index(team))) for team in self.teams}
        mud_indicator = lambda player : " (" + ("⬇" if self._coords_difference(self.player_muds[player]["target"], self.player_locations[player]) == (1, 0) \
                                                else "⬆" if self._coords_difference(self.player_muds[player]["target"], self.player_locations[player]) == (-1, 0) \
                                                else "➡" if self._coords_difference(self.player_muds[player]["target"], self.player_locations[player]) == (0, 1) \
                                                else "⬅") + " " + str(self.player_muds[player]["count"]) + ")" if self.player_muds[player]["count"] > 0 else ""
        players = {player : colorize(player + mud_indicator(player), colored.bg("grey_23") + colored.fg(9 + ["team" if player in team else 0 for team in self.teams.values()].index("team"))) for player in self.player_locations}
        
        # Game info
        environment_str = "Game over" if self.done else "Starting turn %d" % (self.turn) if self.turn > 0 else "Initial configuration"
        team_scores = self._score_per_team()
        scores_str = ""
        for team in self.teams :
            scores_str += "\n" + score_cheese * int(team_scores[team]) + score_half_cheese * int(numpy.ceil(team_scores[team] - int(team_scores[team])))
            scores_str += "[" + teams[team] + "] "
            scores_str += " + ".join(["%s (%s)" % (player, str(round(self.player_scores[player], 3)).rstrip('0').rstrip('.') if self.player_scores[player] > 0 else "0") for player in self.teams[team]])
        environment_str += scores_str

        # Consider cells in lexicographic order
        environment_str += "\n" + wall * (self.maze_width * (cell_width + 1) + 1)
        for row in range(self.maze_height) :
            players_in_row = [self.player_locations[player] for player in self.player_locations if self._i_to_rc(self.player_locations[player])[0] == row]
            cell_height = max([players_in_row.count(cell) for cell in players_in_row] + [max_weight_len]) + 2
            environment_str += "\n"
            for subrow in range(cell_height) :
                environment_str += wall
                for col in range(self.maze_width) :
                    
                    # Check cell contents
                    players_in_cell = [player for player in self.player_locations if self.player_locations[player] == self._rc_to_i(row, col)]
                    cheese_in_cell = self._rc_to_i(row, col) in self.cheese

                    # Find subrow contents (nothing, cell number, cheese, trace, player)
                    unconnected_cell = self._rc_to_i(row, col) not in self.maze
                    background = wall if unconnected_cell else ground
                    cell_contents = ""
                    if subrow == 0 :
                        if background != wall and self.render_details :
                            cell_contents += background
                            cell_contents += cell_number(self._rc_to_i(row, col))
                    elif cheese_in_cell :
                        if subrow == (cell_height - 1) // 2 :
                            cell_contents = background * ((cell_width - colored_len(cheese)) // 2)
                            cell_contents += cheese
                        else :
                            cell_contents = background * cell_width
                    else :
                        first_player_index = (cell_height - len(players_in_cell)) // 2
                        if first_player_index <= subrow < first_player_index + len(players_in_cell) :
                            cell_contents = background * ((cell_width - colored_len(players[players_in_cell[subrow - first_player_index]])) // 2)
                            cell_contents += players[players_in_cell[subrow - first_player_index]]
                        else :
                            cell_contents = background * cell_width
                    environment_str += cell_contents
                    environment_str += background * (cell_width - colored_len(cell_contents))
                    
                    # Right separation
                    right_weight = "0" if unconnected_cell or self._rc_to_i(row, col + 1) not in self.maze[self._rc_to_i(row, col)] else str(self.maze[self._rc_to_i(row, col)][self._rc_to_i(row, col + 1)])
                    if col == self.maze_width - 1 or right_weight == "0" :
                        environment_str += wall
                    else :
                        if right_weight == "1" :
                            environment_str += path_vertical
                        elif self.render_details and int(numpy.ceil((cell_height - len(right_weight)) / 2)) <= subrow < int(numpy.ceil((cell_height - len(right_weight)) / 2)) + len(right_weight) :
                            digit_number = subrow - int(numpy.ceil((cell_height - len(right_weight)) / 2))
                            environment_str += mud_value(right_weight[digit_number])
                        else :
                            environment_str += mud_vertical
                environment_str += "\n"
            environment_str += wall
            
            # Bottom separation
            for col in range(self.maze_width) :
                unconnected_cell = self._rc_to_i(row, col) not in self.maze
                bottom_weight = "0" if unconnected_cell or self._rc_to_i(row + 1, col) not in self.maze[self._rc_to_i(row, col)] else str(self.maze[self._rc_to_i(row, col)][self._rc_to_i(row + 1, col)])
                if bottom_weight == "0" :
                    environment_str += wall * (cell_width + 1)
                elif bottom_weight == "1" :
                    environment_str += path_horizontal * cell_width + wall
                else :
                    cell_contents = mud_horizontal * ((cell_width - colored_len(bottom_weight)) // 2) + mud_value(bottom_weight) if self.render_details else ""
                    environment_str += cell_contents
                    environment_str += mud_horizontal * (cell_width - colored_len(cell_contents)) + wall
        
        # Render
        if use_colors :
            nb_rows = 1 + len(environment_str.splitlines())
            nb_cols = 1 + (cell_width + 1) * self.maze_width
            print("\x1b[8;%d;%dt" % (nb_rows, nb_cols), file=sys.stderr)
        print(environment_str, file=sys.stderr)
        
    #############################################################################################################################################

    def _render_gui (self) :
        
        """
            Renders the current state of the environment in a GUI.
            
            In:
                * None.
                
            Out:
                * None.
        """

        # Define a function to run the GUI in a separate process
        def gui_process_function () :
            try :
                
                # Initialize the library and window
                # TODO
                pygame.init()
                #screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                screen = pygame.display.set_mode((1600, 800))
                
                # We will store elements to display
                maze_elements = []
                avatar_elements = []
                player_elements = {}
                cheese_elements = {}
                
                # Dimensions
                window_width, window_height = pygame.display.get_surface().get_size()
                cell_size = int(min(window_width / self.maze_width, window_height / self.maze_height) * 0.9)
                background_color = (0, 0, 0)
                cell_text_color = (50, 50, 50)
                cell_text_offset = int(cell_size * 0.1)
                wall_size = cell_size // 7
                mud_text_color = (185, 155, 60)
                corner_wall_ratio = 1.2
                flag_size = int(cell_size * 0.4)
                flag_x_offset = int(cell_size * 0.2)
                flag_x_next_offset = int(cell_size * 0.07)
                flag_y_offset = int(cell_size * 0.3)
                game_area_width = cell_size * self.maze_width
                game_area_height = cell_size * self.maze_height
                maze_x_offset = int((window_width - game_area_width) * 0.9)
                maze_y_offset = (window_height - game_area_height) // 2
                avatars_x_offset = window_width - maze_x_offset - game_area_width
                avatars_area_width = maze_x_offset - 2 * avatars_x_offset
                avatars_area_height = min(game_area_height // 2, (game_area_height - (len(self.teams) - 1) * maze_y_offset) // len(self.teams))
                avatars_area_border = 2
                avatars_area_angle = 10
                avatars_area_padding = avatars_area_height // 13
                team_text_size = avatars_area_padding * 3
                player_avatar_size = avatars_area_padding * 3
                player_avatar_horizontal_padding = avatars_area_padding * 4
                player_name_text_size = avatars_area_padding
                cheese_score_size = avatars_area_padding
                text_size = int(cell_size * 0.17)
                cheese_size = int(cell_size * 0.4)
                player_size = int(cell_size * 0.5)
                flag_border_color = (255, 255, 255)
                flag_border_width = 1
                player_border_width = 2
                cheese_border_color = (255, 255, 0)
                cheese_border_width = 1
                avatars_area_color = (255, 255, 255)
                cheese_score_border_color = (100, 100, 100)
                cheese_score_border_width = 1
                trace_size = wall_size // 2
                animation_steps = cell_size
                animation_time = 0.01
                medal_size = min(avatars_x_offset, maze_y_offset) * 2
                icon_size = 50
                main_image_factor = 0.8
                main_image_border_color = (0, 0, 0)
                main_image_border_size = 1
                
                # Function to load an image with some scaling
                # If only 2 arguments are provided, scales keeping ratio specifying the maximum size
                # If first argument is a directory, returns a random image from it
                def surface_from_image (file_or_dir_name, target_width_or_max_size, target_height=None) :
                    full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_or_dir_name)
                    if os.path.isdir(full_path) :
                        full_path = nprandom.choice(glob.glob(os.path.join(full_path, "*")))
                    surface = pygame.image.load(full_path).convert_alpha()
                    if target_height is None :
                        max_surface_size = max(surface.get_width(), surface.get_height())
                        surface = pygame.transform.scale(surface, (surface.get_width() * target_width_or_max_size // max_surface_size, surface.get_height() * target_width_or_max_size // max_surface_size))
                    else :
                        surface = pygame.transform.scale(surface, (target_width_or_max_size, target_height))
                    return surface
                
                # Same function for text
                def surface_from_text (text, target_height, text_color, original_font_size=50) :
                    surface = pygame.font.SysFont(None, original_font_size).render(text, True, text_color)
                    surface = pygame.transform.scale(surface, (surface.get_width() * target_height // surface.get_height(), target_height))
                    return surface

                # Function to colorize an object
                def colorize (surface, color) :
                    final_surface = surface.copy()
                    color_surface = pygame.Surface(final_surface.get_size()).convert_alpha()
                    color_surface.fill(color)
                    final_surface.blit(color_surface, (0, 0), special_flags=pygame.BLEND_MULT)
                    return final_surface
                    
                # Function to add a colored border around an object
                def add_color_border (surface, border_color, border_size, final_rescale=True) :
                    final_surface = pygame.Surface((surface.get_width() + 2 * border_size, surface.get_height() + 2 * border_size)).convert_alpha()
                    final_surface.fill((0, 0, 0, 0))
                    mask_surface = surface.copy()
                    color_surface = pygame.Surface(mask_surface.get_size())
                    color_surface.fill((0, 0, 0, 0))
                    mask_surface.blit(color_surface, (0, 0), special_flags=pygame.BLEND_MIN)
                    color_surface.fill(border_color)
                    mask_surface.blit(color_surface, (0, 0), special_flags=pygame.BLEND_MAX)
                    for offset_x in range(-border_size, border_size + 1) :
                        for offset_y in range(-border_size, border_size + 1) :
                            if numpy.linalg.norm([offset_x, offset_y]) <= border_size :
                                final_surface.blit(mask_surface, (border_size // 2 + offset_x, border_size // 2 + offset_y))
                    final_surface.blit(surface, (border_size // 2, border_size // 2))
                    if final_rescale :
                        final_surface = pygame.transform.scale(final_surface, surface.get_size())
                    return final_surface

                # Function to load the surfaces of a player
                def load_player_surfaces (player_name, scale, border_color=None, border_width=None, add_border=self.render_details) :
                    try :
                        player_neutral = surface_from_image(os.path.join("gui", "players", player_name, "neutral.png"), scale)
                        player_north = surface_from_image(os.path.join("gui", "players", player_name, "north.png"), scale)
                        player_south = surface_from_image(os.path.join("gui", "players", player_name, "south.png"), scale)
                        player_west = surface_from_image(os.path.join("gui", "players", player_name, "west.png"), scale)
                        player_east = surface_from_image(os.path.join("gui", "players", player_name, "east.png"), scale)
                        if add_border :
                            player_neutral = add_color_border(player_neutral, border_color, border_width)
                            player_north = add_color_border(player_north, border_color, border_width)
                            player_south = add_color_border(player_south, border_color, border_width)
                            player_west = add_color_border(player_west, border_color, border_width)
                            player_east = add_color_border(player_east, border_color, border_width)
                        return player_neutral, player_north, player_south, player_west, player_east
                    except :
                        return load_player_surfaces("default", scale, border_color, border_width, add_border)
                
                # Function to play a sound
                def play_sound (player_name, sound_name) :
                    try :
                        sound_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gui", "players", player_name, sound_name)
                        playsound.playsound(sound_file, block=False)
                    except :
                        play_sound("default", sound_name)
                
                # Function to load the avatar of a player
                def load_player_avatar (player_name, scale) :
                    try :
                        return surface_from_image(os.path.join("gui", "players", player_name, "avatar.png"), scale)
                    except :
                        return load_player_avatar("default", scale)
                
                # Function to tell if a cell is in the maze
                def cell_is_in_maze (row, col) :
                    return 0 <= row < self.maze_height and 0 <= col < self.maze_width and self._rc_to_i(row, col) in self.maze

                # Function to get the main color of a surface
                def get_main_color (surface) :
                    colors = pygame.surfarray.array2d(surface)
                    values, counts = numpy.unique(colors, return_counts=True)
                    argmaxes = numpy.argpartition(-counts, kth=2)[:2]
                    max_occurrences = values[argmaxes]
                    main_color = surface.unmap_rgb(max_occurrences[0])
                    if main_color == (0, 0, 0, 0) :
                        main_color = surface.unmap_rgb(max_occurrences[1])
                    return main_color

                # Create colors for the teams
                colors = distinctipy.distinctipy.get_colors(len(self.teams))
                team_colors = {list(self.teams.keys())[i] : tuple([int(c * 255) for c in colors[i]]) for i in range(len(self.teams))}

                # Set window icon
                icon = surface_from_image(os.path.join("gui", "drawings", "pyrat.png"), icon_size)
                pygame.display.set_icon(icon)
                
                # Set background color
                pygame.draw.rect(screen, background_color, pygame.Rect(0, 0, window_width, window_height))
                
                # Add cells
                for row in range(self.maze_height) :
                    for col in range(self.maze_width) :
                        if cell_is_in_maze(row, col) :
                            cell = surface_from_image(os.path.join("gui", "ground"), cell_size, cell_size)
                            cell = pygame.transform.rotate(cell, nprandom.randint(4) * 90)
                            cell = pygame.transform.flip(cell, bool(nprandom.randint(2)), bool(nprandom.randint(2)))
                            cell_x = maze_x_offset + col * cell_size
                            cell_y = maze_y_offset + row * cell_size
                            maze_elements.append((cell_x, cell_y, cell))
                            
                # Add mud
                mud = surface_from_image(os.path.join("gui", "mud", "mud.png"), cell_size)
                for row in range(self.maze_height) :
                    for col in range(self.maze_width) :
                        if cell_is_in_maze(row, col) :
                            if cell_is_in_maze(row, col - 1) :
                                if self._rc_to_i(row, col - 1) in self.maze[self._rc_to_i(row, col)] :
                                    if self.maze[self._rc_to_i(row, col)][self._rc_to_i(row, col - 1)] > 1 :
                                        mud_x = maze_x_offset + col * cell_size - mud.get_width() // 2
                                        mud_y = maze_y_offset + row * cell_size
                                        maze_elements.append((mud_x, mud_y, mud))
                                        if self.render_details :
                                            weight_text = surface_from_text(str(self.maze[self._rc_to_i(row, col)][self._rc_to_i(row, col - 1)]), text_size, mud_text_color)
                                            weight_text_x = maze_x_offset + col * cell_size - weight_text.get_width() // 2
                                            weight_text_y = maze_y_offset + row * cell_size + (cell_size - weight_text.get_height()) // 2
                                            maze_elements.append((weight_text_x, weight_text_y, weight_text))
                            if cell_is_in_maze(row - 1, col) :
                                if self._rc_to_i(row - 1, col) in self.maze[self._rc_to_i(row, col)] :
                                    if self.maze[self._rc_to_i(row, col)][self._rc_to_i(row - 1, col)] > 1 :
                                        mud_horizontal = pygame.transform.rotate(mud, 90)
                                        mud_x = maze_x_offset + col * cell_size
                                        mud_y = maze_y_offset + row * cell_size - mud.get_width() // 2
                                        maze_elements.append((mud_x, mud_y, mud_horizontal))
                                        if self.render_details :
                                            weight_text = surface_from_text(str(self.maze[self._rc_to_i(row, col)][self._rc_to_i(row - 1, col)]), text_size, mud_text_color)
                                            weight_text_x = maze_x_offset + col * cell_size + (cell_size - weight_text.get_width()) // 2
                                            weight_text_y = maze_y_offset + row * cell_size - weight_text.get_height() // 2
                                            maze_elements.append((weight_text_x, weight_text_y, weight_text))

                # Add cell numbers
                for row in range(self.maze_height) :
                    for col in range(self.maze_width) :
                        if cell_is_in_maze(row, col) :
                            if self.render_details :
                                cell_text = surface_from_text(str(self._rc_to_i(row, col)), text_size, cell_text_color)
                                cell_text_x = maze_x_offset + col * cell_size + cell_text_offset
                                cell_text_y = maze_y_offset + row * cell_size + cell_text_offset
                                maze_elements.append((cell_text_x, cell_text_y, cell_text))
                # Add walls
                walls = []
                wall = surface_from_image(os.path.join("gui", "wall", "wall.png"), cell_size)
                for row in range(self.maze_height + 1) :
                    for col in range(self.maze_width + 1) :
                        case_outside_to_inside = not cell_is_in_maze(row, col) and cell_is_in_maze(row, col - 1)
                        case_inside_to_outside = cell_is_in_maze(row, col) and not cell_is_in_maze(row, col - 1)
                        case_inside_to_inside = cell_is_in_maze(row, col) and cell_is_in_maze(row, col - 1) and self._rc_to_i(row, col - 1) not in self.maze[self._rc_to_i(row, col)]
                        if case_outside_to_inside or case_inside_to_outside or case_inside_to_inside :
                            wall_x = maze_x_offset + col * cell_size - wall.get_width() // 2
                            wall_y = maze_y_offset + row * cell_size
                            maze_elements.append((wall_x, wall_y, wall))
                            walls.append((row, col, row, col - 1))
                        case_outside_to_inside = not cell_is_in_maze(row, col) and cell_is_in_maze(row - 1, col)
                        case_inside_to_outside = cell_is_in_maze(row, col) and not cell_is_in_maze(row - 1, col)
                        case_inside_to_inside = cell_is_in_maze(row, col) and cell_is_in_maze(row - 1, col) and self._rc_to_i(row - 1, col) not in self.maze[self._rc_to_i(row, col)]
                        if case_outside_to_inside or case_inside_to_outside or case_inside_to_inside :
                            wall_horizontal = pygame.transform.rotate(wall, 90)
                            wall_x = maze_x_offset + col * cell_size
                            wall_y = maze_y_offset + row * cell_size - wall.get_width() // 2
                            maze_elements.append((wall_x, wall_y, wall_horizontal))
                            walls.append((row, col, row - 1, col))
                    
                # Add corners
                corner = surface_from_image(os.path.join("gui", "wall", "corner.png"), int(wall.get_width() * corner_wall_ratio), int(wall.get_width() * corner_wall_ratio))
                for row, col, neighbor_row, neighbor_col in walls :
                    if col != neighbor_col :
                        corner_x = maze_x_offset + col * cell_size - corner.get_width() // 2
                        if (row - 1, col, neighbor_row - 1, neighbor_col) not in walls or ((neighbor_row, neighbor_col, neighbor_row - 1, neighbor_col) in walls and (row, col, row - 1, col) in walls and (row - 1, col, neighbor_row - 1, neighbor_col) in walls) :
                            corner_y = maze_y_offset + row * cell_size - corner.get_width() // 2
                            maze_elements.append((corner_x, corner_y, corner))
                        if (row + 1, col, neighbor_row + 1, neighbor_col) not in walls :
                            corner_y = maze_y_offset + (row + 1) * cell_size - corner.get_width() // 2
                            maze_elements.append((corner_x, corner_y, corner))
                    if row != neighbor_row :
                        corner_y = maze_y_offset + row * cell_size - corner.get_width() // 2
                        if (row, col - 1, neighbor_row, neighbor_col - 1) not in walls :
                            corner_x = maze_x_offset + col * cell_size - corner.get_width() // 2
                            maze_elements.append((corner_x, corner_y, corner))
                        if (row, col + 1, neighbor_row, neighbor_col + 1) not in walls :
                            corner_x = maze_x_offset + (col + 1) * cell_size - corner.get_width() // 2
                            maze_elements.append((corner_x, corner_y, corner))
                
                # Add flags
                if self.render_details :
                    cells_with_flags = {cell : {} for cell in self.player_locations.values()}
                    for player in self.player_locations :
                        team = [team for team in self.teams if player in self.teams[team]][0]
                        if team not in cells_with_flags[self.player_locations[player]] :
                            cells_with_flags[self.player_locations[player]][team] = 0
                        cells_with_flags[self.player_locations[player]][team] += 1
                    flag = surface_from_image(os.path.join("gui", "flag", "flag.png"), flag_size)
                    max_teams_in_cells = max([len(team) for team in cells_with_flags.values()])
                    max_players_in_cells = max([cells_with_flags[cell][team] for cell in cells_with_flags for team in cells_with_flags[cell]])
                    for cell in cells_with_flags :
                        row, col = self._i_to_rc(cell)
                        for i_team in range(len(cells_with_flags[cell])) :
                            team = list(cells_with_flags[cell].keys())[i_team]
                            flag_colored = colorize(flag, team_colors[team])
                            flag_colored = add_color_border(flag_colored, flag_border_color, flag_border_width)
                            for i_player in range(cells_with_flags[cell][team]) :
                                flag_x = maze_x_offset + (col + 1) * cell_size - flag_x_offset - i_player * min(flag_x_next_offset, (cell_size - flag_x_offset) / (max_players_in_cells + 1))
                                flag_y = maze_y_offset + row * cell_size - flag.get_height() + flag_y_offset + i_team * min(flag_y_offset, (cell_size - flag_y_offset) / (max_teams_in_cells + 1))
                                maze_elements.append((flag_x, flag_y, flag_colored))

                # Add cheese
                cheese = surface_from_image(os.path.join("gui", "cheese", "cheese.png"), cheese_size)
                cheese = add_color_border(cheese, cheese_border_color, cheese_border_width)
                for c in self.cheese :
                    row, col = self._i_to_rc(c)
                    cheese_x = maze_x_offset + col * cell_size + (cell_size - cheese.get_width()) // 2
                    cheese_y = maze_y_offset + row * cell_size + (cell_size - cheese.get_height()) // 2
                    cheese_elements[c] = (cheese_x, cheese_y, cheese)
                
                # Add players
                for player_name in self.player_locations :
                    team = [team for team in self.teams if player_name in self.teams[team]][0]
                    player_neutral, player_north, player_south, player_west, player_east = load_player_surfaces(player_name, player_size, team_colors[team], player_border_width)
                    row, col = self._i_to_rc(self.player_locations[player_name])
                    player_x = maze_x_offset + col * cell_size + (cell_size - player_neutral.get_width()) // 2
                    player_y = maze_y_offset + row * cell_size + (cell_size - player_neutral.get_height()) // 2
                    player_elements[player_name] = (player_x, player_y, player_neutral, player_north, player_south, player_west, player_east)
                
                # Add avatars area
                score_locations = {}
                medal_locations = {}
                for i in range(len(self.teams)) :
                
                    # Box
                    team = list(self.teams.keys())[i]
                    avatars_area_color_box = team_colors[team] if self.render_details else avatars_area_color
                    team_background = pygame.Surface((avatars_area_width, avatars_area_height))
                    pygame.draw.rect(team_background, background_color, pygame.Rect(0, 0, avatars_area_width, avatars_area_height))
                    pygame.draw.rect(team_background, avatars_area_color_box, pygame.Rect(0, 0, avatars_area_width, avatars_area_height), avatars_area_border, avatars_area_angle)
                    team_background_x = avatars_x_offset
                    team_background_y = (1 + i) * maze_y_offset + i * avatars_area_height if len(self.teams) > 1 else (window_height - avatars_area_height) // 2
                    avatar_elements.append((team_background_x, team_background_y, team_background))
                    medal_locations[team] = (team_background_x + avatars_area_width, team_background_y)
                    
                    # Team name
                    team_text = surface_from_text(team, team_text_size, avatars_area_color)
                    if team_text.get_width() > avatars_area_width - 2 * avatars_area_padding :
                        ratio = (avatars_area_width - 2 * avatars_area_padding) / team_text.get_width()
                        team_text = pygame.transform.scale(team_text, (int(team_text.get_width() * ratio), int(team_text.get_height() * ratio)))
                    team_text_x = avatars_x_offset + (avatars_area_width - team_text.get_width()) // 2
                    team_text_y = team_background_y + avatars_area_padding + (team_text_size - team_text.get_height()) // 2
                    avatar_elements.append((team_text_x, team_text_y, team_text))
                    
                    # Players images
                    players = []
                    for j in range(len(self.teams[team])) :
                        player_name = self.teams[team][j]
                        player_avatar = load_player_avatar(player_name, player_avatar_size)
                        players.append((player_name, player_avatar))
                    avatar_area = pygame.Surface((2 * avatars_area_padding + sum([player[1].get_width() for player in players]) + player_avatar_horizontal_padding * (len(players) - 1), player_avatar_size))
                    pygame.draw.rect(avatar_area, background_color, pygame.Rect(0, 0, avatar_area.get_width(), avatar_area.get_height()))
                    player_x = avatars_area_padding
                    centers = []
                    for player_name, player in players :
                        avatar_area.blit(player, (player_x, 0))
                        centers.append(player_x + player.get_width() // 2)
                        player_x += player.get_width() + player_avatar_horizontal_padding
                    if avatar_area.get_width() > avatars_area_width - 2 * avatars_area_padding :
                        ratio = (avatars_area_width - 2 * avatars_area_padding) / avatar_area.get_width()
                        centers = [center * ratio for center in centers]
                        avatar_area = pygame.transform.scale(avatar_area, (int(avatar_area.get_width() * ratio), int(avatar_area.get_height() * ratio)))
                    avatar_area_x = avatars_x_offset + (avatars_area_width - avatar_area.get_width()) // 2
                    avatar_area_y = team_background_y + 2 * avatars_area_padding + team_text_size + (player_avatar_size - avatar_area.get_height()) // 2
                    avatar_elements.append((avatar_area_x, avatar_area_y, avatar_area))

                    # Players names
                    for j in range(len(self.teams[team])) :
                        player_name = self.teams[team][j]
                        while True :
                            player_name_text = surface_from_text(player_name, player_name_text_size, avatars_area_color)
                            if player_name_text.get_width() > (avatars_area_width - 2 * avatars_area_padding) / len(self.teams[team]) :
                                player_name = player_name[:-2] + "."
                            else :
                                break
                        player_name_text_x = avatar_area_x + centers[j] - player_name_text.get_width() // 2
                        player_name_text_y = team_background_y + 3 * avatars_area_padding + team_text_size + player_avatar_size + (player_name_text_size - player_name_text.get_height()) // 2
                        avatar_elements.append((player_name_text_x, player_name_text_y, player_name_text))
                
                    # Score locations
                    cheese_missing = surface_from_image(os.path.join("gui", "cheese", "cheese_missing.png"), cheese_score_size)
                    score_x_offset = avatars_x_offset + avatars_area_padding
                    score_margin = avatars_area_width - 2 * avatars_area_padding - cheese_missing.get_width()
                    if self.nb_cheese > 1 :
                        score_margin /= (self.nb_cheese - 1)
                    score_margin = min(score_margin, cheese_missing.get_width() * 2)
                    estimated_width = cheese_missing.get_width() + (self.nb_cheese - 1) * score_margin
                    if estimated_width < avatars_area_width - 2 * avatars_area_padding :
                        score_x_offset += (avatars_area_width - 2 * avatars_area_padding - estimated_width) / 2
                    score_y_offset = team_background_y + 4 * avatars_area_padding + team_text_size + player_avatar_size + player_name_text_size
                    score_locations[team] = (score_x_offset, score_margin, score_y_offset)

                # Show maze
                def show_maze () :
                    pygame.draw.rect(screen, background_color, pygame.Rect(maze_x_offset, maze_y_offset, game_area_width, game_area_height))
                    for surface_x, surface_y, surface in maze_elements :
                        screen.blit(surface, (surface_x, surface_y))
                show_maze()
                
                # Show cheese
                def show_cheese (cheese) :
                    for c in cheese :
                        cheese_x, cheese_y, surface = cheese_elements[c]
                        screen.blit(surface, (cheese_x, cheese_y))
                show_cheese(self.cheese)
                
                # Show_players at initial locations
                for p in player_elements :
                    player_x, player_y, player_neutral, _, _ , _, _ = player_elements[p]
                    screen.blit(player_neutral, (player_x, player_y))
                
                # Show avatars
                def show_avatars () :
                    for surface_x, surface_y, surface in avatar_elements :
                        screen.blit(surface, (surface_x, surface_y))
                show_avatars()
                
                # Show scores
                def show_scores (team_scores) :
                    cheese_missing = surface_from_image(os.path.join("gui", "cheese", "cheese_missing.png"), cheese_score_size)
                    cheese_missing = add_color_border(cheese_missing, cheese_score_border_color, cheese_score_border_width)
                    cheese_eaten = surface_from_image(os.path.join("gui", "cheese", "cheese_eaten.png"), cheese_score_size)
                    cheese_eaten = add_color_border(cheese_eaten, cheese_score_border_color, cheese_score_border_width)
                    for team in score_locations :
                        score_x_offset, score_margin, score_y_offset = score_locations[team]
                        for i in range(int(team_scores[team])) :
                            screen.blit(cheese_eaten, (score_x_offset + i * score_margin, score_y_offset))
                        if int(team_scores[team]) != team_scores[team] :
                            cheese_partial = surface_from_image(os.path.join("gui", "cheese", "cheese_eaten.png"), cheese_score_size)
                            cheese_partial = colorize(cheese_partial, [(team_scores[team] - int(team_scores[team])) * 255] * 3)
                            cheese_partial = add_color_border(cheese_partial, cheese_score_border_color, cheese_score_border_width)
                            screen.blit(cheese_partial, (score_x_offset + int(team_scores[team]) * score_margin, score_y_offset))
                        for j in range(int(numpy.ceil(team_scores[team])), self.nb_cheese) :
                            screen.blit(cheese_missing, (score_x_offset + j * score_margin, score_y_offset))
                show_scores(self._score_per_team())
                
                # Show preprocessing message
                preprocessing_image = surface_from_image(os.path.join("gui", "drawings", "pyrat_preprocessing.png"), int(min(game_area_width, game_area_height) * main_image_factor))
                preprocessing_image = add_color_border(preprocessing_image, main_image_border_color, main_image_border_size)
                go_image = surface_from_image(os.path.join("gui", "drawings", "pyrat_go.png"), int(min(game_area_width, game_area_height) * main_image_factor))
                go_image = add_color_border(go_image, main_image_border_color, main_image_border_size)
                main_image_x = maze_x_offset + (game_area_width - preprocessing_image.get_width()) / 2
                main_image_y = maze_y_offset + (game_area_height - preprocessing_image.get_height()) / 2
                screen.blit(preprocessing_image, (main_image_x, main_image_y))
                
                # Show initial rendering
                pygame.display.flip()
                
                # Run until the user asks to quit
                current_player_locations = self.player_locations
                current_cheese = self.cheese
                mud_being_crossed = {player : 0 for player in self.player_locations}
                traces = {player : [(player_elements[player][0] + player_elements[player][2].get_width() / 2, player_elements[player][1] + player_elements[player][2].get_height() / 2)] for player in self.player_locations}
                trace_colors = {player : get_main_color(player_elements[player][2]) for player in self.player_locations}
                player_surfaces = {player : player_elements[player][2] for player in self.player_locations}
                running = True
                while running :
                
                    # Stop when the window is closed or escape key is pressed
                    for event in pygame.event.get() :
                        if event.type == pygame.QUIT or (event.type == pglocals.KEYDOWN and event.key == pglocals.K_ESCAPE) :
                            running = False
                    
                    # Update display
                    try :
                        
                        # Get turn info
                        team_scores, new_player_locations, mud_values, new_cheese, done, turn = self.gui_process_queue.get(False)
                        
                        # Enter mud?
                        for player in current_player_locations :
                            if mud_values[player] > 0 and mud_being_crossed[player] == 0 :
                                mud_being_crossed[player] = mud_values[player] + 1

                        # Choose the correct player surface
                        for player in current_player_locations :
                            player_x, player_y, player_neutral, player_north, player_south, player_west, player_east = player_elements[player]
                            row, col = self._i_to_rc(current_player_locations[player])
                            new_row, new_col = self._i_to_rc(new_player_locations[player])
                            player_x += player_surfaces[player].get_width() / 2
                            player_y += player_surfaces[player].get_height() / 2
                            if new_col > col :
                                player_surfaces[player] = player_east
                            elif new_col < col :
                                player_surfaces[player] = player_west
                            elif new_row > row :
                                player_surfaces[player] = player_south
                            elif new_row < row :
                                player_surfaces[player] = player_north
                            else :
                                player_surfaces[player] = player_neutral
                            player_x -= player_surfaces[player].get_width() / 2
                            player_y -= player_surfaces[player].get_height() / 2
                            player_elements[player] = (player_x, player_y, player_neutral, player_north, player_south, player_west, player_east)

                        # Move players
                        for i in range(animation_steps) :
                        
                            # Reset background & cheese
                            show_maze()
                            show_cheese(current_cheese if i != animation_steps - 1 else new_cheese)
                            
                            # Move player with trace
                            for player in current_player_locations :
                                player_x, player_y, player_neutral, player_north, player_south, player_west, player_east = player_elements[player]
                                row, col = self._i_to_rc(current_player_locations[player])
                                new_row, new_col = self._i_to_rc(new_player_locations[player])
                                shift = (i + 1) * cell_size / animation_steps
                                if mud_being_crossed[player] > 0 :
                                    shift /= mud_being_crossed[player]
                                    shift += (mud_being_crossed[player] - mud_values[player] - 1) * cell_size / mud_being_crossed[player]
                                next_x = player_x if col == new_col else player_x + shift if new_col > col else player_x - shift
                                next_y = player_y if row == new_row else player_y + shift if new_row > row else player_y - shift
                                if i == animation_steps - 1 and mud_values[player] == 0 :
                                    player_elements[player] = (next_x, next_y, player_neutral, player_north, player_south, player_west, player_east)
                                if self.trace_length > 0 :
                                    pygame.draw.line(screen, trace_colors[player], (next_x + player_surfaces[player].get_width() / 2, next_y + player_surfaces[player].get_height() / 2), traces[player][-1], width=trace_size)
                                    for j in range(1, self.trace_length) :
                                        if len(traces[player]) > j :
                                            pygame.draw.line(screen, trace_colors[player], traces[player][-j-1], traces[player][-j], width=trace_size)
                                    if len(traces[player]) == self.trace_length + 1 :
                                        final_segment_length = numpy.sqrt((traces[player][-1][0] - (next_x + player_surfaces[player].get_width() / 2))**2 + (traces[player][-1][1] - (next_y + player_surfaces[player].get_height() / 2))**2)
                                        ratio = 1 - final_segment_length / cell_size
                                        pygame.draw.line(screen, trace_colors[player], traces[player][1], (traces[player][1][0] + ratio * (traces[player][0][0] - traces[player][1][0]), traces[player][1][1] + ratio * (traces[player][0][1] - traces[player][1][1])), width=trace_size)
                                screen.blit(player_surfaces[player], (next_x, next_y))
                            
                            # Indicate when preprocessing is over
                            if turn == 1 :
                                screen.blit(go_image, (main_image_x, main_image_y))
                            
                            # Show & wait for animation
                            pygame.display.flip()
                            time.sleep(animation_time / animation_steps)

                        # Exit mud?
                        for player in current_player_locations :
                            if mud_values[player] == 0 :
                                mud_being_crossed[player] = 0
                            if mud_being_crossed[player] == 0 :
                                current_player_locations[player] = new_player_locations[player]
                                player_x, player_y, _, _, _, _, _ = player_elements[player]
                                if traces[player][-1] != (player_x + player_surfaces[player].get_width() / 2, player_y + player_surfaces[player].get_height() / 2) :
                                    traces[player].append((player_x + player_surfaces[player].get_width() / 2, player_y + player_surfaces[player].get_height() / 2))
                                traces[player] = traces[player][-self.trace_length-1:]
                        
                        # Play a sound is a cheese is eaten
                        for player in current_player_locations :
                            if new_player_locations[player] in current_cheese and mud_being_crossed[player] == 0 :
                                play_sound(player, "cheese_eaten.wav")
                        
                        # Update score
                        show_avatars()
                        show_scores(team_scores)
                        pygame.display.flip()
                        current_cheese = new_cheese
                        
                        # Indicate if the game is over
                        if done :
                            sorted_results = sorted([(team_scores[team], team) for team in team_scores], reverse=True)
                            medals = [surface_from_image(os.path.join("gui", "medals", medal_name), medal_size) for medal_name in ["first.png", "second.png", "third.png", "others.png"]]
                            for i in range(len(sorted_results)) :
                                if i > 0 and sorted_results[i][0] != sorted_results[i-1][0] and len(medals) > 1 :
                                    del medals[0]
                                team = sorted_results[i][1]
                                screen.blit(medals[0], (medal_locations[team][0] - medals[0].get_width() / 2, medal_locations[team][1] - medals[0].get_height() / 2))
                            pygame.display.flip()
                        
                    # Ignore exceptions raised due to emtpy queue
                    except queue.Empty :
                        pass
                    
                # Quit PyGame
                pygame.quit()
                
            except :
                pass
            
        # Initialize the GUI in a different process
        if self.turn == 0 :
            self.gui_process = DillProcess(target=gui_process_function)
            self.gui_process.start()
        
        # At each turn, send current info to the thread
        else :
            new_player_locations = {player : self.player_muds[player]["target"] if self._is_in_mud(player) else self.player_locations[player] for player in self.player_locations}
            mud_values = {player : self.player_muds[player]["count"] for player in self.player_locations}
            self.gui_process_queue.put((self._score_per_team(), new_player_locations, mud_values, self.cheese, self.done, self.turn))
            
    #############################################################################################################################################
    #                                                                GAME METHODS                                                               #
    #############################################################################################################################################

    def _create_maze (self) :
        
        """
            Creates a maze.
            
            In:
                * None.
                
            Out:
                * maze .......... dict : int -> (dict : int -> int) ...................... Created maze.
                * maze_public ... numpy.ndarray [or] dict : int -> (dict : int -> int) ... Created maze in a representation for the players.
                * maze_width .... int .................................................... Width of the maze effectively created.
                * maze_height ... int .................................................... Height of the maze effectively created.
        """
        
        # Set random seed
        nprandom.seed(self.game_random_seed_maze)

        # If a fixed maze is provided (as a representation handled by the game), we use it
        if self.fixed_maze is not None :
        
            # Load given maze
            fixed_maze = ast.literal_eval(self.fixed_maze) if isinstance(self.fixed_maze, str) else self.fixed_maze
            if isinstance(self.fixed_maze, list) :
                fixed_maze = numpy.array(fixed_maze)
            
            # If given as a matrix
            if isinstance(fixed_maze, numpy.ndarray) :
                maze = {}
                maze_width = 1
                for vertex in range(fixed_maze.shape[0]) :
                    for neighbor in fixed_maze[vertex].nonzero()[0] :
                        if neighbor - vertex > 1 :
                            maze_width = neighbor - vertex
                        if vertex not in maze :
                            maze[vertex] = {}
                        maze[vertex][neighbor] = fixed_maze[vertex, neighbor]
                maze_height = fixed_maze.shape[0] // maze_width
            
            # If given as a dictionary
            elif isinstance(fixed_maze, dict) :
                maze = fixed_maze
                maze_width = 1
                for vertex in fixed_maze :
                    for neighbor in fixed_maze[vertex] :
                        if neighbor - vertex > 1 :
                            maze_width = neighbor - vertex
                            break
                    if maze_width != 1 :
                        break
                maze_height = int(numpy.ceil((max(fixed_maze) + 1) / maze_width))
            
            else :
                raise Exception("Unhandled type", type(fixed_maze), "when loading fixed maze, should be a matrix")
            
            # Check dimensions
            if maze_width < 1 or maze_height < 1 :
                raise Exception("Invalid maze dimensions in fixed maze")
            
        # Otherwise we generate one
        else :
        
            # We will use the provided dimensions
            maze_width = self.maze_width
            maze_height = self.maze_height
        
            # Initialize an empty maze, and add cells until it reaches the asked density
            maze_sparse = sparse.lil_matrix((maze_width * maze_height, maze_width * maze_height), dtype=int)
            cells = [(maze_height // 2, maze_width // 2)]
            while len(cells) / maze_sparse.shape[0] * 100 < self.cell_percentage :
                row, col = cells[nprandom.randint(len(cells))]
                neighbor_row, neighbor_col = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)][nprandom.randint(4)]
                if 0 <= neighbor_row < maze_height and 0 <= neighbor_col < maze_width :
                    maze_sparse[self._rc_to_i(row, col), self._rc_to_i(neighbor_row, neighbor_col)] = 1
                    maze_sparse[self._rc_to_i(neighbor_row, neighbor_col), self._rc_to_i(row, col)] = 1
                    for next_neighbor_row, next_neighbor_col in [(neighbor_row - 1, neighbor_col), (neighbor_row + 1, neighbor_col), (neighbor_row, neighbor_col - 1), (neighbor_row, neighbor_col + 1)] :
                        if (next_neighbor_row, next_neighbor_col) in cells :
                            maze_sparse[self._rc_to_i(next_neighbor_row, next_neighbor_col), self._rc_to_i(neighbor_row, neighbor_col)] = 1
                            maze_sparse[self._rc_to_i(neighbor_row, neighbor_col), self._rc_to_i(next_neighbor_row, next_neighbor_col)] = 1
                    if (neighbor_row, neighbor_col) not in cells :
                        cells.append((neighbor_row, neighbor_col))
            
            # Add walls
            maze_full = csgraph.minimum_spanning_tree(maze_sparse)
            maze_full += maze_full.transpose()
            walls = sparse.triu(maze_sparse - maze_full).nonzero()
            walls = [(walls[0][i], walls[1][i]) for i in range(walls[0].shape[0])]
            nprandom.shuffle(walls)
            for i in range(int(numpy.ceil(self.wall_percentage / 100.0 * len(walls)))) :
                maze_sparse[walls[i][0], walls[i][1]] = 0
                maze_sparse[walls[i][1], walls[i][0]] = 0

            # Add mud
            paths = sparse.triu(maze_sparse).nonzero()
            paths = [(paths[0][i], paths[1][i]) for i in range(paths[0].shape[0])]
            nprandom.shuffle(paths)
            for i in range(int(numpy.ceil(self.mud_percentage / 100.0 * len(paths)))) :
                mud_weight = nprandom.choice(range(self.mud_range[0], self.mud_range[1] + 1))
                maze_sparse[paths[i][0], paths[i][1]] = mud_weight
                maze_sparse[paths[i][1], paths[i][0]] = mud_weight

            # Convert to dictionary
            maze = {}
            for vertex in range(maze_sparse.shape[0]) :
                neighbors = maze_sparse[vertex].rows[0]
                if len(neighbors) > 0 :
                    maze[vertex] = {}
                    for neighbor in neighbors :
                        maze[vertex][neighbor] = maze_sparse[vertex, neighbor]
            
        # We convert the maze to the asked format to provide to players
        if self.maze_representation == "dictionary" :
            maze_public = maze.copy()
        elif self.maze_representation == "matrix" :
            maze_public = numpy.zeros((maze_width * maze_height, maze_width * maze_height))
            for vertex in maze :
                for neighbor in maze[vertex] :
                    maze_public[vertex, neighbor] = maze[vertex][neighbor]
        else :
            raise Exception("Invalid public representation of the maze %s" % self.maze_representation)

        # Done
        return maze, maze_public, maze_width, maze_height
        
    #############################################################################################################################################

    def _distribute_cheese (self) :
        
        """
            Distributes pieces of cheese in the maze.
            
            In:
                * None.
                
            Out:
                * cheese ... list (int) ... List of pieces of cheese.
        """
        
        # Set random seed
        nprandom.seed(self.game_random_seed_cheese)
        
        # Get free cells
        cells = [i for i in self.maze if i not in self.player_locations.values()]
        
        # If a fixed list of cheese is provided, we use it
        if self.fixed_cheese is not None :
        
            # Load given cheese
            cheese = ast.literal_eval(self.fixed_cheese) if isinstance(self.fixed_cheese, str) else self.fixed_cheese
            if not isinstance(cheese, list) :
                raise Exception("Unhandled type", type(cheese), "when loading fixed cheese, should be a list")
            
            # check compatibility
            if len(cheese) != len(set(cheese)) :
                raise Exception("Duplicates in fixed cheese")
            if len(set(cheese) & set(cells)) != len(cheese) :
                raise Exception("Some cheese cannot be placed")

        # Otherwise we generate them
        else :
        
            # We check there is enough space for the required cheese (and keep the center clear for players)
            if len(cells) < self.nb_cheese :
                raise Exception("Not enough space for asked number of cheese")
            
            # We place the cheese
            nprandom.shuffle(cells)
            cheese = cells[:self.nb_cheese]
        
        # Done
        return cheese
        
    #############################################################################################################################################

    def _register_player (self, name, turn_function, preprocessing_function=None, postprocessing_function=None, team="", location="center") :
        
        """
            Adds a player to the game.
            
            In:
                * name ...................... str .................................................................................................................................................................................................................................................. Name of the player.
                * turn_function ............. function : (numpy.ndarray [or] dict : int -> (dict : int -> int)), int, int, str, (dict : str -> list (str)), (dict : str -> int), (dict : str -> float), (dict : str -> (dict : str -> int)), list (int), list (str) -> str ......................... Function used to control the player at each turn.
                * preprocessing_function .... function : (numpy.ndarray [or] dict : int -> (dict : int -> int)), int, int, str, (dict : str -> list (str)), (dict : str -> int), list (int), list (str) -> None .................................................................................... Preprocessing function used by the player at the beginning of the game (optional).
                * postprocessing_function ... function : (numpy.ndarray [or] dict : int -> (dict : int -> int)), int, int, str, (dict : str -> list (str)), (dict : str -> int), (dict : str -> float), (dict : str -> (dict : str -> int)), list (int), list (str), (dict : str -> Any) -> None ... Function called at the end of the game (optional).
                * team ...................... str .................................................................................................................................................................................................................................................. Team of the player.
                * location .................. str [or] int ......................................................................................................................................................................................................................................... Controls initial location of the player (random, same, center, or a fixed index).
            
            Out:
                * None.
        """

        # Set random seed
        nprandom.seed(self.game_random_seed_players + len(self.player_locations))
        
        # Make sure these are strings
        name = str(name)
        team = str(team)
        
        # Check if the name is unique
        if name in self.player_locations :
            raise Exception("Use distinct names for players")

        # Set initial location
        if location == "random" :
            self.player_locations[name] = nprandom.choice(list(self.maze))
        elif location == "same" :
            self.player_locations[name] = list(self.player_locations.values())[-1]
        elif location == "center" :
            self.player_locations[name] = self._rc_to_i(self.maze_height // 2, self.maze_width // 2)
        elif isinstance(location, int) and 0 <= location < self.maze_height * self.maze_width :
            if location in self.maze :
                self.player_locations[name] = location
            else :
                print("Warning: player %s cannot start at unreachable location %d, starting at closest cell (using Euclidean distance)" % (name, location), file=sys.stderr)
                location_rc = numpy.array(self._i_to_rc(location))
                distances = [numpy.linalg.norm(location_rc - numpy.array(self._i_to_rc(cell))) for cell in self.maze]
                self.player_locations[name] = list(self.maze.keys())[numpy.argmin(distances)]
        else :
            raise Exception("Invalid initial location provided for player %s" % (name))
        self.player_initial_locations[name] = self.player_locations[name]

        # Append to team
        if team not in self.teams :
            self.teams[team] = []
        self.teams[team].append(name)
        
        # Initialize other elements
        self.player_scores[name] = 0
        self.player_muds[name] = {"target" : None,
                                    "count" : 0}
        self.player_traces[name] = []
        self.player_functions[name] = {"preprocessing" : preprocessing_function,
                                        "postprocessing" : postprocessing_function,
                                        "turn" : turn_function}
        self.moves_history[name] = []
        self.stats["players"][name] = {"moves" : {"mud" : 0,
                                                    "error" : 0,
                                                    "miss" : 0,
                                                    "nothing" : 0,
                                                    "north" : 0,
                                                    "east" : 0,
                                                    "south" : 0,
                                                    "west" : 0,
                                                    "wall" : 0},
                                        "score" : 0,
                                        "turn_durations" : [],
                                        "preprocessing_duration" : None}
        
    #############################################################################################################################################

    def start (self, reset=False) :

        """
            Starts a game, asking players for decisions until the game is over.
            
            In:
                * reset ... bool ... Indicates if the game should be reset to avoid reseting twice when the game is created.
                
            Out:
                * stats ... dict : str -> Any ... Game statistics computed during the game.
        """
        
        # Function to execute in a separate process per player
        def player_process_function (player, input_queue, output_queue, turn_start_synchronizer, turn_timeout_lock, turn_end_synchronizer) :
            try :
                while True :
                    
                    # Wait for all players ready
                    turn_start_synchronizer.wait()
                    maze, maze_width, maze_height, teams, possible_actions, player_locations, player_scores, player_muds, cheese, turn, final_stats = input_queue.get()
                    duration = None
                    try :
                        
                        # Call postprocessing once the game is over
                        if final_stats is not None :
                            action = "postprocessing_error"
                            if self.player_functions[player]["postprocessing"] is not None :
                                self.player_functions[player]["postprocessing"](maze, maze_width, maze_height, player, teams, player_locations, player_scores, player_muds, cheese, possible_actions, final_stats)
                            action = "postprocessing"
                            
                        # If in mud, we return immediately (main thread will wait for us in all cases)
                        elif player_muds[player]["target"] is not None :
                            action = "mud"
                        
                        # Otherwise, we ask for a move
                        else :
                            start = time.process_time()
                            if turn == 0 :
                                action = "preprocessing_error"
                                if self.player_functions[player]["preprocessing"] is not None :
                                    self.player_functions[player]["preprocessing"](maze, maze_width, maze_height, player, teams, player_locations, cheese, possible_actions)
                                action = "preprocessing"
                            else :
                                action = "error"
                                a = self.player_functions[player]["turn"](maze, maze_width, maze_height, player, teams, player_locations, player_scores, player_muds, cheese, possible_actions)
                                if a not in possible_actions :
                                    raise Exception("Invalid action %s by player %s" % (str(a), player))
                                action = a
                            duration = time.process_time() - start
                    
                    # Print error message in case of a crash
                    except :
                        print("Player %s has crashed with the following error:" % (player), file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
                            
                    # Turn is over
                    with turn_timeout_lock :
                        output_queue.put((action, duration))
                    turn_end_synchronizer.wait()
                    if action.startswith("postprocessing") :
                        break

            except :
                pass

        # Function to execute in a separate process per player to handle timeouts
        def waiter_process_function (input_queue, turn_start_synchronizer) :
            try :
                while True :
                    _ = input_queue.get()
                    turn_start_synchronizer.wait()
            except :
                pass

        # Reset the game
        if reset :
            self.reset()
        
        # Initial rendering of the maze
        self.render()

        # We catch exceptions that may happen during the game
        try :
        
            # Create a thread per player
            turn_start_synchronizer = multiprocessing.Manager().Barrier(len(self.player_locations) + 1)
            turn_timeout_lock = multiprocessing.Manager().Lock()
            player_processes = {}
            for player in self.player_locations :
                
                # Create associated process
                player_processes[player] = {"process" : None, "input_queue" : multiprocessing.Manager().Queue(), "output_queue" : multiprocessing.Manager().Queue(), "turn_end_synchronizer" : multiprocessing.Barrier(2)}
                player_processes[player]["process"] = DillProcess(target=player_process_function, args=(player, player_processes[player]["input_queue"], player_processes[player]["output_queue"], turn_start_synchronizer, turn_timeout_lock, player_processes[player]["turn_end_synchronizer"],))
                player_processes[player]["process"].start()

            # Get some info once and for all
            maze, maze_width, maze_height, teams, possible_actions = self.get_description()
            action_meanings = self.get_action_meanings()

            # If playing asynchrounously, we create threads to wait instead of missing players
            if not self.synchronous :
                waiter_processes = {}
                for player in player_processes :
                    waiter_processes[player] = {"process" : None, "input_queue" : multiprocessing.Manager().Queue()}
                    waiter_processes[player]["process"] = DillProcess(target=waiter_process_function, args=(waiter_processes[player]["input_queue"], turn_start_synchronizer,))
                    waiter_processes[player]["process"].start()

            # We play until the game is over
            players_ready = list(player_processes.keys())
            players_running = {player : True for player in player_processes}
            while any(players_running.values()) :

                # We communicate the state of the game to the players not in mud
                player_locations, player_scores, player_muds, cheese = self.get_state()
                for player in players_ready :
                    final_stats = self.stats if self.done else None
                    player_processes[player]["input_queue"].put((maze, maze_width, maze_height, teams, possible_actions, player_locations, player_scores, player_muds, cheese, self.turn, final_stats))
                turn_start_synchronizer.wait()

                # Wait some time
                sleep_time = self.preprocessing_time if self.turn == 0 else self.turn_time
                time.sleep(sleep_time)

                # In synchronous mode, we wait for everyone
                actions_as_text = {player : "postprocessing" for player in player_processes}
                if self.synchronous :
                    for player in player_processes :
                        player_processes[player]["turn_end_synchronizer"].wait()
                        actions_as_text[player], duration = player_processes[player]["output_queue"].get()

                # Otherwise, we block the possibility to return an action and check who answered in time
                else :

                    # Wait at least for those in mud
                    for player in player_processes :
                        if self._is_in_mud(player) and players_running[player] :
                            player_processes[player]["turn_end_synchronizer"].wait()
                            actions_as_text[player], duration = player_processes[player]["output_queue"].get()

                    # For others, set timeout and wait for output info of those who passed just before timeout
                    with turn_timeout_lock :
                        for player in player_processes :
                            if not self._is_in_mud(player) and players_running[player] :
                                if not player_processes[player]["output_queue"].empty() :
                                    player_processes[player]["turn_end_synchronizer"].wait()
                                    actions_as_text[player], duration = player_processes[player]["output_queue"].get()
                                else :
                                    actions_as_text[player] = "miss"
                                    duration = None
                        
                # Check which players are ready to continue
                players_ready = []
                for player in player_processes :
                    if actions_as_text[player].startswith("postprocessing") :
                        players_running[player] = False
                    if not self.synchronous and (actions_as_text[player].startswith("postprocessing") or actions_as_text[player] == "miss") :
                        waiter_processes[player]["input_queue"].put(True)
                    else :
                        players_ready.append(player)

                # Check for errors
                if any([actions_as_text[player].endswith("error") for player in player_processes]) and not self.continue_on_error :
                    raise Exception("A player has crashed, exiting")

                # We save the turn info if we are not postprocessing
                if not self.done :
                
                    # Save stats
                    for player in player_processes :
                        if not actions_as_text[player].startswith("preprocessing") :
                            self.stats["players"][player]["moves"][actions_as_text[player]] += 1
                            if actions_as_text[player] != "mud" :
                                self.moves_history[player].append("nothing" if actions_as_text[player] not in action_meanings.values() else actions_as_text[player])
                        if duration is not None :
                            if actions_as_text[player].startswith("preprocessing") :
                                self.stats["players"][player]["preprocessing_duration"] = duration
                            else :
                                self.stats["players"][player]["turn_durations"].append(duration)
                        self.stats["players"][player]["score"] = self.player_scores[player]
                
                    # Apply the actions
                    locations_before = self.player_locations.copy()
                    self.stats["turns"] = self.turn
                    actions = {player : list(action_meanings.keys())[list(action_meanings.values()).index(actions_as_text[player] if actions_as_text[player] in possible_actions else "nothing")] for player in player_processes}
                    self.step(actions)
                    
                    # Correct stats if we went into a wall
                    for player in player_processes :
                        if actions_as_text[player] in ["north", "west", "south", "east"] and locations_before[player] == self.player_locations[player] and not self._is_in_mud(player) :
                            self.stats["players"][player]["moves"]["wall"] += 1
                            self.stats["players"][player]["moves"][actions_as_text[player]] -= 1

                    # rendering of the maze
                    self.render()

        # Stats make no sense in case of an error
        except :
            print(traceback.format_exc(), file=sys.stderr)
            self.stats = {}

        # Cleanup
        self.close()
        
        # Return stats
        return self.stats

    #############################################################################################################################################
    #                                                                MISC METHODS                                                               #
    #############################################################################################################################################

    def get_description (self) :
        
        """
            Returns elements describing the game, that do not vary.
            
            In:
                * None.
                
            Out:
            
                * maze ............... numpy.ndarray [or] dict : int -> (dict : int -> int) ... Map of the maze, in the public format as chosen in constructor.
                * maze_width ......... int .................................................... Width of the maze in number of cells.
                * maze_height ........ int .................................................... Height of the maze in number of cells.
                * teams .............. dict : str -> list (str) ............................... Recap of the teams of players.
                * possible_actions ... list (str) ............................................. Meanings of the possible actions.
        """
        
        # Return elements describing game state
        possible_actions = list(self.get_action_meanings().values())
        return self.maze_public, self.maze_width, self.maze_height, self.teams, possible_actions

    #############################################################################################################################################

    def _coords_difference (self, i1, i2) :
        
        """
            Computes the difference between two cells for each dimension.
            
            In:
                * i1 ... int ... First cell.
                * i2 ... int ... Second cell.
                
            Out:
                * row_diff ... int ... Difference in rows.
                * col_diff ... int ... Difference in cols.
        """
        
        # Conversion
        row1, col1 = self._i_to_rc(i1)
        row2, col2 = self._i_to_rc(i2)
        row_diff = row1 - row2
        col_diff = col1 - col2
        return row_diff, col_diff
    
    #############################################################################################################################################
    
    def _rc_to_i (self, row, col) :
        
        """
            Transforms a (row, col) pair of maze coordiates (lexicographic order) in a maze index.
            
            In:
                * row ... int ... Row of the cell.
                * col ... int ... Column of the cell.
                
            Out:
                * i ... int ... Corresponding index in the adjacency matrix.
        """
        
        # Conversion
        i = row * self.maze_width + col
        return i
    
    #############################################################################################################################################

    def _i_to_rc (self, i) :
        
        """
            Transforms a maze index in a pair (row, col).
            
            In:
                * i ... int ... Index of the cell.
                
            Out:
                * row ... int ... Row of the cell.
                * col ... int ... Column of the cell.
        """
        
        # Conversion
        row = i // self.maze_width
        col = i % self.maze_width
        return row, col
    
    #############################################################################################################################################

    def _is_in_mud (self, player) :
        
        """
            Indicates if a player is in the mud.
            
            In:
                * player ... str ... Player to check.
                
            Out:
                * mud_indicator ... bool ... Boolean telling if the player is in mud.
        """
        
        # Check dictionary
        mud_indicator = self.player_muds[player]["target"] is not None
        return mud_indicator
        
    #############################################################################################################################################

    def _score_per_team (self) :
        
        """
            Returns the score per team.
            
            In:
                * None.
                
            Out:
                * scores ... dict : str -> float ... Dictionary of scores.
        """
        
        # Aggregate players of the team
        scores = {team : round(sum([self.player_scores[player] for player in self.teams[team]]), 5) for team in self.teams}
        return scores

#####################################################################################################################################################
#####################################################################################################################################################