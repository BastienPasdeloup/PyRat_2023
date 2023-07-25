#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This script compares the needed number of actions to complete a game.
    It performs two analyses: a quick average analysis and a formal Mann-Whitney U test.
    Finally, it generates a plot to compare the results.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports
import sys
import matplotlib.pyplot as pyplot
import scipy.stats
import os
import numpy
import types
import tqdm

# Previously developed functions
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "programs"))
import random_1
import random_2
import random_3

#####################################################################################################################################################
############################################################### CONSTANTS & VARIABLES ###############################################################
#####################################################################################################################################################

"""
    Number of games to consider per program.
"""

NB_GAMES = 500

#####################################################################################################################################################

"""
    Games configuration.
"""

MAZE_WIDTH = 15
MAZE_HEIGHT = 10
MUD_PERCENTAGE = 10.0
WALL_PERCENTAGE = 40.0
MUD_RANGE = [2, 10]
NB_CHEESE = 1

#####################################################################################################################################################

"""
    List here the programs you want to compare.
"""

PROGRAMS = [random_1, random_2, random_3]

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def run_one_game ( seed:    int,
                   program: types.ModuleType
                 ) ->       Dict[str, Any]:

    """
        This function runs a PyRat game, with no GUI, for a given seed and program, and returns the obtained stats.
        In:
            * seed:    Random seed used to create the game.
            * program: Program to use in that game.
        Out:
            * stats: Statistics output at the end of the game.
    """
    
    # Map the functions to the character
    players = [{"name": program.__name__, "preprocessing_function": program.preprocessing if "preprocessing" in dir(program) else None, "turn_function": program.turn}]

    # Customize the game elements
    config = {"maze_width": MAZE_WIDTH,
              "maze_height": MAZE_HEIGHT,
              "mud_percentage": MUD_PERCENTAGE,
              "mud_range": MUD_RANGE,
              "wall_percentage": WALL_PERCENTAGE,
              "nb_cheese": NB_CHEESE,
              "render_mode": "no_rendering",
              "preprocessing_time": 0.0,
              "turn_time": 0.0,
              "synchronous": True,
              "random_seed": seed}
        
    # Start the game
    game = PyRat(players, **config)
    stats = game.start()
    return stats
    
#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Run multiple games for each player
    results = {program.__name__: {"turns": [], "preprocessing_duration": [], "turn_durations": []} for program in PROGRAMS}
    for program in tqdm.tqdm(PROGRAMS, desc="Program", position=0, leave=False):
        for seed in tqdm.tqdm(range(NB_GAMES), desc="Game", position=1, leave=False):
        
            # Here we are interested in the number of turns needed to complete the game, as well as the time it takes 
            stats = run_one_game(seed, program)
            results[program.__name__]["turns"].append(stats["turns"])
            results[program.__name__]["preprocessing_duration"].append(stats["players"][program.__name__]["preprocessing_duration"])
            results[program.__name__]["turn_durations"] += stats["players"][program.__name__]["turn_durations"]

    # Show results briefly
    print("#" * 20)
    print("#  Quick analysis  #")
    print("#" * 20)
    for program in PROGRAMS:
        print("Program", program.__name__, "requires on average", numpy.mean(results[program.__name__]["turns"]), "actions, with an average preprocessing duration of", numpy.mean(results[program.__name__]["preprocessing_duration"]), "seconds, and an average turn duration of", numpy.mean(results[program.__name__]["turn_durations"]), "seconds")

    # More formal statistics to check if these curves are statistically significant
    print("#" * 21)
    print("#  Formal analysis  #")
    print("#" * 21)
    for i in range(len(PROGRAMS)):
        for j in range(i + 1, len(PROGRAMS)):
            test_result = scipy.stats.mannwhitneyu(results[PROGRAMS[i].__name__]["turns"], results[PROGRAMS[j].__name__]["turns"], alternative="two-sided")
            print("Mann-Whitney U test between turns of program", PROGRAMS[i].__name__, "and of program", PROGRAMS[j].__name__, ":", test_result)

    # Visualization of histograms of numbers of turns taken per program
    max_turn = max([max(results[program.__name__]["turns"]) for program in PROGRAMS])
    pyplot.figure(figsize=(20, 10))
    for program in PROGRAMS:
        games_completed_per_turn = [0] + [sum(map(lambda turn: turn <= i, results[program.__name__]["turns"])) * 100 / NB_GAMES for i in range(max_turn)]
        pyplot.plot(range(max_turn + 1), games_completed_per_turn, label=program.__name__)
    pyplot.title("Comparison of turns needed to complete all %d games" % (NB_GAMES))
    pyplot.xlabel("Turns")
    pyplot.ylabel("% of games completed")
    pyplot.xscale("log")
    pyplot.legend()
    pyplot.show()
    
#####################################################################################################################################################
#####################################################################################################################################################