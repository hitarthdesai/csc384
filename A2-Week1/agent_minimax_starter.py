###############################################################################
# This file implements various minimax search agents.
#
# CSC 384 Assignment 2 Starter Code
# version 1.0
###############################################################################
from mancala_game import Board, play_move
from utils import *


def minimax_max_basic(board, curr_player, heuristic_func):
    """
    Perform Minimax Search for MAX player.

    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function

    :return the best move and its minimax value according to minimax search.
    """

    raise NotImplementedError


def minimax_min_basic(board, curr_player, heuristic_func):
    """
    Perform Minimax Search for MIN player.

    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function

    :return the best move and its minimax value according to minimax search.
    """

    raise NotImplementedError


def minimax_max_limit(board, curr_player, heuristic_func, depth_limit):
    """
    Perform Minimax Search for MAX player up to the given depth limit.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit

    :return the best move and its minimmax value estimated by our heuristic function.
    """

    raise NotImplementedError


def minimax_min_limit(board, curr_player, heuristic_func, depth_limit):
    """
    Perform Minimax Search for MIN player  up to the given depth limit.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit

    :return the best move and its minimmax value estimated by our heuristic function.
    """

    raise NotImplementedError


def minimax_max_limit_opt(board, curr_player, heuristic_func, depth_limit, optimizations):
    """
    Perform Minimax Search for MAX player up to the given depth limit 
    with the option of using additional optimizations.

    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary in which we keep data structures 
        for additional optimizations. It contains a cache to be used for caching.

    :return the best move and its minimmax value estimated by our heuristic function.
    """

    raise NotImplementedError


def minimax_min_limit_opt(board, curr_player, heuristic_func, depth_limit, optimizations):
    """
    Perform Minimax Search for MIN player up to the given depth limit 
    with the option of using additional optimizations.
    
    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary in which we keep data structures 
        for additional optimizations. It contains a cache to be used for caching.

    :return the best move and its minimmax value estimated by our heuristic function.
    """

    raise NotImplementedError


###############################################################################
## DO NOT MODIFY THE CODE BELOW.
###############################################################################

def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Mancala AI")  # First line is the name of this AI
    arguments = input().split(",")

    player  = int(arguments[0]) # Player color
    limit   = int(arguments[1]) # Depth limit
    opt     = int(arguments[2]) # Optimizations
    hfunc   = int(arguments[3]) # Heuristic Function

    optimizations = {}

    if (opt == 1): 
        opt = True
        optimizations["cache"] = {}
    else: 
        opt = False

    eprint("Running MINIMAX")


    if limit == -1:
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is", limit)

    if opt:
        eprint("Optimizations are ON")
    else:
        eprint("Optimizations are OFF")

    if hfunc == 0:
        eprint("Using heuristic_basic")
        heuristic_func = heuristic_basic
    else:
        eprint("Using heuristic_advanced")
        heuristic_func = heuristic_advanced

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":  # Game is over.
            print()
        else:
            pockets = eval(input())  # Read in the input and turn it into an object
            mancalas = eval(input())  # Read in the input and turn it into an object
            board = Board(pockets, mancalas)

            # Select the move and send it to the manager
            if opt:
                move, value = minimax_max_limit_opt(board, player, heuristic_func, limit, optimizations)
            elif limit >= 0:
                move, value = minimax_max_limit(board, player, heuristic_func, limit)
            else:
                move, value = minimax_max_basic(board, player, heuristic_func)
            print("{},{}".format(move, value))


if __name__ == "__main__":
    run_ai()
