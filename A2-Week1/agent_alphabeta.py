###############################################################################
# This file implements various alpha-beta search agents.
#
# CSC 384 Assignment 2 Starter Code
# version 1.0
###############################################################################
from mancala_game import Board, play_move
from utils import *

def alphabeta_max_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MAX player.

    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function

    :return the best move and its minimax value.
    """

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    best_value, best_move = float('-inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = alphabeta_min_basic(new_board, get_opponent(curr_player), heuristic_func)
        if value > best_value:
            best_value = value
            best_move = move

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_min_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MIN player.

    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function

    :return the best move and its minimax value.
    """

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    best_value, best_move = float('inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = alphabeta_max_basic(new_board, get_opponent(curr_player), heuristic_func)
        if value < best_value:
            best_value = value
            best_move = move

            if value < beta:
                beta = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_max_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MAX player up to the given depth limit.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit

    :return the best move and its estimated minimax value.
    """

    raise NotImplementedError

def alphabeta_min_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MIN player up to the given depth limit.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit

    :return the best move and its estimated minimax value.
    """

    raise NotImplementedError

def alphabeta_max_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Search for MAX player up to the given depth limit 
    with the option of using additional optimizations.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary in which we keep data structures 
        for additional optimizations. It contains a cache to be used for caching.
 
    :return the best move and its estimated minimax value.
    """

    raise NotImplementedError

def alphabeta_min_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Search for MIN player up to the given depth limit 
    with the option of using additional optimizations.

    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary in which we keep data structures 
        for additional optimizations. It contains a cache to be used for caching.

    :return the best move and its estimated minimax value.
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

    eprint("Running Alpha-Beta Search")

    if limit == -1:
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

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
            alpha = float("-Inf")
            beta = float("Inf")
            if opt:
                move, value = alphabeta_max_limit_opt(board, player, alpha, beta, 
                    heuristic_func, limit, optimizations)

            elif limit >= 0:
                move, value = alphabeta_max_limit(board, player, alpha, beta, 
                    heuristic_func, limit)

            else:
                move, value = alphabeta_max_basic(board, player, alpha, beta, 
                    heuristic_func)

            print("{},{}".format(move, value))


if __name__ == "__main__":
    run_ai()
