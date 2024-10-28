###############################################################################
# This file implements various alpha-beta pruning agents.
#
# CSC 384 Assignment 2 Starter Code
# version 2.0
###############################################################################
from wrapt_timeout_decorator import timeout

from mancala_game import play_move
from utils import *


def alphabeta_max_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MAX player.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

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
        _, value = alphabeta_min_basic(new_board, get_opponent(curr_player), alpha, beta, heuristic_func)
        if value > best_value:
            best_value, best_move = value, move

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_min_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MIN player.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

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
        _, value = alphabeta_max_basic(new_board, get_opponent(curr_player), alpha, beta, heuristic_func)
        if value < best_value:
            best_value, best_move = value, move

            if value < beta:
                beta = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_max_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MAX player up to the given depth limit.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its estimated minimax value.
    """

    if depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    best_value, best_move = float('-inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = alphabeta_min_limit(new_board, get_opponent(curr_player), alpha, beta, heuristic_func, depth_limit - 1)
        if value > best_value:
            best_value, best_move = value, move

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_min_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MIN player up to the given depth limit.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its estimated minimax value.
    """

    if depth_limit == 0:
        return None, heuristic_func(board, get_opponent(curr_player))

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    best_value, best_move = float('inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = alphabeta_max_limit(new_board, get_opponent(curr_player), alpha, beta, heuristic_func, depth_limit)
        if value < best_value:
            best_value, best_move = value, move

            if value < beta:
                beta = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_max_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Search for MAX player 
    up to the given depth limit and with additional optimizations.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its estimated minimax value.
    """

    if depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    cache = optimizations['cache']
    best_value, best_move = float('-inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)

        value = None
        if new_board in cache:
            depth, cached_value = cache[new_board]
            if depth <= depth_limit - 1:
                value = cached_value

        if value is None:
            _, value = alphabeta_min_limit_opt(new_board, get_opponent(curr_player), alpha, beta, heuristic_func, depth_limit - 1, optimizations)
            cache[new_board] = depth_limit - 1, value

        if value > best_value:
            best_value, best_move = value, move

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

    return best_move, best_value

def alphabeta_min_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Pruning for MIN player 
    up to the given depth limit and with additional optimizations.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its estimated minimax value.
    """

    if depth_limit == 0:
        return None, heuristic_func(board, get_opponent(curr_player))

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    cache = optimizations['cache']
    best_value, best_move = float('inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)

        value = None
        if new_board in cache:
            depth, cached_value = cache[new_board]
            if depth <= depth_limit - 1:
                value = cached_value

        if value is None:
            _, value = alphabeta_max_limit_opt(new_board, get_opponent(curr_player), alpha, beta, heuristic_func, depth_limit - 1, optimizations)
            cache[new_board] = depth_limit - 1, value

        if value < best_value:
            best_value, best_move = value, move

            if value < beta:
                beta = value

            if alpha >= beta:
                break

    return best_move, best_value

    
###############################################################################
## DO NOT MODIFY THE CODE BELOW.
###############################################################################

@timeout(TIMEOUT, timeout_exception=AiTimeoutError)
def run_alphabeta(curr_board, player, limit, optimizations, hfunc):
    if optimizations is not None:
        opt = True
    else:
        opt = False

    alpha = float("-Inf")
    beta = float("Inf")
    if opt:
        move, value = alphabeta_max_limit_opt(curr_board, player, alpha, beta, hfunc, limit, optimizations)
    elif limit >= 0:
        move, value = alphabeta_max_limit(curr_board, player, alpha, beta, hfunc, limit)
    else:
        move, value = alphabeta_max_basic(curr_board, player, alpha, beta, hfunc)
    
    return move, value

