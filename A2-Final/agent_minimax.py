###############################################################################
# This file implements various minimax search agents.
#
# CSC 384 Assignment 2 Starter Code
# version 2.0
###############################################################################
from wrapt_timeout_decorator import timeout

from mancala_game import play_move
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

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    best_value, best_move = float('-inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = minimax_min_basic(new_board, get_opponent(curr_player), heuristic_func)
        if value > best_value:
            best_value = value
            best_move = move

    return best_move, best_value


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

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    best_value, best_move = float('inf'), None
    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = minimax_max_basic(new_board, get_opponent(curr_player), heuristic_func)
        if value < best_value:
            best_value = value
            best_move = move

    return best_move, best_value


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

    if depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    best_value, best_move = float('-inf'), None
    depth_limit -= 1

    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = minimax_min_limit(new_board, get_opponent(curr_player), heuristic_func, depth_limit)
        if value > best_value:
            best_value = value
            best_move = move

    return best_move, best_value

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

    if depth_limit == 0:
        return None, heuristic_func(board, get_opponent(curr_player))

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    best_value, best_move = float('inf'), None
    depth_limit -= 1

    for move in moves:
        new_board = play_move(board, curr_player, move)
        _, value = minimax_max_limit(new_board, get_opponent(curr_player), heuristic_func, depth_limit)
        if value < best_value:
            best_value = value
            best_move = move

    return best_move, best_value


def minimax_max_limit_opt(board, curr_player, heuristic_func, depth_limit, optimizations):
    """
    Perform Minimax Search for MAX player up to the given depth limit with the option of caching states.
    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its minimmax value estimated by our heuristic function.
    """

    cache = optimizations['cache']
    cache_key = hash(board)
    try:
        cached_depth, cached_result = cache[cache_key]
        if cached_depth <= depth_limit:
            return cached_result
    except KeyError:
        pass

    if depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, curr_player)

    best_value, best_move = float('-inf'), None
    depth_limit -= 1

    for move in moves:
        new_board = play_move(board, curr_player, move)
        result = minimax_min_limit_opt(new_board, get_opponent(curr_player), heuristic_func, depth_limit, optimizations)
        _move, value = result
        if value > best_value:
            best_value = value
            best_move = move
        
        if _move is not None:
            ck = hash(new_board)
            cache[ck] = depth_limit, result

    result = best_move, best_value
    cache[cache_key] = depth_limit + 1, result
    return result

def minimax_min_limit_opt(board, curr_player, heuristic_func, depth_limit, optimizations):
    """
    Perform Minimax Search for MIN player up to the given depth limit with the option of caching states.
    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its minimmax value estimated by our heuristic function.
    """

    cache = optimizations['cache']
    cache_key = hash(board)
    try:
        cached_depth, cached_result = cache[cache_key]
        if cached_depth <= depth_limit:
            return cached_result
    except KeyError:
        pass

    if depth_limit == 0:
        return None, heuristic_func(board, get_opponent(curr_player))

    moves = board.get_possible_moves(curr_player)
    if len(moves) == 0:
        return None, heuristic_func(board, get_opponent(curr_player))
    
    best_value, best_move = float('inf'), None
    depth_limit -= 1

    for move in moves:
        new_board = play_move(board, curr_player, move)
        result = minimax_max_limit_opt(new_board, get_opponent(curr_player), heuristic_func, depth_limit, optimizations)
        _move, value = result
        if value < best_value:
            best_value = value
            best_move = move
            
        if _move is not None:
            ck = hash(new_board)
            cache[ck] = depth_limit, result

    result = best_move, best_value
    cache[cache_key] = depth_limit + 1, result
    return result


###############################################################################
## DO NOT MODIFY THE CODE BELOW.
###############################################################################

@timeout(TIMEOUT, timeout_exception=AiTimeoutError)
def run_minimax(curr_board, player, limit, optimizations, hfunc):
    if optimizations is not None:
        opt = True
    else:
        opt = False

    if opt:
        move, value = minimax_max_limit_opt(curr_board, player, hfunc, limit, optimizations)
    elif limit >= 0:
        move, value = minimax_max_limit(curr_board, player, hfunc, limit)
    else:
        move, value = minimax_max_basic(curr_board, player, hfunc)
    
    return move, value
