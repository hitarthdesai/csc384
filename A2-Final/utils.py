###############################################################################
# This file contains helper functions and the heuristic functions
# for our AI agents to play the Mancala game.
#
# CSC 384 Assignment 2 Starter Code
# version 2.0
###############################################################################

import sys

###############################################################################
### DO NOT MODIFY THE CODE BELOW

### Global Constants ###
TOP = 0
BOTTOM = 1
TIMEOUT = 60

### Errors ###
class InvalidMoveError(RuntimeError):
    pass

class AiTimeoutError(RuntimeError):
    pass

### Functions ###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_opponent(player):
    if player == BOTTOM:
        return TOP
    return BOTTOM

### DO NOT MODIFY THE CODE ABOVE
###############################################################################


def heuristic_basic(board, player):
    """
    Compute the heuristic value of the current board for the current player 
    based on the basic heuristic function.

    :param board: the current board.
    :param player: the current player.
    :return: an estimated utility of the current board for the current player.
    """

    opponent = get_opponent(player)

    return board.mancalas[player] - board.mancalas[opponent]


def heuristic_advanced(board, player): 
    """
    Compute the heuristic value of the current board for the current player
    based on the advanced heuristic function.

    :param board: the current board object.
    :param player: the current player.
    :return: an estimated heuristic value of the current board for the current player.
    """
    
    opponent = get_opponent(player)
    own_pockets = board.pockets[player]
    opp_pockets = board.pockets[opponent]
    pockets_count = len(own_pockets)

    # prefer placing stones in own mancala
    basic = board.mancalas[player] - board.mancalas[opponent]

    # prefer to have more stones in own
    # pockets, and less in opponent's pockets
    stone_diff = sum(own_pockets) - sum(opp_pockets)

    # prefer to have more potential captures
    capture_potential = 0
    for i, stones in enumerate(own_pockets):
        if stones == 0:
            continue

        # idea is we hypothetically move all stones back to the first pocket
        # by adding the index of the pocket to the number of stones
        # then we modulate it by the total number of pockets plus our mancala
        # to find the ending pocket
        end = (i + stones) % (2 * pockets_count + 1)

        # if ending pocket is on the player's side, and it's empty, then we're golden
        if end < len(own_pockets) and own_pockets[end] == 0:
            capture_potential = max(capture_potential, opp_pockets[end])


    # Weighted sum of all factors
    return (
        3 * basic +
        2 * stone_diff +
        5 * capture_potential
    )