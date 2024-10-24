###############################################################################
# This is an AI that randomly chooses a legal move. 
# 
# Play against this AI to get familiar with the game. 
# This is not the best AI opponent, so don't get flattered if you win! 
# You can also have your AI compete against your AI to test its performance. :D
#
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on
#
# CSC 384 Assignment 2
# version 2.0
###############################################################################

import random
import time
from wrapt_timeout_decorator import timeout

from utils import *


def select_move(board, player):
    """
    Given a board and a player, decide on a move. 
    The return value is an integer i.
    """

    # We just get a list of all permitted moves in this state and select a random one!
    i = None
    moves = board.get_possible_moves(player)
    if (len(moves) > 0):
        i = random.choice(moves)

    time.sleep(0.5) # Delay, so the random agent doesn't look as simple as it really is.  
    return i


@timeout(TIMEOUT, timeout_exception=AiTimeoutError)
def run_random(curr_board, player, limit, cache, hfunc):
    move = select_move(curr_board, player)
    value = None
    return move, value
