##########################################################
# This module contains the main Mancala game which maintains the board, score, and players.  
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on
# 
# CSC 384 Assignment 2
# version 2.0
##########################################################

from utils import *


# Reading initial board from file
def read_initial_board(init_board):
    if isinstance(init_board, list):
        return read_initial_board_list(init_board)
    elif isinstance(init_board, str):
        return read_initial_board_file(init_board)
    else:
        raise TypeError("init_board should be either a string filename or a list for the board")


def read_initial_board_list(init_board):
    """
    Creates a new board from the given list. The list should be 3 elements where

    Element 1: List representing # stones in Top Player's pockets
    Element 2: List representing # stones in Bottom Player's pockets
    Element 3: List representing Mancalas [TOP, BOTTOM]
    """
    return len(init_board[0]), Board([init_board[0], init_board[1]], init_board[2])


def read_initial_board_file(filename):
    """
    Reads the starting state from a file. File should be in this format:

    Line 1: # stones in Top Player's pockets
    Line 2: # stones in Bottom Player's pockets
    Line 3: # stones in Top Player's mancala
    Line 4: # stones in Bottom Player's mancala

    You can also check the example file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    top_stones = [int(x.strip()) for x in lines[0].split(",")]
    bottom_stones = [int(x.strip()) for x in lines[1].split(",")]
    top_mancala = int(lines[2].strip())
    bottom_mancala = int(lines[3].strip())

    assert len(top_stones) == len(bottom_stones)
    dimension = len(top_stones)

    pockets = [top_stones, bottom_stones]
    mancalas = [top_mancala, bottom_mancala]
    return dimension, Board(pockets, mancalas)


def create_initial_board(dimension=None, initial_board=None):
    assert dimension is not None or initial_board is not None
    if initial_board is not None:
        # if dimension is not None:
        #     print("Initializing game from", initial_board, ", dimension parameter ignored.")
        dimension, board = read_initial_board(initial_board)
    elif dimension is not None:
        board = Board([[4] * dimension, [4] * dimension], [0, 0])
    return board


class Board(object):
    """
    Board class that represents a Mancala board
    """

    def __init__(self, pockets, mancalas):
        """
        Create a Mancala game board.
        """
        self.dimension = len(pockets[TOP])
        self.pockets = pockets
        self.mancalas = mancalas

    def __eq__(self, other):
        return self.pockets == other.pockets and self.mancalas == other.mancalas

    def __hash__(self):
        self.pockets = tuple(tuple(sublist) for sublist in self.pockets)
        self.mancalas = tuple(self.mancalas)
        return hash((self.pockets, self.mancalas))
    
    def draw_board(self, return_str=False):
        """
        Print the Mancala game board in a readable format.
        """
        dim = self.dimension
        mancalas = self.mancalas
        pockets = self.pockets

        max_num = max(max(mancalas), max(pockets[TOP]), max(pockets[BOTTOM]))

        grid_size = 3 + len(str(max_num))

        topper_row = " " * (grid_size - 1)
        for i in range(len(pockets[TOP])):
            extra_space = grid_size - 2 - len(str(i+1))
            topper_row += " T" + (str(i + 1)) + " " * extra_space

        top_row = " " * (grid_size - 1)
        for i in pockets[TOP]:
            extra_space = grid_size - 3 - len(str(i))
            top_row += "| " + (" " * extra_space) + str(i) + " "
        top_row += "|"

        middle_row = str(mancalas[TOP])
        extra_space = grid_size - 1 - len(str(mancalas[TOP]))
        middle_row += " " * extra_space + "|"
        middle_row += " " * (len(top_row) - grid_size - 1)
        extra_space = grid_size - 1 - len(str(mancalas[BOTTOM]))
        middle_row += "|" + " " * extra_space + str(mancalas[BOTTOM])

        bottom_row = " " * (grid_size - 1)
        for i in pockets[BOTTOM]:
            extra_space = grid_size - 3 - len(str(i))
            bottom_row += "| " + (" " * extra_space) + str(i) + " "
        bottom_row += "|"

        bottomer_row = " " * (grid_size - 1)
        for i in range(len(pockets[TOP])):
            num = len(pockets[TOP]) - i
            extra_space = grid_size - 2 - len(str(num))
            bottomer_row += " B" + (str(num)) + " " * extra_space

        msg = topper_row
        msg += "\n" + "-" * (dim + 2) * grid_size
        msg += "\n" + top_row
        msg += "\n" + middle_row
        msg += "\n" + bottom_row
        msg += "\n" + "-" * (dim + 2) * grid_size
        msg += "\n" + bottomer_row
        
        if return_str:
            return msg
        else:
            print(msg)
    
    def get_board_list(self):
        """
        Get the board in the format of a list.
        """
        data = [", ".join([str(x) for x in self.pockets[0]]), 
                ", ".join([str(x) for x in self.pockets[1]]), 
                ", ".join([str(x) for x in self.mancalas])]
        return data
    
    def get_possible_moves(self, player):
        """
        Return a list of all possible indices (representing pockets) that the 
        current player can play on the current board.
        """
        moves = []
        for j in range(self.dimension):
            #if the pocket has at least one piece
            if self.pockets[player][j] > 0: 
                moves.append(j)
        return moves
   

def get_winner(board):
    """
    Returns the player number of the player with more stones in their mancala.
    """
    if board.mancalas[TOP] > board.mancalas[BOTTOM]:
        return "Top Player"
    elif board.mancalas[TOP] < board.mancalas[BOTTOM]:
        return "Bottom Player"
    else:
        return "Tie"


def play_move(board, player, move):
    """
    Play a move on the current board. 
    :param board: the current board
    :param player: the player to move.
    :param move: the move to perform. the index of the pocket.
    """  
    side = player

    new_board = []
    for row in board.pockets: 
        new_board.append(list(row[:]))
    new_mancalas = [board.mancalas[TOP], board.mancalas[BOTTOM]]

    stone_count = board.pockets[side][move] # find the number of stones in the pocket
    new_board[side][move] = 0 # set to 0

    if (player == BOTTOM):
        direction = True
        ind = move + 1
    else:
        direction = False
        ind = move - 1

    while stone_count > 0: #deposit stones around the board
        # we are at the end of the board
        if ind > (len(board.pockets[side])-1) or ind < 0:
            # swap side and change direction
            side = TOP if side == BOTTOM else BOTTOM 
            direction = not direction 

            #if we are at the end of the board, 
            #deposit stone in a mancala before we continue
            if ind > (len(board.pockets[side]) - 1): 
                if player == BOTTOM:
                    stone_count -= 1
                    new_mancalas[player] += 1
                ind = len(board.pockets[side])-1
            else:
                if player == TOP:
                    stone_count -= 1
                    new_mancalas[player] += 1
                ind = 0

        # ran out of stones
        if stone_count == 0: 
            break

        #but if not, put a stone in a pocket and decrement stone count
        new_board[side][ind] = new_board[side][ind] + 1
        stone_count -= 1

        #do we have a capture?
        if stone_count == 0 and new_board[side][ind] == 1 and side == player: 
            captures = new_board[get_opponent(side)][ind] #if yes capture stones in the opposite pit
            new_board[get_opponent(side)][ind] = 0
            new_mancalas[player] += captures

        if direction: 
            ind += 1
        else: 
            ind -= 1

    # make rows tuples
    final_pockets = []
    for row in new_board: 
        final_pockets.append(tuple(row))
    final_board = Board(final_pockets, new_mancalas)

    # end the game if done
    if sum(new_board[TOP]) == 0 or sum(new_board[BOTTOM]) == 0:
        final_board = end_game(final_board)

    return final_board


def end_game(board):
    """
    Call this function at the end of the game to move all remaining stones
    on the board.
    Opponent just moved and should have no moves left, only current player
    has stones to move.
    Modifies the input board.
    """
    new_board = []
    new_mancalas = board.mancalas.copy()
    for row in board.pockets: 
        new_board.append(list(row[:]))

    for player in range(len(board.pockets)):
        value = 0
        for j in range(len(new_board[player])):
            value +=  new_board[player][j]
            new_board[player][j] = 0
        new_mancalas[player] += value

    final = []
    for row in new_board: 
        final.append(tuple(row))

    board.pockets = tuple(final)
    board.mancalas = tuple(new_mancalas)
    return board


# Player Classes
class Player(object):
    def __init__(self, player, name="Human"):
        """
        Initialize a player of the game.

        player_num: TOP or BOTTOM representing which player they are.
        name: a cool name for this player (give a string) 
        """
        self.name = name
        self.player = player

    def get_move(self, board, player):
        pass


class AiPlayerInterface(Player):
    def __init__(self, player, algorithm, limit, optimizations, heuristic):
        """
        Initializes an AI player that uses minimax or alphabeta.

        player    str: for notation 
        algorithm str: [random, minimax, alphabeta]
        limit     int: >0 -> using depth limit
        optimizations  bool: whether to use additional optimizations
        heuristic str: [basic, advanced] 
        """
        super().__init__(player, algorithm.__name__)
        self.algorithm = algorithm
        self.hfunc = heuristic

        self.limit = limit
        if optimizations:
            self.optimizations = {}
            self.optimizations["cache"] = {}
        else:
            self.optimizations = None
        
    def get_move(self, board, player):
        move, value = self.algorithm(board, player, self.limit, self.optimizations, self.hfunc)
        return move, value
