###############################################################################
# Play a game of Mancala on the command line.
# Author: Shirley Wang
# 
# CSC 384 Assignment 2
# version 2.0
###############################################################################

import argparse
import sys
import random
from datetime import datetime

from agent_alphabeta import run_alphabeta
from agent_minimax import run_minimax
from agent_random import run_random
from mancala_game import *
from utils import *

class MancalaCommandLine(object):

    def __init__(self, dimension, initial_board, player1, player2):
        self.board = create_initial_board(dimension, initial_board)
        self.players = [player1, player2]
        self.curr_player = TOP

    def user_input_move(self):
        player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"
        prompt = "" + player + " Please Input Move: "
        text = input(prompt).strip()

        if text[0] == "T" or text[0] == "B":
            # in case students also input the row in the move
            text = text[1:]

        try:
            move_num = int(text)
        except:
            raise InvalidMoveError
        
        if self.curr_player == TOP:
            move_num = move_num - 1
        else:
            move_num = self.board.dimension - move_num

        if move_num not in self.board.get_possible_moves(self.curr_player):
            raise InvalidMoveError

        self.board = play_move(self.board, self.curr_player, move_num)
        self.curr_player = get_opponent(self.curr_player)
    
    def ai_move(self):
        player_obj = self.players[self.curr_player]
        move, value = player_obj.get_move(self.board, self.curr_player)
        player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"

        if move is not None: 
            move_view = int(move)
            if self.curr_player == BOTTOM:
                move_view = self.board.dimension - move_view
            else:
                move_view = move_view + 1

            # print("{}: {} ({})".format(player, move_view, move))
            print("{} Move: {}".format(player, move_view))
            print("")
            self.board = play_move(self.board, self.curr_player, move)
            self.curr_player = get_opponent(self.curr_player)
        else:
            print("Returned None for move, this shouldn't be possible")
            raise InvalidMoveError

    def run(self):
        self.board.draw_board()

        # while game is not over
        possible_moves = self.board.get_possible_moves(self.curr_player)
        other_moves = self.board.get_possible_moves(get_opponent(self.curr_player))
        while len(possible_moves) > 0 and len(other_moves) > 0:
            # run game
            player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"
            print("")
            print("Turn:", player)

            success = True
            if isinstance(self.players[self.curr_player], AiPlayerInterface):
                # call AI to make a move, if timeout then end game
                try:
                    self.ai_move()
                except AiTimeoutError:
                    print("{} lost due to timeout".format(self.players[self.curr_player].name))
                    success = False
                    break
            else:
                # get move from user, continue if incorrect
                try:
                    self.user_input_move()
                except InvalidMoveError:
                    print("Invalid Move")
                    success = False

            if success:
                print("")
                self.board.draw_board()

            possible_moves = self.board.get_possible_moves(self.curr_player)
            other_moves = self.board.get_possible_moves(get_opponent(self.curr_player))

        winner = get_winner(self.board)
        print("GAME OVER: winner is {}".format(winner))
    
    def save_board(self, filename):
        data = [
            ", ".join([str(x) for x in self.board.pockets[0]]) + "\n",
            ", ".join([str(x) for x in self.board.pockets[1]]) + "\n",
            str(self.board.mancalas[0]) + "\n",
            str(self.board.mancalas[1]) + "\n"
        ]

        with open(filename, "w") as f:
            f.writelines(data)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MancalaCmdline",
        description="Run this code to start a game of mancala"
    )

    parser.add_argument("-d", "--dimension", type=int, default=4,
                        help="Dimension of mancala board. Default is 4.")
    parser.add_argument("-t", "--agentTop", type=str,
                        help="Algorithm for the top player. Options are [random, minimax, alphabeta]. If not specified, user inputs moves.")
    parser.add_argument("-b", "--agentBottom", type=str,
                        help="Algorithm for the bottom player. Options are [random, minimax, alphabeta]. If not specified, user inputs moves.")
    parser.add_argument("-ht", "--heuristicTop", type=str, default="basic",
                        help="Heuristic for top player to use. Options are [basic, advanced]. Default is basic.")
    parser.add_argument("-hb", "--heuristicBottom", type=str, default="basic",
                        help="Heuristic for bottom player to use. Options are [basic, advanced]. Default is basic.")

    parser.add_argument("-l", "--limit", type=int, default=-1,
                        help="(Optional) Depth limit for agent to use. Default is -1, which means no depth limit.")

    parser.add_argument("-i", "--initialBoard", type=str,
                        help="File storing the initial state of the board. Overwrites dimension.")

    parser.add_argument("-o", "--optimizations", action="store_true",
                        help="Use flag if agent should use additional optimizations.")

    args = parser.parse_args()    
    return args

def get_algorithm(algorithm):
    if algorithm == "minimax":
        return run_minimax
    elif algorithm == "alphabeta":
        return run_alphabeta
    elif algorithm == "random":
        return run_random
    else:
        raise TypeError("Algorithm not recognized. Options are [random, minimax, alphabeta].")
    
def get_heuristic(heuristic):
    if heuristic == "basic":
        return heuristic_basic
    elif heuristic == "advanced":
        return heuristic_advanced
    else:
        raise TypeError("Heuristic not recognized. Options are [basic, advanced].")

def main():
    random.seed(datetime.now().timestamp())
    args = parse_args()

    if args.dimension is not None and args.dimension <= 0 and args.initialBoard is None: #if no dimension provided
        print('Please provide a valid dimension or a valid initial board.')
        sys.exit(2)
    
    if args.agentTop != None:
        p1 = AiPlayerInterface(TOP, get_algorithm(args.agentTop), args.limit, args.optimizations, get_heuristic(args.heuristicTop))
    else:
        p1 = Player(TOP)

    if args.agentBottom != None:
        p2 = AiPlayerInterface(BOTTOM, get_algorithm(args.agentBottom), args.limit, args.optimizations, get_heuristic(args.heuristicBottom))
    else:
        p2 = Player(BOTTOM)
        
    cmdline = MancalaCommandLine(args.dimension, args.initialBoard, p1, p2) 
    cmdline.run()


if __name__ == "__main__":
    main()
