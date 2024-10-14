###############################################################################
# This module contains a simple graphical user interface for Mancala. 
#
# Thanks to Daniel Bauer, Columbia University, for a version of Othello that this was based on.
# Modified by Shirley Wang.
#
# CSC 384 Assignment 2
# version 2.0
###############################################################################

import argparse
import sys
import random
from datetime import datetime

from tkinter import *
from tkinter import scrolledtext

from agent_alphabeta import run_alphabeta
from agent_minimax import run_minimax
from agent_random import run_random
from mancala_game import *
from utils import *

class MancalaGui(object):

    def __init__(self, dimension, initial_board, player1, player2):
        self.board = create_initial_board(dimension, initial_board)
        self.players = [player1, player2]
        self.curr_player = TOP

        # GUI details 
        self.height = 2  #2 sides to the board
        self.width = self.board.dimension #pit count
        
        self.offset = 3
        self.cell_size = 100
        self.stone_size = 10

        root = Tk()
        root.wm_title("Mancala")
        root.lift()
        root.attributes("-topmost", True)
        self.root = root
        self.canvas = Canvas(root,height = self.cell_size * (self.height+1) + self.offset,width = self.cell_size * (self.width+2) + self.offset)
        self.move_label = Label(root)
        self.score_label = Label(root)
        self.text = scrolledtext.ScrolledText(root, width=70, height=10)
        self.move_label.pack(side="top")
        self.score_label.pack(side="top")
        self.canvas.pack()
        self.text.pack()
        self.draw_board()

    def get_position(self, x, y):
        i = (x -self.offset) // self.cell_size
        j = (y -self.offset) // self.cell_size
        return i,j
    
    def shutdown(self, text):
        self.move_label["text"] = text 
        self.root.unbind("<Button-1>")

    def mouse_pressed(self, event):
        """
        Correlated to the human player clicking to make moves
        """
        # get the human move
        i, j = self.get_position(event.x, event.y)
        player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"
        self.log("{}: {},{}".format(player, i-1, j))

        if j != self.curr_player:
            self.log("Invalid move. {},{}".format(i,j))
            raise InvalidMoveError("Invalid move: Not the current player.")

        # play move and display
        self.board = play_move(self.board, self.curr_player, i - 1)
        self.curr_player = get_opponent(self.curr_player)
        self.draw_board()

        # check if game is over
        possible_moves = self.board.get_possible_moves(self.curr_player)
        if not possible_moves:
            winner = get_winner(self.board)
            print('{} {} {}\n'.format(winner, self.board.mancalas[TOP], self.board.mancalas[BOTTOM]))

            self.log("GAME OVER: winner is {}".format(winner))
            self.shutdown("Game Over")

        # check if we don't allow mouse clicking for the next player
        elif isinstance(self.players[self.curr_player], AiPlayerInterface):
            self.root.unbind("<Button-1>")
            self.root.after(100, lambda: self.ai_move())
        
    def ai_move(self):
        """
        For when the AI agent gets moves
        """
        # get the ai move
        player_obj = self.players[self.curr_player]
        try:
            move, value = player_obj.get_move(self.board, self.curr_player)
        except AiTimeoutError:
            self.log("Game Over, {} lost (timeout)".format(player_obj.name))
            self.shutdown("Game Over, {} lost (timeout)".format(player_obj.name))
            return

        # log
        player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"
        self.log("{}: {}".format(player, move))
        
        # play move and display
        self.board = play_move(self.board, self.curr_player, move)
        self.curr_player = get_opponent(self.curr_player)
        self.draw_board()

        # check if game is over
        possible_moves = self.board.get_possible_moves(self.curr_player)
        if not possible_moves:
            winner = get_winner(self.board)
            print('{} {} {}\n'.format(winner, self.board.mancalas[TOP], self.board.mancalas[BOTTOM]))
            
            self.log("GAME OVER: winner is {}".format(winner))
            self.shutdown("Game Over")

        # next player is ai
        elif isinstance(self.players[self.curr_player], AiPlayerInterface):
            self.root.after(1, lambda: self.ai_move())
        else:
            # next player is human
            self.root.bind("<Button-1>",lambda e: self.mouse_pressed(e)) 

    def run(self):
        if isinstance(self.players[TOP], AiPlayerInterface):
            self.root.after(10, lambda: self.ai_move())
        else: 
            self.root.bind("<Button-1>",lambda e: self.mouse_pressed(e))        
        self.draw_board()
        self.canvas.mainloop()

    def draw_board(self):
        self.draw_pits()
        self.draw_stones()
        player = "Bottom Player" if self.curr_player == BOTTOM else "Top Player"
        self.move_label["text"]= player
        self.score_label["text"]= "Top Player {} : {} Bottom Player".format(*self.board.mancalas) 
   
    def log(self, msg, newline = True): 
        self.text.insert("end","{}{}".format(msg, "\n" if newline else ""))
        self.text.see("end")
 
    def draw_pits(self):
        colors = ("light green", "light blue") if self.curr_player == BOTTOM else ("light blue", "light green")
        for i in range(1, self.width+1):
            self.canvas.create_oval(i*self.cell_size + self.offset, self.offset, 
                                    (i+1)*self.cell_size + self.offset, self.cell_size + self.offset, 
                                    fill=colors[0])
            self.canvas.create_oval(i*self.cell_size + self.offset, self.cell_size + self.offset, 
                                    (i+1)*self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                    fill=colors[1])
        
        #pits for players
        self.canvas.create_oval(self.offset, self.offset, 
                                self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                fill="white")
        self.canvas.create_oval((self.width+1)*self.cell_size + self.offset, self.offset, 
                                (self.width+2)*self.cell_size + self.offset, 2*self.cell_size + self.offset, 
                                fill="white")

    def draw_stone(self, i, j):
        x = (i + 0.5) * self.cell_size - self.stone_size/2 + random.randint(0,20) - 10
        y = (j + 0.5) * self.cell_size - self.stone_size/2 + random.randint(0,20) - 10
        
        self.canvas.create_oval(x, y, x+self.stone_size, y+self.stone_size, fill="green")
        
    def draw_stones(self):       
        for i in range(2):
            for j in range(1, len(self.board.pockets[i])+1):
                x = (j + 0.5) * self.cell_size + self.offset
                y = (i+1)*self.cell_size - 2*self.offset
                for k in range(self.board.pockets[i][j-1]):
                    self.draw_stone(j, i)
                self.canvas.create_text(x, y, font="Arial", text=str(self.board.pockets[i][j-1]))

        #draw disks on the top
        for i in range(self.board.mancalas[TOP]):
            x = self.cell_size/2 + random.randint(0,20) - 10
            y = self.cell_size + random.randint(0,20) - 10
            self.canvas.create_oval(x, y, x + self.stone_size, y + self.stone_size, fill="blue")
        x = self.cell_size/2
        y = 2*self.cell_size - 2*self.offset
        self.canvas.create_text(x, y,font="Arial", text=str(self.board.mancalas[TOP]))

        #draw disks on the bottom
        for i in range(self.board.mancalas[BOTTOM]):
            x = (self.width+1.5)*self.cell_size + random.randint(0,20) - 10
            y = self.cell_size + random.randint(0,20) - 10 
            self.canvas.create_oval(x, y, x + self.stone_size, y + self.stone_size, fill="red")
        x = (self.width+1.5)*self.cell_size
        y = 2*self.cell_size - 2*self.offset
        self.canvas.create_text(x, y,font="Arial", text=str(self.board.mancalas[BOTTOM]))

def parse_args():
    parser = argparse.ArgumentParser(
        prog="MancalaGUI",
        description="Run this code to start a game of mancala"
    )

    parser.add_argument("-d", "--dimension", type=int, default=4,
                        help="Dimension of mancala board.")
    parser.add_argument("-t", "--agentTop", type=str,
                        help="Algorithm for the top player. If not specified, user inputs moves. [random, minimax, alphabeta]")
    parser.add_argument("-b", "--agentBottom", type=str,
                        help="Algorithm for the bottom player. If not specified, user inputs moves. [random, minimax, alphabeta]")
    parser.add_argument("-ht", "--heuristicTop", type=str, default="basic",
                        help="Heuristic for top player to use. [basic, advanced]")
    parser.add_argument("-hb", "--heuristicBottom", type=str, default="basic",
                        help="Heuristic for bottom player to use. [basic, advanced]")
    parser.add_argument("-c", "--caching", action="store_true",
                        help="Use flag if agent should use caching.")
    parser.add_argument("-l", "--limit", type=int, default=-1,
                        help="(Optional) Depth limit for agent to use.")
    parser.add_argument("-i", "--initialBoard", type=str,
                        help="File storing the initial state of the board. Overwrites dimension.")

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
        raise TypeError("Algorithm not recognized (only minimax or alphabeta)")
    
def get_heuristic(heuristic):
    if heuristic == "basic":
        return heuristic_basic
    elif heuristic == "advanced":
        return heuristic_advanced
    else:
        raise TypeError("Heuristic not recognized (only basic or advanced)")

def main():
    random.seed(datetime.now().timestamp())
    args = parse_args()

    if args.dimension is not None and args.dimension <= 0 and args.initialBoard is None: #if no dimension provided
        print('Please provide a valid board size (at least 1).')
        sys.exit(2)
    
    if args.agentTop != None:
        p1 = AiPlayerInterface(TOP, get_algorithm(args.agentTop), args.limit, args.caching, get_heuristic(args.heuristicTop))
    else:
        p1 = Player(TOP)

    if args.agentBottom != None:
        p2 = AiPlayerInterface(BOTTOM, get_algorithm(args.agentBottom), args.limit, args.caching, get_heuristic(args.heuristicBottom))
    else:
        p2 = Player(BOTTOM)
        
    gui = MancalaGui(args.dimension, args.initialBoard, p1, p2) 
    gui.run()

if __name__ == "__main__":
    main()
