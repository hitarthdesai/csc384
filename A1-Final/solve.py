############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.0
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity

from board import *

DIRECTIONS = [
    (0, 1),   # right
    (0, -1),  # left
    (1, 0),   # down
    (-1, 0),  # up
]

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    # We have reached the goal state if all boxes are in the storage spaces
    for box in state.board.boxes:
        if box not in state.board.storage:
            return False
        
    return True


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    path = []

    # We need to keep traversing up the path of state.parent until we reach the initial state
    while state is not None:
        path.insert(0, state)
        state = state.parent
    return path


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    successors = []
    
    # We need to check all possible moves for the robot
    for direction in DIRECTIONS:
        # Create a new board to represent the new state
        new_board = Board(
            state.board.name,
            state.board.width,
            state.board.height,
            state.board.robots.copy(),
            state.board.boxes.copy(),
            state.board.storage.copy(),
            state.board.obstacles.copy()
        )
        
        # Robot cannot move on top of another robot or an obstacle
        new_robot_location = (
            state.board.robots[0][0] + direction[0],
            state.board.robots[0][1] + direction[1]
        )
        if new_robot_location in state.board.robots or new_robot_location in state.board.obstacles:
            continue
        new_board.robots[0] = new_robot_location

        # Check if there is a box at the new robot location
        gen = (
            box for box in state.board.boxes
            if box == new_robot_location
        )
        box_at_new_robot_location = next(gen, None)

        # If there is a box at the new robot location, check if the box can move in the direction
        if box_at_new_robot_location is not None:
            new_box_location = (box_at_new_robot_location[0] + direction[0], box_at_new_robot_location[1] + direction[1])
            # Moved box cannot be on top of another box, robot, or an obstacle
            if new_box_location in state.board.robots or new_box_location in state.board.boxes or new_box_location in state.board.obstacles:
                continue

            new_board.boxes.remove(box_at_new_robot_location)
            new_board.boxes.append(new_box_location)
        

        successors.append(State(new_board, state.hfn, state.f, state.depth + 1, state))
    
    return successors


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    
    frontier = [init_board]
    states = {}

    states[init_board.__hash__()] = State(init_board, heuristic_zero, 0, 0, None)

    while frontier:
        current = frontier.pop()
        current_state = states[current.__hash__()]

        if is_goal(current_state):
            return get_path(current_state), current_state.depth
        
        successor_states = get_successors(current_state)
        for state in successor_states:
            states[state.board.__hash__()] = state
            frontier.append(state.board)

    return [], -1


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    raise NotImplementedError


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box 
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()