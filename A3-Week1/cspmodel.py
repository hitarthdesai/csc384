############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
############################################################

from board import *
from cspbase import *

def kropki_model(board):
    """
    Create a CSP for a Kropki Sudoku Puzzle given a board of dimension.

    If a variable has an initial value, its domain should only contain the initial value.
    Otherwise, the variable's domain should contain all possible values (1 to dimension).

    We will encode all the constraints as binary constraints.
    Each constraint is represented by a list of tuples, representing the values that
    satisfy this constraint. (This is the table representation taught in lecture.)

    Remember that a Kropki sudoku has the following constraints.
    - Row constraint: every two cells in a row must have different values.
    - Column constraint: every two cells in a column must have different values.
    - Cage constraint: every two cells in a 2x3 cage (for 6x6 puzzle) 
            or 3x3 cage (for 9x9 puzzle) must have different values.
    - Black dot constraints: one value is twice the other value.
    - White dot constraints: the two values are consecutive (differ by 1).

    Make sure that you return a 2D list of variables separately. 
    Once the CSP is solved, we will use this list of variables to populate the solved board.
    Take a look at csprun.py for the expected format of this 2D list.

    :returns: A CSP object and a list of variables.
    :rtype: CSP, List[List[Variable]]

    """

    dim = board.dimension
    variables = create_variables(dim, board)
    
    all_diff_tuples = satisfying_tuples_difference_constraints(dim)
    row_and_col_constraints = create_row_and_col_constraints(dim, all_diff_tuples, variables)
    cage_constraints = create_cage_constraints(dim, all_diff_tuples, variables)
    
    white_tuples = satisfying_tuples_white_dots(dim)
    black_tuples = satisfying_tuples_black_dots(dim)
    dot_constraints = create_dot_constraints(board.dots, white_tuples, black_tuples, variables)
    
    no_dot_tuples = satisfying_tuples_no_dots(dim)
    no_dot_constraints = create_no_dot_constraints(dim, board.dots, no_dot_tuples, variables)
    
    csp = CSP("Kropki Sudoku")
    all_constraints = row_and_col_constraints + cage_constraints + dot_constraints + no_dot_constraints
    for constraint in all_constraints:
        csp.add_constraint(constraint)
    
    return csp, variables


def create_variables(dim, board):
    """
    Return a list of variables for the board, and initialize their domain appropriately.

    We recommend that your name each variable Var(row, col).

    :param dim: Size of the board
    :type dim: int

    :returns: A list of variables with an initial domain, one for each cell on the board
    :rtype: List[Variables]
    """

    variables = [None for _ in range(dim*dim)]
    for i in range(dim):
        for j in range(dim):
            name = f"Var({i}, {j})"
            initial_value = board.cells[i][j]
            if initial_value != 0:
                domain = [initial_value]
            else:
                domain = list(range(1, dim + 1))

            var = Variable(name, domain)
            variables[i*dim + j] = var

    return variables

def satisfying_tuples_difference_constraints(dim):
    """
    Return a list of satifying tuples for binary difference constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, j) for i in range(1, dim + 1) for j in range(1, dim + 1) if i != j]

def satisfying_tuples_white_dots(dim):
    """
    Return a list of satifying tuples for white dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, i+1) for i in range(1, dim)] + [(i+1, i) for i in range(1, dim)]

def satisfying_tuples_black_dots(dim):
    """
    Return a list of satifying tuples for black dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, 2*i) for i in range(1, (dim//2)+1)] + [(2*i, i) for i in range(1, (dim//2)+1)]

def create_row_and_col_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different row/column constraints.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """
   
    constraints = []
    for i in range(dim):
        for j in range(dim):
            for k in range(j + 1, dim):
                name = f"Row({i},{j},{k})"
                scope = [variables[i*dim+j], variables[i*dim+k]]
                con = Constraint(name, scope)
                con.add_satisfying_tuples(sat_tuples)
                constraints.append(con)

                name = f"Col({j},{i},{k})"
                scope = [variables[j*dim+i], variables[k*dim+i]]
                con = Constraint(name, scope)
                con.add_satisfying_tuples(sat_tuples)
                constraints.append(con)
    return constraints

def create_cage_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different constraints for all cages.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    constraints = []
    cage_size = 3 if dim == 9 else 2
    for i in range(0, dim, cage_size):
        for j in range(0, dim, cage_size):
            for x in range(cage_size):
                for y in range(cage_size):
                    for a in range(x, cage_size):
                        for b in range(y + 1 if a == x else 0, cage_size):
                            name = f"Cage({i+x},{j+y},{i+a},{j+b})"
                            scope = [variables[(i+x)*dim+j+y], variables[(i+a)*dim+j+b]]
                            con = Constraint(name, scope)
                            con.add_satisfying_tuples(sat_tuples)
                            constraints.append(con)
    
    return constraints
    
def create_dot_constraints(dim, dots, white_tuples, black_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]

    :param white_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the white dot constraint.
    :type white_tuples: List[(int, int)]
    
    :param black_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the black dot constraint.
    :type black_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary dot constraints
    :rtype: List[Constraint]
    """

    constraints = []
    for dot in dots:
        name = f"Dot({dot.row1},{dot.col1},{dot.row2},{dot.col2})"
        scope = [variables[dot.row1*dim+dot.col1], variables[dot.row2][dot.col2]]
        con = Constraint(name, scope)
        if dot.color == "white":
            con.add_satisfying_tuples(white_tuples)
        else:
            con.add_satisfying_tuples(black_tuples)
        constraints.append(con)
    return constraints

def satisfying_tuples_no_dots(dim):
    """
    Return a list of satifying tuples for no dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    no_dot_tuples = []
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            if abs(i - j) != 1 and i != 2*j and j != 2*i:
                no_dot_tuples.append((i, j))
    return no_dot_tuples

def create_no_dot_constraints(dim, dots, no_dot_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]
 
    :param no_dot_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the no dot constraint.
    :type no_dot_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary no dot constraints
    :rtype: List[Constraint]
    """

    constraints = []
    dot_positions = set((dot.row1, dot.col1, dot.row2, dot.col2) for dot in dots)

    for i in range(dim):
        for j in range(dim):
            if j < dim - 1 and (i, j, i, j+1) not in dot_positions and (i, j+1, i, j) not in dot_positions:
                name = f"NoDot({i},{j},{i},{j+1})"
                scope = [variables[i*dim+j], variables[i*dim+j+1]]
                con = Constraint(name, scope)
                con.add_satisfying_tuples(no_dot_tuples)
                constraints.append(con)

            if i < dim - 1 and (i, j, i+1, j) not in dot_positions and (i+1, j, i, j) not in dot_positions:
                name = f"NoDot({i},{j},{i+1},{j})"
                scope = [variables[i*dim+j], variables[(i+1)*dim+j]]
                con = Constraint(name, scope)
                con.add_satisfying_tuples(no_dot_tuples)
                constraints.append(con)

    return constraints

