############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
##
############################################################

from collections import deque
from cspmodel import *

def prop_FC(csp, last_assigned_var=None):
    """
    This is a propagator to perform forward checking. 

    First, collect all the relevant constraints.
    If the last assigned variable is None, then no variable has been assigned 
    and we are performing propagation before search starts.
    In this case, we will check all the constraints.
    Otherwise, we will only check constraints involving the last assigned variable.

    Among all the relevant constraints, focus on the constraints with one unassigned variable. 
    Consider every value in the unassigned variable's domain, if the value violates 
    any constraint, prune the value. 

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: The boolean indicates whether forward checking is successful.
        The boolean is False if at least one domain becomes empty after forward checking.
        The boolean is True otherwise.
        Also returns a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """

    constraints = csp.get_all_cons() if last_assigned_var is None else csp.get_cons_with_var(last_assigned_var)
    single_unassigned = list(filter(lambda c: c.get_num_unassigned_vars() == 1, constraints))

    pruned = []
    for c in single_unassigned:
        values_to_check = [v.get_assigned_value() for v in c.get_scope()]
        unassigned_idx = values_to_check.index(None)
        unassigned_var = c.get_scope()[unassigned_idx]
        for val in unassigned_var.cur_domain():
            values_to_check[unassigned_idx] = val
            values_to_check_tuple = tuple(values_to_check)

            if not c.check(values_to_check_tuple):
                unassigned_var.prune_value(val)
                pruned.append((unassigned_var, val))
                if unassigned_var.cur_domain_size() == 0:
                    return False, pruned

    return True, pruned


def prop_AC3(csp, last_assigned_var=None):
    """
    This is a propagator to perform the AC-3 algorithm.

    Keep track of all the constraints in a queue (list). 
    If the last_assigned_var is not None, then we only need to 
    consider constraints that involve the last assigned variable.

    For each constraint, consider every variable in the constraint and 
    every value in the variable's domain.
    For each variable and value pair, prune it if it is not part of 
    a satisfying assignment for the constraint. 
    Finally, if we have pruned any value for a variable,
    add other constraints involving the variable back into the queue.

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes 
        all the constraints and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """

    def revise(var, con):
        revised = False
        pruned = []
        for val in var.cur_domain():
            is_consistent = False
            tuples = con.sup_tuples.get((var, val), [])
            for t in tuples:
                all_domains_satisfied = True
                for stuff in zip(con.get_scope(), t):
                    other_var, other_val = stuff
                    if other_var.in_cur_domain(other_val):
                        continue

                    all_domains_satisfied = False
                    break

                if all_domains_satisfied:
                    is_consistent = True
                    break
            if not is_consistent:
                var.prune_value(val)
                pruned.append((var, val))
                revised = True

        return revised, pruned

    cons = csp.get_all_cons() if last_assigned_var is None else csp.get_cons_with_var(last_assigned_var)
    queue = deque([(v, c) for c in cons for v in c.get_scope()])

    pruned = []
    while queue:
        var, con = queue.popleft()
        revised, pruned_vals = revise(var, con)
        if not revised:
            continue

        pruned.extend(pruned_vals)
        if var.cur_domain_size() == 0:
            return False, pruned
        for con_prime in csp.get_cons_with_var(var):
            for neighbor in con_prime.get_scope():
                if neighbor != var:
                    queue.append((neighbor, con_prime))

    return True, pruned

def ord_mrv(csp):
    """
    Implement the Minimum Remaining Values (MRV) heuristic.
    Choose the next variable to assign based on MRV.

    If there is a tie, we will choose the first variable. 

    :param csp: A CSP problem
    :type csp: CSP

    :returns: the next variable to assign based on MRV

    """

    return min(csp.get_all_unasgn_vars(), key=lambda v: v.cur_domain_size())


###############################################################################
# Do not modify the prop_BT function below
###############################################################################


def prop_BT(csp, last_assigned_var=None):
    """
    This is a basic propagator for plain backtracking search.

    Check if the current assignment satisfies all the constraints.
    Note that we only need to check all the fully instantiated constraints 
    that contain the last assigned variable.
    
    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes all the constraints 
        and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]

    """
    
    # If we haven't assigned any variable yet, return true.
    if not last_assigned_var:
        return True, []
        
    # Check all the constraints that contain the last assigned variable.
    for c in csp.get_cons_with_var(last_assigned_var):

        # All the variables in the constraint have been assigned.
        if c.get_num_unassigned_vars() == 0:

            # get the variables
            vars = c.get_scope() 

            # get the list of values
            vals = []
            for var in vars: #
                vals.append(var.get_assigned_value())

            # check if the constraint is satisfied
            if not c.check(vals): 
                return False, []

    return True, []
