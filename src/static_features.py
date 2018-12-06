import cplex
from cplex.exceptions import CplexSolverError
import sys
import numpy as np


def lpex2(filename, method):
    c = cplex.Cplex(filename)

    alg = c.parameters.lpmethod.values

    if method == "o":
        c.parameters.lpmethod.set(alg.auto)
    elif method == "p":
        c.parameters.lpmethod.set(alg.primal)
    elif method == "d":
        c.parameters.lpmethod.set(alg.dual)
    elif method == "b":
        c.parameters.lpmethod.set(alg.barrier)
        c.parameters.barrier.crossover.set(c.parameters.barrier.crossover.values.none)
    elif method == "h":
        c.parameters.lpmethod.set(alg.barrier)
    elif method == "s":
        c.parameters.lpmethod.set(alg.sifting)
    elif method == "c":
        c.parameters.lpmethod.set(alg.concurrent)
    else:
        print "Unrecognized option, using automatic"
        c.parameters.lpmethod.set(alg.auto)


    features = []

    for j,val in enumerate(c.objective.get_linear()):
        if val>0:
            features.append([val,val,0])
        elif val<0:
            features.append([val,0,val])
        else:
            features.append([0,0,0])


    #column_count = c.variables.get_num()
    #print c.variables.get_names()
    #row_count = c.linear_constraints.get_num()
    #print c.linear_constraints.get_names()


    num_of_var_in_a_constraint = []

    for i, row in enumerate(c.linear_constraints.get_names()):
        non_zero_in_a_row = 0
        for j, var in enumerate(c.variables.get_names()):
            if c.linear_constraints.get_coefficients(row, var) != 0:
                non_zero_in_a_row = non_zero_in_a_row + 1
        num_of_var_in_a_constraint.append(non_zero_in_a_row)


    #number_of_contraints_a_variable_participate
    #stats of constraints a variable participate
    #stats of coefficients ( pos  and neg )  a variable particiapte


    for i, var in enumerate(c.variables.get_names()):
        stats = np.array([])
        count = 0
        pos = np.array([])
        neg = np.array([])
        for j, row in enumerate(c.linear_constraints.get_names()):
            constraint_coeff = c.linear_constraints.get_coefficients(row, var)
            if constraint_coeff != 0:
                count = count + 1
                stats = np.append(stats,num_of_var_in_a_constraint[j])
                if constraint_coeff < 0:
                    neg = np.append(neg,constraint_coeff)
                else:
                    pos = np.append(pos,constraint_coeff)


        features[i].append(count)
        features[i].append(np.mean(stats,axis=0))
        features[i].append(np.std(stats, axis=0))
        features[i].append(np.amin(stats, axis=0))
        features[i].append(np.amax(stats, axis=0))
        if len(pos)>0:
            features[i].append(len(pos))
            features[i].append(np.mean(pos,axis=0))
            features[i].append(np.std(pos, axis=0))
            features[i].append(np.amin(pos, axis=0))
            features[i].append(np.amax(pos, axis=0))
        else:
            for i in range(5):
                features[i].append(0)
        if len(neg)>0:
            features[i].append(len(neg))
            features[i].append(np.mean(neg, axis=0))
            features[i].append(np.std(neg, axis=0))
            features[i].append(np.amin(neg, axis=0))
            features[i].append(np.amax(neg, axis=0))
        else:
            for i in range(5):
                features[i].append(0)




    print features

    #print number_of_contraints_a_variable_participate



    '''
    try:
        c.solve()
    except CplexSolverError:
        print "Exception raised during solve"
        return

    # solution.get_status() returns an integer code
    status = c.solution.get_status()
    print c.solution.status[status]
    if status == c.solution.status.unbounded:
        print "Model is unbounded"
        return
    if status == c.solution.status.infeasible:
        print "Model is infeasible"
        return
    if status == c.solution.status.infeasible_or_unbounded:
        print "Model is infeasible or unbounded"
        return

    s_method = c.solution.get_method()
    s_type = c.solution.get_solution_type()

    print "Solution status = ", status, ":",
    # the following line prints the status as a string
    print c.solution.status[status]
    print "Solution method = ", s_method, ":",
    print c.solution.method[s_method]

    if s_type == c.solution.type.none:
        print "No solution available"
        return
    print "Objective value = ", c.solution.get_objective_value()

    if s_type == c.solution.type.basic:
        basis = c.solution.basis.get_col_basis()
    else:
        basis = None

    print

    x = c.solution.get_values(0, c.variables.get_num() - 1)
    # because we're querying the entire solution vector,
    # x = c.solution.get_values()
    # would have the same effect
    for j in range(c.variables.get_num()):
        print "Column %d: Value = %17.10g" % (j, x[j])
        if basis is not None:
            if basis[j] == c.solution.basis.status.at_lower_bound:
                print "  Nonbasic at lower bound"
            elif basis[j] == c.solution.basis.status.basic:
                print "  Basic"
            elif basis[j] == c.solution.basis.status.at_upper_bound:
                print "  Nonbasic at upper bound"
            elif basis[j] == c.solution.basis.status.free_nonbasic:
                print "  Superbasic, or free variable at zero"
            else:
                print "  Bad basis status"

    infeas = c.solution.get_float_quality(c.solution.quality_metric.max_primal_infeasibility)
    print "Maximum bound violation = ", infeas

    '''

import sys
'''
if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ["o", "p", "d", "b", "h", "s", "c"]:
        print "Usage: lpex2.py filename algorithm"
        print "  filename   Name of a file, with .mps, .lp, or .sav"
        print "             extension, and a possible, additional .gz"
        print "             extension"
        print "  algorithm  one of the letters"
        print "             o default"
        print "             p primal simplex"
        print "             d dual simplex"
        print "             b barrier"
        print "             h barrier with crossover"
        print "             s sifting"
        print "             c concurrent"
        sys.exit(-1)
    lpex2(sys.argv[1], sys.argv[2])
else:
    prompt = """Enter the path to a file with .mps, .lp, or .sav
extension, and a possible, additional .gz extension:
The path must be entered as a string; e.g. "my_model.mps"\n """
    fname = input(prompt)
    prompt = """Enter the letter indicating what optimization method
should be used:
    o default
    p primal simplex
    d dual simplex
    b barrier
    h barrier with crossover
    s sifting
    c concurrent \n"""
    o = "o"
    p = "p"
    d = "d"
    b = "b"
    h = "h"
    s = "s"
    c = "c"
    lpex2(fname, input(prompt))
    '''
lpex2('aligninq.mps','o')