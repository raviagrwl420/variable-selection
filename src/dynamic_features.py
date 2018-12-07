# Main file

import torch
import itertools

import numpy as np

#from src.lop import lop
#from src.nn_rank import NNRank

import math

# lop('RandB/N-p40-01').solve_instance()

# model = NNRank(100, 5, 1)

# inputs = torch.randn(10000).reshape(100, 100)
# targets = torch.cat((torch.zeros(50), torch.ones(50)), dim=0)

# model.train(inputs, targets, 100)

import cplex as CPX
import cplex.callbacks as CPX_CB

c = CPX.Cplex("aligninq.mps")

c.parameters.mip.cuts.bqp.set(-1)
c.parameters.mip.cuts.cliques.set(-1)
c.parameters.mip.cuts.covers.set(-1)
c.parameters.mip.cuts.disjunctive.set(-1)
c.parameters.mip.cuts.flowcovers.set(-1)
c.parameters.mip.cuts.pathcut.set(-1)
c.parameters.mip.cuts.gomory.set(-1)
c.parameters.mip.cuts.gubcovers.set(-1)
c.parameters.mip.cuts.implied.set(-1)
c.parameters.mip.cuts.localimplied.set(-1)
c.parameters.mip.cuts.liftproj.set(-1)
c.parameters.mip.cuts.mircut.set(-1)
c.parameters.mip.cuts.mcfcut.set(-1)
c.parameters.mip.cuts.rlt.set(-1)
c.parameters.mip.cuts.zerohalfcut.set(-1)

# c.parameters.preprocessing.presolve.set(0)
# c.parameters.mip.display.set(4)


constraint_degree_stat = [[]]*(c.linear_constraints.get_num())

class MyBranch(CPX_CB.BranchCallback):
    def __call__(self):
        objval = self.get_objective_value()
        obj = self.get_objective_coefficients()
        feas = self.get_feasibilities()

        # print(objval)

        node_data = self.get_node_data()
        if node_data is None:
        	node_data = []

        print "this is what's in nodes", node_data

        clone = CPX.Cplex(self.cplex)
        # clone.parameters.mip.limits.nodes.set(1)
        clone.set_problem_type(clone.problem_type.LP)
        for b in node_data:
        	b_var = b[0]
        	b_type = b[1]
        	b_val = b[2]

        	if b_type == 'L':
        		clone.variables.set_lower_bounds(b_var, b_val)
        	elif b_type == 'U':
        		clone.variables.set_upper_bounds(b_var, b_val)

    	clone.set_results_stream(None)

        num_of_var_in_a_constraint = []

        for i, row in enumerate(clone.linear_constraints.get_names()):
            non_zero_in_a_row = 0
            for j, var in enumerate(c.variables.get_names()):
                if clone.linear_constraints.get_coefficients(row, var) != 0:
                    non_zero_in_a_row = non_zero_in_a_row + 1
            num_of_var_in_a_constraint.append(non_zero_in_a_row)

        for i, var in enumerate(clone.variables.get_names()):
            for j, row in enumerate(clone.linear_constraints.get_names()):
                constraint_coeff = clone.linear_constraints.get_coefficients(row, var)
                if constraint_coeff != 0:
                    constraint_degree_stat[i].append(num_of_var_in_a_constraint[j])


    	clone.solve()

        print("Actual: %.3f, Solution: %.3f" % (objval, clone.solution.get_objective_value()))

        print self.get_num_branches()

        for i in range(self.get_num_branches()):
        	node_data_clone = node_data[:]
        	node_data_clone.append(self.get_branch(i)[1][0])

        	self.make_cplex_branch(i, node_data=node_data_clone)

c.register_callback(MyBranch)
MyBranch.cplex = c

# c.set_problem_type(c.problem_type.LP)

c.solve()

dynamic_features =[]

sol_vec = c.solution.get_values()

for i,sol in enumerate(sol_vec):
    dynamic_features.append([min(sol-math.floor(sol),math.ceil(sol)-sol),sol])

for i,sol in enumerate(sol_vec):
    dynamic_features[i].append(np.mean(constraint_degree_stat[i]))
    dynamic_features[i].append(np.std(constraint_degree_stat[i]))
    dynamic_features[i].append(np.amin(constraint_degree_stat[i]))
    dynamic_features[i].append(np.amax(constraint_degree_stat[i]))

print "dynamic features:", dynamic_features


print(c.solution.get_objective_value())