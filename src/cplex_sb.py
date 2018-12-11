from math import floor

import numpy as np
from scipy.sparse import csr_matrix

import cplex as CPX
import cplex.callbacks as CPX_CB

from static_features import StaticFeatures
from dynamic_features import DynamicFeatures

cplex = CPX.Cplex("data/miplib/30n20b8.mps")

NUM_CANDIDATES = 20
INFEASIBILITY = 1e6
EPSILON = 1e-6
LOWER_BOUND = 'L'
UPPER_BOUND = 'U'
OPTIMAL = 1

def turn_cuts_off(cplex_instance):
	cplex_instance.parameters.mip.cuts.bqp.set(-1)
	cplex_instance.parameters.mip.cuts.cliques.set(-1)
	cplex_instance.parameters.mip.cuts.covers.set(-1)
	cplex_instance.parameters.mip.cuts.disjunctive.set(-1)
	cplex_instance.parameters.mip.cuts.flowcovers.set(-1)
	cplex_instance.parameters.mip.cuts.pathcut.set(-1)
	cplex_instance.parameters.mip.cuts.gomory.set(-1)
	cplex_instance.parameters.mip.cuts.gubcovers.set(-1)
	cplex_instance.parameters.mip.cuts.implied.set(-1)
	cplex_instance.parameters.mip.cuts.localimplied.set(-1)
	cplex_instance.parameters.mip.cuts.liftproj.set(-1)
	cplex_instance.parameters.mip.cuts.mircut.set(-1)
	cplex_instance.parameters.mip.cuts.mcfcut.set(-1)
	cplex_instance.parameters.mip.cuts.rlt.set(-1)
	cplex_instance.parameters.mip.cuts.zerohalfcut.set(-1)

def set_parameters(cplex_instance):
    cplex_instance.parameters.mip.strategy.variableselect.set(2)
    cplex_instance.parameters.mip.limits.nodes.set(1)
    cplex_instance.parameters.preprocessing.presolve.set(0)
    cplex_instance.parameters.mip.tolerances.integrality.set(EPSILON)

def disable_output(cplex_instance):
    cplex_instance.set_log_stream(None)
    cplex_instance.set_error_stream(None)
    cplex_instance.set_warning_stream(None)
    cplex_instance.set_results_stream(None)

def apply_branch_history(cplex_instance, branch_history):
    for b in branch_history:
        b_var = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == LOWER_BOUND:
            cplex_instance.variables.set_lower_bounds(b_var, b_val)
        elif b_type == UPPER_BOUND:
            cplex_instance.variables.set_upper_bounds(b_var, b_val)

def solve_it_as_lp(cplex_instance):
    disable_output(cplex_instance)
    cplex_instance.set_problem_type(cplex_instance.problem_type.LP)
    cplex_instance.solve()
    status = cplex_instance.solution.get_status()
    objective = cplex_instance.solution.get_objective_value() if status == OPTIMAL else INFEASIBILITY
    return status, objective

def get_candidates(pseudocosts, values):
    scores = [pseudocost[0]*pseudocost[1] for pseudocost in pseudocosts]
    variables = sorted(range(len(scores)), key=lambda i: -scores[i])

    candidates = []
    for i in variables:
        if len(candidates) == NUM_CANDIDATES:
            break

        value = values[i]
        if not abs(value-round(value)) <= EPSILON:
            candidates.append(i)

    return candidates

class MyBranch(CPX_CB.BranchCallback):
    def get_data(self):
        node_data = self.get_node_data()
        if node_data is None:
            node_data = {}
            node_data['branch_history'] = []

        return node_data

    def get_clone(self):
        clone = CPX.Cplex(self.cplex)

        node_data = self.get_data()
        apply_branch_history(clone, node_data['branch_history'])

        return clone

    def get_branch_solution(self, clone, var, bound_type):
        value = self.get_values(var)

        get_bounds = None
        set_bounds = None
        new_bound = None
        if bound_type == LOWER_BOUND:
            get_bounds = self.get_lower_bounds
            set_bounds = clone.variables.set_lower_bounds
            new_bound = floor(value) + 1
        elif bound_type == UPPER_BOUND:
            get_bounds = self.get_upper_bounds
            set_bounds = clone.variables.set_upper_bounds
            new_bound = floor(value)

        original_bound = get_bounds(var)

        set_bounds(var, new_bound)
        status, objective = solve_it_as_lp(clone)
        set_bounds(var, original_bound)

        return status, objective

    def get_features(self, candidates):
        clone = self.get_clone()

        _, node_objective = solve_it_as_lp(clone)

        max_var = None
        max_sb_score = 0
        for var in candidates:
            _, lower_objective = self.get_branch_solution(clone, var, LOWER_BOUND)
            _, upper_objective = self.get_branch_solution(clone, var, UPPER_BOUND)

            delta_lower = max(lower_objective-node_objective, EPSILON)
            delta_upper = max(upper_objective-node_objective, EPSILON)

            sb_score = delta_lower*delta_upper
            if sb_score >= max_sb_score:
                max_sb_score = sb_score
                max_var = var

        return max_var

    def get_dynamic_features(self, candidates):
        dyn_feat = DynamicFeatures(self, self.stat_feat, candidates)

        # values = np.array(self.get_values())
        # slacks = np.array(self.get_linear_slacks())
        # # duals = self.get_dual_values()

        # matrix, rhs = get_static_features(self.cplex)

        # active_constraints = slacks == 0
        # active_matrix = matrix[active_constraints, :]

        # positive_rhs = rhs > 0
        # negative_rhs = rhs < 0
        # print positive_rhs
        # print negative_rhs


    def __call__(self):
        # Turn cuts off after root node
        if not self.cuts_off:
           turn_cuts_off(self.cplex)
           self.cuts_off = True

        # Get candidate variables
        pseudocosts = self.get_pseudo_costs()
        values = self.get_values()
        candidates = get_candidates(pseudocosts, values)

        # Get features
        self.get_dynamic_features(candidates)

        sb_var = self.get_features(candidates)
        sb_val = self.get_values(sb_var)
        obj_val = self.get_objective_value()

        node_data = self.get_data()

        branches = [(sb_var, LOWER_BOUND, floor(sb_val) + 1), (sb_var, UPPER_BOUND, floor(sb_val))]

        for i in range(len(branches)):
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            
            branch = branches[i]
            node_data_clone['branch_history'].append(branch)
            # node_data_clone['branch_history'].append(self.get_branch(i)[1][0])

            self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)
            # self.make_cplex_branch(i, node_data=node_data_clone)

stat_feat = StaticFeatures(cplex)

set_parameters(cplex)

cplex.register_callback(MyBranch)
MyBranch.cplex = cplex
MyBranch.cuts_off = False
MyBranch.stat_feat = stat_feat

cplex.solve()

print(cplex.solution.get_objective_value())
