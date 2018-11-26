# Implements the class for LOP instances

import os
import sys
import itertools
import cplex as CPX
import cplex.callbacks as CPX_CB

data_dir = os.path.join(os.path.dirname(__file__), '../data/lop/')

class lop(object):
	def __init__(self, instance_name):
		file_path = os.path.join(data_dir, instance_name)
		file = open(file_path, 'r')

		self.size = int(file.readline())
		self.w = []
		self.w_flattened = []
		
		for line in file:
			weights = map(int, line.split())
		 	self.w.append(weights)

		self.w_flattened = list(itertools.chain.from_iterable(self.w))

	def construct_objective(self):
		ubs = [1] * (self.size ** 2)
		lbs = [0] * (self.size ** 2)

		return self.w_flattened, ubs, lbs

	def construct_columns(self):
		columns = []
		
		for i in range(self.size):
			cols = []
			for j in range(self.size):
				colname = '_'.join(map(str, ['x', i, j]))
				cols.append(colname)

			columns.append(cols)

		ctypes = 'B' * (self.size ** 2)

		return columns, ctypes

	def construct_rows_for_edges(self, columns):
		rows = []
		rownames = []
		rhs = []

		for i in range(self.size):
			for j in range(self.size):
				rowname = '_'.join(map(str, ['e', i, j]))
				rownames.append(rowname) 
				if i == j:
					rows.append([[columns[i][j]], [1]])
					rhs.append(0)
				else:
					rows.append([[columns[i][j], columns[j][i]], [1, 1]])
					rhs.append(1)

		senses = 'E' * (self.size ** 2)

		return rows, rownames, rhs, senses

	def construct_rows_for_cycles(self, columns):
		rows = []
		rownames = []
		rhs = []

		for i in range(self.size):
			for j in range(i, self.size):
				for k in range(i, self.size):
					if not j == k:
						rowname = '_'.join(map(str, ['c', i, j, k]))
						rownames.append(rowname)
						rows.append([[columns[i][j], columns[j][k], columns[k][i]], [1, 1, 1]])

		rhs = [2] * len(rows)
		senses = 'L' * len(rows)

		return rows, rownames, rhs, senses

	def solve_instance(self):
		c = CPX.Cplex()

		obj, ub, lb = self.construct_objective()
		columns, ctypes = self.construct_columns()

		columns_flattened = list(itertools.chain.from_iterable(columns))

		c.objective.set_sense(c.objective.sense.maximize)
		c.variables.add(obj=obj, lb=lb, ub=ub, types=ctypes, names=columns_flattened)

		edge_rows, edge_rownames, edge_rhs, edge_senses = self.construct_rows_for_edges(columns)
		cycle_rows, cycle_rownames, cycle_rhs, cycle_senses = self.construct_rows_for_cycles(columns)

		c.linear_constraints.add(lin_expr=edge_rows+cycle_rows, 
			senses=edge_senses+cycle_senses, 
			rhs=edge_rhs+cycle_rhs, 
			names=edge_rownames+cycle_rownames)

		c.solve()
