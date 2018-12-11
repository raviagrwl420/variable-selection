import cplex
import numpy as np

from scipy.sparse import csr_matrix

class StaticFeatures:
    def __init__(self, cplex_instance):
        self.num_vars = cplex_instance.variables.get_num()
        self.var_names = cplex_instance.variables.get_names() # Simple list of strings
        self.var_types = cplex_instance.variables.get_types() # Simple list of 'B', 'I' or 'C'
        self.columns = cplex_instance.variables.get_cols() # List of SparsePair objects - SparsePair(ind=[], val=[])
        self.num_rows = cplex_instance.linear_constraints.get_num()
        self.rows = cplex_instance.linear_constraints.get_rows() # List of SparsePair objects - SparsePair(ind=[], val=[])
        self.rhs = np.array(cplex_instance.linear_constraints.get_rhs()) # Simple list of values
        self.obj = np.array(cplex_instance.objective.get_linear())

        # Generate the row x column matrix
        self.matrix = csr_matrix((self.num_rows, self.num_vars))
        for i, row in enumerate(self.rows):
            data = np.array(row.val)
            row_indices = np.empty(len(row.ind))
            row_indices.fill(i)
            col_indices = np.array(row.ind)
            self.matrix += csr_matrix((data, (row_indices, col_indices)), shape=(self.num_rows, self.num_vars))

        ## Part 1: Objective function coefficients
        # 1. Raw obj function coefficients
        raw = self.obj.copy().reshape(-1, 1)

        # 2. Positive only obj function coefficients
        positive_only = self.obj.copy().reshape(-1, 1)
        positive_only[positive_only < 0] = 0

        # 3. Negative only obj function coefficients
        negative_only = self.obj.copy().reshape(-1, 1)
        negative_only[negative_only > 0] = 0

        # Add 1, 2, 3 to features
        self.features = np.c_[raw, positive_only, negative_only]

        ## Part 2: Num. Constraints
        # 4. Number of constraints that the variable participates in (with a non-zero coefficient)
        non_zeros = self.matrix != 0
        num_const_for_var = np.transpose(non_zeros.sum(0))

        # Add 4 to features
        self.features = np.c_[self.features, num_const_for_var]

        ## Part 3: Stats for constraint degrees
        num_var_for_const = non_zeros.sum(1)
        degree_matrix = non_zeros.multiply(csr_matrix(num_var_for_const)).todense()

        # 5. Mean of degrees
        mean_degrees = np.transpose(np.mean(degree_matrix, axis=0))

        # 6. Stdev of degrees
        std_degrees = np.transpose(np.std(degree_matrix, axis=0))

        # 7. Min of degrees
        min_degrees = np.transpose(np.min(degree_matrix, axis=0))

        # 8. Max of degrees
        max_degrees = np.transpose(np.max(degree_matrix, axis=0))

        # Add 5, 6, 7, 8 to features
        self.features = np.c_[self.features, mean_degrees, std_degrees, min_degrees, max_degrees]

        ## Part 4: Stats for constraint coefficients
        pos_coeffs = self.matrix > 0
        pos_matrix = self.matrix.todense()
        pos_matrix[pos_matrix < 0] = 0
        neg_coeffs = self.matrix < 0
        neg_matrix = self.matrix.todense()
        neg_matrix[neg_matrix > 0] = 0

        # 9, 10, 11, 12, 13
        count_pos_coeffs = np.transpose(pos_coeffs.sum(0))
        mean_pos_coeffs = np.transpose(np.mean(pos_matrix, axis=0))
        std_pos_coeffs = np.transpose(np.std(pos_matrix, axis=0))
        min_pos_coeffs = np.transpose(np.min(pos_matrix, axis=0))
        max_pos_coeffs = np.transpose(np.max(pos_matrix, axis=0))

        # Add 9, 10, 11, 12, 13 to features
        self.features = np.c_[self.features, count_pos_coeffs, mean_pos_coeffs, std_pos_coeffs, min_pos_coeffs, max_pos_coeffs]

        # 14, 15, 16, 17, 18
        count_neg_coeffs = np.transpose(neg_coeffs.sum(0))
        mean_neg_coeffs = np.transpose(np.mean(neg_matrix, axis=0))
        std_neg_coeffs = np.transpose(np.std(neg_matrix, axis=0))
        min_neg_coeffs = np.transpose(np.min(neg_matrix, axis=0))
        max_neg_coeffs = np.transpose(np.max(neg_matrix, axis=0))

        # Add 14, 15, 16, 17, 18 to features
        self.features = np.c_[self.features, count_neg_coeffs, mean_neg_coeffs, std_neg_coeffs, min_neg_coeffs, max_neg_coeffs]
