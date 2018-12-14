import cplex
import numpy as np

from scipy.sparse import csr_matrix

class DynamicFeatures:
    def __init__(self, branch_instance, static_features, candidates):
        # Part 1: Slack and ceil distances
        self.values = np.array(branch_instance.get_values()).reshape(-1, 1)
        self.values = self.values[candidates] # Filter by candidates
        
        # 1. Min of slack and ceil
        ceil = np.ceil(self.values)
        floor = np.floor(self.values)
        fractionality = np.minimum(self.values - floor, ceil - self.values)

        # 2. Distance from ceil
        dist_ceil = ceil - self.values

        # Add 1, 2 to features
        self.features = np.c_[fractionality, dist_ceil]

        # Part 2: Pseudocosts

        # 3. Upwards and downwards pseudocosts weighted by fractionality
        self.pseudocosts = np.array(branch_instance.get_pseudo_costs())
        self.pseudocosts = self.pseudocosts[candidates]
        up_down_pc = self.pseudocosts * fractionality
        
        # 4. Sum of pseudocosts weighted by fractionality
        sum_pc = np.sum(self.pseudocosts, axis=1).reshape(-1, 1) * fractionality

        # 5. Ratio of pseudocosts weighted by fractionality
        ratio_pc = (self.pseudocosts[:, 0] / self.pseudocosts[:, 1]).reshape(-1, 1) * fractionality
        ratio_pc[np.isnan(ratio_pc)] = 0
        ratio_pc[np.isinf(ratio_pc)] = 0

        # 6. Prod of pseudocosts weighted by fractionality
        prod_pc = np.prod(self.pseudocosts, axis=1).reshape(-1, 1) * fractionality

        # Add 3, 4, 5, 6 to features
        self.features = np.c_[self.features, up_down_pc, sum_pc, ratio_pc, prod_pc]

        # Skipped Parts 3 and 4
        # Part 5: Min/max ratios of constraint coeffs to RHS
        rhs = static_features.rhs.reshape(-1, 1)
        pos_rhs = rhs[rhs > 0]
        neg_rhs = rhs[rhs < 0]

        mat = static_features.matrix.todense()
        candidate_matrix = static_features.matrix[:, candidates].todense()
        pos_ratio_matrix = np.divide(candidate_matrix[(rhs > 0).ravel(), :], pos_rhs.reshape(-1, 1))
        pos_ratio_matrix = pos_ratio_matrix if pos_ratio_matrix.size else np.zeros((1, candidate_matrix.shape[1]))
        neg_ratio_matrix = np.divide(candidate_matrix[(rhs < 0).ravel(), :], neg_rhs.reshape(-1, 1))
        neg_ratio_matrix = neg_ratio_matrix if neg_ratio_matrix.size else np.zeros((1, candidate_matrix.shape[1]))

        # 7. Min ratio for positive RHS
        min_ratio_pos = np.transpose(np.min(pos_ratio_matrix, axis=0))
        
        # 8. Max ratio for positive RHS
        max_ratio_pos = np.transpose(np.max(pos_ratio_matrix, axis=0))

        # 9. Min ratio for negative RHS
        min_ratio_neg = np.transpose(np.min(neg_ratio_matrix, axis=0))

        # 10. Max ratio for negative RHS
        max_ratio_neg = np.transpose(np.max(neg_ratio_matrix, axis=0))

        # Add 7, 8, 9, 10 to features
        self.features = np.c_[self.features, min_ratio_pos, max_ratio_pos, min_ratio_neg, max_ratio_neg]

        # Part 6: Min/max for one-to-all coefficient ratios
        pos_coeff_matrix = static_features.pos_coeff_matrix[:, candidates]
        neg_coeff_matrix = static_features.neg_coeff_matrix[:, candidates]

        sum_pos_coeffs = static_features.sum_pos_coeffs
        sum_neg_coeffs = static_features.sum_neg_coeffs

        pos_pos_ratio_matrix = pos_coeff_matrix.todense() / sum_pos_coeffs
        pos_pos_ratio_matrix[np.isnan(pos_pos_ratio_matrix)] = 0
        pos_pos_ratio_matrix[np.isinf(pos_pos_ratio_matrix)] = 0
        pos_neg_ratio_matrix = pos_coeff_matrix.todense() / sum_neg_coeffs
        pos_neg_ratio_matrix[np.isnan(pos_neg_ratio_matrix)] = 0
        pos_neg_ratio_matrix[np.isinf(pos_neg_ratio_matrix)] = 0
        neg_neg_ratio_matrix = neg_coeff_matrix.todense() / sum_neg_coeffs
        neg_neg_ratio_matrix[np.isnan(neg_neg_ratio_matrix)] = 0
        neg_neg_ratio_matrix[np.isinf(neg_neg_ratio_matrix)] = 0
        neg_pos_ratio_matrix = neg_coeff_matrix.todense() / sum_pos_coeffs
        neg_pos_ratio_matrix[np.isnan(neg_pos_ratio_matrix)] = 0
        neg_pos_ratio_matrix[np.isinf(neg_pos_ratio_matrix)] = 0        

        pos_pos_ratio_min = np.transpose(np.min(pos_pos_ratio_matrix, axis=0))
        pos_pos_ratio_max = np.transpose(np.max(pos_pos_ratio_matrix, axis=0))
        pos_neg_ratio_min = np.transpose(np.min(pos_neg_ratio_matrix, axis=0))
        pos_neg_ratio_max = np.transpose(np.max(pos_neg_ratio_matrix, axis=0))
        neg_neg_ratio_min = np.transpose(np.min(neg_neg_ratio_matrix, axis=0))
        neg_neg_ratio_max = np.transpose(np.max(neg_neg_ratio_matrix, axis=0))
        neg_pos_ratio_min = np.transpose(np.min(neg_pos_ratio_matrix, axis=0))
        neg_pos_ratio_max = np.transpose(np.max(neg_pos_ratio_matrix, axis=0))

        self.features = np.c_[self.features, pos_pos_ratio_min, pos_pos_ratio_max, pos_neg_ratio_min, pos_neg_ratio_max,
            neg_neg_ratio_min, neg_neg_ratio_max, neg_pos_ratio_min, neg_pos_ratio_max]

        # Part 7: Stats for active constraints
        slacks = np.array(branch_instance.get_linear_slacks())
        active_constraints = slacks == 0

        active_matrix = static_features.matrix[active_constraints, :]
        active_matrix = active_matrix[:, candidates].todense()
        count_active_matrix = active_matrix != 0

        # Unit weighting
        unit_sum = np.transpose(np.sum(active_matrix, axis=0))
        unit_mean = np.transpose(np.mean(active_matrix, axis=0))
        unit_std = np.transpose(np.std(active_matrix, axis=0))
        unit_min = np.transpose(np.min(active_matrix, axis=0))
        unit_max = np.transpose(np.max(active_matrix, axis=0))
        unit_count = np.transpose(np.sum(count_active_matrix, axis=0))

        # Add unit weighting features
        self.features = np.c_[self.features, unit_sum, unit_mean, unit_std, unit_min, unit_max, unit_count]

        # Inverse sum all weighting
        inverse_sum_all = 1 / static_features.sum_coeffs[active_constraints]
        inverse_sum_all[np.isnan(inverse_sum_all)] = 0
        inverse_sum_all[np.isinf(inverse_sum_all)] = 0
        inverse_sum_all_matrix = np.multiply(active_matrix, inverse_sum_all)
        count_inverse_sum_all_matrix = np.multiply(count_active_matrix, inverse_sum_all)

        inv_sum_all_sum = np.transpose(np.sum(inverse_sum_all_matrix, axis=0))
        inv_sum_all_mean = np.transpose(np.mean(inverse_sum_all_matrix, axis=0))
        inv_sum_all_std = np.transpose(np.std(inverse_sum_all_matrix, axis=0))
        inv_sum_all_min = np.transpose(np.min(inverse_sum_all_matrix, axis=0))
        inv_sum_all_max = np.transpose(np.max(inverse_sum_all_matrix, axis=0))
        inv_sum_all_count = np.transpose(np.sum(count_inverse_sum_all_matrix, axis=0))

        # Add inverse sum all weighting features
        self.features = np.c_[self.features, inv_sum_all_sum, inv_sum_all_mean, inv_sum_all_std, inv_sum_all_min, inv_sum_all_max, inv_sum_all_count]

        # Inverse sum candidate weighting
        inverse_sum_candidate = 1 / np.sum(active_matrix, axis=1)
        inverse_sum_candidate[np.isnan(inverse_sum_candidate)] = 0
        inverse_sum_candidate[np.isinf(inverse_sum_candidate)] = 0
        inverse_sum_candidate_matrix = np.multiply(active_matrix, inverse_sum_candidate)
        count_inverse_sum_candidate_matrix = np.multiply(count_active_matrix, inverse_sum_candidate)

        inv_sum_candidate_sum = np.transpose(np.sum(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_mean = np.transpose(np.mean(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_std = np.transpose(np.std(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_min = np.transpose(np.min(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_max = np.transpose(np.max(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_count = np.transpose(np.sum(count_inverse_sum_candidate_matrix, axis=0))

        self.features = np.c_[static_features.features[candidates, :], self.features, inv_sum_candidate_sum, inv_sum_candidate_mean, inv_sum_candidate_std, inv_sum_candidate_min, inv_sum_candidate_max, inv_sum_candidate_count]
