import numpy as np

def pareto_efficient(costs, maximize=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array


    """
    original_costs = costs
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        if maximize:
            nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        else:
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    return  costs, is_efficient  