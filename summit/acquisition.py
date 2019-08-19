from abc import ABC, abstractmethod

from math import log, floor
import random
import warnings
import numpy as np

class Acquisition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_max(self, samples, num_evaluations):
        pass

class HvI(Acquisition):
    ''' Hypervolume Improvement Acquisition Function

    This acquisition functions selects points based on the hypervolume improvement.
    The hypervolume improvement function is a modified version of the one proposed
    in Bradford et al.
    
    Parameters
    ---------- 
    reference: array
        The reference point used in the calculation of the hypervolume. 
    data: np.ndarray, optional
        A numpy array with the initial data used for comparison of hypervolume
    random_rate: `float`, optional
        The rate at which points will be selected at random instead of using 
        hypervolume. Defaults to 0.0s
    
    Attributes
    ----------
    data    
    
    Notes
    -----

    References:
    @article{Bradford2018,
        author = {Bradford, Eric and Schweidtmann, Artur M. and Lapkin, Alexei},
        doi = {10.1007/s10898-018-0609-2},
        issn = {0925-5001},
        journal = {Journal of Global Optimization},
        month = {jun},
        number = {2},
        pages = {407--438},
        publisher = {Springer US},
        title = {{Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm}},
        url = {http://link.springer.com/10.1007/s10898-018-0609-2},
        volume = {71},
        year = {2018}
        }
    
    ''' 
    def __init__(self, reference, data = [], random_rate=0.0):
        self._reference = reference
        self._data = data
        self._random_rate = random_rate

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, y):
        self._data = y

    def select_max(self, samples, num_evals=1):
        '''  Returns the point(s) that maximimize hypervolume improvement 
        
        Parameters
        ---------- 
        samples: np.ndarray
             The samples on which hypervolume improvement is calculated
        num_evals: `int`
            The number of points to return (with top hypervolume improvement)
        
        Returns
        -------
        hv_imp, index
            Returns a tuple with lists of the best hypervolume improvement
            and the indices of the corresponding points in samples       
        
        ''' 
        #Get the reference point, r
        # r = self._reference + 0.01*(np.max(samples, axis=0)-np.min(samples, axis=0)) 
        r = self._reference
        index = []
        mask = np.ones(samples.shape[0], dtype=bool)
        n = samples.shape[1]
        Ynew = self._data

        assert (self._random_rate <=1.) | (self._random_rate >=0.)
        if self._random_rate>0:
            num_random = round(self._random_rate*num_evals)
            random_selects = np.random.randint(0, num_evals, size=num_random)
        else:
            random_selects = np.array([])
        
        for i in range(num_evals):
            masked_samples = samples[mask, :]
            Yfront, _ = pareto_efficient(Ynew, maximize=True)
            if len(Yfront) ==0:
                raise ValueError('Pareto front length too short')

            hv_improvement = []
            hvY = HvI.hypervolume(-Yfront, [0, 0])
            #Determine hypervolume improvement by including
            #each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1,n)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = pareto_efficient(A, maximize=True)
                hv = HvI.hypervolume(-Afront, [0,0])
                hv_improvement.append(hv-hvY)
            
            hvY0 = hvY if i==0 else hvY0

            if i in random_selects:
                masked_index = np.random.randint(0, masked_samples.shape[0])
            else:
                #Choose the point that maximizes hypervolume improvement
                masked_index = hv_improvement.index(max(hv_improvement))

            samples_index = np.where((samples == masked_samples[masked_index, :]).all(axis=1))[0][0]
            new_point = samples[samples_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[samples_index] = False
            index.append(samples_index)

        if len(hv_improvement)==0:
            hv_imp = 0
        elif len(index) == 0:
            index = []
            hv_imp = 0
        else:
            #Total hypervolume improvement
            #Includes all points added to batch (hvY + last hv_improvement)
            #Subtracts hypervolume without any points added (hvY0)
            hv_imp = hv_improvement[masked_index] + hvY-hvY0
        return hv_imp, index

    @staticmethod
    def hypervolume(pointset, ref):
        """Compute the absolute hypervolume of a *pointset* according to the
        reference point *ref*.
        """
        hv = _HyperVolume(ref)
        return hv.compute(pointset)


class _HyperVolume:
    """
    This code is copied from the GA library DEAP. 
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.
    Minimization is implicitly assumed here!
    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []


    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.
        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].
        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        #######
        # fmder: Here it is assumed that every point dominates the reference point
        # for point in front:
        #     # only consider points that dominate the reference point
        #     if weaklyDominates(point, referencePoint):
        #         relevantPoints.append(point)
        relevantPoints = front
        # fmder
        #######
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            
            #######
            # fmder: Assume relevantPoints are numpy array
            # for j in range(len(relevantPoints)):
            #     relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in range(dimensions)]
            relevantPoints -= referencePoint
            # fmder
            #######

        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume


    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.
        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo != None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol


    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = _MultiList(dimensions)
        nodes = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList


    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]

class _MultiList: 
    """A special data structure needed by FonsecaHyperVolume. 
    
    It consists of several doubly linked lists that share common nodes. So, 
    every node has multiple predecessors and successors, one in every list.
    """

    class Node: 
        
        def __init__(self, numberLists, cargo=None): 
            self.cargo = cargo 
            self.next  = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists
    
        def __str__(self): 
            return str(self.cargo)

        def __lt__(self, other):
            return all(self.cargo < other.cargo)
        
    def __init__(self, numberLists):  
        """Constructor. 
        
        Builds 'numberLists' doubly linked lists.
        """
        self.numberLists = numberLists
        self.sentinel = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists  
        
        
    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr
    
    
    def __len__(self):
        """Returns the number of lists that are included in this _MultiList."""
        return self.numberLists
    
    
    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length
            
            
    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node
        
        
    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node
        
        
    def remove(self, node, index, bounds): 
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index): 
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor  
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node
    
    
    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous 
        nodes of the node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]

def _pareto_front(points):
    '''Calculate the pareto front of a 2 dimensional set'''
    try:
        assert points.all() == np.atleast_2d(points).all()
        assert points.shape[1] == 2
    except AssertionError:
        raise ValueError("Points must be 2 dimensional.")

    sorted_indices = np.argsort(points[:, 0])
    sorted = points[sorted_indices, :]
    front = np.atleast_2d(sorted[-1, :])
    front_indices = sorted_indices[-1]
    for i in range(2, sorted.shape[0]+1):
        if np.greater(sorted[-i, 1], front[:, 1]).all():
            front = np.append(front, 
                              np.atleast_2d(sorted[-i, :]),
                              axis=0)
            front_indices = np.append(front_indices,
                                      sorted_indices[-i])
    return front, front_indices


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


def remove_points_above_reference(Afront, r):
    A = sortrows(Afront)
    for p in range(len(Afront[0, :])):
        A = A[A[:,p]<= r[p], :]
    return A

def sortrows(A, i=0):
    '''Sort rows from matrix A by column i'''
    I = np.argsort(A[:, i])
    return A[I, :]
            
__all__ = ["hypervolume_kmax", "hypervolume"]