############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

from slim_gsgp.algorithms.GP.representations.tree import Tree
import torch

class MultiObjectiveTree(Tree):
    """
    The MultiObjectiveTree class representing candidate solutions for multi-objective 
    problems in genetic programming. (adds pareto front and crowding distance attributes)

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    fitness : torch.Tensor
        1D tensor of fitness values of the tree (one for each objective).
    pareto_front : int, optional
        The rank of the Pareto front to which this individual belongs.
    crowding_distance : float, optional
        The crowding distance value.
    """
    def __init__(self, repr_):
        """
        Initializes a MultiObjectiveTree object inherited from Tree.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        super().__init__(repr_)
        
        # Add attributes for multi-objective optimization: pareto front and crowding distance
        self.pareto_front = None
        self.crowding_distance = None

    # Define the is_better method
    def is_better(self, other_solution):
        """
        Determines if this individual is better than another.

        The comparison logic is as follows:
        Primary Criterion: The Pareto front rank.
        Secondary Criterion: If both individuals are on 
        the same Pareto front, we use crowding distance.

        Parameters
        ----------
        other_solution : MultiObjectiveTree
            The other individual to compare against.

        Returns
        -------
        bool
            True if this individual is better than `other_solution`, False otherwise.
        """
        if self.pareto_front is None or other_solution.pareto_front is None:
            raise ValueError("Cannot compare individuals that do not have a Pareto front ranking.")
       
        # Primary Criterion: Pareto front rank
        # Secondary Criterion: Crowding distance
        if self.pareto_front < other_solution.pareto_front:
            return True
        elif self.pareto_front == other_solution.pareto_front:
            if self.crowding_distance is None or other_solution.crowding_distance is None:
                 raise ValueError("Crowding Distance is None. It should be calculated before comparison.")
            return self.crowding_distance > other_solution.crowding_distance
        return False


    def evaluate(self, fitness_functions: list, X, y, testing=False, new_data = False):
        """
        Evaluates the tree given the fitness functions, input data (X), and target data (y).

        Parameters
        ----------
        fitness_functions : list
            List of fitness functions to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        testing : bool, optional
            Flag indicating if the data is testing data. Default is False.
        new_data : bool, optional
            Flag indicating that the input data is new and the model is being used outside the training process.

        Returns
        -------
        None
            If the data is training or testing data, the fitness vector is attributed to the individual.
        torch.Tensor
            If exposed to new data, the fitness vector is returned.
        """
        
        # getting the predictions (i.e., semantics) of the individual
        preds = self.apply_tree(X)
        
        # Fitness for each objective
        fits_for_new_data = []
        
        for ffunction in fitness_functions:
            fit_value = ffunction(y, preds)
            fits_for_new_data.append(fit_value)
            
        fitness_vector = torch.tensor(fits_for_new_data)
        # if new (testing data) is being used, return the fitness vector for this new data
        if new_data:
            return fitness_vector

        # if not, attribute the fitness vector to the individual
        else:
            if testing:
                self.test_fitness = fitness_vector
            else:
                self.fitness = fitness_vector