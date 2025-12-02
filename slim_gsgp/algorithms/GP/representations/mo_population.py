############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

from joblib import Parallel, delayed
import torch
from slim_gsgp.algorithms.GP.representations.population import Population
from slim_gsgp.algorithms.GP.representations.tree_utils import _execute_tree
from slim_gsgp.algorithms.GP.representations.mo_tree import MultiObjectiveTree


class MultiObjectivePopulation(Population):
    def __init__(self, pop):
        """
        Initializes a population of Multi-Objective Trees.

        Parameters
        ----------
        pop : list
            The list of individual MultiObjectiveTree objects that make up the population.

        Returns
        -------
        None
        """
        if not all(isinstance(ind, MultiObjectiveTree) for ind in pop):
            raise TypeError("MultiObjectivePopulation can only contain MultiObjectiveTree individuals.")

        super().__init__(pop)

    def evaluate(self, fitness_functions: list, X, y, n_jobs=1):
        """
        Evaluates the population given a list of fitness functions, input data (X), and target data (y).

        Attributes a fitness vector to each individual and stores a
        complete fitness matrix in `self.fit`.

        Parameters
        ----------
        fitness_functions : list
            A list of callable fitness function objects.
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The expected output (target) values.
        n_jobs : int
            The maximum number of concurrently running jobs for joblib parallelization.

        Returns
        -------
        None          

        """
        # Evaluate each individual for all objectives in parallel
        y_preds = Parallel(n_jobs=n_jobs)(
            delayed(_execute_tree)(
                individual.repr_, X,
                individual.FUNCTIONS, individual.TERMINALS, individual.CONSTANTS,
            ) for individual in self.population
        )

        # Compute fitness for each objective
        fits_per_objective = []
        for ffunc in fitness_functions:

            func_name = ffunc.__name__ if hasattr(ffunc, "__name__") else str(ffunc)
            if "size" in func_name.lower():
                fits_for_this_objective = [ffunc(y, individual.node_count) for individual in self.population]
            else:
                fits_for_this_objective = [ffunc(y, y_pred_ind) for y_pred_ind in y_preds]

            fits_per_objective.append(torch.tensor(fits_for_this_objective))
        
        fitness_matrix = torch.stack(fits_per_objective).T
        
        # Assign the corresponding fitness vector to each individual
        for i, individual in enumerate(self.population):
            individual.fitness = fitness_matrix[i]
            
        self.fit = fitness_matrix

    def non_dominated_sorting(self, minimization_flags: list):
        """
        Sorts the population into different non-domination levels.

        Assigns a `pareto_front` attribute to each individual based on their dominance level.
        

        Parameters
        ----------
        minimization_flags : list of bool
            The list indicating if each objective is for minimization (True)
            or maximization (False).

        Returns
        -------
        list of lists
            A list where each element is a list of individuals belonging to a
            specific Pareto front.
        """
        domination_counts = {id(ind): 0 for ind in self.population}
        dominated_solutions = {id(ind): [] for ind in self.population}
        fronts = [[]]

        # Determine domination relationships
        for i, p in enumerate(self.population):
            for q in self.population[i+1:]:
                if self._dominates(p, q, minimization_flags):
                    dominated_solutions[id(p)].append(q)
                    domination_counts[id(q)] += 1
                elif self._dominates(q, p, minimization_flags):
                    dominated_solutions[id(q)].append(p)
                    domination_counts[id(p)] += 1

        # Build the first front
        for p in self.population:
            if domination_counts[id(p)] == 0:
                p.pareto_front = 0
                fronts[0].append(p)

        # Build subsequent fronts
        i = 0
        while i < len(fronts):
            current_front = fronts[i]

            next_front = []
            for p in current_front:
                for q in dominated_solutions[id(p)]:
                    domination_counts[id(q)] -= 1
                    if domination_counts[id(q)] == 0:
                        q.pareto_front = i + 1
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts

    
    def _dominates(self, ind1, ind2, minimization_flags: list):
        """
        Checks if individual 1 dominates individual 2.

        Parameters
        ----------
        ind1 : MultiObjectiveTree
            The first individual.
        ind2 : MultiObjectiveTree
            The second individual to compare against.
        minimization_flags : list of bool
            A list where each boolean indicates if the corresponding objective
            is a minimization problem (True) or a maximization problem (False).

        Returns
        -------
        bool
            True if ind1 dominates ind2, False otherwise.
        """
        fit1 = ind1.fitness
        fit2 = ind2.fitness

        # Check if all objectives are no worse
        all_no_worse = True
        for i in range(len(fit1)):
            if minimization_flags[i]:
                if fit1[i] > fit2[i]:
                    all_no_worse = False
                    break
            else: # maximization
                if fit1[i] < fit2[i]:
                    all_no_worse = False
                    break
    
        if not all_no_worse:
            return False

        # Check if any objective is better
        any_better = False
        for i in range(len(fit1)):
            if minimization_flags[i]:
                if fit1[i] < fit2[i]:
                    any_better = True
                    break
            else: # maximization
                if fit1[i] > fit2[i]:
                    any_better = True
                    break
                
        return any_better


    def calculate_crowding_distance(self, front):
        """
        Assigns a crowding distance to each individual in a given Pareto front.

        Parameters
        ----------
        front : list
            A list of MultiObjectiveTree individuals that form a single Pareto front.
        """
        if len(front) == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        num_objectives = len(front[0].fitness)

        for i in range(num_objectives):
            front.sort(key=lambda ind: ind.fitness[i])
            min_val = front[0].fitness[i]
            max_val = front[-1].fitness[i]
            
            if max_val == min_val:
                continue
            
            scale = float(max_val - min_val)

            # Infinite distance to the boundary solutions
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            for j in range(1, len(front) - 1):
                distance = front[j+1].fitness[i] - front[j-1].fitness[i]
                
                # Add the normalized distance to the individual's total crowding distance.
                # The total distance is the sum of distances for each objective.
                front[j].crowding_distance += distance / scale