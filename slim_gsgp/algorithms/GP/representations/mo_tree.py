# Author: Filipa Pereira
# Date: March 2026

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


    def evaluate(self, fitness_functions: list, X, y, testing=False):
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
        # if we already have training fitness and it's not testing, do nothing
        if self.fitness is not None and not testing:
            return


        # find which fitness functions are structural vs performance-based
        metric_names = [f.__name__.lower() if hasattr(f, '__name__') else str(f).lower() for f in fitness_functions]
        
        #list of metrics that we know do not use data, but the structure
        structural_keywords = ['size', 'node', 'feature', 'depth', 'nao']
        
        preds = None
        calculated_fitness = []
        
        for i, ffunc in enumerate(fitness_functions):
            fname = metric_names[i]
            is_structural = any(k in fname for k in structural_keywords)
            
            val = None
            
            if is_structural:
                # get the right value to pass to the fitness function
                # 'gs_size' function expects (y, size), not (y, preds)
                
                if 'size' in fname or 'node' in fname:
                    # get the number of nodes as "y_pred"
                    val_input = float(self.node_count)
                    val = ffunc(y, val_input)
                    
                elif 'feature' in fname:
                    
                    if 'ind_repr' in ffunc.__code__.co_varnames: 
                        # the function can handle the tree
                        val = ffunc(y, self.repr_)
                    else:
                        # Fallback: extract features ourselves
                        from slim_gsgp.utils.utils import _traverse_get_features
                        f_set = set()
                        _traverse_get_features(self.repr_, f_set)
                        val_input = float(len(f_set))
                        val = val_input
                        
                elif 'nao' in fname:
                     # if the function asks for ind_repr, we pass the tree
                     if 'ind_repr' in ffunc.__code__.co_varnames:
                         val = ffunc(y, self.repr_)
                     else:
                         raise ValueError(f"Function {fname} requires tree structure but signature unknown.")
                         
            else:
                # --- PERFORMANCE LOGIC (RMSE, MAE) ---
                # These require actual predictions
                if preds is None:
                    preds = self.predict(X)
                    if isinstance(preds, torch.Tensor) and preds.requires_grad:
                        preds = preds.detach()
                
                # Standard call: f(y_true, y_pred)
                val = ffunc(y, preds)
            
            # Final sanity check
            if val is None:
                raise ValueError(f"Fitness function {fname} returned None.")
                
            calculated_fitness.append(val)

        # get final tensor
        final_tensor = torch.tensor(calculated_fitness) if not isinstance(calculated_fitness, torch.Tensor) else torch.tensor(calculated_fitness)
        
        if testing:
            self.test_fitness = final_tensor
        else:
            self.fitness = final_tensor