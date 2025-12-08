# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import random

import numpy as np
import torch
from slim_gsgp.algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                                create_grow_random_tree)
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from sklearn.metrics import root_mean_squared_error

def protected_div(x1, x2):
    """Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(
        torch.abs(x2) > 0.001,
        torch.div(x1, x2),
        torch.tensor(1.0, dtype=x2.dtype, device=x2.device),
    )


def mean_(x1, x2):
    """
    Compute the mean of two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        The first tensor.
    x2 : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The mean of the two tensors.
    """
    return torch.div(torch.add(x1, x2), 2)


def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
        Indices representing the test partition.
    """
    torch.manual_seed(seed)
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


def tensor_dimensioned_sum(dim):
    """
    Generate a sum function over a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension to sum over.

    Returns
    -------
    function
    A function that sums tensors over the specified dimension.
    """

    def tensor_sum(input):
        return torch.sum(input, dim)

    return tensor_sum

def verbose_reporter(
        dataset, generation, pop_val_fitness, pop_test_fitness, timing, nodes
):
    """
    Prints a formatted report of generation, fitness values, timing, and node count.

    Parameters
    ----------
    generation : int
        Current generation number.
    pop_val_fitness : float
        Population's validation fitness value.
    pop_test_fitness : float
        Population's test fitness value.
    timing : float
        Time taken for the process.
    nodes : int
        Count of nodes in the population.

    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    digits_dataset = len(str(dataset))
    digits_generation = len(str(generation))
    digits_val_fit = len(str(float(pop_val_fitness)))
    if pop_test_fitness is not None:
        digits_test_fit = len(str(float(pop_test_fitness)))
        test_text_init = (
                "|"
                + " " * 3
                + str(float(pop_test_fitness))
                + " " * (23 - digits_test_fit)
                + "|"
        )
        test_text = (
                " " * 3 + str(float(pop_test_fitness)) + " " * (23 - digits_test_fit) + "|"
        )
    else:
        digits_test_fit = 4
        test_text_init = "|" + " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
        test_text = " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
    digits_timing = len(str(timing))
    digits_nodes = len(str(nodes))

    if generation == 0:
        print("Verbose Reporter")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|         Dataset         |  Generation  |     Train Fitness    |       Test Fitness       |        "
            "Timing          |      Nodes       |"
        )
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(pop_val_fitness))
            + " " * (20 - digits_val_fit)
            + test_text_init
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
            + " " * 6
            + str(nodes)
            + " " * (12 - digits_nodes)
            + "|"
        )
    else:
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(pop_val_fitness))
            + " " * (20 - digits_val_fit)
            + "|"
            + test_text
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
            + " " * 6
            + str(nodes)
            + " " * (12 - digits_nodes)
            + "|"
        )

############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
#                                                                          #
#                                                                          #
############################################################################
def mo_verbose_reporter(
        dataset, generation, pop_val_fitness_vector, pop_test_fitness_vector, timing, nodes
):
    """
    Prints a formatted report of generation, multi-objective fitness values, timing, and node count.

    Parameters
    ----------
    generation : int
        Current generation number.
    pop_val_fitness_vector : torch.Tensor or list
        Elite's vector of fitness values (Train).
    pop_test_fitness_vector : torch.Tensor or list, optional
        Elite's vector of fitness values (Test/Validation).
    timing : float
        Time taken for the process.
    nodes : int
        Count of nodes in the elite individual.

    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    
    # Determine number of objectives
    if pop_val_fitness_vector is not None:
        num_objs = len(pop_val_fitness_vector)
    elif pop_test_fitness_vector is not None:
        num_objs = len(pop_test_fitness_vector)
    else:
        num_objs = 1 

    # Set column width dynamically
    width_per_obj = 10 
    fit_col_width = max(20, (num_objs * width_per_obj) + (num_objs - 1) * 3)

    # Helper to format fitness vectors
    def format_vec(vec):
        if vec is None:
            return "N/A"
        if isinstance(vec, torch.Tensor):
            vec = vec.tolist()
        return " | ".join([f"{f:.4f}" for f in vec])

    train_str = format_vec(pop_val_fitness_vector)
    test_str = format_vec(pop_test_fitness_vector)

    # dataset + gen + 2*fit(width) + time + nodes + formatting chars
    total_width = 15 + 6 + 2 * fit_col_width + 10 + 8 + 19 
    sep_line = "-" * total_width

    # Header
    if generation == 0:
        print("\n" + sep_line)
        header = (
            f"| {'Dataset':<15} | {'Gen':^6} | "
            f"{'Train Fitness':<{fit_col_width}} | "
            f"{'Test Fitness':<{fit_col_width}} | "
            f"{'Time':^10} | {'Nodes':^8} |"
        )
        print(header)
        print(sep_line)

    row = (
        f"| {dataset:<15} | {generation:^6} | "
        f"{train_str:<{fit_col_width}} | "
        f"{test_str:<{fit_col_width}} | "
        f"{timing:^10.4f} | {nodes:^8} |"
    )
    print(row)

def get_terminals(X):
    """
    Get terminal nodes for a dataset.

    Parameters
    ----------
    X : (torch.Tensor)
        An array to get the set of TERMINALS from, it will correspond to the columns.

    Returns
    -------
    dict
        Dictionary of terminal nodes.
    """

    return  {f"x{i}": i for i in range(len(X[0]))}


def get_best_min(population, n_elites):
    """
    Get the best individuals from the population with the minimum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, n_elites)
        elites = [population.population[i] for i in idx[:n_elites]]
        return elites, elites[np.argmin([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmin(population.fit)]
        return [elite], elite


def get_best_max(population, n_elites):
    """
    Get the best individuals from the population with the maximum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        The list of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, -n_elites)
        elites = [population.population[i] for i in idx[-n_elites:]]
        return elites, elites[np.argmax([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmax(population.fit)]
        return [elite], elite
    



############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
# find_mo_elites_default function                                          #
#                                                                          #
############################################################################

def find_mo_elites_ideal_candidate(population, n_elites, minimization_flags, ideal_candidate_values: list):
    """
    Function to find MO elites based on the Euclidean Distance to an Ideal Point.

    Parameters
    ----------
    population : MultiObjectivePopulation
        The population of individuals.
    n_elites : int
        Number of elites to return.
    minimization_flags : list of bool 
        Minimization flags for each objective.
    ideal_candidate_values : list of float
        The user-defined ideal values for each objective.

    Returns
    -------
    list
        The list of elite individuals.
    MultiObjectiveTree
        Best elite individual.    
    """
    
    if not population.population:
        return [], None
    
    if population.fit is None or len(population.fit) != len(population.population):
        # Recalculate fitness matrix if not present or inconsistent
        fits = [ind.fitness for ind in population.population]
        # Ensure no None values before stacking
        if any(f is None for f in fits):
             raise ValueError("Some individuals in the population have not been evaluated (fitness is None).")
        population.fit = torch.stack(fits)
        
    num_objectives_pop = population.fit.shape[1]
    num_objectives_ideal = len(ideal_candidate_values)
    if num_objectives_ideal != num_objectives_pop:
        raise ValueError(
            f"Dimension mismatch: 'ideal_candidate_values' has {num_objectives_ideal} objectives, "
            f"but the population fitness has {num_objectives_pop} objectives."
        )

    
    dtype = population.fit.dtype
    device = population.fit.device
    ideal_point = torch.tensor(ideal_candidate_values, dtype=dtype, device=device)
    
    fit_matrix = population.fit
    
    min_vals = fit_matrix.min(dim=0).values
    max_vals = fit_matrix.max(dim=0).values
    
    ranges = max_vals - min_vals
    non_zero_range_mask = ranges > 0
    
    # Initialize with zeros for constant objectives
    normalized_fit = torch.zeros_like(fit_matrix, dtype=dtype, device=device)
    normalized_ideal_point = torch.zeros_like(ideal_point, dtype=dtype, device=device)
    
    if non_zero_range_mask.any():
        valid_indices = torch.where(non_zero_range_mask)[0]
        valid_ranges = ranges[valid_indices]
        valid_min_vals = min_vals[valid_indices]
        
        # Apply Min-Max normalization for valid objectives and ideal point
        normalized_fit[:, valid_indices] = (
            fit_matrix[:, valid_indices] - valid_min_vals
        ) / valid_ranges
        
        normalized_ideal_point[valid_indices] = (
            ideal_point[valid_indices] - valid_min_vals
        ) / valid_ranges
    
    distances = torch.sqrt(torch.pow(normalized_fit - normalized_ideal_point, 2).sum(dim=1) )
    
    
    sorted_indices = torch.argsort(distances)
    
    elites_indices = sorted_indices[:n_elites].tolist()
    elites = [population.population[i] for i in elites_indices]
    
    elite = elites[0] if elites else None
    
    return elites, elite
######################

def find_mo_elites_default(population, n_elites, minimization_flags, use_first_obj=False, fronts=None):
    if not population.population:
        return [], None
    
    if use_first_obj:
        # First-objective logic (Mantém-se igual para o Cenário 4)
        obj_index = 0
        is_min = minimization_flags[obj_index]
        
        if is_min:
            best_ind = min(population.population, key=lambda ind: ind.fitness[obj_index])
        else:
            best_ind = max(population.population, key=lambda ind: ind.fitness[obj_index])
            
        elites = [best_ind] * n_elites
        elite = best_ind
        
    else:
        # NSGA-II logic (Atualizada para Multi-Frente)
        if fronts is None:
            fronts = population.non_dominated_sorting(minimization_flags)

        elites = []
        
        # Iterar pelas frentes (0, 1, 2...) até ter elites suficientes
        for front in fronts:
            if len(elites) >= n_elites:
                break
                
            # Calcular Crowding Distance se necessário (para desempatar dentro da frente)
            if len(front) > 0:
                if front[0].crowding_distance is None:
                    population.calculate_crowding_distance(front)
                
                # Ordenar por CD decrescente (maior diversidade é melhor)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            
            # Calcular quantos faltam para encher o pedido
            missing = n_elites - len(elites)
            
            # Adicionar os melhores desta frente
            elites.extend(front[:missing])

        # O "Melhor Elite" é sempre o primeiro da lista (Frente 0, Maior CD)
        elite = elites[0] if elites else None  
            
    return elites, elite


def find_mo_elites_default(population, n_elites, minimization_flags, use_first_obj=False, fronts=None):
    """
    Default function to find MO elites (NSGA-II or first-objective logic).

    Parameters
    ----------
    population : MultiObjectivePopulation
        The population of individuals.
    n_elites : int
        NNumber of elites to return.
    minimization_flags : list of bool
        Minimization flags for each objective.
    use_first_obj : bool, optional
        If True, selects elites based only on the first objective. 
        If False (NSGA-II default), uses Pareto Front 1 and Crowding Distance.
    fronts : list of lists, optional
        Pre-calculated Pareto fronts.

    Returns
    -------
    list
        The list of elite individuals.
    MultiObjectiveTree
        Best individual from the elites.
    """
    if not population.population:
        return [], None
    
    if use_first_obj:
        # First-objective logic
        obj_index = 0
        is_min = minimization_flags[obj_index]
        
        # Find the best individual based on the first objective
        if is_min:
            best_ind = min(population.population, key=lambda ind: ind.fitness[obj_index])
        else:
            best_ind = max(population.population, key=lambda ind: ind.fitness[obj_index])
            
        # Repeat the best individual to fill the elites list
        elites = [best_ind] * n_elites
        elite = best_ind
        
    else:
        # NSGA-II logic
        if fronts is None:
            # Recalculate ranking if fronts were not provided
            fronts = population.non_dominated_sorting(minimization_flags)

        elites = []

        # Iterate through fronts (0, 1, 2...) until enough elites are collected
        for front in fronts:
            if len(elites) >= n_elites:
                break
            
            # Calculate Crowding Distance if needed (to break ties within the front)
            if len(front) > 0:
                if front[0].crowding_distance is None:
                    population.calculate_crowding_distance(front)
                
                # Sort by descending Crowding Distance
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            
            missing = n_elites - len(elites) #how many are still needed
            
            # Add the best individuals from this front
            elites.extend(front[:missing])

        # The "Best Elite"
        elite = elites[0] if elites else None  
            
    return elites, elite


def get_random_tree(
        max_depth,
        FUNCTIONS,
        TERMINALS,
        CONSTANTS,
        inputs,
        p_c=0.3,
        grow_probability=1,
        logistic=True,
):
    """
    Get a random tree using either grow or full method.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    FUNCTIONS : dict
        Dictionary of functions.
    TERMINALS : dict
        Dictionary of terminals.
    CONSTANTS : dict
        Dictionary of constants.
    inputs : torch.Tensor
        Input tensor for calculating semantics.
    p_c : float, default=0.3
        Probability of choosing a constant.
    grow_probability : float, default=1
        Probability of using the grow method.
    logistic : bool, default=True
            Whether to use logistic semantics.

    Returns
    -------
    Tree
        The generated random tree.
    """
    if random.random() < grow_probability:
        tree_structure = create_grow_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
        )
    else:
        tree_structure = create_full_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
        )

    tree = Tree(
        structure=tree_structure,
        train_semantics=None,
        test_semantics=None,
        reconstruct=True,
    )
    tree.calculate_semantics(inputs, testing=False, logistic=logistic)
    return tree


def generate_random_uniform(lower, upper):
    """
    Generate a random number within a specified range using numpy random.uniform.

    Parameters
    ----------
    lower : float
        The lower bound of the range for generating the random number.
    upper : float
        The upper bound of the range for generating the random number.

    Returns
    -------
    Callable
        A function that when called, generates a random number within the specified range.
    Notes
    -----
    The returned function takes no input and returns a random float between lower and upper whenever called.
    """

    def generate_num():
        """
        Generate a random number within a specified range.

        Returns
        -------
        float
            A random number between the defined lower and upper bounds.
        """
        return random.uniform(lower, upper)

    generate_num.lower = lower
    generate_num.upper = upper
    return generate_num


def show_individual(tree, operator):
    """
    Display an individual's structure with a specified operator.

    Parameters
    ----------
    tree : Tree
        The tree representing the individual.
    operator : str
        The operator to display ('sum' or 'prod').

    Returns
    -------
    str
        The string representation of the individual's structure.
    """
    op = "+" if operator == "sum" else "*"

    return f" {op} ".join(
        [
            (
                str(t.structure)
                if isinstance(t.structure, tuple)
                else (
                    f"f({t.structure[1].structure})"
                    if len(t.structure) == 3
                    else f"f({t.structure[1].structure} - {t.structure[2].structure})"
                )
            )
            for t in tree.collection
        ]
    )


def gs_rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        The root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred[0])


def gs_size(y_true, y_pred):
    """
    Get the size of the predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    int
        The size of the predicted values.
    """
    if isinstance(y_pred, (int, float)):
        return y_pred

    return y_pred[1]

############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
# adapted       validate_inputs                                            #
#                                                                          #
############################################################################

def validate_inputs(X_train, y_train, X_test, y_test, pop_size, n_iter, elitism, n_elites, init_depth, log_path,
                    prob_const, tree_functions, tree_constants, log, verbose, minimization, n_jobs, test_elite,
                    fitness_function, initializer, tournament_size, offspring_size):
    """
    Validates the inputs based on the specified conditions.

    Parameters
    ----------
    tournament_size : int OR LIST, optional
        Tournament size(s) to utilize during selection. Can be a single integer for GP 
        or a list of integers (one per objective) for MOGP.
    offspring_size : int, optional
        The number of offspring to produce in each generation. If None, defaults to the population size.
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved.
    log : int, optional
        Level of detail to utilize in logging.
    verbose : int, optional
        Level of detail to include in console output.
    minimization : bool OR LIST, optional
        If True, the objective is to minimize the fitness function. If False, maximize it. 
        Can be a single bool for GP or a list of bools (one per objective) for MOGP. (default is True)
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.
    fitness_function : str OR LIST, optional
        The fitness function(s) used for evaluating individuals. Can be a single string for GP 
        or a list of strings (one per objective) for MOGP. (default is from gp_solve_parameters)
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    prob_const : float, optional
        The probability of introducing constants into the trees during evolution.
    tree_functions : list, optional
        List of allowed functions that can appear in the trees Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    """
    from slim_gsgp.config.gp_config import fitness_function_options

    if not isinstance(X_train, torch.Tensor):
        raise TypeError("X_train must be a torch.Tensor")
    if not isinstance(y_train, torch.Tensor):
        raise TypeError("y_train must be a torch.Tensor")
    if X_test is not None and not isinstance(X_test, torch.Tensor):
        raise TypeError("X_test must be a torch.Tensor")
    if y_test is not None and not isinstance(y_test, torch.Tensor):
        raise TypeError("y_test must be a torch.Tensor")
    if not isinstance(pop_size, int):
        raise TypeError("pop_size must be an int")
    if not isinstance(n_iter, int):
        raise TypeError("n_iter must be an int")
    if not isinstance(elitism, bool):
        raise TypeError("elitism must be a bool")
    if not isinstance(n_elites, int):
        raise TypeError("n_elites must be an int")
    if not isinstance(init_depth, int):
        raise TypeError("init_depth must be an int")
    if not isinstance(log_path, str):
        raise TypeError("log_path must be a str")
    

    if offspring_size is not None and not isinstance(offspring_size, int):
        raise TypeError("offspring_size must be an int or None")
############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

    is_mogp = isinstance(fitness_function, list)
    if is_mogp:
        if not isinstance(minimization, list) or not isinstance(tournament_size, list):
            raise TypeError("MOGP requires 'minimization' and 'tournament_size' to be lists.")
        
        funcs, flags, sizes = fitness_function, minimization, tournament_size
        if not (len(funcs) == len(flags) == len(sizes) and len(funcs) > 0):
             raise ValueError("MOGP lists ('fitness_function', 'minimization', 'tournament_size') must have the same, non-zero length.")

        for i, f in enumerate(funcs):
            if not isinstance(f, str) or f.lower() not in fitness_function_options:
                valid_funcs = list(fitness_function_options.keys())
                raise ValueError(f"Fitness function at index {i} ('{f}') is invalid. Valid options: {', '.join(valid_funcs)}")
            
            if not isinstance(flags[i], bool):
                 raise TypeError(f"Minimization flag at index {i} must be a boolean.")
            
            if not isinstance(sizes[i], int) or sizes[i] <= 1:
                raise ValueError(f"Tournament size at index {i} must be at least 2.")
        
        if not isinstance(initializer, str):
            raise TypeError("initializer must be a str")
            
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be an int")
        assert n_jobs >= 1, "n_jobs must be at least 1"

        if not isinstance(test_elite, bool):
            raise TypeError("test_elite must be a bool")


    else: # is_mogp é False, it's traditional GP
        if isinstance(minimization, list) or isinstance(tournament_size, list):
            raise TypeError("For single-objective GP, 'minimization' must be a bool and 'tournament_size' an int, not lists.")
        
        if not isinstance(minimization, bool):
            raise TypeError("minimization must be a bool")
        if not isinstance(fitness_function, str):
            raise TypeError("fitness_function must be a str")
        if fitness_function.lower() not in fitness_function_options:
            valid_funcs = list(fitness_function_options.keys())
            raise ValueError(f"fitness function must be one of: {', '.join(valid_funcs)}")
        if not isinstance(initializer, str):
            raise TypeError("initializer must be a str")
        if not isinstance(tournament_size, int):
            raise TypeError("tournament_size must be an int")
        if tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be an int")
        assert n_jobs >= 1, "n_jobs must be at least 1"
        if not isinstance(test_elite, bool):
            raise TypeError("test_elite must be a bool")


    # assuring the prob_const is valid
    if not (isinstance(prob_const, float) or isinstance(prob_const, int)):
        raise TypeError("prob_const must be a float (or an int when probability is 1 or 0)")

    if not 0 <= prob_const <= 1:
        raise ValueError("prob_const must be a number between 0 and 1")

    if n_iter < 1:
        raise ValueError("n_iter must be greater than 0")

    # Ensuring the functions and constants passed are valid
    if not isinstance(tree_functions, list) or len(tree_functions) == 0:
        raise TypeError("tree_functions must be a non-empty list")

    if not isinstance(tree_constants, list) or len(tree_constants) == 0:
        raise TypeError("tree_constants must be a non-empty list")

    assert all(isinstance(elem, (int, float)) and not isinstance(elem, bool) for elem in tree_constants), \
    "tree_constants must be a list containing only integers and floats"

    if not isinstance(log, int):
        raise TypeError("log_level must be an int")

    assert 0 <= log <= 4, "log_level must be between 0 and 4"

    if not isinstance(verbose, int):
        raise TypeError("verbose level must be an int")

    assert 0 <= verbose <= 1, "verbose level must be either 0 or 1"



def check_slim_version(slim_version):
    """
    Validate the slim_gsgp version given as input bu the users and assign the correct values to the parameters op, sig and trees
    Parameters
    ----------
    slim_version : str
        Name of the slim_gsgp version.

    Returns
    -------
    op, sig, trees
        Parameters reflecting the kind of operation considered, the use of the sigmoid and the use of multiple trees.
    """
    if slim_version == "SLIM+SIG2":
        return "sum", True, True
    elif slim_version == "SLIM*SIG2":
        return "mul", True, True
    elif slim_version == "SLIM+ABS":
        return "sum", False, False
    elif slim_version == "SLIM*ABS":
        return "mul", False, False
    elif slim_version == "SLIM+SIG1":
        return "sum", True, False
    elif slim_version == "SLIM*SIG1":
        return "mul", True, False
    else:
        raise Exception('Invalid SLIM configuration')

def _evaluate_slim_individual(individual, ffunction, y, testing=False, operator="sum"):
    """
    Evaluate the individual using a fitness function.

    Args:
        ffunction: Fitness function to evaluate the individual.
        y: Expected output (target) values as a torch tensor.
        testing: Boolean indicating if the evaluation is for testing semantics.
        operator: Operator to apply to the semantics ("sum" or "prod").

    Returns:
        None
    """
    if operator == "sum":
        operator = torch.sum
    else:
        operator = torch.prod

    if testing:
        individual.test_fitness = ffunction(
            y,
            torch.clamp(
                operator(individual.test_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

    else:
        individual.fitness = ffunction(
            y,
            torch.clamp(
                operator(individual.train_semantics, dim=0),
                -1000000000000.0,
                1000000000000.0,
            ),
        )

        # if testing is false, return the value so that training parallelization has effect
        return ffunction(
                y,
                torch.clamp(
                    operator(individual.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )