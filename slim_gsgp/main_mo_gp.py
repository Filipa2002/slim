############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

"""
This script runs the Multi-Objective Genetic Programming (MOGP) algorithm 
on various datasets and configurations, logging the results for further analysis.

It uses Nested Tournament Selection for parent selection and NSGA-II for 
survival selection.
"""
import uuid
import os
import warnings
import torch
from slim_gsgp.algorithms.GP.mogp import MOGP
from slim_gsgp.algorithms.GP.operators.mutators import mutate_tree_subtree
from slim_gsgp.algorithms.GP.representations.tree_utils import tree_depth
from slim_gsgp.config.gp_config import * 
from slim_gsgp.selection.selection_algorithms import nested_tournament_selection, tournament_selection_nsga2
from slim_gsgp.selection.survival_strategies import nsga2_survival, generational_survival
from slim_gsgp.utils.logger import log_settings
from slim_gsgp.utils.utils import (get_terminals, validate_inputs, find_mo_elites_default, find_mo_elites_ideal_candidate) 

def mo_gp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
          dataset_name: str = None,
          pop_size: int = gp_parameters["pop_size"],
          
          selector_strategy: str = "nested_tournament",
          survival_strategy: str = "nsga2",
          offspring_size: int | None = None,

          n_iter: int = gp_solve_parameters["n_iter"],
          p_xo: float = gp_parameters['p_xo'],
          n_elites: int = gp_solve_parameters["n_elites"],
          max_depth: int | None = gp_solve_parameters["max_depth"],
          init_depth: int = gp_pi_init["init_depth"],
          log_path: str = None, seed: int = gp_parameters["seed"],
          log_level: int = gp_solve_parameters["log"],
          verbose: int = gp_solve_parameters["verbose"],
          
          # Specific MOGP Parameters
          fitness_functions: list = mo_parameters["mo_fitness_functions"], 
          minimization_flags: list = mo_parameters["mo_minimization_flags"], 
          tournament_sizes: list = mo_parameters["mo_tournament_sizes"],
          elitism_strategy: str = "nsga2",
    
          
          
          initializer: str = gp_parameters["initializer"],     
          n_jobs: int = gp_solve_parameters["n_jobs"],
          prob_const: float = gp_pi_init["p_c"],
          tree_functions: list = list(FUNCTIONS.keys()),
          tree_constants: list = [float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
          test_elite: bool = gp_solve_parameters["test_elite"]):

    """
    Main function to execute the Multi-Objective Genetic Programming (MOGP) algorithm on specified datasets
    
    Parameters
    ----------
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    selector_strategy : str, optional
        The selection strategy for parent selection. Options are "nested_tournament" or "nsga2" (default is "nested_tournament").
    survival_strategy : str, optional
        The survival selection strategy. Options are "nsga2" or "generational" (default is "nsga2").
    offspring_size : int, optional
        The size of the offspring population to be generated in each generation. If None, it defaults to pop_size.
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
    n_elites : int, optional
        The number of elites.
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved. Defaults to `os.path.join(os.getcwd(), "log", "mo_gp.csv")`
    seed : int, optional
        Seed for the randomness
    log_level : int, optional
        Level of detail to utilize in logging.
    verbose : int, optional
        Level of detail to include in console output.
    fitness_functions : list, optional
        A list of fitness function names, one for each objective. (Default is from mo_parameters)
    minimization_flags : list, optional
        A list of booleans indicating if each corresponding objective is for minimization (True) or maximization (False). (Default is from mo_parameters)
    tournament_sizes : list, optional
        A list of integers defining the tournament size for each objective during Nested Tournament Selection. (Default is from mo_parameters)
    elitism_strategy : str, optional
        The elitism strategy to use. Options are "nsga2" (Rank+CD), "first_obj", "ideal_point".
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    prob_const : float, optional
        The probability of a constant being chosen rather than a terminal in trees creation (default: 0.2).
    tree_functions : list, optional
        List of allowed functions that can appear in the trees. Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.

    Returns
    -------
    MultiObjectiveTree
        Returns the best individual according to the tracking strategy at the last generation.
        """

    # ================================
    #           Input Validation
    # ================================

    # Setting the log_path
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "log", "mo_gp.csv")

    validate_inputs(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter, 
        elitism=True, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const, 
        tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose, 
        minimization=minimization_flags, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_functions, 
        initializer=initializer, tournament_size=tournament_sizes, offspring_size=offspring_size
    )

    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    if test_elite and (X_test is None or y_test is None):
        warnings.warn("If test_elite is True, a test dataset must be provided. test_elite has been set to False")
        test_elite = False
    
    if not isinstance(max_depth, int) and max_depth is not None:
         raise TypeError("max_depth value must be a int or None")

    assert max_depth is None or init_depth <= max_depth, f"max_depth must be at least {init_depth}"
        
    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset_1"

    # creating a list with the valid available initializers
    valid_initializers = list(initializer_options)

    # assuring the chosen initializer is valid
    assert initializer.lower() in initializer_options.keys(), \
        "initializer must be " + f"{', '.join(valid_initializers[:-1])} or {valid_initializers[-1]}" \
             if len(valid_initializers) > 1 else valid_initializers[0]

    # ================================
    #         Parameter Definition
    # ================================
    unique_run_id = uuid.uuid1()

    if offspring_size is None:
        if survival_strategy == "generational":
            offspring_size = pop_size - n_elites
        else:
            offspring_size = pop_size
    if offspring_size <= 0:
        raise ValueError(f"Calculated offspring_size is {offspring_size} (<=0). Check your pop_size and n_elites.")

    if n_elites > pop_size:
        warnings.warn(f"The number of elites ({n_elites}) cannot exceed the population size ({pop_size}). n_elites has been capped to {pop_size}.", UserWarning)
        n_elites = pop_size

    #   *************** GP_PI_INIT ***************
    TERMINALS = get_terminals(X_train)
    gp_pi_init["TERMINALS"] = TERMINALS
    try:
        gp_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS.keys())
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])

    try:
        gp_pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: torch.tensor(num, dtype=torch.float32) 
                                   for n in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS.keys())
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])
            
    gp_pi_init["p_c"] = prob_const
    gp_pi_init["init_pop_size"] = pop_size
    gp_pi_init["init_depth"] = init_depth
    
   #  *************** GP_PARAMETERS ***************
    gp_parameters["p_xo"] = p_xo
    gp_parameters["p_m"] = 1 - gp_parameters["p_xo"]
    gp_parameters["pop_size"] = pop_size    


    if selector_strategy == "nsga2": 
        gp_parameters["selector"] = tournament_selection_nsga2(pool_size=2)
    else:
        gp_parameters["selector"] = nested_tournament_selection(
            tournament_sizes=tournament_sizes, 
            minimization_flags=minimization_flags
        )

    
    ####################APAGAR
    # elitism_state = {"current_ideal": None}

    # def stateful_ideal_wrapper(pop, n, min_flags, ideal_candidate_values=None, fronts=None):
    #     """
    #     Função inteligente que:
    #     - Se receber um novo valor (do mogp.py), guarda-o.
    #     - Se não receber valor (do survival), usa o guardado.
    #     """
    #     # Se o MOGP passar um valor novo, atualizamos a memória
    #     if ideal_candidate_values is not None:
    #         elitism_state["current_ideal"] = ideal_candidate_values
        
    #     # Recuperamos o valor da memória se não for passado
    #     val_to_use = ideal_candidate_values if ideal_candidate_values is not None else elitism_state["current_ideal"]
        
    #     if val_to_use is None:
    #          raise ValueError("[CRITICAL] Ideal Point falhou! O Survival foi chamado sem haver um Ponto Ideal definido.")

    #     return find_mo_elites_ideal_candidate(pop, n, min_flags, val_to_use)

    # # 2. Atribuir a função correta ao dicionário de parâmetros
    # if elitism_strategy == "ideal_point":
    #     gp_parameters["find_elit_func"] = stateful_ideal_wrapper
    


    # if elitism_strategy == "ideal_point":
    #     # We use a dictionary to maintain state between calls
    #     # This way, when generational_survival calls the function without extra arguments,
    #     # it uses the last known 'ideal'
    #     elitism_state = {"last_ideal": None}
        
    #     def stateful_ideal_finder(pop, n, min_flags, ideal_candidate_values=None, fronts=None):
    #         # If we receive a new value (from mogp.py), we update the memory
    #         if ideal_candidate_values is not None:
    #             elitism_state["last_ideal"] = ideal_candidate_values
            
    #         # If we do not receive (from survival), we use the memory
    #         current_ideal = ideal_candidate_values if ideal_candidate_values is not None else elitism_state["last_ideal"]     
            
    #         if current_ideal is None:
    #             raise ValueError(
    #                 "[CRITICAL ERROR] Ideal Point Elitism: 'ideal_candidate_values' is None. "
    #                 "This means that '_update_dynamic_ideal_point' did not run before selection."
    #             )

    #         return find_mo_elites_ideal_candidate(pop, n, min_flags, current_ideal)

    #     gp_parameters["find_elit_func"] = find_mo_elites_ideal_candidate
        
    if elitism_strategy == "ideal_point":
        gp_parameters["find_elit_func"] = find_mo_elites_ideal_candidate

    elif elitism_strategy == "first_obj":
        gp_parameters["find_elit_func"] = lambda pop, n, min_flags, fronts=None, ideal_candidate_values=None: \
             find_mo_elites_default(pop, n, min_flags, use_first_obj=True, fronts=fronts)
             
    elif elitism_strategy == "nsga2": #Rank + Crowding Distance elitism
        gp_parameters["find_elit_func"] = lambda pop, n, min_flags, fronts=None, ideal_candidate_values=None: \
             find_mo_elites_default(pop, n, min_flags, use_first_obj=False, fronts=fronts)
             
    else:
        raise ValueError(f"Unknown elitism_strategy '{elitism_strategy}'. Options are: 'nsga2', 'first_obj', 'ideal_point'.")


    if survival_strategy == "nsga2":
        algo = "MOGP_NSGAII"
        survival_op = nsga2_survival(minimization_flags)
        
    elif survival_strategy == "generational":
        algo = "MOGP_GENERATIONAL"
        survival_op = generational_survival(
            n_elites=n_elites,
            find_elit_func=gp_parameters["find_elit_func"], # Uses the same elite definition as logging
            minimization_flags=minimization_flags
        )
    else:
        raise ValueError(f"Invalid survival_strategy: '{survival_strategy}'. Use 'nsga2' or 'generational'.")

    
    gp_parameters["mutator"] = mutate_tree_subtree(
        gp_pi_init['init_depth'],  gp_pi_init["TERMINALS"], gp_pi_init['CONSTANTS'], gp_pi_init['FUNCTIONS'],
        p_c=gp_pi_init['p_c']
    )
    gp_parameters["initializer"] = initializer_options[initializer]    
    gp_parameters["seed"] = seed
    #   *************** GP_SOLVE_PARAMETERS ***************

    gp_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name]
    gp_solve_parameters["log"] = log_level
    gp_solve_parameters["verbose"] = verbose
    gp_solve_parameters["log_path"] = log_path
    gp_solve_parameters["n_elites"] = n_elites
    gp_solve_parameters["max_depth"] = max_depth
    gp_solve_parameters["n_iter"] = n_iter
    gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=gp_pi_init['FUNCTIONS'])


    gp_solve_parameters["ffunction"] = [fitness_function_options[f] for f in fitness_functions]
    gp_solve_parameters["offspring_size"] = offspring_size
    
    
    gp_solve_parameters["n_jobs"] = n_jobs
    gp_solve_parameters["test_elite"] = test_elite
    
    # ================================
    #         Running the Algorithm
    # ================================
    
    optimizer = MOGP(
        minimization_flags=minimization_flags,
        find_mo_elit_func=gp_parameters["find_elit_func"],
        survival_strategy=survival_op,
        pi_init=gp_pi_init, 
        elitism_strategy=elitism_strategy,
        **gp_parameters
    )

    optimizer.solve(
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test,
        curr_dataset=dataset_name,
        **gp_solve_parameters
    )
    if log_level > 0:
        log_settings(
            path=log_path[:-4] + "_settings.csv",
            settings_dict=[gp_solve_parameters,
                           gp_parameters,
                           gp_pi_init,
                           {'minimization_flags': minimization_flags,
                            'tournament_sizes': tournament_sizes,
                            'fitness_functions_names': fitness_functions,
                            'survival_strategy': survival_strategy,
                            'selector_strategy': selector_strategy}
                        ],
            unique_run_id=unique_run_id,
        )

    return optimizer.elite

if __name__ == "__main__":
    from slim_gsgp.datasets.data_loader import load_resid_build_sale_price
    from slim_gsgp.utils.utils import train_test_split
    from slim_gsgp.evaluators.fitness_functions import rmse

    # --- Example of MOGP Execution ---
    mo_ffuncs = ["rmse", "size"]
    mo_min_flags = [True, True]
    mo_tsizes = [4, 2]

    X, y = load_resid_build_sale_price(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = mo_gp(
        X_train=X_train, y_train=y_train,
        X_test=X_val, y_test=y_val,
        dataset_name='resid_build_sale_price_mo', 
        pop_size=100, n_iter=1000, prob_const=0,
        
        fitness_functions=mo_ffuncs,
        minimization_flags=mo_min_flags,
        tournament_sizes=mo_tsizes,
        selector_strategy="nested_tournament",
        survival_strategy="nsga2",
        n_elites=1,
        n_jobs=2, verbose=1, log_level=0)

    final_tree.print_tree_representation() 
    predictions = final_tree.predict(X_test)
    test_rmse = float(rmse(y_true=y_test, y_pred=predictions))
    print(f"\nRMSE Test (1º Objective): {test_rmse:.6f}")