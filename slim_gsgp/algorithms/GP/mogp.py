############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

"""
Multi-Objective Genetic Programming (MOGP) module.
"""

import random
import time
import numpy as np
import torch
import copy

from slim_gsgp.algorithms.GP.gp import GP
from slim_gsgp.algorithms.GP.representations.mo_population import MultiObjectivePopulation
from slim_gsgp.algorithms.GP.representations.mo_tree import MultiObjectiveTree

from slim_gsgp.utils.logger import logger
from slim_gsgp.utils.utils import mo_verbose_reporter
from slim_gsgp.utils.diversity import niche_entropy


class MOGP(GP):
    def __init__(
        self,
        minimization_flags: list,
        find_mo_elit_func: callable,
        survival_strategy: callable,
        elitism_strategy: str = "nsga2",
        **kwargs
    ):
        """
        Initialize the Multi-Objective Genetic Programming (MOGP) algorithm.

        Parameters
        ----------
        minimization_flags : list of bool
            A list indicating if each objective is a minimization problem (True)
            or a maximization problem (False).
        find_mo_elit_func : callable
            Function used to determine the best elite(s) for logging and tracking.
        survival_strategy : callable
            Function used to determine the survival strategy for the population.
        **kwargs :
            All standard parameters inherited from the base GP class (pi_init, selector, ...).
        """
        super().__init__(**kwargs)
        self.minimization_flags = minimization_flags
        self.find_mo_elit_func = find_mo_elit_func
        self.survival_strategy = survival_strategy
        self.elitism_strategy = elitism_strategy
        self.best_ideal_point = None # to track the best ideal point found (ideal candidate elitism)
        
        MultiObjectiveTree.FUNCTIONS = self.pi_init["FUNCTIONS"]
        MultiObjectiveTree.TERMINALS = self.pi_init["TERMINALS"]
        MultiObjectiveTree.CONSTANTS = self.pi_init["CONSTANTS"]

        
    def _update_dynamic_ideal_point(self, population):
        """
        Update the best ideal point found so far based on the current population.

        Parameters
        ----------
        population : MultiObjectivePopulation
            The current population of individuals.
            
        Returns
        -------
        None
        """
        from slim_gsgp.utils.utils import get_current_ideal_point
        
        current_gen_ideal = get_current_ideal_point(population, self.minimization_flags)
        if current_gen_ideal is None: return

        if self.best_ideal_point is None:
            self.best_ideal_point = current_gen_ideal
        else:
            # Compares and updates the best historical ideal point
            new_ideal = []
            for i, is_min in enumerate(self.minimization_flags):
                if is_min:
                    new_ideal.append(min(self.best_ideal_point[i], current_gen_ideal[i]))
                else:
                    new_ideal.append(max(self.best_ideal_point[i], current_gen_ideal[i]))
            self.best_ideal_point = np.array(new_ideal)
        

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        max_depth=None,
        ffunction=None,
        n_elites=1,
        depth_calculator=None,
        n_jobs=1,
        offspring_size=None,
        **kwargs
    ):
        """
        Execute the Multi-Objective Genetic Programming algorithm.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data features.
        X_test : torch.Tensor
            Test data features.
        y_train : torch.Tensor
            Training data labels.
        y_test : torch.Tensor
            Test data labels.
        curr_dataset : str
            Current dataset name.
        n_iter : int, optional
            Number of iterations. Default is 20.
        log : int, optional
            Logging level. Default is 0.
        verbose : int, optional
            Verbosity level. Default is 0.
        test_elite : bool, optional
            Whether to evaluate elite individuals on test data. Default is False.
        log_path : str, optional
            Path to save logs. Default is None.
        run_info : list, optional
            Information about the current run. Default is None.
        max_depth : int, optional
            Maximum depth of the tree. Default is None.
        ffunction : list, optional
            A list of fitness functions, one for each objective. Default is None.
        n_elites : int, optional
            Number of elites to track. Default is 1.
        depth_calculator : function, optional
            Function to calculate tree depth. Default is None.
        n_jobs : int, optional
            The number of jobs for parallel processing. Default is 1.
        offspring_size : int, optional
            The number of offspring to produce in each generation. If None, defaults to the population size
        """
        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population
        initial_trees = [
            MultiObjectiveTree(tree) for tree in self.initializer(**self.pi_init)
        ]
        population = MultiObjectivePopulation(initial_trees)

        # evaluating the intial population
        population.evaluate(fitness_functions=ffunction, X=X_train, y=y_train, n_jobs=n_jobs)
        
        #updating ideal point if using ideal candidate elitism
        if self.elitism_strategy == "ideal_point":
            self._update_dynamic_ideal_point(population)

        # getting the non-dominated fronts of the initial population
        fronts = population.non_dominated_sorting(self.minimization_flags)
            
        end = time.time()

        # getting the elite(s) from the initial population
        self.elites, self.elite = self.find_mo_elit_func(
            population, max(1, n_elites), self.minimization_flags
        )
        
        ###################Created by me: just for our experiments
        rmse_elite = min(population.population, key=lambda ind: ind.fitness[0])
        ###################
        
            
        # testing the elite on testing data, if applicable
        if test_elite:
            self.elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
            if rmse_elite != self.elite:
                rmse_elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
            else:
                rmse_elite.test_fitness = self.elite.test_fitness
        
        # logging the results if the log level is not 0
        if log != 0:
            self.log_generation(
                0, population, end - start, log, log_path, run_info
            )
        
        #displaying the results on console if verbose level is not 0
        if verbose != 0:
            mo_verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.fitness,
                self.elite.test_fitness if self.elite.test_fitness is not None else None,
                end - start,
                self.elite.node_count,
            )

        # EVOLUTIONARY PROCESS
        if offspring_size is None:
            n_offspring = self.pop_size
        else:
            n_offspring = offspring_size

        for it in range(1, n_iter + 1):
            # generate offsprings
            offs_pop, start = self.evolve_population(
                    population, 
                    ffunction, 
                    max_depth, 
                    depth_calculator,
                    X_train, 
                    y_train, 
                    n_jobs=n_jobs,
                    offspring_size=n_offspring
                )
            
            #Apply survival strategy
            population = self.survival_strategy(population, offs_pop, self.pop_size)

            end = time.time()

            # updating ideal point if using ideal candidate elitism
            if self.elitism_strategy == "ideal_point":
                self._update_dynamic_ideal_point(population)

            population.non_dominated_sorting(self.minimization_flags)
            self.elites, self.elite = self.find_mo_elit_func(population, max(1, n_elites), self.minimization_flags)

            ###################Created by me: just for our experiments
            rmse_elite = min(population.population, key=lambda ind: ind.fitness[0])
            ###################

            if test_elite:
                self.elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
                if rmse_elite != self.elite:
                    rmse_elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
                else:
                    rmse_elite.test_fitness = self.elite.test_fitness

            # logging the results if log != 0
            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info, secondary_elite=rmse_elite
                )

            # displaying the results on console if verbose != 0    
            if verbose != 0:
                mo_verbose_reporter(
                    run_info[-1],
                    it, 
                    self.elite.fitness, 
                    self.elite.test_fitness if self.elite.test_fitness is not None else None,
                    end - start, 
                    self.elite.node_count,
                )


    
    def log_generation(
        self, generation, population, elapsed_time, log, log_path, run_info, secondary_elite=None
    ):
        """
        Log the results for the current generation (adapted for Multi-Objective data).
        
        Args:
            generation (int): Current generation (iteration) number.
            population (MultiObjectivePopulation): Current population.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.
            secondary_elite (Individual, optional): Secondary elite individual for logging.

        Returns:
            None
        """        
        
        # A. Formart the Main Elite (ex: rank)
        if self.elite is not None:
            main_test_str = "|".join([f"{f:.6f}" for f in self.elite.test_fitness.tolist()]) if self.elite.test_fitness is not None else "N/A"  # Ex: "5.2|10.0|3.0"
            main_nodes = self.elite.node_count
        else:
            main_test_str = "N/A"; main_nodes = 0

        # B. Format the RMSE Elite
        if secondary_elite is not None:
            sec_test_str = "|".join([f"{f:.6f}" for f in secondary_elite.test_fitness.tolist()]) if secondary_elite.test_fitness is not None else "N/A"
            sec_nodes = secondary_elite.node_count
        else:
            sec_test_str = main_test_str
            sec_nodes = main_nodes

        # C. Population standard deviation
        if population.fit is not None: pop_np = population.fit.numpy()
        else: pop_np = torch.stack([ind.fitness for ind in population.population]).numpy()
        std_train_rmse = np.std(pop_np[:, 0])

        # for ideal point logging
        if self.best_ideal_point is not None:
            ideal_str = "|".join([f"{val:.6f}" for val in self.best_ideal_point])
        else:
            ideal_str = "N/A"

        if log == 5:
            # default logger: [Gen, Time, Main_Train_Fit, Main_Nodes]
            # and then add:
            add_info = [
                main_test_str,    # Main Elite Test (for all objectives)
                
                sec_test_str,     # RMSE Elite Test (for all objectives)
                sec_nodes,        # RMSE Elite Size          
                f"{std_train_rmse:.6f}", # StdDev
                ideal_str,        # Ideal Point
                log
            ]
        else:
            add_info = [main_test_str, main_nodes, log]

        main_train_str = "|".join([f"{f:.6f}" for f in self.elite.fitness.tolist()]) if self.elite else "N/A"

        logger(
            log_path, generation, main_train_str, elapsed_time, float(main_nodes),
            additional_infos=add_info, run_info=run_info, seed=self.seed,
        )


    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        X_train,
        y_train,
        n_jobs=1,
        offspring_size=None
    ):
        """
        Evolve the population for one iteration (generation).

        Parameters
        ----------
        population : MultiObjectivePopulation
            The current population of individuals to evolve.
        ffunction : list
            A list of fitness functions used to evaluate individuals.
        max_depth : int
            Maximum allowable depth for trees in the population.
        depth_calculator : Callable
            Function used to calculate the depth of trees.
        X_train : torch.Tensor
            Input training data features.
        y_train : torch.Tensor
            Target values for the training data.
        n_jobs : int, optional
            Number of parallel jobs to use. Default is 1.
        offspring_size : int, optional
            The number of offspring to produce in each generation. If None, defaults to the population size.

        Returns
        -------
        MultiObjectivePopulation
            The offspring population (P') after one generation.
        float
            The start time of the evolution process.
        """
        # creating an empty offspring population list
        offs_pop_list = []
        start = time.time()

        if offspring_size is None:
         offspring_size = population.size

        # filling the offspring population
        while len(offs_pop_list) < offspring_size:
            # choosing between crossover and mutation
            if random.random() < self.p_xo: # if crossover is selected
                # choose two parents
                p1, p2 = self.selector(population), self.selector(population)
                # make sure that the parents are different
                while p1 == p2:
                    p1, p2 = self.selector(population), self.selector(population)

                # generate offspring from the chosen parents
                offs1, offs2 = self.crossover(
                    p1.repr_, 
                    p2.repr_, 
                    p1.node_count, 
                    p2.node_count,
                )

                # assuring the offspring do not exceed max_depth
                if max_depth is not None:
                    offspring = []
                    while len(offspring) == 0:
                        d1 = depth_calculator(offs1)
                        d2 = depth_calculator(offs2)
                        # if at least one offspring is valid, we keep it (avoids repeated calculations)
                        if d1 <= max_depth:
                            offspring.append(offs1)
                        if d2 <= max_depth:
                            offspring.append(offs2)
                        
                        if len(offspring) > 0:
                            break  
                        # if both are invalid, retry crossover
                        offs1, offs2 = self.crossover(
                            p1.repr_, 
                            p2.repr_, 
                            p1.node_count, 
                            p2.node_count
                        )
                # if no valid offspring found after attempts, offspring remains empty and the loop continues
                else:
                    offspring = [offs1, offs2]
            
            else: # if mutation was chosen
                # choosing a parent
                p1 = self.selector(population)
                #generating a mutated offspring from the parent
                offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)
                
                # making sure the offspring does not exceed max_depth
                if max_depth is not None:
                    while depth_calculator(offs1) > max_depth:
                        offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)
                
                #adding the offspring to the list, to be added to the offspring population
                offspring = [offs1]

            # Adding offspring as MultiObjectiveTree instances
            offs_pop_list.extend([MultiObjectiveTree(child) for child in offspring])

        # making sure the offspring population is of the correct size
        if len(offs_pop_list) > offspring_size:
            offs_pop_list = offs_pop_list[: offspring_size]
        
        offs_pop = MultiObjectivePopulation(offs_pop_list)

        # evaluating the offspring population
        offs_pop.evaluate(fitness_functions=ffunction, X=X_train, y=y_train, n_jobs=n_jobs)
        
        # retuning the offspring population and the time control variable
        return offs_pop, start