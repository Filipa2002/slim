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
        
        MultiObjectiveTree.FUNCTIONS = self.pi_init["FUNCTIONS"]
        MultiObjectiveTree.TERMINALS = self.pi_init["TERMINALS"]
        MultiObjectiveTree.CONSTANTS = self.pi_init["CONSTANTS"]

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
        
        # getting the non-dominated fronts of the initial population
        fronts = population.non_dominated_sorting(self.minimization_flags)
            
        end = time.time()

        # getting the elite(s) from the initial population
        n_elites_to_log = max(1, n_elites)
        self.elites, self.elite = self.find_mo_elit_func(population, n_elites_to_log, self.minimization_flags, fronts=fronts)

        # testing the elite on testing data, if applicable
        if test_elite:
            self.elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
        
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

            current_fronts = population.non_dominated_sorting(self.minimization_flags)

            self.elites, self.elite = self.find_mo_elit_func(population, n_elites_to_log, self.minimization_flags, fronts=current_fronts)

            # testing the elite on testing data, if applicable
            if test_elite:
                self.elite.evaluate(fitness_functions=ffunction, X=X_test, y=y_test, testing=True)
            
            # logging the results if log != 0
            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info
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


    def log_generation(
        self, generation, population, elapsed_time, log, log_path, run_info
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

        Returns:
            None
        """
        # 1st: formatting the required strings

        if self.elite is not None:
            # Elite Fitness (1D vector : [obj1, obj2, ...])
            elite_fit_str = " ".join([f"{f:.6f}" for f in self.elite.fitness.tolist()])
            if self.elite.test_fitness is not None:
                elite_test_fit_str = " ".join([f"{f:.6f}" for f in self.elite.test_fitness.tolist()])
            else:
                elite_test_fit_str = "N/A"
            elite_nodes = self.elite.node_count
        else:
            elite_fit_str = "N/A"
            elite_test_fit_str = "N/A"
            elite_nodes = 0 

        # Population Fit (2D matrix: [individuals x objectives]). Flattened into a single string.
        if population.fit is not None:
            pop_fit_numpy = population.fit.numpy()
        else:
            fits = [ind.fitness for ind in population.population]
            pop_fit_numpy = torch.stack(fits).numpy()
        pop_fit_str = " ".join([f"{f:.6f}" for f in pop_fit_numpy.flatten().tolist()])

        # 2nd: adapting add_info 
        if log == 2:
            add_info = [
                elite_test_fit_str,
                elite_nodes,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.mean(np.std(pop_fit_numpy, axis=0)), # Average standard deviation per objective
                log,
            ]
        elif log == 3:
            add_info = [
                elite_test_fit_str,
                elite_nodes,
                " ".join([str(ind.node_count) for ind in population.population]),
                pop_fit_str, # gives the fitness of all individuals
                log,
            ]
        elif log == 4:
            add_info = [
                elite_test_fit_str,
                elite_nodes,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.mean(np.std(pop_fit_numpy, axis=0)),
                " ".join([str(ind.node_count) for ind in population.population]),
                pop_fit_str,
                log,
            ]
        else:
            add_info = [elite_test_fit_str, elite_nodes, log]

        logger(
            log_path,
            generation,
            elite_fit_str, # Passes the formatted string
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )