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
"""
Selection operator implementation.
"""

import random

import numpy as np


def tournament_selection_min(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts


############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

def nested_tournament_selection(tournament_sizes: list, minimization_flags: list):
    """
    Returns a function that performs nested tournament selection (only for multi-objective problems).

    Parameters
    ----------
    tournament_sizes : list of int
        A list containing the tournament size for each objective.
    minimization_flags : list of bool
        A list indicating if each objective is for minimization (True) or maximization (False).

    Returns
    -------
    Callable
        A function ('nts') that, returns a single winning individual.
    """
    
    # error checks
    if len(tournament_sizes) != len(minimization_flags):
        raise ValueError("tournament_sizes and minimization_flags must have the same length.")
    if len(tournament_sizes) < 2:
        raise ValueError("tournament_sizes list must have at least two elements.")
    
    # Calculate the number of tournaments needed at each objective level
    num_winners_needed = [1]
    for size in reversed(tournament_sizes[1:]):
        num_winners_needed.append(num_winners_needed[-1] * size)
    num_winners_needed.reverse()

    #Util
    def _run_single_tournament(pool, objective_index, is_min):
        """
        Runs a single tournament on a given pool of individuals.
        Returns the winning individual object.
        """
        if is_min:
            winner = min(pool, key=lambda ind: ind.fitness[objective_index])
        else:
            winner = max(pool, key=lambda ind: ind.fitness[objective_index])
        return winner

    def nts(pop):
        """
        Selects a single individual from the population using the nested tournament selection.

        Parameters
        ----------
        pop : Population
            The population from which to select an individual.

        Returns
        -------
        Individual
            The single winning individual object.
        """

        ### For the tournaments of the first objective ###
        # Consider the entire population as competitors
        competitors_pool = pop.population
        current_tournament_size = tournament_sizes[0]
        is_min = minimization_flags[0]
        required_winners = num_winners_needed[0]
        winners_of_stage_1 = []

        for _ in range(required_winners):
            tournament_pool = random.choices(competitors_pool, k=current_tournament_size)
            winner = _run_single_tournament(tournament_pool, 0, is_min)
            winners_of_stage_1.append(winner)
        competitors_pool = winners_of_stage_1


        ### For subsequent objectives ###
        for i in range(1, len(tournament_sizes)):
            
            current_tournament_size = tournament_sizes[i]
            is_min = minimization_flags[i]
            
            winners_of_this_stage = []

            for j in range(0, len(competitors_pool), current_tournament_size):
                tournament_pool = competitors_pool[j : j + current_tournament_size]
                # single tournament for this group
                winner = _run_single_tournament(tournament_pool, i, is_min)
                winners_of_this_stage.append(winner)
            # The winners of this stage become the competitors for the next stage
            competitors_pool = winners_of_this_stage

        return competitors_pool[0]

    return nts


def tournament_selection_nsga2(pool_size=2):
    """
    Returns a function that performs tournament selection based on 
    Pareto Rank and Crowding Distance (Standard NSGA-II Parent Selection).

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament (default 2).

    Returns
    -------
    Callable
        A function ('ts') that elects the best individual based on Rank (min) and CD (max).
    """
    def ts(pop):
        # Select k random individuals
        pool = random.choices(pop.population, k=pool_size)
        
        # The 'winner' is determined by the .is_better() method of MultiObjectiveTree
        best = pool[0]
        for ind in pool[1:]:
            if ind.pareto_front != best.pareto_front:
                if ind.is_better(best):
                    best = ind
            else:
                if ind.crowding_distance == None:
                    current_rank = ind.pareto_front

                    same_front_individuals = [
                        indiv for indiv in pop.population 
                        if indiv.pareto_front == current_rank
                    ]
                    pop.calculate_crowding_distance(same_front_individuals)
                if ind.is_better(best):
                    best = ind
        return best

    return ts