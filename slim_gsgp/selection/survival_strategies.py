############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

import copy
from slim_gsgp.algorithms.GP.representations.mo_population import MultiObjectivePopulation

def nsga2_survival(minimization_flags):
    """
    Returns a survival function based on NSGA-II (Rank + Crowding Distance).
    Merges parents and offspring and selects the N best

    Parameters
    ----------
    minimization_flags : List[bool]
        List indicating for each objective whether it is to be minimized (True) or maximized (False).
    
    Returns
    -------
    Callable
        A function ('survival') that performs NSGA-II survival selection.
  
        The returned function merges the old population and offspring, sorts them into non-dominated fronts,
        and fills the new population based on rank and crowding distance until the desired population size is reached.

        Parameters
        ----------
        old_pop : MultiObjectivePopulation
            The old population before survival selection.
        offspring_pop : MultiObjectivePopulation
            The offspring population generated in the current generation.
        pop_size : int
            The desired size of the new population.
        
        Returns
        -------
        MultiObjectivePopulation
            The new population after survival selection.
    """
    def survival(old_pop, offspring_pop, pop_size):
        """
        Performs NSGA-II survival selection by merging old and offspring populations,
        sorting them into non-dominated fronts, and selecting individuals based on rank and crowding distance.

        Parameters
        ----------
        old_pop : MultiObjectivePopulation
            The old population before survival selection.
        offspring_pop : MultiObjectivePopulation
            The offspring population generated in the current generation.
        pop_size : int
            The desired size of the new population.
        
        Returns
        -------
        MultiObjectivePopulation
            The new population after survival selection.
        """ 
        # combine populations (P + Q= old population + offspring population)
        combined_pop_list = old_pop.population + offspring_pop.population
        combined_pop = MultiObjectivePopulation(combined_pop_list)
        
        # sorting the combined population into non-dominated fronts
        fronts = combined_pop.non_dominated_sorting(minimization_flags)
        
        # Fill new population using NSGA-II (Rank + Crowding Distance)
        new_pop_list = []
        for front in fronts:
            if len(new_pop_list) + len(front) <= pop_size:
                # If the whole front fits, calculate CD and add it
                combined_pop.calculate_crowding_distance(front)
                new_pop_list.extend(front)
            else:
                # If the front doesn't fit entirely, sort by CD and truncate
                combined_pop.calculate_crowding_distance(front)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                needed = pop_size - len(new_pop_list)
                new_pop_list.extend(front[:needed])
                break
                
        return MultiObjectivePopulation(new_pop_list)
    
    return survival


def generational_survival(n_elites=0, find_elit_func=None, minimization_flags=None):
    """
    Returns a function for Generational Survival Strategy.
    The offspring replaces the parents preserving N elites from the parents.

    Parameters
    ----------
    n_elites : int
        Number of elite individuals to preserve from the old population (default 0).
    find_elit_func : Callable
        Function to find elite individuals in the old population.
    minimization_flags : List[bool]
        List indicating for each objective whether it is to be minimized (True) or maximized (False).
    
    Returns
    -------
    Callable
        A function ('survival') that performs generational survival selection.

        The returned function preserves N elite individuals from the old population
        and fills the rest of the new population with offspring individuals.

        Parameters
        ----------
        old_pop : MultiObjectivePopulation
            The old population before survival selection.
        offspring_pop : MultiObjectivePopulation
            The offspring population generated in the current generation.
        pop_size : int
            The desired size of the new population.
        
        Returns
        -------
        MultiObjectivePopulation
            The new population after survival selection.
        """
    def survival(old_pop, offspring_pop, pop_size):
        """
        Performs generational survival selection by preserving N elite individuals
        from the old population and filling the rest with offspring individuals.
        
        Parameters
        ----------
        old_pop : MultiObjectivePopulation
            The old population before survival selection.
        offspring_pop : MultiObjectivePopulation
            The offspring population generated in the current generation.
        pop_size : int
            The desired size of the new population.
        
        Returns
        -------
        MultiObjectivePopulation
            The new population after survival selection.
        """
        new_pop_list = []
        
        # Elitism
        if n_elites > 0 and find_elit_func is not None:
            elites, _ = find_elit_func(old_pop, n_elites, minimization_flags)
            new_pop_list.extend([copy.deepcopy(e) for e in elites])
            
        #Add Offspring
        new_pop_list.extend(offspring_pop.population)
        
        #Truncate if necessary
        if len(new_pop_list) > pop_size:
            new_pop_list = new_pop_list[:pop_size]
            
        return MultiObjectivePopulation(new_pop_list)

    return survival