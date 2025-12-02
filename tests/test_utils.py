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

############################################################################
#                                                                          #
# Created by me: test_find_mo_elites_default_first_obj()                   #
#                test_find_mo_elites_default_nsga2_logic()                 #
#                test_find_mo_elites_ideal_candidate()                     #
#                test_find_mo_elites_ideal_candidate_recalculates_fit()    #
#                                                                          #
############################################################################
import pytest
import torch
import random
from unittest.mock import MagicMock
from slim_gsgp.utils.utils import get_best_max, get_best_min, find_mo_elites_default, find_mo_elites_ideal_candidate
    
class MockIndiv:
    def __init__(self, fitness, repr_val=None):
        self.fitness = fitness
        self.repr_ = repr_val
        self.pareto_front = 0
        self.crowding_distance = 0.0

    def __eq__(self, other):
        if isinstance(self.fitness, torch.Tensor) and isinstance(other.fitness, torch.Tensor):
             return torch.equal(self.fitness, other.fitness)
        return self.fitness == other.fitness
    
    def __repr__(self):
        return f"Indiv({self.fitness})"


class MockPop:
    def __init__(self, population):
        self.population = population
        # try to stack fitness if they are tensors
        try:
             self.fit = torch.stack([ind.fitness for ind in population])
        except:
             self.fit = [ind.fitness for ind in population]

    # Placeholder methods so they can be mocked in tests
    def non_dominated_sorting(self, minimization_flags):
        return []
    
    def calculate_crowding_distance(self, front):
        pass

def test_get_best_max():
    class IndivTest:
        def __init__(self, fitness):
            self.fitness = fitness

        def __eq__(self, other):
            return self.fitness == other.fitness

    class PopTest:
        def __init__(self, population):
            self.population = population
            self.fit = [indiv.fitness for indiv in self.population]

    example1 =  IndivTest(1)
    example2 =  IndivTest(2)
    example3 =  IndivTest(3)
    example4 =  IndivTest(4)
    example5 =  IndivTest(5)

    example_list = [example1, example2, example3, example4, example5]
    expected_top = sorted(example_list, key=lambda x: x.fitness, reverse=True)[:3]

    for i in range(30):
        random.shuffle(example_list)
        example_pop = PopTest(example_list)
        result1, result2 = get_best_max(example_pop, 3)

        assert (example4 in expected_top and example5 in expected_top and
                result2 == example5)

def test_get_best_min():
    class indiv_test:
        def __init__(self, fitness):
            self.fitness = fitness

        def __eq__(self, other):
            return self.fitness == other.fitness

    class pop_test:
        def __init__(self, population):
            self.population = population
            self.fit = [indiv.fitness for indiv in self.population]

    example1 = indiv_test(1)
    example2 = indiv_test(2)
    example3 = indiv_test(3)
    example4 = indiv_test(4)
    example5 = indiv_test(5)

    example_list = [example1, example2, example3, example4, example5]

    for i in range(30):
        random.shuffle(example_list)
        example_pop = pop_test(example_list)
        result1, result2 = get_best_min(example_pop, 3)

        assert (example1 in result1 and example2 in result1 and example3 in result1 and
                result2 == example1)        

def test_find_mo_elites_default_first_obj():
    ind_a = MockIndiv(torch.tensor([1.0, 10.0]))
    ind_b = MockIndiv(torch.tensor([5.0, 5.0]))
    pop = MockPop([ind_a, ind_b])   
    elites, elite = find_mo_elites_default(
        pop, n_elites=1, minimization_flags=[True, True], use_first_obj=True
    )
    assert elite == ind_a, "Expected ind_a to be the elite based on the first objective minimization."
    assert len(elites) == 1, "Expected only one elite individual."

def test_find_mo_elites_default_nsga2_logic():
    ind_a = MockIndiv(torch.tensor([1.0, 10.0]))
    ind_a.pareto_front = 0
    ind_a.crowding_distance = None 
    ind_b = MockIndiv(torch.tensor([10.0, 1.0]))
    ind_b.pareto_front = 0
    ind_b.crowding_distance = None
    ind_c = MockIndiv(torch.tensor([5.0, 5.0]))
    ind_c.pareto_front = 1
    
    pop = MockPop([ind_a, ind_b, ind_c])
    
    pop.non_dominated_sorting = MagicMock(return_value=[[ind_a, ind_b], [ind_c]])
    
    def mock_calc_cd(front):
        if ind_a in front: ind_a.crowding_distance = 10.0
        if ind_b in front: ind_b.crowding_distance = 5.0
            
    pop.calculate_crowding_distance = MagicMock(side_effect=mock_calc_cd)

    elites, elite = find_mo_elites_default(
        pop, n_elites=1, minimization_flags=[True, True], use_first_obj=False
    )

    pop.calculate_crowding_distance.assert_called()
    assert elite == ind_a, "Expected ind_a to be the elite based on NSGA-II logic."

def test_find_mo_elites_ideal_candidate():    
    ind_a = MockIndiv(torch.tensor([1.0, 1.0]))
    ind_b = MockIndiv(torch.tensor([4.0, 4.0]))
    pop = MockPop([ind_a, ind_b])
    
    ideal_values = [0.0, 0.0]
    
    elites, elite = find_mo_elites_ideal_candidate(
        pop, n_elites=1, minimization_flags=[True, True], ideal_candidate_values=ideal_values
    )
    
    assert elite == ind_a, "Expected ind_a to be the elite based on proximity to the ideal candidate."

def test_find_mo_elites_ideal_candidate_recalculates_fit():
    ind_a = MockIndiv(torch.tensor([1.0, 1.0]))
    ind_b = MockIndiv(torch.tensor([4.0, 4.0]))
    pop = MockPop([ind_a, ind_b])
    
    # Forcing fit to be None to simulate the parallel execution edge case
    pop.fit = None
    
    ideal_values = [0.0, 0.0]
    
    # This should NOT crash, and should calculate fit internally
    elites, elite = find_mo_elites_ideal_candidate(
        pop, n_elites=1, minimization_flags=[True, True], ideal_candidate_values=ideal_values
    )
    
    assert elite == ind_a
    assert pop.fit is not None, "Population fit tensor should have been updated/calculated."
    assert len(pop.fit) == 2