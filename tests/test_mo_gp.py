############################################################################
#                                                                          #
# Created by me                                                            #
#                                                                          #
############################################################################

import pytest
import torch
import random
import numpy as np
from unittest.mock import patch

from slim_gsgp.algorithms.GP.representations.mo_tree import MultiObjectiveTree
from slim_gsgp.algorithms.GP.representations.mo_population import MultiObjectivePopulation
from slim_gsgp.selection.selection_algorithms import nested_tournament_selection, tournament_selection_nsga2

from slim_gsgp.main_mo_gp import mo_gp
from slim_gsgp.datasets.data_loader import load_ppb
from slim_gsgp.evaluators.fitness_functions import rmse 
from slim_gsgp.utils.utils import train_test_split 

# Dummy valid inputs to use in tests
valid_mo_functions = ["rmse", "size"]
valid_mo_min_flags = [True, True]
valid_mo_tournament_sizes = [3, 2]

valid_X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
valid_y_train = torch.tensor([1, 0], dtype=torch.float32)
valid_X_test = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
valid_y_test = torch.tensor([1, 0], dtype=torch.float32)
valid_n_iter = 3
valid_seed = 42
MOCK_REPR = ('add', 'x0', 'x1')

# ================ Unit tests ================
# Utils
# Mock Individual to test NSGA-II rules
class MockMOIndividual(MultiObjectiveTree):
    def __init__(self, fitness, pareto_front=None, crowding_distance=0.0):        
        self.fitness = torch.tensor(fitness, dtype=torch.float32)
        self.test_fitness = None
        self.pareto_front = pareto_front
        self.crowding_distance = crowding_distance
        self.repr_ = MOCK_REPR
        self.node_count = 3 
        self.depth = 2
        self.FUNCTIONS = {}
        self.TERMINALS = {}
        self.CONSTANTS = {}

    # Mock _dominates and is_better in mo_population
    def __lt__(self, other):
        return self.is_better(other)

    def is_better(self, other_solution):
        # Repeat the is_better logic from MultiObjectiveTree
        if self.pareto_front is None or other_solution.pareto_front is None:
            raise ValueError("Cannot compare individuals that do not have a Pareto front ranking.")
            
        if self.pareto_front < other_solution.pareto_front:
            return True
        elif self.pareto_front == other_solution.pareto_front:
            return self.crowding_distance > other_solution.crowding_distance
        return False
        
    def __setattr__(self, name, value):
         object.__setattr__(self, name, value)

# Mock Population to test NSGA-II/NTS rules
class MockMOPopulation(MultiObjectivePopulation):
    def __init__(self, pop):
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.node_count for ind in pop)
        self.fit = None

    # we don't need evaluate, but we neeed 'fit' for Dominance/NTS
    def fill_fit_matrix(self):
        if self.population:
            fits = [ind.fitness for ind in self.population]
            self.fit = torch.stack(fits)
        else:
            self.fit = None


#################### MultiObjectiveTree Tests ####################
# 1. Test Initialization
def test_mo_tree_initialization_defaults():
    """Checks if MultiObjectiveTree initializes fitness as None and MO attributes as None"""
    with patch('slim_gsgp.algorithms.GP.representations.tree.Tree.FUNCTIONS', {'add': {'function': lambda x, y: x + y, 'arity': 2}}):
        with patch('slim_gsgp.algorithms.GP.representations.tree.Tree.TERMINALS', {'x0': 0, 'x1': 1}):
            with patch('slim_gsgp.algorithms.GP.representations.tree.Tree.CONSTANTS', {'constant_2': lambda _: torch.tensor(2.0)}):
                mo_ind = MultiObjectiveTree(repr_=MOCK_REPR)
                assert mo_ind.fitness is None, "self.fitness must be initialized as None."
                assert mo_ind.test_fitness is None, "self.test_fitness must be initialized as None."
                assert mo_ind.pareto_front is None, "self.pareto_front must be initialized as None."
                assert mo_ind.crowding_distance is None, "self.crowding_distance must be initialized as None."    

#2. Test MultiObjectiveTree is_better Method
def test_mo_tree_is_better_rank():
    # 2.1. Lower Rank (Pareto Front) is better
    ind_better = MockMOIndividual(fitness=[0, 0], pareto_front=0, crowding_distance=1.0)
    ind_worse = MockMOIndividual(fitness=[1, 1], pareto_front=1, crowding_distance=5.0)
    assert ind_better.is_better(ind_worse) == True, "Individual with a lower Rank should be better."
    assert ind_worse.is_better(ind_better) == False, "Individual with a higher Rank should be worse."

def test_mo_tree_is_better_crowding_distance():
    # 2.2. Higher Crowding Distance (ind_a) is better when Ranks are equal
    ind_a = MockMOIndividual(fitness=[1, 1], pareto_front=1, crowding_distance=10.0)
    ind_b = MockMOIndividual(fitness=[2, 2], pareto_front=1, crowding_distance=5.0)
    assert ind_a.is_better(ind_b) == True, "Higher Crowding Distance should be better when Ranks are equal."
    assert ind_b.is_better(ind_a) == False, "Lower Crowding Distance should be worse when Ranks are equal."

def test_mo_tree_is_better_equal():
    # 2.3. Equal Ranks and Crowding Distances
    ind_a = MockMOIndividual(fitness=[1, 1], pareto_front=1, crowding_distance=5.0)
    ind_b = MockMOIndividual(fitness=[2, 2], pareto_front=1, crowding_distance=5.0)
    assert ind_a.is_better(ind_b) == False, "Equal Ranks and Crowding Distances should not be considered 'better'."

def test_mo_tree_is_better_no_rank_raises_error():
    # 2.4. No Rank Assigned Raises Error
    ind_a = MockMOIndividual(fitness=[1, 1], pareto_front=None, crowding_distance=5.0)
    ind_b = MockMOIndividual(fitness=[2, 2], pareto_front=1, crowding_distance=5.0)
    with pytest.raises(ValueError, match="Cannot compare individuals that do not have a Pareto front ranking."):
        ind_a.is_better(ind_b)


#################### MultiObjectivePopulation Tests ####################
# 3. Test Initialization
def test_mo_population_initialization_defaults():
    """Checks if MultiObjectivePopulation initializes self.fit as None."""
    ind1 = MockMOIndividual(fitness=[1, 1], pareto_front=0)
    ind2 = MockMOIndividual(fitness=[2, 2], pareto_front=1)
    mo_pop = MockMOPopulation([ind1, ind2])
    assert mo_pop.fit is None, "self.fit (fitness matrix) must be initialized as None."
    assert mo_pop.size == 2, "self.size must be initialized correctly."

#4. Test MultiObjectivePopulation _dominates method
# (Obj 0: Min, Obj 1: Min)
def test_dominance_rules():
    # 4.1. A dominates B
    #  A (5, 5) and B (10, 10)
    ind_a = MockMOIndividual(fitness=[5, 5]) 
    ind_b = MockMOIndividual(fitness=[10, 10])
    
    pop = MockMOPopulation([ind_a, ind_b])
    
    assert pop._dominates(ind_a, ind_b, minimization_flags=[True, True]) == True, "Individual A (5,5) should dominate B (10,10) in Min-Min."
    assert pop._dominates(ind_b, ind_a, minimization_flags=[True, True]) == False, "Individual B (10,10) should not dominate A (5,5) in Min-Min."
    
    # 4.2. D dominates C
    #  C (5, 10) and D (5, 5)
    ind_c = MockMOIndividual(fitness=[5, 10])
    ind_d = MockMOIndividual(fitness=[5, 5])
    pop = MockMOPopulation([ind_c, ind_d])

    assert pop._dominates(ind_d, ind_c, minimization_flags=[True, True]) == True, "Individual D (5,5) should dominate C (5,10) in Min-Min."

    #4.3. E and F do not dominate each other
    # E (5, 10) and F (10, 5)
    ind_e = MockMOIndividual(fitness=[5, 10])
    ind_f = MockMOIndividual(fitness=[10, 5])
    pop = MockMOPopulation([ind_e, ind_f])
    assert pop._dominates(ind_e, ind_f, minimization_flags=[True, True]) == False, "Individual E (5,10) should not dominate F (10,5) in Min-Min."
    assert pop._dominates(ind_f, ind_e, minimization_flags=[True, True]) == False, "Individual F (10,5) should not dominate E (5,10) in Min-Min."


#5. Test MultiObjectivePopulation non_dominated_sorting method
def test_non_dominated_sorting():
    # (Obj 0: Min, Obj 1: Min)   
    ind_a = MockMOIndividual(fitness=[1, 10]) # F1: A(1), B(2)
    ind_b = MockMOIndividual(fitness=[10, 1]) # F1: A(1), B(2)
    ind_c = MockMOIndividual(fitness=[3, 12]) # F2: C(3), D(4)
    ind_d = MockMOIndividual(fitness=[12, 3]) # F2: C(3), D(4)
    ind_e = MockMOIndividual(fitness=[5, 15]) # F3: E(5)
    pop_list = [ind_a, ind_b, ind_c, ind_d, ind_e]
    random.shuffle(pop_list)
    pop = MockMOPopulation(pop_list)
    fronts = pop.non_dominated_sorting(minimization_flags=[True, True])
    
    #5.1. Check number of fronts
    assert len(fronts) == 3, "There should be 3 Pareto fronts."

    # 5.2. Check Ranks of individuals
    assert ind_a.pareto_front == 0, "Individual A should be in Pareto front 0."
    assert ind_b.pareto_front == 0, "Individual B should be in Pareto front 0."
    assert ind_c.pareto_front == 1, "Individual C should be in Pareto front 1."
    assert ind_d.pareto_front == 1, "Individual D should be in Pareto front 1."
    assert ind_e.pareto_front == 2, "Individual E should be in Pareto front 2."
    
    # 5.3. Check contents of Fronts
    f1 = fronts[0]
    f2 = fronts[1]
    f3 = fronts[2]
    assert len(f1) == 2, "Front 1 should have 2 individuals."
    assert ind_a in f1 and ind_b in f1, "Front 1 should contain individuals A and B."
    assert len(f2) == 2, "Front 2 should have 2 individuals."
    assert ind_c in f2 and ind_d in f2, "Front 2 should contain individuals C and D."
    assert len(f3) == 1, "Front 3 should have 1 individual."
    assert ind_e in f3, "Front 3 should contain individual E."


#6. Test MultiObjectivePopulation calculate_crowding_distance method
def test_crowding_distance_calculation():    
    # (Obj 0: Min, Obj 1: Min) 
    # A (1, 10), C (5, 5), B (10, 1)
    # Scale_O1 = 9, Scale_O2 = 9
    
    ind_a = MockMOIndividual(fitness=[1, 10]) # Boundary
    ind_c = MockMOIndividual(fitness=[5, 5])
    ind_b = MockMOIndividual(fitness=[10, 1]) # Boundary
    
    front = [ind_a, ind_c, ind_b]
    pop = MockMOPopulation(front)
    pop.calculate_crowding_distance(front)
    
    # 6.1. Check if Boundary individuals have Infinite Crowding Distance
    assert ind_a.crowding_distance == float("inf"), "Boundary individual A should have infinite crowding distance."
    assert ind_b.crowding_distance == float("inf"), "Boundary individual B should have infinite crowding distance."
    
    # 6.2. Check if Interior individual C has correct Crowding Distance
    # CD(C) = [Dist_O1(C)/Scale_O1] + [Dist_O2(C)/Scale_O2]= (9/9) + (9/9) = 1+1= 2
    assert ind_c.crowding_distance == pytest.approx(2.0), "Interior individual C should have a crowding distance of approximately 2.0."



#################### Nested Tournament Selection Tests ####################
#7. Test Nested Tournament Selection Functionality
def test_nts_functionality_deterministic(mocker):
    # (Obj 1: Min, Obj 2: Min)
    # W_1 (1, 20), L_1 (10, 10), W_2 (2, 30), L_2 (12, 5)
    #NTS Sizes: [2, 2] (3 tournaments in total)
    ind_w1_o1 = MockMOIndividual(fitness=[1, 20])
    ind_l1_o1 = MockMOIndividual(fitness=[10, 10])
    ind_w2_o1 = MockMOIndividual(fitness=[2, 30])
    ind_l2_o1 = MockMOIndividual(fitness=[12, 5])
    
    pop_list = [ind_w1_o1, ind_l1_o1, ind_w2_o1, ind_l2_o1]
    pop = MockMOPopulation(pop_list)
    pop.fill_fit_matrix() #needs to fill fitness matrix for domination checks

    # Nested Tournament Selection
    nts_selector = nested_tournament_selection(tournament_sizes=[2, 2], minimization_flags=[True, True])

    mocker.patch('random.choices', side_effect=[
        # 1st Tournament Pools (Obj 1)
        [ind_w1_o1, ind_l1_o1], 
        # 2nd Tournament Pools (Obj 2)
        [ind_w2_o1, ind_l2_o1]
    ])
    # W_1 and W_2 are expected to win their first tournaments based on Obj 1
    # W_1 is expected to win the final tournament based on Obj 2
    winner = nts_selector(pop)
    
    assert winner is ind_w1_o1, "Nested Tournament Selection should select W_1 as the overall winner."

#################### NSGA-II Selection Tests ####################
#8. NSGA-II Selection Functionality
def test_tournament_selection_nsga2_logic(mocker):
    # prioritize Rank
    ind_rank_0 = MockMOIndividual(fitness=[1,1], pareto_front=0, crowding_distance=0.1)
    ind_rank_1 = MockMOIndividual(fitness=[2,2], pareto_front=1, crowding_distance=100.0)
    pop = MockMOPopulation([ind_rank_0, ind_rank_1])
    selector = tournament_selection_nsga2(pool_size=2)
    mocker.patch('random.choices', return_value=[ind_rank_0, ind_rank_1])
    winner = selector(pop)
    assert winner == ind_rank_0, "NSGA-II Selector should prioritize lower Rank over higher Crowding Distance."

    #Equal Ranks
    ind_cd_low = MockMOIndividual(fitness=[1,1], pareto_front=0, crowding_distance=0.1)
    ind_cd_high = MockMOIndividual(fitness=[1,1], pareto_front=0, crowding_distance=0.5)
    
    pop2 = MockMOPopulation([ind_cd_low, ind_cd_high])
    mocker.patch('random.choices', return_value=[ind_cd_low, ind_cd_high])
    winner2 = selector(pop2)
    assert winner2 == ind_cd_high, "NSGA-II Selector should prioritize higher Crowding Distance when Ranks are equal."

#9. Test mo_gp elitism slot reservation logic
@patch('slim_gsgp.algorithms.GP.mogp.MOGP.evolve_population')
def test_mo_gp_elitism_calculates_correct_offspring_count(mock_evolve):
    mock_pop = MockMOPopulation([])
    mock_pop.population = []
    mock_evolve.return_value = (mock_pop, 0.0)
    pop_size = 20
    n_elites = 2
    
    #execute mo_gp with survival_strategy="generational" and n_elites=2
    mo_gp(
        X_train=valid_X_train, 
        y_train=valid_y_train, 
        pop_size=pop_size,
        n_elites=n_elites,
        survival_strategy="generational",
        n_iter=1,
        fitness_functions=valid_mo_functions,
        minimization_flags=valid_mo_min_flags,
        tournament_sizes=valid_mo_tournament_sizes,
        dataset_name="test_ds",
        test_elite=False
    )
    
    # confirm that evolve_population was called with offspring_size = pop_size - n_elites
    assert mock_evolve.called
    _, kwargs = mock_evolve.call_args
    assert kwargs['offspring_size'] == (pop_size - n_elites)

#10. Test mo_gp elitism with n_elites=0 (no elitism)
def test_mo_gp_allows_zero_elites_generational():
    try:
        result_tree = mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            n_elites=0,
            n_iter=1,
            survival_strategy="generational",
            fitness_functions=valid_mo_functions, 
            minimization_flags=valid_mo_min_flags, 
            tournament_sizes=valid_mo_tournament_sizes,
            pop_size=30, 
            seed=valid_seed,
            dataset_name="test_ds",
            test_elite=False
        )
    except Exception as e:
        pytest.fail(f"mo_gp crashed with n_elites=0: {e}")

    assert result_tree is not None, "Algorithm should complete and return a tree even with 0 elites."

# ================ Integration tests ================
#1. Test if it returns MultiObjectiveTree with fitness vector of size 2 for the 2 objectives defined in valid_mo_functions ("rmse", "size")
def test_mo_gp_valid_inputs_returns_tree():
    """Checks if mo_gp runs successfully and returns a MultiObjectiveTree"""
    result_tree = mo_gp(
        X_train=valid_X_train, 
        y_train=valid_y_train, 
        n_iter=valid_n_iter,
        fitness_functions=valid_mo_functions,
        minimization_flags=valid_mo_min_flags,
        tournament_sizes=valid_mo_tournament_sizes,
        pop_size=30,
        seed=valid_seed,
        dataset_name="test_ds",
        test_elite=False
    )
    assert result_tree is not None
    assert isinstance(result_tree, MultiObjectiveTree)
    assert result_tree.fitness.shape[0] == 2
    
#2.
def test_mo_gp_invalid_mo_list_length():
    """Checks if mo_gp fails when the objective lists have different lengths."""
    with pytest.raises(ValueError, match="MOGP lists .* must have the same, non-zero length."):
        mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            n_iter=valid_n_iter,
            fitness_functions=["rmse", "size", "mae"],
            minimization_flags=[True, True],
            tournament_sizes=[2, 2],
            pop_size=30,
            dataset_name="test_ds",
            test_elite=False
        )

#3.
def test_mo_gp_invalid_minimization_flag_type():
    """Checks if mo_gp fails when a minimization flag is not boolean."""
    with pytest.raises(TypeError, match="Minimization flag at index 0 must be a boolean."):
        mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            n_iter=valid_n_iter,
            fitness_functions=valid_mo_functions,
            minimization_flags=["invalid", True],
            tournament_sizes=valid_mo_tournament_sizes,
            pop_size=30,
            dataset_name="test_ds",
            test_elite=False
        )

#4.
def test_mo_gp_invalid_offspring_size():
    """Checks if mo_gp fails when offspring_size is less than 1."""
    with pytest.raises(ValueError, match=r"Calculated offspring_size is .* \(<=0\)"):
        mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            n_iter=valid_n_iter,
            offspring_size=0,
            fitness_functions=valid_mo_functions,
            minimization_flags=valid_mo_min_flags,
            tournament_sizes=valid_mo_tournament_sizes,
            pop_size=30,
            dataset_name="test_ds",
            test_elite=False
        )

#5.
def test_mo_gp_n_elites_exceeds_pop_size_is_handled():
    """Checks if n_elites > pop_size does not crash (limited internally or relies on initialization checks)."""
    result_tree = mo_gp(
        X_train=valid_X_train, 
        y_train=valid_y_train, 
        n_elites=50, # 50>30
        n_iter=valid_n_iter,
        fitness_functions=valid_mo_functions,
        minimization_flags=valid_mo_min_flags,
        tournament_sizes=valid_mo_tournament_sizes,
        pop_size=30,
        seed=valid_seed,
        dataset_name="test_ds",
        test_elite=False
    )
    assert result_tree is not None

#6. 
@patch('slim_gsgp.main_mo_gp.find_mo_elites_ideal_candidate')
@patch('slim_gsgp.main_mo_gp.find_mo_elites_default')
def test_mo_gp_slot_reservation_elitism_active(mock_default_finder, mock_ideal_finder):
    """Checks if the MOGP solve logic correctly activates the ELITISM SLOT RESERVATION path."""
    
    elite_A = MockMOIndividual(fitness=[0.1, 0.1])
    elite_B = MockMOIndividual(fitness=[0.2, 0.2])
    # Mock find_mo_elites function to always return A and B as elites
    mock_return = ([elite_A, elite_B], elite_A)
    mock_default_finder.return_value = mock_return
    mock_ideal_finder.return_value = mock_return

    # ideal_candidate_values=None since we are using mock_default_finder 
    result_tree = mo_gp(
        X_train=valid_X_train, 
        y_train=valid_y_train, 
        n_elites=2,
        n_iter=valid_n_iter,
        survival_strategy="generational",
        fitness_functions=valid_mo_functions,
        minimization_flags=valid_mo_min_flags,
        tournament_sizes=valid_mo_tournament_sizes,
        pop_size=30,
        seed=valid_seed,
        dataset_name="test_ds",
        test_elite=False
    )
    
    # It doesn't verify the exact content of the final pool but ensures that the elite function was called and that the algorithm ran
    assert result_tree is not None
    assert mock_default_finder.called or mock_ideal_finder.called, "Elitism function should have been called."

#7.
def test_mo_gp_ideal_candidate_selection_path():
    """
    Checks if the mo_gp can be configured with ideal_candidate_values 
    and that the find_mo_elites_ideal_candidate logic is used.
    """

    with patch('slim_gsgp.main_mo_gp.find_mo_elites_ideal_candidate') as mock_ideal_finder:
        # Mock the return to have a predictable elite
        mock_individual = MockMOIndividual(fitness=[0.1, 0.1])
        mock_ideal_finder.return_value = ([mock_individual], mock_individual)
        
        ideal_points = [0.0, 0.0]
        
        mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            n_elites=1,
            n_iter=1,
            fitness_functions=valid_mo_functions,
            minimization_flags=valid_mo_min_flags,
            tournament_sizes=valid_mo_tournament_sizes,
            pop_size=30,
            ideal_candidate_values=ideal_points,
            seed=valid_seed,
            dataset_name="test_ds",
            test_elite=False
        )
        
        assert mock_ideal_finder.call_count > 0, "find_mo_elites_ideal_candidate should have been called."

        args, kwargs = mock_ideal_finder.call_args[0], mock_ideal_finder.call_args[1]
        assert args[3] == ideal_points, "Ideal candidate values should be passed to the elite finder."

# 8.
def test_mo_gp_invalid_X_train():
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="X_train must be a torch.Tensor"):
        mo_gp("invalid_type", y_train)

# 9.
def test_mo_gp_invalid_pop_size():
    with pytest.raises(TypeError, match="pop_size must be an int"):
        mo_gp(valid_X_train, valid_y_train, pop_size="invalid_type")

# 10.
def test_mo_gp_invalid_prob_const():
    with pytest.raises(TypeError, match="prob_const must be a float"):
        mo_gp(valid_X_train, valid_y_train, prob_const="invalid_type")
# 11.
def test_mo_gp_out_of_range_prob_const():
    with pytest.raises(ValueError, match="prob_const must be a number between 0 and 1"):
        mo_gp(valid_X_train, valid_y_train, prob_const=1.5)

# 12.
def test_mo_gp_min_n_iter():
    with pytest.raises(ValueError, match="n_iter must be greater than 0"):
        mo_gp(valid_X_train, valid_y_train, n_iter=0)

# 13.
def test_mo_gp_seed_reproducibility():
    """Checks if two runs with the same seed produce the same result."""
    result1 = mo_gp(
        valid_X_train, valid_y_train, seed=valid_seed, n_iter=valid_n_iter, 
        fitness_functions=valid_mo_functions, minimization_flags=valid_mo_min_flags, tournament_sizes=valid_mo_tournament_sizes,
        dataset_name="test_ds",
        test_elite=False
    )
    result2 = mo_gp(
        valid_X_train, valid_y_train, seed=valid_seed, n_iter=valid_n_iter, 
        fitness_functions=valid_mo_functions, minimization_flags=valid_mo_min_flags, tournament_sizes=valid_mo_tournament_sizes,
        dataset_name="test_ds",
        test_elite=False
    )

    predictions1 = result1.predict(valid_X_test)
    predictions2 = result2.predict(valid_X_test)
    assert torch.equal(predictions1, predictions2), "Results should be reproducible with the same seed."
    assert torch.equal(result1.fitness, result2.fitness), "Final fitness vectors should be identical."

# 14.
def test_mo_gp_n_jobs_parallelism():
    """Checks if the algorithm runs successfully using parallel processing."""
    result = mo_gp(
        valid_X_train, valid_y_train, n_jobs=4, n_iter=valid_n_iter, 
        fitness_functions=valid_mo_functions, minimization_flags=valid_mo_min_flags, tournament_sizes=valid_mo_tournament_sizes,
        dataset_name="test_ds",
        test_elite=False
    )
    assert result is not None, "The function should run successfully in parallel (n_jobs > 1)."

#15. execute mo_gp with offspring_size different than pop_size
def test_mo_gp_custom_offspring_size():
    # population: 20 offsprings: 10
    # survival should reduce from 30 to 20
    try:
        mo_gp(
            X_train=valid_X_train, 
            y_train=valid_y_train, 
            pop_size=20,
            offspring_size=10,
            n_iter=2,
            survival_strategy="nsga2",
            fitness_functions=valid_mo_functions,
            minimization_flags=valid_mo_min_flags,
            tournament_sizes=valid_mo_tournament_sizes,
            dataset_name="test_ds",
            test_elite=False
        )
    except Exception as e:
        pytest.fail(f"mo_gp failed with custom offspring_size: {e}")