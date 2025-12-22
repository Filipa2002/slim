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
This script sets up the configuration dictionaries for the execution of the GP algorithm
"""
from slim_gsgp.algorithms.GP.operators.crossover_operators import crossover_trees
from slim_gsgp.initializers.initializers import rhh, grow, full
from slim_gsgp.selection.selection_algorithms import tournament_selection_min

from slim_gsgp.evaluators.fitness_functions import *
import slim_gsgp.utils.utils as utils
import torch
import random

# Define functions and constants
FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': utils.protected_div, 'arity': 2},
    'mod': {'function': utils.protected_mod, 'arity': 2},
    'pow': {'function': utils.protected_pow, 'arity': 2},
}

random.seed(47)
CONSTANTS = {
    f'constant_{i}': lambda _, val=random.uniform(-1, 1): torch.tensor(val)
    for i in range(10)
}

# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
gp_solve_parameters = {
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "rmse",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites": 1,
    "elitism": True,
    "n_iter": 1000
}

# GP parameters
gp_parameters = {
    "initializer": "rhh",
    "selector": tournament_selection_min(2),
    "crossover": crossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "p_xo": 0.8,
    "pop_size": 100,
    "seed": 74
}

gp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.2,
    "init_depth": 6
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors,
    "r2_score": r2_score,
    "size": utils.gs_size,
    "nao": utils.num_nao,
    "naoc": utils.num_consecutive_nao,
    "features": utils.num_features
}
###################################################################################################
#                                                                                                 #
# Created by me (this and "size","r2_score","nao", "c...nao", "features" into ffunction_options)  #
#                                                                                                 #
###################################################################################################
mo_parameters = {
    "mo_fitness_functions": ["rmse", "size"], 
    "mo_minimization_flags": [True, True],
    "mo_tournament_sizes": [2, 2], 
}
initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full
}