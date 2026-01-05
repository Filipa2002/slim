# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
from slim_gsgp.main_mo_gp import mo_gp
from slim_gsgp.datasets.data_loader import (
     load_efficiency_cooling, load_ld50, 
     load_boston
)

# Configuration
RANDOM_SEED = 37
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Datasets to Test (we need to import them first)
DATASETS = {
    # 'Toxicity': load_ld50,
    'Cooling': load_efficiency_cooling, 
    # 'Boston': load_boston
}


# Objective Configurations
OBJECTIVE_SETS = {
    '2_Objs': {
        'funcs': ["rmse", "size"],
        'flags': [True, True] # Min, Min
    },
    '3_Objs': {
        'funcs': ["rmse", "size", "features"],
        'flags': [True, True, True]
    },
    '5_Objs': {
        'funcs': ["rmse", "size", "features", "nao", "naoc"],
        'flags': [True, True, True, True, True]
    }
}

SCENARIOS = {
    # 'NSGA-II_Pure': {
    #     'selector': 'nsga2',
    #     'survival': 'nsga2',
    #     'n_elites': 0, 
    #     'elitism_strategy': 'nsga2'
    # },
    # 'NT_NSGA-II': {
    #     'selector': 'nested_tournament',
    #     'survival': 'nsga2',
    #     'n_elites': 0,
    #     'elitism_strategy': 'nsga2'
    # },
    'NT_No_Elitism': {
        'selector': 'nested_tournament',
        'survival': 'generational',
        'n_elites': 0,
        'elitism_strategy': 'nsga2'
    },
    # 'NT_1st_Obj_Elitism': {
    #     'selector': 'nested_tournament',
    #     'survival': 'generational',
    #     'n_elites': 1,
    #     'elitism_strategy': 'first_obj' #uses RMSE
    # },
    'NT_Rank_CD_Elitism': {
        'selector': 'nested_tournament',
        'survival': 'generational',
        'n_elites': 1,
        'elitism_strategy': 'nsga2' #Rank + CD
    },
    'NT_Ideal_Cand_Elitism': {
        'selector': 'nested_tournament',
        'survival': 'generational',
        'n_elites': 1,
        'elitism_strategy': 'ideal_point' #dynamic ideal point
    }
}

#Hyperparameter Grid for Inner CV
FIXED_PARAMS = {
    'pop_size': 400,
    'n_iter': 250,
    'p_xo': 0.8,           #default  
    'prob_const': 0.2,     #default  
    'max_depth': 17,       #default  
    'initializer': 'rhh',  #default
    "init_depth": 4,
    "seed": 74,            #default
    "n_jobs": 1,           #default
    "test_elite": True    
}

#usado para teste pequeno
# FIXED_PARAMS = {
#     'pop_size': 10,
#     'n_iter': 5,
#     'p_xo': 0.8,
#     'prob_const': 0.2,    
#     'max_depth': 6,
#     'initializer': 'rhh', 
#     "init_depth": 6,      
#     "seed": 74,           
#     "n_jobs": 1,
#     "test_elite": True    
# }

##################################JUST COMMENTS##################################
#Other parameters will remain default:
###offspring_size###
# if offspring_size is None:
#     n_offspring = self.pop_size

###n_elites###
# it depends on the scenario being used

###log_path & log_level###
#posso criar um log =5 e fazer hard coded o que quero armazenar    APAGAR
# it depends on dataset, scenario, fold, hyperparams
# APAGAR: find out what information we need to log (its better to log more info and then ignore what is not needed than the opposite)

###fitness_functions, minimization_flags###
#it depends on the scenario being used: OBJECTIVE_SETS (Ok)

###tournament_sizes###
# Tournament sizes need to be dynamic based on n_objectives, handled in loop. When using Nested Tournament Selection, but for now we will fix it 

###ideal_candidate_values###
# it can no longer be user defined      

### "test_elite": True###

### tree_functions ###
# FUNCTIONS = {
#     'add': {'function': torch.add, 'arity': 2},
#     'subtract': {'function': torch.sub, 'arity': 2},
#     'multiply': {'function': torch.mul, 'arity': 2},
#     'divide': {'function': utils.protected_div, 'arity': 2},
#     'mod': {'function': utils.protected_mod, 'arity': 2},
#     'pow': {'function': utils.protected_pow, 'arity': 2},
# }

### tree_constants ###
# I changed what originaly was in gp_config.py to the following:
# random.seed(47)
# CONSTANTS = {
#     f'constant_{i}': lambda _, val=random.uniform(-1, 1): torch.tensor(val)
#     for i in range(10)
# }
#################################################################################

N_SPLITS_MC = 30 
TEST_SIZE = 0.3

# ### Main Evaluation Loop
for ds_name, loader_func in DATASETS.items():
    print(f"\n{'='*40}\nDataset: {ds_name}\n{'='*40}")

    # Load Data (X: features, y: target)
    X, y = loader_func(X_y=True)

    # Monte Carlo CV (30 random splits)
    rs = ShuffleSplit(n_splits=N_SPLITS_MC, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    for split_idx, (train_idx, test_idx) in enumerate(rs.split(X, y)):
            
        print(f"\n  > MC Split {split_idx+1}/{N_SPLITS_MC}")

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        for obj_set_name, obj_config in OBJECTIVE_SETS.items():
            n_objs = len(obj_config['funcs'])
            print(f"    > Objectives: {obj_set_name}")
            
            for scen_name, scen_config in SCENARIOS.items():
                print(f"      > Scenario: {scen_name}")
                
                fold_dir = f"./log_mo/{ds_name}/{scen_name}/{obj_set_name}/fold_{split_idx+1}"
                
                if not os.path.exists(fold_dir): os.makedirs(fold_dir)
                
                log_path = os.path.join(fold_dir, "execution_log.csv")

                #if we already ran it, skip
                if os.path.exists(log_path):
                    continue
                    
                t_sizes_final = [2] * n_objs
                try:
                    mo_gp(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        dataset_name=ds_name,
                        fitness_functions=obj_config['funcs'],
                        minimization_flags=obj_config['flags'],
                        tournament_sizes=t_sizes_final,
                        elitism_strategy=scen_config['elitism_strategy'],
                        selector_strategy=scen_config['selector'],
                        survival_strategy=scen_config['survival'],
                        n_elites=scen_config['n_elites'],
                            
                        #fixed params
                        pop_size=FIXED_PARAMS['pop_size'],
                        n_iter=FIXED_PARAMS['n_iter'],
                        p_xo=FIXED_PARAMS['p_xo'],
                        prob_const=FIXED_PARAMS['prob_const'],
                        max_depth=FIXED_PARAMS['max_depth'],
                        initializer=FIXED_PARAMS['initializer'],
                        init_depth=FIXED_PARAMS['init_depth'],
                        verbose=0, 
                        log_level=5,
                        log_path=log_path,
                        n_jobs=FIXED_PARAMS['n_jobs'],
                        test_elite=FIXED_PARAMS['test_elite']
                    )
                    print(f"        > Log saved: {scen_name} @ Split {split_idx+1}")
                
                except Exception as e:
                        print(f"      [Error] {ds_name} {scen_name} Split {split_idx+1}: {e}")
                        import traceback
                        traceback.print_exc()
