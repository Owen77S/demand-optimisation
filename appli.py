# %%

import algo_package as m
import numpy as np
import os
from plot_package import *
import concurrent.futures

def initialisation(filepath, column, time_stamp, time_stamp_unit, type_profile):
    """
    To import the power data list, and return it into a power profile.

    Parameters
    ----------
    filepath : str
        The path of the file.
    column : str or int
        Contains the column or the index of the column containing the data.
    time_stamp : TYPE
        DESCRIPTION.
    time_stamp_unit : TYPE
        DESCRIPTION.
    type_profile : TYPE
        DESCRIPTION.

    Returns
    -------
    power_profile : dictionnary.
        The monthly power profile.

    """
    
    power = m.importation(filepath, column)
    
    power_profile = m.power_profile_creation(power, time_stamp, time_stamp_unit, type_profile)
    
    return power_profile

def optimisations(len_pop, n_iter, n_sample, power, configs, fixeds, printed):
    best_schedules = []
    fixeds = fixeds.copy()
    
    for config in configs:
        # For each configuration, we obtain the best_schedule
        schedules, objs = m.optimisation(len_pop, n_iter, n_sample, power, config, fixeds, False)
        best_schedule = schedules[objs.index(min(objs))]
        
        # Then we add that schedule as a constant demand
        demand = [s*config[4] for s in best_schedule]
        fixeds.append([demand, config[5]])
        
        # Best schedule
        best_schedules.append(best_schedule)
        
    return best_schedules, min(objs)

def final(len_pop, n_iter, n_sample, power, configs, fixeds, printed):
    
    # We let two CPUs remaining
    if os.cpu_count() > 2:
        nb_proc = os.cpu_count()-2
    else:
        nb_proc = 1

    # We create the arguments
    n_sample = (n_sample//nb_proc)+1
    args = [len_pop, n_iter, n_sample, power, configs, fixeds, printed]
    
    # We prepare the outputs
    best_schedules_objs = []
    
    # Multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(optimisations, *args) for _ in range(nb_proc)]
        
        for f in concurrent.futures.as_completed(results):
            schedules, obj = f.result()
            best_schedules_objs.append([schedules, obj])
              
    # Best schedule with its objective
    best_schedule, objective = sorted(best_schedules_objs, key=lambda x:x[1])[0]
    
    return best_schedule

def final_2(len_pop, n_iter, n_sample, power, configs, fixeds, printed):
    # correct one
    best_schedules = []
    fixeds = fixeds.copy()
    
    for config in configs:
        # For each configuration, we obtain the best_schedule
        
        # schedules and objs contains respectively the list of each schedule and objective
        schedules, objs = m.optimisation_mp(len_pop, n_iter, n_sample, power, config, fixeds, False)
        
        # best_schedule contains the best_schedule for the configuration config
        best_schedule = schedules[objs.index(min(objs))]
                
        # Best schedule is added
        best_schedules.append(best_schedule)
        
        # We add that schedule as a constant demand for the next config optimisation
        demand = [s*config[4] for s in best_schedule]
        fixeds.append([demand, config[5]])
        
    return best_schedules, min(objs)

def calculation(filepath, column, type_profile, configs, fixeds, development, parameters=None):
    """
    To run the optimisation in the UI.

    Parameters
    ----------
    Defined in the previous functions.
    
        development : bool
        To determine if we are in development phase or in using phase.
            If True :
                The optimisation will be quicker, for the sake of verifying that everything works.
            If False :
                The optimisation will be as usual.
        
        params : list or None
            If development is False : parameters = [len_pop, n_iter, n_sample]
            
            If development is True : parameters = None
            

    Returns
    -------
    Defined in the previous functions.

    """
    power_profile = initialisation(filepath, column, 1, 'hour', type_profile)
    if development:
        len_pop = 20
        n_iter = 1
        n_sample = 1
    else:
        len_pop, n_iter, n_sample = parameters
    
    best_schedule, min_obj = final_2(len_pop, n_iter, n_sample, power_profile, configs, fixeds, development)
    return power_profile, best_schedule
# %%
