# Ajouter la documentation
# Ne pas oublier de faire des tests : opt ou opt2 ? Et quel len_pop, n_iter, 
# n_sample choisir ?

# test power_profile_creation
# ("1 hour - monthly" fonctionne)

from base_package import *
import pandas as pd
import numpy as np
import os
import concurrent.futures

def importation(filepath, column):
    """
    To import the power data file.

    Parameters
    ----------
    filepath : str
        The file path of the power data.
    column : int or str
        The index of the column (starts from 1) or the name 
        of the column.

    Returns
    -------
    power : list
        The power data as a list.

    """
    ext = filepath[filepath.rindex('.')+1:]
    
    if ext == "csv":
        df = pd.read_csv(filepath)
    elif ext == "xlsx":
        df = pd.read_excel(filepath)
        
    if type(column) is str:    
        power = df[column].tolist()
    else:
        power = df.iloc[:,column-1].tolist()
        
    return power

def power_profile_creation(power, time_stamp, time_stamp_unit, type_profile):
    """
    Create the average monthly yield power of the system.

    Parameters
    ----------
    power : list
        List containing of the power produced per time stamp 
        selected.
        
    time_stamp : int
        Value of the time stamp.
        
    time_stamp_unit : string
        Unit of the time stamp.
        Can be :
            - "second",
            - "minute",
            - "hour".
        
    type_profile : string
        Type of profile desired. Can be :
            - "daily"
            - "weekly"
            - "monthly"
            - "seasonnaly"
    Return
    ------
    power_m : dictionnary.
        The power profile. Each value is the power average 
        in relation to the selected profile type.
        
        If type_profile = "daily" :
            power_m[i] is the average power produced at each hour of a
            day i.
        If type_profile = "weekly" :
            power_m[i] is the average power produced at each hour of a 
            day of the week i.
        If type_profile = "monthly" :
            power_m[i] is the average power produced at each hour of a
            day of the month i.
        If type_profile = "yearly" :
            power_m[i] is the average power produced at each hour of a
            day of the year.
    """
        
    #per year
    nb_weeks = 52
    nb_days = 364
    nb_hours = 8736
    nb_months = 12
    
    # number of samples of power used to compute the power for an hour/day
    if time_stamp_unit == "seconds":
        nb_samples_per_hour = 3600//time_stamp
        nb_samples_per_day = nb_samples_per_hour*24

    elif time_stamp_unit == "minute":
        nb_samples_per_hour = 60//time_stamp
        nb_samples_per_day = nb_samples_per_hour*24
    
    elif time_stamp_unit == "hour":
        nb_samples_per_hour = -1
        nb_samples_per_day = 24//time_stamp
    

    # power_per_hour is a list containing the hourly yield power of the system
    power_per_hour = []
    
    # power_per_day is a dictionnary indexed by the number of days, and containing
    # the hourly yield power of the system.
    power_per_day = {}
    
    if nb_samples_per_hour > 0:
        nb_samples_from_power = min(nb_samples_per_hour*nb_hours, len(power))
        nb_hours_created = nb_samples_from_power//nb_samples_per_hour
    
        for i in range(nb_hours_created):
            tmp = power[i*nb_samples_per_hour:(i+1)*nb_samples_per_hour]
            power_per_hour.append(np.average(np.array(tmp)))
            
        nb_days_created = nb_hours_created//24
        
        for i in range(nb_days_created):
            power_per_day[i] = power_per_hour[24*i:24*(i+1)]
    
    else:
        nb_samples_from_power = min(nb_samples_per_day*nb_days, len(power))
        nb_days_created = nb_samples_from_power//nb_samples_per_day
            
        for i in range(nb_days_created):
            tmp = power[i*nb_samples_per_day:(i+1)*nb_samples_per_day]
            # tmp = reg(tmp)
            power_per_day[i] = tmp

    power_res = {}
    
    if type_profile == 'daily':
        power_res = power_per_day
        
    elif type_profile == 'week':
        power_res = {}
        
        actual_day = 0
        len_power_res = nb_days_created//7
        
        for w in range(len_power_res):
            list_power_one_week = []
            for d in range(7):
                list_power_one_week.append(power_per_day[actual_day])
                actual_day += 1
            power_res[w+1] = [np.average(np.array(x)) for x in zip(*list_power_one_week)]
    
    elif type_profile == "monthly":
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 29]
        
        actual_day = 0
        for m in range(len(months)):
            list_power_one_month = []
            for d in range(months[m]):
                list_power_one_month.append(power_per_day[actual_day])
                actual_day += 1
            power_res[m+1] = [np.average(np.array(x)) for x in zip(*list_power_one_month)]
    
    elif type_profile == "yearly":
        list_power_a_year = []
        for key, value in power_per_day.items():
            list_power_a_year.append(value)
        power_res[1] = [np.average(np.array(x)) for x in zip(*list_power_a_year)]
    
    return power_res

file_name = 'data/power_fixed.csv'
power_supplied = importation(file_name, 2)   
power_m = power_profile_creation(power_supplied, 1, 'hour', 'monthly')

# --------------- Generic optimisation

"""
Definitions :
    
    1. A config is a list describing the configuration of a device.
    It is composed of :
        - deb : int,
            the first hour at which the device can be run,
        - fin : int,
            the last hour at which the device can be run,
        - nb_device : int,
            the number of devices that can be used per hour,
        - total_devices : int,
            the total number of devices that can be run during a
            day,
        - P_device : float,
            the power capacity of the device IN KW,
        - name : str,
            the name of the device (for graphic purposes)

    config = [deb, fin, nb_device, total_devices, P_device, name]
    
    2. A fixed is a list describing the configuration of a device,
    t
fixed = [schedule, name]
"""

def algo(len_pop, n_iter, power, config, fixeds, printed):
    """
    
    An unique genetic algorithm run, to obtain the best schedule,
    with an unique config and several fixed in [fixeds].
    
    Parameters
    ----------
    len_pop : int > 5
        The size of a population (the number of the 
        schedule in the initialize population).
    n_iter : int
        The number of iterations of the algorithm.
    power : dictionnary
        The power profile.
    config : list
        An unique configuration.
    fixeds : list
        A list of fixed.
    printed : boolean
        If True : print the algorithm progression.

    Returns
    -------
    best_el : schedule
        The best schedule obtained
    best_obj : float
        Its associated value from the objective function.
    sort_pop : list of schedule
        A list of the schedules sorted objective-function-associated.

    """
    pop_init = create_pop(len_pop, config)

    new_gene = pop_init
    
    nb_mut_from_best = 4
    nb_others = len_pop - 1 - nb_mut_from_best
    
    p_crossover = 0.9
    p_mut = 0.3
    
    for k in range(n_iter):
        # Sort/selection
        sort_pop, best_el, best_obj = sort_selection(power, new_gene, config, fixeds)
        # breakpoint()

        # We add the best one
        new_gene = [best_el]
        
        # We add mutations of the best one
        for i in range(nb_mut_from_best):
            muted =  mutation(best_el, p_mut, config)
            muted = to_be_constrained(muted, config)
            new_gene.append(muted)
        
        # We do crossovers
        for i in range(nb_others):
            P1 = parents_selection(sort_pop)
            P2 = parents_selection(sort_pop)
            
            child = crossover(P1, P2, p_crossover, config)
            child = to_be_constrained(child, config)
            new_gene.append(child)
        
        # We do mutations 
        for i in range(nb_mut_from_best+1, len_pop):
            P = parents_selection(new_gene)
            p_muted = mutation(P, p_mut, config)
            p_muted = to_be_constrained(p_muted, config)
            new_gene[i] = p_muted
        
        if printed == True:
            print(f"{k+1}/{n_iter} iterations, best objective {best_obj}")
        
        sort_pop, best_el, best_obj = sort_selection(power, new_gene, config, fixeds)
        
    return best_el, best_obj, sort_pop

def optimisation(len_pop, n_iter, n_sample, power, config, fixeds, printed):
    """
    Run algo n_sample times.
    Return the best schedule for each run of algo.

    Parameters
    ----------
    Like usual

    Returns
    -------
    best_els : list of schedule
        List of the best schedules.
    best_objs : list of float
        List of the objective function associated with each schedule.

    """
    
    best_els, best_objs = [], []

    for k in range(n_sample):
        best_el, best_obj, sort_pop = algo(len_pop, n_iter, power, config, fixeds, False)
        best_els.append(best_el)
        best_objs.append(best_obj)

        if printed == True:
            print(f"{k+1}/{n_sample} iterations.")

    return best_els, best_objs

def optimisation_mp(len_pop, n_iter, n_sample, power, config, fixeds, printed):
    """
    Run algo approximatively n_sample times by multiprocessing
    Return the best schedule for each run of algo.

    Parameters
    ----------
    Like usual

    Returns
    -------
    best_els : list of schedule
        List of the best schedules.
    best_objs : list of float
        List of the objective function associated with each schedule.

    """

    # We let two CPUs remaining
    if os.cpu_count() > 2:
        nb_proc = os.cpu_count()-2
    else:
        nb_proc = 1

    # We get the number of samples
    n_sample = (n_sample//nb_proc)+1
    args = [len_pop, n_iter, n_sample, power, config, fixeds, printed]
    
    # We prepare the outputs
    total_best_schedules = []
    total_best_objs = []
    
    # Multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(optimisation, *args) for _ in range(nb_proc)]
        
        for f in concurrent.futures.as_completed(results):
            best_els, best_objs = f.result()
            
            total_best_schedules += best_els
            total_best_objs += best_objs
              
    return total_best_schedules, total_best_objs

