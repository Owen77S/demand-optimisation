import random

def f(power, P, configs, fixeds, time_range=1):
    """
    The objective function.

    Parameters
    ----------
    power 
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.
    configs : TYPE
        DESCRIPTION.
    fixeds : TYPE
        DESCRIPTION.
    time_range : str

    Returns
    -------
    energy_stored_required : TYPE
        DESCRIPTION.

    """
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    energy_stored_required = 0
    if type(P[0]) != list:
        P = [P]
        
    if type(configs[0]) != list:
        # If there is only one configuration
        configs = [configs]

    if len(fixeds) == 0:
        # If there is no fixed devices.
        demand_per_hour = []
        capacities = [config[4] for config in configs]
        for m in range(1, 13):
            for h in range(24):
                if m == 1:
                    demand_per_hour.append(sum([capacities[index]*schedule[h] for index, schedule in enumerate(P)]))
                energy_stored_required += max(demand_per_hour[h] - power[m][h], 0)*months[m-1]
    
    else:
        demand_per_hour = []
        demand_per_hour_fixed = []
        capacities = [config[4] for config in configs]
        
        d_fixed = [f[0] for f in fixeds]

        for m in range(1, 13):
            for h in range(24):
                if m == 1:
                    demand_per_hour.append(sum([capacities[index]*schedule[h] for index, schedule in enumerate(P)]))
                    demand_per_hour_fixed.append(sum([f[h] for f in d_fixed]))
                energy_stored_required += max(demand_per_hour[h] + demand_per_hour_fixed[h] - power[m][h], 0)*months[m-1]
                
    return energy_stored_required    

def to_be_constrained(p, config):
    """
    Function used to constraint an already existing schedule.

    Parameters
    ----------
    p : list of 24 integers.
        A schedule.
        
    config : a config

    Returns
    -------
    p : list of 24 integers.
        A constrained schedule.

    """
    deb, fin, nb_device, total_device, P_device, name = config
    fin -= 1
    p = p.copy()
    if sum(p) > total_device:
        while sum(p) > total_device:
            i = random.randint(deb, fin)
            if p[i] != 0:
                p[i] -= 1
    elif sum(p) < total_device:
        while sum(p) < total_device:
            i = random.randint(deb, fin)
            if p[i] != nb_device:
                p[i] += 1
    return p

def create_unit(config):
    """
    VERIFIED
    Create a schedule. A demand exists between the hour numero deb
    and end at the hour fin. Each hour, there can be nb_pump
    pumps at most.
    
    For example if deb = 8 and fin = 11, we can have a demand :
        - between 8 and 9,
        - between 9 and 10,
        - between 10 and 11.

    Parameters
    ----------
    nb_pump : int.
        The number of pumps allowed per hour.
    deb : int
        Between 0 and 23.
    fin : int
        Between 1 and 24.

    Returns
    -------
    P : list of 24 elements
        A schedule
        For example : 
            P = [0, 0, 0, 0, 0, 3, 6, 2, 9, 8, 7, 6,
             1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    """
    deb, fin, nb_pump, total_pump, P_pump, name = config
    P = [0]*(deb)
    for i in range(fin-deb):
        P.append(random.randint(0, nb_pump))
    P += [0]*(24-fin)
    return P

def create_pop(len_pop, config):
    "VERIFIED"
    pop = []
    for _ in range(len_pop):
        p = create_unit(config)
        pop.append(to_be_constrained(p, config))
    return pop

def sort_selection(power, pop, config, fixeds):
    "VERIFIED"
    pop_obj = []
    
    # Get all the objective associated with the pop
    for schedule in pop:
        obj_associated = f(power, schedule, config, fixeds)
        pop_obj.append([schedule, obj_associated])
    
    # Sort
    sort_pop_obj = sorted(pop_obj, key=lambda x:x[1])
    sort_pop = [el_obj[0] for el_obj in sort_pop_obj]
    
    best_el, best_obj = sort_pop_obj[0]
    return sort_pop, best_el, best_obj

def parents_selection(pop):
    "VERIFIED"
    return random.choice(pop)

def crossover(p1, p2, probability, config):
    """
    The crossover function.

    Parameters
    ----------
    p1 : list.
        The first schedule for the crossover.
    p2 : list.
        The second schedule for the crossover.
    probability : float between 0 and 1.
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    child : TYPE
        DESCRIPTION.

    """
    deb, fin, nb_pump, total_pump, P_pump, name = config
    # Here, we consider subarrays of the schedule. If we have 
    # two parents with deb = 0 and fin = 2, keeping fin = 2
    # help ensure that we can have child = P1, child = P1[0]+P2[1]
    # or child = P2
    P1 = p1.copy()
    P2 = p2.copy()
    p = random.uniform(0, 1)
    if p<probability:
        i = random.randint(deb, fin)
        child = []
        child = P1[:i] + P2[i:]
    else:
        d = random.randint(0, 1)
        child = d*P1 + (1-d)*P2
    return child

def mutation(p, probability, config):
    """
    Mutation function.

    Parameters
    ----------
    p : list of 24 elements.
        The schedule of a device.
    probability : between 0 and 1.
        The probability for the schedule to mute.
    config : list of 6 elements.
        The configuration of the device.

    Returns
    -------
    mut : list of 24 elements
        The muted schedule.

    """
    deb, fin, nb_pump, total_pump, P_pump, name = config  
    fin -= 1 # If we only allow to run pump between 0 and 1 for example, we can only modify the very first 
    # element of the schedule. To ensure this proprerty, fin needs to be reduce to one.
    mut = p.copy()
    p = random.uniform(0, 1)
    if p < probability:
        nb_mutations = random.randint(1, int(fin-deb/2))
        for i in range(nb_mutations):
            i_mut= random.randint(deb, fin)
            mut[i_mut] = random.randint(0, nb_pump)
    return mut