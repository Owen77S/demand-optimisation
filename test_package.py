import numpy as np

def create_fixeds(nb):
    """
    To create a random fixed(s).

    Parameters
    ----------
    nb : int
        Number of fixed in the returned list.

    Returns
    -------
    fixeds : list
        List of fixed

    """
    fixeds = []
    letters = ['a', 'b', 'c', 'd', 'e']
    
    for _ in range(nb):
        schedule = [np.random.randint(0, 11) for i in range(24)]
        
        len_name = np.random.randint(1, 11)
        name = ""
        for i in range(len_name):
            name += letters[np.random.randint(0, 5)]
            
        fixeds.append([schedule, name])

    return fixeds

def create_configs(nb):
    """
    To create a random config(s).

    Parameters
    ----------
    nb : int
        Number of config in the returned list.

    Returns
    -------
    configs : list
        List of config

    """
    
    configs = []
    letters = ['a', 'b', 'c', 'd', 'e']
    
    for _ in range(nb):
        deb = np.random.randint(0, 24)
        fin = np.random.randint(deb+1, 25)
        nb_devices = np.random.randint(1, 4)
        total_devices = ((fin-deb)*nb_devices)//2
        P_device = np.random.randint(1, 10)*5
        
        len_name = np.random.randint(1, 11)
        name = ""
        for i in range(len_name):
            name += letters[np.random.randint(0, 5)]
           
        config = [deb, fin, nb_devices, total_devices, P_device, name]
        configs.append(config)
        
    return configs