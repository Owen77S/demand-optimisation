import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import test_package as test

def plot_profile(power, P, configs, fixeds, GUI=False):
    """
    To plot the profiles.

    Parameters
    ----------
    power : dictionary.
        Dictionary containing the monthly power profile.
    P : TYPE
        DESCRIPTION.
    configs : list
        List of config.
    fixeds : list
        List of fixeds.
    GUI : bool, optional
        If GUI = True : the function returns a figure.
        If GUI = False : the functions plot the profile only.
        The default is False.

    Returns
    -------
    figure : matplotlib.Figure
        The figure containing the profiles.

    """
    # if type(P[0]) != list:
    #     P = [P]
        
    # if type(configs[0]) != list:
    #     # If there is only one configuration
    #     configs = [configs]
    # else:
    #     configs = sorted(configs, key=lambda x:x[4], reverse=True)
    
    name_month = ["January",
                  "February",
                  "March",
                  "April",
                  "May",
                  "June",
                  "July",
                  "August",
                  "September",
                  "October",
                  "November",
                  "December"]
    
    demands = []
    for p, config in zip(P, configs):
        capacity = config[4]
        demand = [h*capacity for h in p]
        demands.append(demand)

    t = range(1,25)
    
    # Sorting for better plot
    names = [config[5] for config in configs]
    
    if len(fixeds) == 0:
        demands_fixed = []
        names_fixed = []
    else:
        demands_fixed_names = [x for x in sorted(fixeds, reverse=True, key = lambda x:sum(x[0]))]
        
        demands_fixed = [demand_name[0] for demand_name in demands_fixed_names]
        names_fixed = [demand_name[1] for demand_name in demands_fixed_names]

    if GUI == True:
        figure = Figure(figsize=(5, 4), dpi=100)
        
        for month in range(12):
            axis = figure.add_subplot(3, 4, month+1)
            if month == 11:
                axis.plot(t, power[month+1],'r', label="Power produced")
                axis.stackplot(t, demands_fixed+demands, labels=names_fixed+names)
            else:
                axis.plot(t, power[month+1],'r', label="_Power produced")
                axis.stackplot(t, demands_fixed+demands, labels=['_'+name for name in names_fixed+names])
            if month%4 == 0:
                axis.set_ylabel("Power [kW]")
            if month//4 == 2    :
                axis.set_xlabel("Hour")
            axis.title.set_text(name_month[month])
            axis.title.set_fontsize(8)
            
            figure.legend()
        
        return figure
    
    else:
        figure, axis = plt.subplots(4, 3)
        for month in range(12):
            axis[month//3, month%3].plot(t, power[month+1],'r', label="Power produced")
            axis[month//3, month%3].stackplot(t, demands_fixed+demands, labels=names_fixed+names)
            axis[month//3, month%3].set_title(name_month[month])
            if month%3 == 0:
                axis[month//3, month%3].set_ylabel("Power [kW]")
            if month//3 == 3:
                axis[month//3, month%3].set_xlabel("Hour")
        
        plt.legend()
        plt.show()
    
def show_fixed(fixed):
    """
    To display a fixed.

    Parameters
    ----------
    fixed : list.
        A fixed
    
    Returns
    -------
    figure : matplotlib.Figure.
        Figure containing the plots. Required by tkinter (GUI)
        to display figures in windows.

    """
        
    figure = Figure(figsize=(5, 4), dpi=100)
    axis = figure.add_subplot()
    
    schedule, name_device = fixed
    axis.stairs(schedule, label=name_device)

    axis.set_title(f"Daily power consumption of {name_device}")
    axis.set_xlabel("Hour")
    axis.set_ylabel("Power [kW]")
    axis.legend()
    
    return figure
    
def plot_configs(schedules, configs):
    """
    To plot different schedules

    Parameters
    ----------
    schedules : list
        List of schedules. 
        The schedules are assumed to be in ascendent order,
        capacity wise.
    configs : list
        List of config associated with the schedule.
        The configs are not assumed to be in ascendent order,
        capacity wise.

    Returns
    -------
    list_figures : list
        List of the figures displaying each schedule.

    """
    list_figures = []
    
    for schedule, config in zip(schedules, configs):
        
        figure = Figure(figsize=(5, 4), dpi=100)
        axis= figure.add_subplot()
        axis.stairs(schedule)
        axis.set_xlabel("Hour")
        axis.set_ylabel("Number of devices run per hour")
        axis.set_title(f"Optimised schedule for {config[5]}")
        
        list_figures.append(figure)
        
    return list_figures

def figure_creation():
    """
    To create a figure. 
    
    Used by the GUI.
    Created to avoid mixing plotting functions in the GUI.

    Returns
    -------
    fig : matplotlib.Figure
        An empty figure, used by the GUI.
        
    ax : matplotlib.axes
        An empty axe with its x_label used by the GUI.

    """
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    
    ax.set_xlabel("Hour")
    
    return fig, ax

def change_name_device(ax, name):
    """
    To change the y_label and the title of the figure,
    according to the name of the device of the displayed schedule.

    Used by the GUI.
    Created to avoid mixing plotting functions in the GUI.
    
    Parameters
    ----------
    ax : matplotlib.axes
        The used axe.
    name : string
        The name of the device.

    """
    ax.set_ylabel(f"Number of {name} per hour")
    ax.set_title(f"Optimised daily schedule for {name}")