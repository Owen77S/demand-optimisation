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
    

    2. A fixed is a list describing the configuration of a device with a specific 
    consumption
fixed = [schedule, name]