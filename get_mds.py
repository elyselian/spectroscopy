# -*- coding: utf-8 -*-
"""
Created on Sun May  4 23:36:30 2025

@author: aqilk
"""
# This code consists of a function to import various MDSPlus data for use in data analysis.

# Imports
import MDSplus as mds
import matplotlib.pyplot as plt
import numpy as np

def get_currents(shot, time=None):
    
    """
Imports discharge current data from MDSPlus.

Args:
    shot (integer): The shot number
    time (float): Optional. The time value for which a specific current value is desired.

Returns:
    t_ip: The time basis for the plasma current
    ic: The plasma current in kiloamperes
    i_ind: The index of the specific current value if a particular time is specified.
"""    
    # Select shot
    # shot = 250428045
    
    # Connect to the zappa server
    c = mds.Connection('zappa.zap')
    c.get("getenv('zaphd_path')")
    # Open the tree for a given shot
    c.openTree('zaphd',shot)
    

    # Rogowski currents in kA
    t_ip = np.array(c.get(r'dim_of(\i_p)')*1e6)
    ip = np.array(c.get(r'\i_p')/-1000)
    ia = np.array(c.get(r'\i_accel')/-1000)
    ic = np.array(c.get(r'\i_compress')/-1000)
    
    # Current calculated with m0 at P0
    i_p0 = np.array(c.get(r'\m_0_p0')*1e6*5*0.100838)
    
    # Time base in us
    t_m0 = np.array(c.get(r'dim_of(\m_0_p0)')*1e6)
    
    # Current calculated with m0 at P10
    i_p10 = np.transpose(np.array(c.get(r'\m_0_p10')*1e6*5*0.100838))
    
    # Time base in us
    t_m0_p10 = np.array(c.get(r'dim_of(\m_0_p10)')*1e6)

    # Get PMT trace
    # pmt =np.array(c.get(r'\pmt')*1e6*5*0.100838)

    if time != None:
        t_ind = np.argmin(np.abs(t_m0-time))
        # i_ind = t_ip[t_ind]
        i_ind = i_p10[t_ind]
    else:
        i_ind = None
    
    # integrate to get total charge for each shot
    # charge = np.trapz(m0_p0*1e6*5*0.100838,t_m0)

    return t_ip, ip, i_p10, t_m0_p10, i_ind, ic

def get_dhi(shot):
    
    """
Imports DHI data from MDSPlus. 

Args:
    shot (integer): The shot number

Returns:
    t_ip: The time basis for the plasma current

"""    
    # Select shot
    # shot = 250428045
    
    # Connect to the zappa server
    c = mds.Connection('zappa.zap')
    c.get("getenv('zaphd_path')")
    # Open the tree for a given shot
    c.openTree('zaphd',shot)
    
    # Densities
    
    # Line-integrated density
    dhi_int = np.array(c.get(r'\dhi_int'))
    
    # Abel inverted number density
    dhi_inverted = np.array(c.get(r'\dhi_inverted'))
    
    # Horizontal axis of dhi_int and dhi_inverted data
    x_twin = np.array(c.get(r'\x_twin'))
    
    # Vertical axis of dhi_int and dhi_inverted data
    y_twin = np.array(c.get(r'\y_twin'))


    return dhi_int, dhi_inverted, x_twin, y_twin