# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:55:11 2025

@author: aqilk
"""

# This script imports the S/XB coefficient data from .csv files obtained via the OPEN-ADAS databse. Based on a Matlab script I wrote in 2022.


# EDIT NOTES
# 4/6/25 - Completed script to extract SXB coefficient data from the .csv file. This is not generalized to any .csv from ADAS. Rudimentary 3d plotting
# of the values for temperature and density profiles, and a basic 2d plot with densities as separate traces.  Want to implement this script as a callable
# function that takes in density and temperature values as inputs, and outputs interpolated SXB values for this species.

# 4/19/25
# TODO: set up so that code outputs an S/XB value for a particular density and temperature input
# Improved look of contour plot

##########################################################################################

# Import required packages

import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import csv
import pandas as pd
from scipy.optimize import curve_fit

#%% Create class 

class read_sxb:
    def __init__(self, ion):
        self.ion = 'C-III'
        
    #%% Import data from selected .csv files:

    # Enter the file path to the .csv containing S/XB data for C-III:
    file_paths = ['G:\\Shared drives\\Shumlak Lab\\Diagnostics\\Spectroscopy\\S_XB\\Data\\SXB Coefficients\\sxb96#c_pju#c2.csv']
    
    # Initialize an empty list to store DataFrames
    data_arrays = []
    
    # Define the strings to be treated as NaN
    na_values = ['inf',' -inf',float("inf")]
    
    # Iterate over the file paths
    for file_path in file_paths:
        # Read the CSV file into a DataFrame. This imports the header row with wavelength information as row 0.
        df = pd.read_csv(file_path, na_values = na_values, delimiter=',', header=0, skiprows=3555, nrows=79)
    
        # Convert the DataFrame to a Numpy array and append it to the list
        data_array = df.to_numpy() # Each array is 999999 rows, 4 columns 
        data_arrays.append(data_array)
    
    # This is a LIST of NumPy ARRAYS
    sxb_import = data_arrays
    
    # What does this line do?
    sxbdata = sxb_import[0][1:79,0:7]
    
    # Get the wavelength in Angstroms
    wlng = sxb_import[0][0,0]
    
    # Number of densities
    ndens = int(sxb_import[0][0,2])
    
    # Number of temperatures
    nte = int(sxb_import[0][0,3])
    
    # black magic stuff copied from matlab code which was generalized for ADAS SXB data files of any species. Not necessary if only looking at one species.
    nrowdens = math.ceil(ndens/8) #determine minimum number of rows for the density values
    nrowte = math.ceil(nte/8)   # Determine minimum number of rows for the temperature values
    dens_end = 1+nrowdens-1 # Index of the last row that contains density data
    te_start = 1+nrowdens   # Index of the first row that contains temperature data
    te_end = nrowdens+nrowte # Index of the last row that contains temperature data
    sxb_start = te_end+1 # Index of first row that contains sxb data
    sxb_end = sxb_start + (ndens*nrowdens)-1 # Index of last row that contains sxb data
    
    dens_2d = sxb_import[0][1:4,0:8]    # Extract the density values from the array
    te_2d = sxb_import[0][4:7,0:8]  # Extract the temperature values from the arrray
    
    dens = np.zeros(24) #Preallocate numpy array so that values are floats and not strings
    dens[:] = dens_2d.flatten()  # Turns 2d array of density values into single column, convert from cm-3 to m-3
    dens = dens*1e6 
    
    te = np.zeros(24)
    te[:] = te_2d.flatten()  # Turns 2d array of temperature values into single column
    
    # There are 24 densities and 24 temperatures. Starting from row 7, the next 24 values are all S/XB coefficients for increasing temperature
    # for the first density value, and so on...
    
    # Preallocate array for S/XB values. Rows are densities and columns are temperatures.
    sxbdata = np.zeros([int(ndens),int(nte)])
    
    jind = np.arange(sxb_start, sxb_end+1)
    j = np.arange(sxb_start,(ndens+2)*nrowdens,nrowdens)
    
    for i in np.arange(ndens):
        sxb_2d = sxb_import[0][j[i]:j[i]+3,0:8]
        sxbdata[i,:] = sxb_2d.flatten()
    
    
    
    #%% Interpolate sxb values
    
    # Define array of temperature values for interpolation
    interp_size = 100
    te_interp = np.linspace(0.388,2330,interp_size)
    # te_interp = np.linspace(0,2330,interp_size)

    
    # Define array of density values for interpolation
    dens_interp = np.linspace(2.19e10,2.19e24,interp_size)
    
    #Pre-allocate array for interpolated values in temperature
    sxb_interp_1 = np.zeros([ndens,interp_size])
    
    # Interpolate S/XB values at existing density values, but for interp_size temperature values
    for i in np.arange(ndens):
        sxb_interp_1[i] = np.interp(te_interp, te, sxbdata[i])
        
    
    #Pre-allocate array for interpolated values in temperature and density
    sxb_interp_2 = np.zeros([interp_size,interp_size])
        
    # Interpolate S/XB values at each new temperature value for interp_size density values
    for i_d in np.arange(interp_size):
        sxb_interp_2[:,i_d] = np.interp(dens_interp, dens, sxb_interp_1[:,i_d])
    
    #%% Plot sxb data
    
    def plot_2d(self):
        fig1 = plt.figure()
        
        plt.pcolormesh(self.te_interp, self.dens_interp, self.sxb_interp_2, vmin = 0, vmax = 150, cmap = 'plasma')
        # plt.pcolormesh(self.te_interp, np.log10(self.dens_interp), self.sxb_interp_2, vmin = 0, vmax = 150, cmap = 'plasma')

        plt.xlim([10, 1000])
        # plt.xlim([np.log10(1), np.log10(1000)])

        # plt.ylim([np.log10(1e20), np.log10(2e23)])
        plt.ylim([0, 2e23])
        
        # # after plotting the data, format the labels
        # current_values = plt.gca().get_yticks()
        # plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values]) 
        
        cbar = plt.colorbar()
        cbar.set_label('S/XB coefficient [ionizations per photon]', rotation=90, labelpad=15)

        # Plot 2d colormap using pcolormesh (faster than pcolor with easy customization)
        # flip = np.flipud(frame[0])
        # plt.pcolormesh(self.te_interp, np.log10(self.dens), self.sxb_interp_1, cmap = 'plasma')
        # plt.gca().invert_yaxis()
        
        # plt.title( "S/XB Coefficients for C-III 229.7 nm", fontsize = 18)
        plt.xlabel('$T_e$ [eV]')
        plt.ylabel('$n_e$ [m$^{-3}$]')
        # plt.zlabel('S/XB coefficient [ionizations per photon]')

        plt.tight_layout()
        # plt.show()
        return fig1    
    #%% SXB 3D plot
    
    def plot_3d(self):
        fig2 = plt.figure(figsize=(14,10))
        
        #Syntax for 3D projection
        ax = plt.axes(projection='3d')
        # ax = mplot3d.Axes3D(fig2)
        
        # Define 3 axes
        x = self.te_interp    # Use [::-1] to reverse the order of elements in the array, but ticks don't follow
        y = np.log10(self.dens_interp)
        z = self.sxb_interp_2
        
        # Meshgrid?
        X, Y = np.meshgrid(x,y)
        Z = z
        
        # Plot the surface
        surf = ax.plot_surface(X,Y,Z, cmap='plasma', linewidth=0.3, edgecolor='black')
        # surf = ax.plot_trisurf(x,y,z,cmap=plt.cm.jet, linewidth=0.1)
        ax.set_title('S/XB Coefficients for C-III 229.7 nm', fontsize=18)
        # ax.view_init(elev=20, azim=-45)
        # ax.set_zlim([0, 2000])
        fig2.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
        
        # Set axis labels and size
        ax.set_xlabel('Electron Temperature (eV)', fontsize=14)
        ax.set_ylabel('Log of Electron Density (m$^{-3}$)', fontsize=14)
        ax.set_zlabel('S/XB Coefficient (ionizations per photon)', fontsize=14)
        # plt.show()
        return fig2
    
    #%% 2d plots with density as parameter
    
    def plot_dens_param(self):
        fig3 = plt.figure(figsize=(12,6))
        
        for i in np.arange(0,100,10):
            plt.plot(self.te_interp,self.sxb_interp_2[i], label=str(self.dens_interp[i]))
            
        plt.title( "S/XB Coefficients for C-III 229.7 nm", fontsize=18)
        plt.xlabel('Electron Temperature (eV)', fontsize=14)
        plt.ylabel('S/XB Coefficient (ionizations per photon)', fontsize=14)
        plt.legend(title="Electron Density (m$^{-3}$) ")
        plt.show()
        return fig3
    
    #%% When called from outside function, interpolate sxb data to the desired density and temperature values
    # Two approaches: Can either match the closest value in the interpolated sxb array, or create a fit and evaluate at desired plasma parameters
    
    def get_sxb(self, Te, ne):
        """
    Returns the value of the S/XB coefficient for the provided electron temperature and electron number density.
    
    Args:
        Te (float): The electron temperature in eV.
        ne (float): The electron number density in m^-3
    
    Returns:
        float: The S/XB coefficient corresponding to the plasma parameters.
    """
        # Find interpolated Te closest to desired Te
        te_ind = np.argmin(np.abs(self.te_interp-Te))
        te_val = self.te_interp[te_ind]
        
        # Find interpolated density closest to desired ne
        ne_ind = np.argmin(np.abs(self.dens_interp-ne))
        ne_val = self.dens_interp[ne_ind]
        
        sxb_out = self.sxb_interp_2[ne_ind,te_ind]
        
        return sxb_out
