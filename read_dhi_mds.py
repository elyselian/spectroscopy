# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:10:52 2025

@author: aqilk
"""

# Description:
# This script imports DHI data stored within MDSPlus data structures for the ZaP-HD experiment. Specifically, the DHI data
# that has been processed to produce Abel-inverted electron number densities, as well as associated data, was stored in the
# "DHIHD" tree within the "ANALYSISHD" tree. 

# Further details can be found in Appendix C of Michael Ross's PhD thesis.


import MDSplus as mds
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz, cumtrapz, quad, cumulative_trapezoid, trapezoid
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy.io as sio
# from sympy import symbols, integrate

def dhi_profiles(chordconfig, radius):
    """
Outputs the radial profiles of electron number density, azimuthal magnetic field, and electron temperature. 

Args:
    chordconfig (string): Either 'lateral' or 'axial'.
    radius (float): The desired radius in meters for the sharp pinch assumption for evaluating electron temperature.
Returns:
    

"""    
    def get_dhi(shot):
        
        """
    Imports DHI data from MDSPlus. 
    
    Args:
        shot (integer): The shot number
    
    Returns:
        dhi_int: The line-integrated density
        dhi_inverted: Abel inverted number density
        x_twin, y_twin: Horizontal and vertical axis of dhi_int and dhi_inverted data
    
    """    
       
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
        
        # Centroid locations in units of pixel number
        dhi_centroids = np.array(c.get(r'\centroid_abs'))
    
        return dhi_int, dhi_inverted, x_twin, y_twin, dhi_centroids
    
    
    # Select shot number
    shot = 160524021
    
    # Import DHI data stored in MDSPlus
    dhi_data = get_dhi(shot)
    
    # Horizontal axis in cm
    z = dhi_data[2]*100
    
    # Vertical axis in cm
    y = dhi_data[3]*100
    
    # Line-integrated density
    ne_int = dhi_data[0]
    
    # Abel inverted density in m^-3
    ne_inv = dhi_data[1]
    
    #%% Contour plot of number density 
    
    # Centroids in pixels. Change centroids array elements to integer
    centroids = dhi_data[4].astype(int)
    
    # Get min and max values for contour levels
    pmin = ne_inv.min().min()
    pmax = ne_inv.max().max()
    
    # Contour plot of electron density
    levels = np.linspace(pmin/1e23, pmax/1e23, 180)
    
    # # Plot contour fills
    # plt.figure()
    # contourf = plt.contourf(z,y,ne_inv/1e23, levels = levels, cmap='jet')
    
    # # Plot contour lines
    # contour = plt.contour(z,y,ne_inv/1e23, levels = levels, cmap='jet')
    
    # # Plot centroids. 
    # plt.plot(z,y[centroids],'k.')
    
    # # Plot formatting and labels
    # plt.title('DHI - ' + str(shot))
    # plt.xlabel('Axial position [cm]', fontsize=12)
    # plt.ylabel('Impact Parameter [cm]', fontsize=12)
    # # plt.ylim([-1, 1])
    
    # # Color bar settings
    # cbar = plt.colorbar(contourf)
    # cbar.set_label('Electron Density [10$^{23}$ m$^{-3}$]', fontsize=12)
    
    #%% Get radial profiles of number density at specific axial position
    
    # Set the axial position from which to get the radial slice of density data
    pos = np.array([8])
    
    plt.figure()
    
    for i in range(len(pos)):
        # Get index of axial position array (z) that corresponds to desired axial position
        pos_ind = np.argmin(np.abs(z - pos[i]))
        
        # Get index of the centroid at desired axial position [pixel]
        y_cent = centroids[pos_ind]
        
        # Extract number density data at this axial position only [m^-3]
        ne_radial = ne_inv[:,pos_ind]
        
        # Plot number density from centroid in positive y
        ne_radial_pos = ne_radial[y_cent:-1]/1e23 # radial density profile in positive y
        r_ne_pos = y[y_cent:-1]-y[y_cent]
        plt.plot(r_ne_pos, ne_radial_pos, 'k', label = 'z = ' + str(pos[i]) + ' cm, +y')
    
        # Plot number density from centroid to negative y (longer extent of data)
        ne_radial_neg = ne_radial[y_cent::-1]/1e23 # radial density profile in negative y
        plt.plot(-y[y_cent::-1]+y[y_cent], ne_radial_neg, 'r', label = 'z = ' + str(pos[i]) + ' cm, -y')
        
        n_e = ne_radial[y_cent::-1] # Density in m^-3
        r_ne = -y[y_cent::-1]+y[y_cent] # Corresponding radial positions in cm
            
        # Pads the shorter positive side with data from the negative side 
        ne_neg_pad = ne_radial_neg[len(ne_radial_pos):]
        ne_radial_pos_pad = np.concatenate((ne_radial_pos, ne_neg_pad))
            
        # Plot average number density
        ne_radial_avg = 0.5 * (ne_radial_pos_pad + ne_radial_neg)*1e23 # In m^-3
                
        plt.plot(r_ne, ne_radial_avg/1e23, 'b', label = 'z = ' + str(pos[i]) + ' cm, Mean')
    
    plt.title('Number density radial profiles: ' + str(shot))
    plt.ylabel('$n_e$ [10$^{23}$ m$^{-3}$]')
    plt.xlabel('Radial position [cm]')
    # plt.grid()
    # plt.legend()
    plt.xlim([0,1])
    plt.ylim([0, 4])
    # plt.show()
    # plt.close()
    
    #%% Importing the error data for shot 160524021
    
    # FOR DHI DATA STORED IN DHIDATA TREE BASELINE SHOT IS 160524020
    # These error bars seem too small compared to plots in publications
    # Import .mat file with errror values for Ross shot. First shotnum is the deformed shot, second shotnum is the baseline. The three arrays are for each axial cross-section
    # dhi_import = sio.loadmat('G:\\Shared drives\\Shumlak Lab\\Users\\Current\\Aqil Khairi\\Python\\ne_cross_sections_160524021_160524020.mat')
    
    # This one maybe better?
    # These seem to be the correct magnitudes: https://www.dropbox.com/scl/fo/7t95undjeej1ntfgmwr16/AAh4VcZSKXiAyhHq9B9uyio?rlkey=3403qtq3k8dumm5qeyqpvr31t&st=k8liyxrs&dl=0
    dhi_import = sio.loadmat('G:\\Shared drives\\Shumlak Lab\\Users\\Current\\Aqil Khairi\\Python\\dhi_error_data_160524021.mat')
    
    # This selects the array for z = 8 cm
    i = 1
    
    # Get the error values for left and right of the centroid
    
    # ne_error_l = np.abs(dhi_import["n_error_l"][0][i].flatten())
    # ne_error_r = np.abs(dhi_import["n_error_r"][0][i].flatten())
    
    ne_error_l = np.abs(dhi_import["n_error_out_interp_l"][0][i].flatten())
    ne_error_r = np.abs(dhi_import["n_error_out_interp_r"][0][i].flatten())
    
    # Get the number density values for left and right of the centroid
    ne_l = dhi_import["den_num_axial_l_plt"][0][i].flatten()
    ne_r = dhi_import["den_num_axial_r_plt"][0][i].flatten()
    
    # Get the corresponding radial values
    rad_l = 100*np.transpose(dhi_import["rad_l"][0][i]).flatten()
    rad_r = 100*np.transpose(dhi_import["rad_r"][0][i]).flatten()
    
    # Use density values from this file instead of MDSPlus
    r_ne = rad_l
    
    # Pads the shorter positive side with data from the negative side 
    ne_l_pad = ne_l[len(ne_r):]
    ne_r_pos_pad = np.concatenate((ne_r, ne_l_pad))
    
    ne_radial_avg = 0.5 * (ne_r_pos_pad + ne_l) # In m^-3
    
    fig = plt.figure()
    plt.title('Number density with error bars: ' + str(shot))

    plt.plot(r_ne[0:85], ne_radial_avg[0:85]/1e23, 'k')
    plt.errorbar(r_ne[::5], ne_radial_avg[::5]/1e23, ne_error_l[::5]/1e23, fmt = 'k', label = 'Error', ecolor = 'black', capsize = 5, linestyle = 'none')

    plt.xlim([0, 1])
    plt.ylim([0, 4])
    plt.xlabel('Radial position [cm]')
    plt.ylabel('$n_e$ [10$^{23}$ m$^{-3}$]')
    # plt.legend()
    # plt.close()
    
    # Rename error bar array for outside use
    ne_error_out = ne_error_l
    
    #%% # Get currents at m0 at P10
    
    # Connect to the zappa server
    c = mds.Connection('zappa.zap')
    c.get("getenv('zaphd_path')")
    
    # Open the tree for a given shot
    c.openTree('zaphd',shot)
    
    # Get m0 at P10
    m0_p10 = np.ravel(np.transpose(np.array(c.get(r'\m_0_p10'))))
    i_p10 = m0_p10*1e6*5*0.100838
    
    m0_p10_filt = savgol_filter(m0_p10, window_length=100, polyorder=2)
    
    # Get the time base for m0 at P10
    t_m0_p10 = np.array(c.get(r'dim_of(\m_0_p10)')*1e6)
    
    # Get index of closest value in time array to DHI trig time (50 us)
    B_ind = np.argmin(np.abs(t_m0_p10 - 50))
    
    # Get value of B field at DHI trig time
    B_dhi = m0_p10_filt[B_ind]
    
    # plt.figure()
    # plt.plot(t_m0_p10, m0_p10, label = 'Magnetic field')
    # plt.plot(t_m0_p10, m0_p10_filt, label = 'Smoothed magnetic field')
    # # plt.plot(t_m0_p10, i_p10, label = 'Current') # Current at P10
    # plt.plot(t_m0_p10[B_ind], B_dhi, 'rx')
    # plt.title('P10 Magnetic field: ' + str(shot))
    # plt.xlabel('Time [$\\mu$s]')
    # plt.ylabel('Magnetic field [T]')
    # plt.xlim([0,120])
    # plt.grid()
    # plt.legend()
    
    
    #%% Prescribe the density profile
    
    # Integrate the existing number density profile over the area to get linear density
    a_true = r_ne[-1]/100 # Extent of DHI measured radial points in meters 
    a_p = a_true # Prescribed radius of pinch in m
    
    # a_true = radius
    # a_p = radius
    
    r_w = 0.100838 # meters, radius of outer electrode wall
    
    # Prescribed full range of radial positions in meters, out to 100 mm
    r_ne_p = np.linspace(0, 100/1e3, 10000)
    
    # Radius of nose cone cross-section at axial position of chords in meters
    r_c = 7.027369631779123/1e3
    
    # Choose integration limits in meters
    # a_exp = 0.003
    a_exp = radius
    A, B = r_c, r_c + a_exp
    
    # Choose limits for expanded density profile in meters. Inner limit is radius of nose cone
    A1, B1 = r_c, r_c + a_true # Outer limit is same extent as original
    # A1, B1 = r_c, r_w # Outer limit is the wall radius (inner diameter of outer electrode)
    
    # Mask for expanded profile. This removes all radial positions less than the nose cone radius
    mask_core = (r_ne_p >= A1) & (r_ne_p <= B1)
    
    # Radial positions for expanded profile in meters
    r_pres = r_ne_p[mask_core]

    # Override chord configuration
    chordconfig = 'lateral'    

    if chordconfig == 'lateral':
        # Chord positions in mm for LATERAL chords
        chord_pos = np.linspace(-23.6/2, 23.6/2, 20)
    
    elif chordconfig == 'axial':
        # Chord positions in mm for AXIAL chords. From nosecone_profile.py
        chord_pos = np.array([21.44733208, 20.60827119, 19.66379088, 18.57321808, 17.32547784,
        15.9108877 , 14.23324798, 12.17906691,  9.57152072,  5.64817876,
         0.02674082,  0.02674082,  0.02674082,  0.02674082,  0.02674082,
         0.02674082,  0.02674082,  0.02674082,  0.02674082,  0.02674082])
    
# =============================================================================
#     Function to integrate number density to get linear density
# =============================================================================
    def getLinearDensity(r_density, density, a):
        """
        This function integrates a number density profile to calculate the linear density out to a specified radius.
        
        Parameters
        ----------
        r_density : Array of float64
            The radial positions for the expanded number density profile. Must be in units of meters.
        density : Array of float64
            The number density values.
        a : float
            The radial extent for the integration to calculate linear density. Must be in units of meters.

        ReturnsW
        -------
        Ne_linear : float
            The linear density of the profile in m$^{-1}$.

        """
        # Get index of closest value in radius array to the selected pinch radius (meters)
        a_ind = np.argmin(np.abs(r_density - a))
        
        # Radial positions up to chosen pinch radius
        r_ne_lim = r_density[0:a_ind+1]
        
        # Corresponding electron densities in m^-3
        n_e_lim = density[0:a_ind+1]
        
        # Integrand ( ne(r) * r )
        int_ne = n_e_lim * r_ne_lim
        
        # Numerically integrate the radial density profile with r with integrand defined above
        integral = 2 * np.pi * cumulative_trapezoid(int_ne,r_ne_lim,initial=0)
        
        # Last value of integral is the total linear density in m^-1
        Ne_linear = integral[-1]
        
        # Calculate the sharp pinch linear density
        # Ne_linear = density[0] * ( (a_c + a_p)**2 - a_c**2)

        return Ne_linear
    
    # Get the linear density of DHI number density data, convert radial points to meters
    Ne_linear = getLinearDensity(r_ne/100, ne_radial_avg, a_true)
    print(f"{shot} linear density is {Ne_linear:.0f}: ")
        
# =============================================================================
#     Define a Lorentzian function to generate prescribed density function
# =============================================================================
    def lorentzian(x, A, x0, gamma):
        return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2)
    
# =============================================================================
#     Guess the prescribe Lorentzian elements
# =============================================================================
    A_Lfit = max(ne_radial_avg)
    x0_Lfit = 0
    gamma_Lfit = 0.00634 # FOr a_p = 0.003 m
    # gamma_Lfit = 0.01125 # For a_p = a_true
    
    # Get coefficients for lorenztian fit to density data
    popt = A_Lfit, x0_Lfit, gamma_Lfit
    
    # Evaluate Lorentzian function over prescribed radial positions
    ne_dhi_fit = lorentzian(r_pres, *popt)
    
    # Evaluate Lorentzian function for axial chord positions
    # ne_dhi_chords = lorentzian(chord_pos, *popt)
     
    # Calculate linear density of prescribed number density, convert radial points to meters
    # Ne_linear_p = getLinearDensity(r_ne_p/1e3, ne_dhi_fit, a=a_p)
    # print(f"Prescribed linear density is {Ne_linear_p}: ")
    
    # Ne_linear_p2 = getLinearDensity(r_pres, ne_dhi_fit, a = B1)
    # Ne_sharp_p2 = ne_dhi_fit[0] * ((r_c + a_p)**2 - (r_c)**2)
    
# =============================================================================
#   Fit a Lorentzian to the original density data
# =============================================================================

    # Initial guesses
    A_fit = max(ne_radial_avg)
    x0_fit = 0
    gamma_fit = 1
    p0 = [A_fit, x0_fit, gamma_fit]
    
    # Apply curve fitting
    ne_dhi_params, ne_dhi_var = curve_fit(lorentzian, r_ne/100, ne_radial_avg, p0)

    # Evaluate Lorentzian function over prescribed radial positions
    ne_dhi_curve_fit = lorentzian(r_ne/100, *ne_dhi_params)
    
    # Calculate R-squared
    ss_res = np.sum((ne_radial_avg - ne_dhi_curve_fit) ** 2)
    ss_tot = np.sum((ne_radial_avg - np.mean(ne_radial_avg)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate linear density of Lorentzian fit
    Ne_linear_f = getLinearDensity(r_ne/100, ne_dhi_curve_fit, a_true)

# =============================================================================
# Function that is iterated for expanded density profile. Tests different Lorentzian widths for a fixed guess of peak
# =============================================================================
    def IterativeLinearDensity(r_density, density, gamma, a):
        """
        This function generates a Lorentzian profile for the expanded number density profile, and outputs the linear density of this profile.
        Requires the function called 'lorentzian' to be defined.
        
        Parameters
        ----------
        r_density : Array of float64
            The radial positions for the expanded number density profile.
        density : Array of float64
            The original number density profile.
        gamma : float
            The width of the Lorentzian.
        a : float
            The radial extent for the integration to calculate linear density.

        Returns
        -------
        Ne_linear_p : float
            The linear density of the expanded profile in m$^{-1}$.
        ne_dhi_fit : Array of float64
            The expanded number density values.
        ne_dhi_chords : Array of float64
            The Lorentzian profile evaluated at chord positions. Unused for now.
        a_p : float
            DESCRIPTION.

        """
        # Prescribe Lorentzian elements
        
        A_Lfit = max(density) # Amplitude
        x0_Lfit = 0 # Center
        gamma_Lfit = gamma # Width parameter
        
        # Combine initial guesses for Lorentzian parameters
        popt = A_Lfit, x0_Lfit, gamma_Lfit
        
        # Evaluate Lorentzian function over input radial positions
        ne_dhi_fit = lorentzian(r_density, *popt)
        
        # Evaluate Lorentzian function for axial chord positions
        ne_dhi_chords = lorentzian(chord_pos, *popt)
        
        # Upper limit for integration to get linear density value (meters)
        a_iter = (r_c) + a
         
        # Calculate linear density of prescribed number density. 
        Ne_linear_p = getLinearDensity(r_density, ne_dhi_fit, a_iter)
        
        return Ne_linear_p, ne_dhi_fit, ne_dhi_chords, a_p
    
    # Set parameters for iterative function
    target = Ne_linear
    tolerance = 0.1e18
    gamma_guess = 0.01
    step = 0.00001
    
    # Iterate over different widths 
    for i in range(1000):
        output = IterativeLinearDensity(r_pres, ne_radial_avg, gamma_guess, B1)
        Ne_linear_p = output[0]
        error = target - Ne_linear_p

        if abs(error) < tolerance:
            print(f"Converged in {i} iterations: gamma = {gamma_guess:.6f}, Linear density = {Ne_linear_p:.0f}")
            break
    
        # Adjust input based on sign of error
        if Ne_linear_p < target:
            gamma_guess += step
        else:
            gamma_guess -= step
    
        # Optionally reduce step size for finer approach
        # step *= 1
    else:
        print("Did not converge")
    
# =============================================================================
# PLOTTING THE ORIGINAL AND EXPANDED PROFILES
# =============================================================================
    fig = plt.figure(figsize = (6,3))
    
    # Plot the radius of the nose cone
    # plt.axvline(r_c, ls = 'dashed', color = 'gray')
    
    # Prescribed density profile output from iteration
    n_pres = output[1]

    # Mask to show limits of integration
    # mask = (r_pres >= A) & (r_pres <= B)
    
    # r_sel = r_pres[mask]
    # n_sel = n_pres[mask]
    
    # "Sharp pinch" area as linear density. This assumes a constant density out to the radius
    # Ne_sharp = ne_radial_avg[0] * a_p**2
    # Ne_sharp_p = n_pres[0] * ((r_c + a_p)**2 - (r_c)**2)
    
    # Plot sharp pinch lines
    # plt.hlines(ne_radial_avg[0], 0, a_p*1e3, linestyle = '--', color = 'black')
    # plt.vlines(a_p*1e3, 0, ne_radial_avg[0], linestyle = '--', color = 'black')
    
    # plt.hlines(n_pres[0], r_c*1e3, (r_c + a_p)*1e3, linestyle = '--', color = 'green')
    # plt.vlines((r_c + a_p)*1e3, 0, n_pres[0], linestyle = '--', color = 'green')
    
# =============================================================================
#   Plot the density data from DHI
# =============================================================================
    plt.plot(r_ne*10, ne_radial_avg/1e23, 'k', label = f'Original $n_e$: {shot}')
    plt.plot(r_ne*10, ne_dhi_curve_fit/1e23, 'b', label = 'Lorentzian fit')

    # plt.fill_between(r_ne*10, ne_radial_avg, 0, color='black', alpha=0.3)
    # plt.text(5, 2.1, f"Measured $N_e = $ {Ne_linear:.2e} m$^{-1}$", fontsize=8,bbox=dict(facecolor='white', edgecolor='black')) # integral Ne
    # plt.axvline(a_true*1e3, ls = 'dashed', color = 'black', label = f' {shot} pinch radius')
    # plt.hlines(max(ne_radial_avg), 0, a_p*1e3, linestyle = '--', color = 'black')
    # plt.vlines(a_p*1e3, 0, max(ne_radial_avg), linestyle = '--', color = 'black')
    
    # plt.text(8, 1.1e23, f" Sharp pinch $N_e = $ {Ne_sharp:.2e} m$^{-1}$", fontsize=8) # Sharp Ne

# =============================================================================
#     Plot manual method of prescribed density
# =============================================================================
    # plt.plot(r_pres*1e3, ne_dhi_fit/1e23, 'green', label = 'Prescribed $n_e$')
    # plt.axvline(r_c*1e3, ls = 'dotted', color = 'gray', label = 'Nose cone radius')

    # plt.fill_between(r_pres*1e3, ne_dhi_fit, 0, color='green', alpha=0.3)
    # plt.axvline(a_p*1e3, ls = 'dashed', color = 'green', label = 'Prescribed pinch radius')
    # plt.plot(chord_pos, ne_dhi_chords, 'g.', label = f'{chordconfig} chord positions')
    # plt.text(5, 1.1, f"Prescribed linear density is {Ne_linear_p2:.2e} m$^{-1}$", fontsize=8) # integral Ne
    # plt.text(11, 0.8e23, f"Prescribed sharp $N_e = $ {Ne_sharp_p2:.2e} m$^{-1}$", fontsize=8) # Sharp Ne
    # plt.hlines(ne_dhi_fit[0], r_c*1e3, (r_c + a_p)*1e3, linestyle = '--', color = 'green')
    # plt.vlines((r_c + a_p)*1e3, 0, ne_dhi_fit[0], linestyle = '--', color = 'green')

    
# =============================================================================
#     Plot from iterative method of prescribed density
# =============================================================================
    plt.plot(r_pres*1e3, n_pres/1e23, 'green', label = 'Expanded $n_e$')

    plt.axvline(r_c*1e3, ls = 'dotted', color = 'gray', label = 'Nose cone radius')
    # # plt.fill_between(r_sel*1e3, n_sel, 0, color='green', alpha=0.3)
    # # plt.fill_between(r_pres*1e3, n_pres, 0, color='green', alpha=0.3)
    # plt.plot(chord_pos, IterativeLinearDensity(r_pres, gamma_guess, a_p)[2], 'g.', label = 'Axial chord positions')
    # plt.text(3, 1.1, f"Prescribed $N_e = $ {Ne_linear_p:.2e} m$^{-1}$", fontsize=8, bbox=dict(facecolor='white', edgecolor='black'))
    # # plt.axvline(IterativeLinearDensity(r_pres, gamma_guess, a_p)[3]*1e3, ls = 'dashed', color = 'green', label = 'Prescribed pinch radius')
        
    # plt.title('Prescribed electron density')
    plt.xlabel('Radius [mm]')
    plt.ylabel('$n_e$ [10$^{23}$ m$^{-3}$]')
    plt.xlim([0,17])
    plt.ylim([0,2.1])
    # plt.gca().set_yscale('log')  # Set y-axis to log scale
    # plt.grid()
    plt.legend(fontsize = 8)
    # plt.close()
    
    # Toggle the interative output as the assumed prescribed profile for overall script output
    ne_dhi_fit = output[1]

    #%% Calculate the magnetic field and temperature profiles
    
    def get_BandT(r_density, density, a, B_dhi):
    
        # Constants
        mu0 = 4*np.pi*10**-7 # Magnetic permeability of free space
        k_b = 1.38e-23 # J/K
        k_b_eV = 8.617e-5 #eV/K
        eVtoJ = k_b/k_b_eV
        e = 1.602e-19 # Electron charge, Coulombs
        m_e = 9.11e-31 # electron mass, kg
        m_i = 1.67e-27 # proton mass, kg
        r_w = 0.100838 # meters, radius of outer electrode wall where bdot probes are located
        
        # From Ross PhD Chapter 4
        
        # Total current from bdot probe data
        # B_dhi = 0.25
        I_tot = 2*np.pi*B_dhi*r_w / mu0
        
        # Get index of closest value in radius array to the selected pinch radius (meters)
        a_ind = np.argmin(np.abs(r_density - a))
        
        # Radial positions up to pinch radius (meters)
        r_ne_lim = r_density[0:a_ind+1]
        
        # Corresponding electron densities in m^-3, using average of two sides of centroid
        n_e_lim = density[0:a_ind+1]
        
        # Radial positions beyond pinch radius, convert to meters
        r2 = r_density[a_ind:]
        
        # Corresponding electron densities in m^-3 beyond pinch radius
        n_e_2 = density[a_ind:]
        
        # Integrand ( ne(r) * r )
        int_ne = n_e_lim * r_ne_lim
        
        # Numerically integrate the radial density profile with r with integrand defined above
        integral = cumulative_trapezoid(int_ne,r_ne_lim,initial=0)
        
        # Calculate the drift velocity
        # Use the definite of the integral at r = a
        v_d = - I_tot / (2 * np.pi * e * integral[a_ind])
        
        # Calculate the azimuthal magnetic field inside and outside of the pinch radius
        
        # r < a
        Btheta = ((mu0 * e * v_d) / r_ne_lim) * integral
        Btheta[0] = 0
        
        # r > a
        Btheta2 = mu0 * I_tot / (2 * np.pi * r2)
        
        # Calculate the radial temperature profile
    
        # Integrand ( ne(r) * B_theta(r) * )
        temp_int_ne = n_e_lim * -Btheta
        # temp_int_ne = n_e_2 * -Btheta2
        
        # For integration from a to r as in Ross thesis:
        r_ne_lim_rev = r_ne_lim[-1::-1]
        # r2_rev = r2[-1::-1]
        
        # Numerically integrate the radial density profile with Btheta(r)
        temp_integral = cumulative_trapezoid(temp_int_ne[-1::-1],r_ne_lim_rev,initial=0)
        # temp_integral = cumulative_trapezoid(temp_int_ne, r2, initial=0)
        
        T_rad = (e * v_d / (2 * n_e_lim[-1::-1] * eVtoJ)) * temp_integral
        # T_rad = (e * v_d / (2 * n_e_2 * eVtoJ)) * temp_integral
        
        T_rad2 = np.zeros([len(r2)])
        
        return (Btheta, Btheta2, r_ne_lim, r2, T_rad, T_rad2)
    
    #%% Call function to get B and T and error for lateral chords
    
    # Calculate the B and T for density profile
    B_fun = get_BandT(r_density = r_ne/100, density = ne_radial_avg, a = radius , B_dhi = B_dhi)
    
    Btheta = B_fun[0]
    Btheta2 = B_fun[1]
    r_ne_lim = B_fun[2]
    r2 = B_fun[3]
    T_rad = B_fun[4]
    T_rad2 = B_fun[5]
    
    # Propagate the error bars for electron temperature
    # See Ross PhD thesis page 82. This sort of replicates Figure 4.2

    # Apply error bars to density profile to get hi and low profiles for propagation
    ne_radial_hi = ne_radial_avg + ne_error_out
    ne_radial_low = ne_radial_avg - ne_error_out
    
    # Calculate B and T for hi density profile
    B_fun_error_hi = get_BandT(r_density = r_ne/100, density = ne_radial_hi, a = radius , B_dhi = B_dhi)
    Btheta_err_hi = B_fun_error_hi[0]
    Btheta2_err_hi = B_fun_error_hi[1]
    T_rad_err_hi = B_fun_error_hi[4]
    T_rad2_err_hi = B_fun_error_hi[5]
    
    # Calculate B and T for low density profile
    B_fun_error_low = get_BandT(r_density = r_ne/100, density = ne_radial_low, a = radius , B_dhi = B_dhi)
    T_rad_err_low = B_fun_error_low[4]
    T_rad2_err_low = B_fun_error_low[5]

    # Calculate B and T for error bar values directly 
    B_fun_error = get_BandT(r_density = r_ne/100, density = ne_error_out, a = radius , B_dhi = B_dhi)
    T_rad_err = B_fun_error[4]
    T_rad2_err = B_fun_error[5]
    
    #%% Make arrays of density, temperature, and Bfield for use in other scripts if lateral, plot
    
    # Radial positions in meters
    r_final = np.append(r_ne_lim, r2[1:])
    
    # Electron density radial profile
    ne_final = ne_radial_avg
    
    # Electron density error
    ne_error_final = ne_error_out
    
    # Electron temperature profile calculated with hi density profile
    Te_hi_final = np.append(T_rad_err_hi[-1::-1], T_rad2_err_hi[1:])
    
    # Electron temperature profile calculated with low density profile
    Te_low_final = np.append(T_rad_err_low[-1::-1], T_rad2_err_low[1:])
    
    # Define array of errors for export
    
    # Te errors by directly propagating error bar values (big bars)
    Te_error_final = np.append(T_rad_err[-1::-1], T_rad2_err[1:])
    
    # Te errors by equilibrium analysis of scaled density profiles and taking difference
    # Te_error_final = Te_error_low_final - Te_error_hi_final

    # Azimuthal magnetic field radial profile
    Btheta_final = np.append(-Btheta, Btheta2[1:])
    Btheta_err_final_hi = np.append(-Btheta_err_hi, Btheta2_err_hi[1:])
    
    # Electron temperature radial profile
    Te_final = np.append(T_rad[-1::-1], T_rad2[1:])
    
    # Te errors by difference to propagated hi and low Te profiles to actual Te 
    
    # Low Te errors, replace negative values with nan
    Te_error_low_final = Te_final - Te_hi_final
    Te_error_low_final[Te_error_low_final < 0] = np.nan
    
    # Hi Te errors, replace negative values with nan
    Te_error_hi_final = Te_low_final - Te_final
    Te_error_hi_final[Te_error_hi_final < 0] = np.nan
    
    # Control error bar density
    err_dens = 5
    
    # Create (2,n) array for error bar plotting
    Te_error_plot_bars = np.vstack([Te_error_low_final[::err_dens] , Te_error_hi_final[::err_dens]])

# =============================================================================
#     Plotting
# =============================================================================
    # fig5a, ax1 = plt.subplots(figsize = (6,2.5)) # Use for individual plotting
    fig5, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    
    # Plot the density profile(s) and error bars
    
    ax1.plot(r_final*1e3, ne_final/1e23, 'black', label = '$n_e$')
    # ax1.fill_between(r_final*1e3, (ne_radial_low)/1e23, (ne_radial_hi)/1e23, color='black', alpha=0.3)
    # ax1.plot(r_final*1e3, (ne_radial_hi)/1e23, 'k--', label = '1.25$n_e$')
    # ax1.plot(r_final*1e3, (ne_radial_low)/1e23, 'k:', label = '0.75$n_e$')
    # ax1.axvline(0.62) # Radial location of chord closest to axis for reference
    ax1.text(9.5, 3.5, '(a)', fontsize = 10)
    ax1.text(8, 3, 'Pulse ' + str(shot), fontsize = 8)
    ax1.errorbar(r_final[::err_dens]*1e3, ne_final[::err_dens]/1e23, ne_error_final[::err_dens]/1e23, fmt = 'k', ecolor = 'black', capsize = 2, linestyle = 'none')
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 4])
    ax1.set_ylabel('$n_e$ [10$^{23}$ m$^{-3}$]')
    plt.tight_layout()
    
    # # after plotting the data, format the labels
    # current_values = plt.gca().get_yticks()
    # plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values]) 
    
    # (UNUSED) Plot the magnetic field profiles
    ax2.plot(r_final*1e3, Btheta_final, 'blue', label = 'Azimuthal field')
    # ax2.plot(r_final*1e3, Btheta_err_final_hi, 'b--', label = 'Azimuthal field')
    ax2.set_ylabel('$B_{\\theta}$ [T]')
    ax2.set_ylim([0, 10])
    ax2.grid()
    
    # fig5a, ax3 = plt.subplots(figsize = (6,2.5)) # Use for individual plotting

    # Plot the temperature profile(s) and error bars
    
    ax3.plot(r_final*1e3, Te_final, 'red', label = '')
    # ax3.fill_between(r_final*1e3, (Te_error_low_final), (Te_error_hi_final), color='red', alpha=0.3)
    # ax3.plot(r_final*1e3, Te_hi_final, 'r--', label = '')
    # ax3.plot(r_final*1e3, Te_low_final, 'r:', label = '')
    ax3.text(9.5, 1350, '(b)', fontsize = 10)
    ax3.errorbar(r_final[::err_dens]*1e3, Te_final[::err_dens], Te_error_plot_bars, fmt = 'r', ecolor = 'red', capsize = 2, linestyle = 'none')
    # ax3.errorbar(r_final*1e3, Te_final, Te_error_final, fmt = 'r', ecolor = 'red', capsize = 2, linestyle = 'none')

    ax3.set_xlim([0, 10])
    ax3.set_ylim([0, 1500])
    ax3.set_ylabel('$T_e$ [eV]')
    ax3.set_xlabel('Radius [mm]')
    plt.tight_layout()

    
    # plt.suptitle(f'Radial profiles: {shot}, a = {radius*1e3} mm ')
    # plt.tight_layout()
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # plt.legend()
    # plt.show()
    # plt.close()
    
    #%% 
# =============================================================================
#   Attempt to propagate hi and low density profiles for errors in expanded profiles
# =============================================================================

    # Get the linear density of DHI number density data, convert radial points to meters
    
    # Somehow nan value in ne_radial_hi/low is messing everything up? It literally worked thef irst time
    ne_radial_hi[-1] = ne_radial_hi[-2]
    ne_radial_low[-1] = ne_radial_low[-2]

    Ne_linear_hi = getLinearDensity(r_ne/100, ne_radial_hi, a_true)
    Ne_linear_low = getLinearDensity(r_ne/100, ne_radial_low, a_true)
    
    # Scaling of profiles by linear density?
    scale_hi = Ne_linear_hi/Ne_linear
    scale_low = Ne_linear_low/Ne_linear
    
    ne_radial_mod_hi = scale_hi * ne_dhi_fit
    ne_radial_mod_low = scale_low * ne_dhi_fit
    
    # Iterate based on hi and low linear densities
    
    ne_radials = [ne_radial_hi, ne_radial_low]
    ne_targets = [Ne_linear_hi, Ne_linear_low]
    ne_dhi_fits = [[], []]
 
    for jj in [0,1]:
        
        # Set parameters for iterative function
        target = ne_targets[jj]
        tolerance = 0.1e18
        gamma_guess = 0.01
        step = 0.00001
        
        # Iterate over different widths 
        for i in range(1000):
            output2 = IterativeLinearDensity(r_pres, ne_radials[jj], gamma_guess, B1)
            Ne_linear_out = output2[0]
            error = target - Ne_linear_out
    
            if abs(error) < tolerance:
                print(f"Converged in {i} iterations: gamma = {gamma_guess:.6f}, Linear density = {Ne_linear_out:.0f}")
                break
        
            # Adjust input based on sign of error
            if Ne_linear_out < target:
                gamma_guess += step
            else:
                gamma_guess -= step
        
            # Optionally reduce step size for finer approach
            # step *= 1
        else:
            print("Did not converge")
            
        ne_dhi_fits[jj] = output2[1]

    # Use the iterated hi and low profiles
    ne_radial_mod_hi = ne_dhi_fits[0]
    ne_radial_mod_low = ne_dhi_fits[1]

    #%% Call function to get B and T and error for modified profiles
    
    B_fun_ax = get_BandT(r_density = r_pres, density = ne_dhi_fit, a = r_c + a_exp , B_dhi = B_dhi)
    
    Btheta_ax = B_fun_ax[0]
    Btheta2_ax = B_fun_ax[1]
    r_ne_lim_ax = B_fun_ax[2]
    r2_ax = B_fun_ax[3]
    T_rad_ax = B_fun_ax[4]
    T_rad2_ax = B_fun_ax[5]
    
    # Propagate the error bars for electron temperature in prescribed profile
    # See Ross PhD thesis page 82. This sort of replicates Figure 4.2
    
    # Calculate B and T for hi modified density profile
    B_fun_error_hi = get_BandT(r_density = r_pres, density = ne_radial_mod_hi, a = r_c + a_exp , B_dhi = B_dhi)
    Btheta_err_hi = B_fun_error_hi[0]
    Btheta2_err_hi = B_fun_error_hi[1]
    T_rad_err_hi = B_fun_error_hi[4]
    T_rad2_err_hi = B_fun_error_hi[5]
    
    # Calculate B and T for low density profile
    B_fun_error_low = get_BandT(r_density = r_pres, density = ne_radial_mod_low, a = r_c + a_exp , B_dhi = B_dhi)
    T_rad_err_low = B_fun_error_low[4]
    T_rad2_err_low = B_fun_error_low[5]

    # Calculate B and T for error bar values directly 
    # B_fun_error = get_BandT(r_density = r_pres, density = ne_error_out, a = r_c + a_true , B_dhi = B_dhi)
    # T_rad_err = B_fun_error[4]
    # T_rad2_err = B_fun_error[5]

    
    #%% Make arrays of density, temperature, and Bfield for use in other scripts for modified profiles

    # Radial positions in meters
    r_final_ax = np.append(r_ne_lim_ax, r2_ax[1:])
    
    # Modified electron density radial profile
    ne_final_ax = ne_dhi_fit
    
    # Electron density error (use difference of scaled profiles)
    ne_error_final_ax = (ne_radial_mod_hi - ne_radial_mod_low)/2
    
    # Electron temperature profile calculated with hi density profile
    Te_hi_final = np.append(T_rad_err_hi[-1::-1], T_rad2_err_hi[1:])
    
    # Electron temperature profile calculated with low density profile
    Te_low_final = np.append(T_rad_err_low[-1::-1], T_rad2_err_low[1:])
    
    # Te errors by directly propagating error bar values (Unused)
    Te_error_final_ax = np.append(T_rad_err_low[-1::-1], T_rad2_err_low[1:])
    
    # Azimuthal magnetic field radial profile
    Btheta_final_ax = np.append(-Btheta_ax, Btheta2_ax[1:])
    Btheta_err_final_hi = np.append(-Btheta_err_hi, Btheta2_err_hi[1:])
    
    # Electron temperature radial profile
    Te_final_ax = np.append(T_rad_ax[-1::-1], T_rad2_ax[1:])
    
    # Te errors by difference to propagated hi and low Te profiles to actual Te 
    
    # Low Te errors, replace negative values with nan
    Te_error_low_final_ax = Te_final_ax - Te_hi_final
    Te_error_low_final_ax[Te_error_low_final_ax < 0] = np.nan
    
    # Hi Te errors, replace negative values with nan
    Te_error_hi_final_ax = Te_low_final - Te_final_ax
    Te_error_hi_final_ax[Te_error_hi_final_ax < 0] = np.nan
    
    # Control error bar density
    err_dens = 50
    
    # Create (2,n) array for error bar plotting
    Te_error_plot_bars_ax = np.vstack([Te_error_low_final_ax[::err_dens] , Te_error_hi_final_ax[::err_dens]])
    
# =============================================================================
#     PLOTTING
# =============================================================================
    fig6 = plt.figure()
    
    # No magnetic field
    # fig6, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
    
    # With magnetic field
    fig6, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)


    # plt.subplot(3,1,1)
    ax1.plot(r_final_ax*1e3, ne_final_ax/1e23, 'black', label = '$n_e$')
    # ax1.plot(r_final_ax*1e3, (ne_radial_mod_hi)/1e23, 'k--')
    # ax1.plot(r_final_ax*1e3, (ne_radial_mod_low)/1e23, 'k:')
    ax1.axvline(r_c*1e3, ls = 'dotted', color = 'gray')
    ax1.errorbar(r_final_ax[::err_dens]*1e3, ne_final_ax[::err_dens]/1e23, ne_error_final_ax[::err_dens]/1e23, fmt = 'k', ecolor = 'black', capsize = 2, linestyle = 'none')
    ax1.set_ylabel('$n_e$ [10$^{23}$ m$^{-3}$]')
    ax1.text(0.95, 0.9, '(a)', transform=ax1.transAxes, fontsize = 10)
    # ax1.text(8, 3, 'Pulse ' + str(shot), fontsize = 8)
    ax1.set_ylim([0,1])
    # ax1.grid()
    # ax1.legend(fontsize = 8)
    
    plt.subplot(3,1,2)
    ax2.plot(r_final_ax*1e3, Btheta_final_ax, 'blue', label = 'Azimuthal field')
    ax2.plot(r_final_ax*1e3, Btheta_err_final_hi, 'b--', label = 'Azimuthal field')
    ax2.axvline(r_c*1e3, ls = 'dotted', color = 'gray')
    ax2.set_ylabel('$B_{\\theta}$ [T]')
    ax2.grid()
    
    # plt.subplot(3,1,3)
    ax3.plot(r_final_ax*1e3, Te_final_ax, 'red', label = 'Electron temperature')
    # ax3.plot(r_final_ax*1e3, Te_hi_final, 'r--', label = '')
    # ax3.plot(r_final_ax*1e3, Te_low_final, 'r:', label = '')
    ax3.axvline(r_c*1e3, ls = 'dotted', color = 'gray')
    ax3.errorbar(r_final_ax[::err_dens]*1e3, Te_final_ax[::err_dens], Te_error_plot_bars_ax, fmt = 'r', ecolor = 'red', capsize = 2, linestyle = 'none')
    ax3.text(0.95, 0.9, '(b)', transform=ax3.transAxes, fontsize = 10)
    ax3.set_ylabel('$T_e$ [eV]')
    # ax3.grid()
    ax3.set_ylim([0, 500])
    plt.xlabel('Radius [mm]')
    plt.xlim([7, 10])
    
    # plt.suptitle(f'Expanded profiles, a = {a_p*1e3} mm')
    plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.close()
    #%% Outputs

    'Returns the radial profiles of electron density, azimuthal magnetic field, and electron temperature from DHI analysis'
    if chordconfig == 'lateral':
        # return r_final, ne_final, Btheta_final, Te_final, ne_error_final, Te_error_final, Te_error_low_final, Te_error_hi_final
        
        # Uncomment to use modified profile for SXB calcs. Keep lateral chords
        return r_final_ax, ne_final_ax, Btheta_final_ax, Te_final_ax, ne_error_final_ax, Te_error_final_ax, Te_error_low_final_ax, Te_error_hi_final_ax
    
    elif chordconfig == 'axial':
        return r_final_ax, ne_final_ax, Btheta_final_ax, Te_final_ax, ne_error_final_ax, Te_error_final_ax, Te_error_low_final_ax, Te_error_hi_final_ax

plt.close('all')


if __name__ == "__main__":
    dhi_profiles('lateral', radius = 0.003)
    