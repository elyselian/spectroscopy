# -*- coding: utf-8 -*-
"""
Created on Thu May  1 22:23:14 2025

@author: aqilk
"""

# This script calls the SXB_Analysis script which calculates the erosion flux for one shot via the ionizations per photon method. 
# This script allows for plotting and further processing over many shots.

# TODO: Using the expanded dhi profiles results in poor fitting since only two chord radial positions are within the range of profile values

#%% Import required packages and functions

# Change to the correct directory if needed:
# os.chdir(r"G:\Shared drives\Shumlak Lab\Users\Current\Aqil Khairi\Python")

from matplotlib import pyplot as plt
from scipy import integrate
from scipy.integrate import quad, cumulative_trapezoid
import numpy as np

# Imported classes

# Import the read_sxb function which returns S/XB values for a temperature and density input
from read_sxb_class import read_sxb

# Create an instance of the read_sxb_class for ion C-III
sxb_vals = read_sxb("C-III")

# Function for calculating erosion flux via SXB method
from SXB_Analysis import sxb_shot

# Function for importing pinch currents
from get_mds import get_currents

# Function for getting plasma parameters based on pinch current scaling
# from zpinch_scaling import pinch_scale

#%% Shot selection

# NOTE: Axially aligned chord only from 250617

########## PASTE SHOT_RANGE AND SHOT_REM LINES HERE ##########

# shot_range = [250618005] # Axial chords
shot_range = [250424006] # Lateral chords
# shot_range = [250620002]

# Case I
# sr1 = np.array([]) # 10/200 shots
# srem1 = np.array([])

# sr2 = np.arange(250617006, 250617046, 1) # 40/200 shots
# srem2 = np.array([250617032])

# sr3 = np.arange(250618006, 250618043, 1) # 30/200 shots
# srem3 = np.arange(250618015, 250618022)

# sr4 = np.arange(250620007, 250620027, 1) # 20/200 shots
# srem4 = np.array([])

# sr5 = np.arange(250623006, 250623030, 1) # 20/200 shots
# srem5 = np.array([250623009, 250623010, 250623019, 250623021, 250623023, 250623024, 250623025, 250623026])

# sr6 = np.arange(250625006, 250625087, 1) # 80/200 shots
# srem6 = np.array([250625065, 250625070])

# shot_rem = np.concatenate((srem1, srem2, srem3, srem4, srem5, srem6), axis=0).flatten()

# shot_range = np.concatenate((sr1, sr2, sr3, sr4, sr5, sr6), axis=0).flatten()

# # Array of shots to analyze
# shots =  np.setdiff1d(shot_range, shot_rem)

# Case II
# shot_range = np.arange(250714007, 250714057, 1) # 50/50 shots
# srem1 = np.array([250714008, 250714009, 250714014, 250714024, 250714026, 250714027, 250714050, 250714056])
# srem2 = np.arange(250714016, 250714022) 
# shot_rem = np.concatenate((srem1, srem2), axis=0).flatten()

# Case III
# shot_range = np.arange(250728007, 250728053, 1) 
# shot_rem = np.array([250728021, 250728022, 250728024, 250728032, 250728033, 250728034, 250728039, 250728041, 250728042, 250728048, 250728053 ])
# shots =  np.setdiff1d(shot_range, shot_rem)


# chordconfig = 'axial' 
chordconfig = 'lateral'

# Select radius of pinch in meters for calcs for prescribed profiles.
radius = 0.003

shot_rem = np.array([])

# CHECK DICT OUTPUTS AND RADIUS IN READ_DHI_MDS.PY
# CHECK chord_pos in SXB_Analysis

# Save variables to .npz file? True or False
save = False
savename = 'testt.npz'
##############################################################

# Axial chords
# shot_range = [250618005] # 9, 7 kV shots are similar in currents to 9,8 kV

# Lateral chords
# shot_range = [250424017] # 250424 has 9,8 kV shots comparable to Ross DHI data

# Array of shots to analyze
shots =  np.setdiff1d(shot_range, shot_rem)
#%%
# # sxb_data is an array containing the outputs of the sxb_shot function in SXB_Analysis 
# sxb_data = [0]*len(shots)

# if len(shots) > 1:
#     for i in np.arange(len(shots)):
#         sxb_data[i] = sxb_shot(shots[i])

# else:
#     # sxb_data is a tuple with each element an array of floats for each output of sxb_shot in SXB_Analaysis
#     # sxb_data is a list of tuples. Each tuple has arrays for each output of sxb_shot
#     sxb_data = [()]
#     sxb_data[0] = sxb_shot(shots[0])

    
#Get impact parameter defined by fiber bundle

# The 20 chords of the wide fiber image across 23.6 mm across the plasma diameter
impa = np.arange(-23.6/2,23.6/2,1.24)

#%% Array pre-allocation

# Pre-allocate all metadata arrays
wave_cal = np.zeros([1024, len(shots)])
gate_delay = np.zeros(len(shots))
gate_width = np.zeros(len(shots))
cwl = np.zeros(len(shots))

# Pre_allocate all MDS data arrays

t_ip = np.zeros([32768, len(shots)])
i_p = np.zeros_like(t_ip)
t_ip10 = np.zeros([20481, len(shots)])
i_p10 = np.zeros_like(t_ip10)
i_c = np.zeros_like(t_ip)

# Pre-allocate all calculated data arrays
flux_constant_sxb = np.zeros([20, len(shots)])
flux_variable_sxb = np.zeros_like(flux_constant_sxb)
intensity = np.zeros_like(flux_constant_sxb)
sxb_chords = np.zeros_like(flux_constant_sxb)
flux_err = np.zeros_like(flux_constant_sxb)
flux_err_low = np.zeros_like(flux_constant_sxb)
flux_err_hi = np.zeros_like(flux_constant_sxb)
ne_err = np.zeros_like(flux_constant_sxb)

# For loop to extract data for all selected shots

for i in np.arange(len(shots)):
    
    # Get all data from analysis script, including dictionaries
    sxb_data = sxb_shot(shots[i], radius = radius, plot = True, chordconfig = chordconfig)

    # # Get all MDS data arrays
    # mdsdict = sxb_data[2]
    
    # # Get currents, time in us, current in kA
    # t_ip[:,i] = mdsdict['t_ip']
    # i_p[:,i] = mdsdict['i_p']
    # t_ip10[:,i] = mdsdict['t_ip10']
    # i_p10[:,i] = mdsdict['i_p10']
    # i_c[:,i] = mdsdict['i_c']
    
    # Get all calculated data
    sxbdict = sxb_data[0]
    flux_constant_sxb[:,i] = sxbdict['flux_constant_sxb']
    flux_variable_sxb[:,i] = sxbdict['flux_variable_sxb']
    intensity[:,i] = sxbdict['intensity']
    sxb_chords[:,i] = sxbdict['sxb_chords']
    flux_err[:,i] = sxbdict['flux_err']
    flux_err_low[:,i] = sxbdict['flux_err_low']
    flux_err_hi[:,i] = sxbdict['flux_err_hi']

    ne_err[:,i] = sxbdict['ne_err']

    # Get all metadata
    metadict = sxb_data[1]
    
    # Get all wavelength calibration data for all shots
    wave_cal[:,i] = metadict['wave_cal']
    
    # Get array of gate delay and width, convert to us
    gate_delay[i] = metadict['gate_delay'][0]/1e3
    gate_width[i] = metadict['gate_width'][0]/1e3
    
    # Get array of center wavelengths for all shots, nm
    cwl[i] = metadict['cwl'][0]

    
#%% Plot the carbon erosion flux at each chord for each shot

# if len(shots) == 1:

# Distinguish chords that view a surface and those that miss the surface

# 400 micron diameter fibers, so 800 micron spot size at focal length?

# Factor of area increase due to elliptical spot size on angled surface of nose cone (nosecone_profile.py)

if chordconfig == 'lateral':
    # Only accounts for change in angle along x direction, not z
    A_fac = np.array([0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714,
           0.60299089, 0.77371385, 0.88272479, 0.96409869, 0.99477139,
           0.99477139, 0.96409869, 0.90329992, 0.77371385, 0.60299089,
           0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714])

elif chordconfig == 'axial':
    # For axial chords, after chord 11 chords are downstream of tip
    A_fac = np.array([0.84438502, 0.81135106, 0.77416671, 0.73123068, 0.68210694,
           0.62641429, 0.56036534, 0.47949189, 0.37683236, 0.22236974,
           0.00210557, 0.00210557, 0.00210557, 0.00210557, 0.00210557,
           0.00210557, 0.00210557, 0.00210557, 0.00210557, 0.00210557])

fig = plt.figure()
xchord = np.arange(1,21)

# Raw intensity from spectra
# plt.plot(impa, intensity[:,i], 'o-', label = 'Intensity')

# S/XB coefficients at each chord
# plt.plot(impa, sxb_chords[:,i], 'o-', label = 'S/XB')

# Erosion rates with constant S/XB coefficient from core parameters
# plt.plot(impa, flux_constant_sxb[:,i], 'o-', label = 'Constant S/XB value - Core')

# Remove large error bars for prescribed profiles: (check 3 things to change for prescribed)
# flux_err[:,i][4] = np.nan
# flux_err[:,i][15] = np.nan

flux_err_stack = np.vstack([flux_err_low[:,i], flux_err_hi[:,i]])
flux_plot = A_fac * flux_variable_sxb[:,i]

# Plot erosion rates with errorbars (equal)
# plt.errorbar(impa, A_fac * flux_variable_sxb[:,i], flux_err[:,i], fmt = 'b.-', capsize = 5)

# Plot erosion rates with errorbars (not equal)
plt.errorbar(impa, A_fac * flux_variable_sxb[:,i], flux_err_stack, fmt = 'b.-', capsize = 5)
plt.axvline(7.5, color = 'gray', linestyle = 'dotted', label = 'Nose cone edge')
plt.axvline(-7.5, color = 'gray', linestyle = 'dotted')
plt.text(0.83, 0.96, 'Pulse ' + str(shots[i]), transform=plt.gca().transAxes, fontsize = 6)
# plt.text(0.6, 0.93, 'Chord 20 $\longrightarrow$ Chord 1', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'), fontsize = 6)
plt.text(0.2, 0.96, 'Chord 20 $\\longrightarrow$ Chord 1', transform=plt.gca().transAxes, fontsize = 6)


if chordconfig == 'lateral':
    plt.xlabel('x [mm]')
elif chordconfig == 'axial':
    plt.xlabel('z [mm]')

# plt.ylabel('Flux [C atoms/cm$^2$/s]')
plt.ylabel('Flux [C atoms m$^{-2}$ s$^{-1}$]')

plt.xlim([-12, 12])

# plt.ylabel('S/XB Coefficient')
# plt.grid()
# plt.legend(loc = 'upper right', fontsize = 8)
plt.show()
    
    #%% 2D plot of currents with time as parameter

# if len(shots) > 1:
#     for i in np.arange(len(shots)):

#         fig2a = plt.figure(figsize=(8,8))
#         ax1 = fig2a.add_subplot(212)
#         ax2 = fig2a.add_subplot(211)
#         plt.title( 'Nose cone C-III emission intensity, $V_A$ = 9 kV, $V_C$ = 9 kV', fontsize=18)
        
#         ax1.plot(t_ip[:,i], i_p[:,i], 'b', label = 'Plasma current')
#         ax1.plot(t_ip[:,i], i_c[:,i], 'g', label = 'Compression Current')
#         ax1.plot(t_ip10[:,i], i_p10[:,i]/1e3, 'k', label = 'P10 Current')
#         ax1.annotate('Pulse ' + str(shots[0]) + '-' + str(shots[-1]), fontsize = 12, xy = (0, -0.2), xycoords='axes fraction')
    
#         ax1.set_xlim([0,100])
#         ax1.set_ylabel('Current [kA]')
#         ax1.grid()
#         ax1.legend()

#%% Convert ersion flux to mass loss

# Calculate mass loss for chord area and pulse duration. Assume normal observation angle and constant mass loss rate over pulse duration
r_chord = 400e-6 # Radius of chord viewing area on nose cone in meters. Fiber is 400 um diameter, magnified x2 on nosecone
A_chord = np.pi * r_chord**2 # area of chord viewing area in m^2

# Projected area
# A_chord = A_fac * np.pi * r_chord**2 # area of chord viewing area in m^2

# What should be the duration of plasma exposure used? If we use the time that we measure a magnetic field, then 100us
# However, using this time 
t_pulse = 100e-6 # pulse duration in seconds

C_mass = 1.9926e-23 # mass of carbon atom in grams
C_molar = 12.011 # Carbon mass in grams per moL
avo = 6.022e23 # Avogadros number

# Spectroscopy collection gate width, convert us to seconds
t_gate = gate_width*1e-6

# Preallocate mass flux 
mass_flux = np.zeros([20,len(shots)])

# Preallocate mass flux using molar mass
mass_flux_mol = np.zeros([20,len(shots)])

# Preallocate mass loss 
mass_loss = np.zeros([20,len(shots)])
mass_shot = np.zeros([20,len(shots)])

for ii in np.arange(len(flux_variable_sxb)):
    
    # eroded mass in mg per square meter per second
    mass_flux[ii] = C_mass * flux_variable_sxb[ii] * 1e3
    
    # Eroded mass in mg per sq meter per second using molar mass
    mass_flux_mol[:,i] = flux_variable_sxb[:,i]/avo * C_molar * 1e3

# mass_flux = mass_flux.ravel()
for i_shot in np.arange(len(shots)):

    mass_loss[:,i_shot] = mass_flux[:,i_shot] * t_gate[i_shot] * A_chord # eroded mass in mg over the gate width
    mass_shot[:,i_shot] = mass_flux[:,i_shot] * t_pulse * A_chord # eroded mass in mg over the shot, assuming constant erosion rate


mass_nc = 73.122 # Mass of nose cone coupon in grams (PMIC0)
mass_b = 691.4 # Mass of nose cone base in grams
mass_pmic0_final = 73.111 # Mass of PMIC0 after removal

mass_change = mass_nc - mass_pmic0_final # mass loss meaasured in grams
mass_change_shot = mass_change / 139

mass_pct = 100*mass_shot/mass_nc # Calculate percentage of nose cone coupon mass eroded each shot
mass_shots = 100/mass_pct # How many shots to erode some percentage of the coupon mass

# 3D plot of mass loss with radial position and time

# if len(shots) > 1:
    
#     fig3 = plt.figure(figsize=(14,10))

#     #Syntax for 3D projection
#     ax = plt.axes(projection='3d')
#     # ax = mplot3d.Axes3D(fig2)
    
#     # Define 3 axes
#     y = gate_delay    # Use [::-1] to reverse the order of elements in the array, but ticks don't follow
#     # x = np.arange(1,21,1) # By chord
#     x = impa
    
#     # z = np.transpose(sxb_data)
#     z = np.transpose(mass_loss)
    
#     # Meshgrid?
#     X, Y = np.meshgrid(x,y)
#     Z = z
    
#     # Plot the surface
#     surf = ax.scatter(X,Y,Z, cmap='plasma', linewidth=0.3, edgecolor='black')
#     # surf = ax.plot_surface(X,Y,Z, cmap='plasma', linewidth=0.3, edgecolor='black')
    
#     ax.set_title('Nose cone carbon emission via S/XB measurements of C-III', fontsize=18)
#     # ax.view_init(elev=20, azim=-45)
#     # ax.set_zlim([0, 2000])
#     fig3.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
    
#     # Set axis labels and size
#     ax.set_ylabel('Time ($\\mu$s)', fontsize=14)
#     # ax.set_xlabel('Chord', fontsize=14)
#     ax.set_xlabel('Radial position (mm)', fontsize=14)
#     ax.set_zlabel(' Mass loss (g)', fontsize=14)
#     plt.show()


#%% Comparison of spectroscopic mass loss to actual mass loss, synthetic mass loss profiles

# Measured mass loss of coupons [mg]
mass_loss = np.array([11, 17, 32, 4])

# Uncertainty is 1 mg
mass_loss_err = 1

# Number of shots in each run campaign
shot_count = np.array([155, 200, 50, 42])

# Mass loss per shot [mg]
mass_loss_shot = mass_loss / shot_count

m_C = 1.9926e-20 # Carbon atom mass in mg

# t_shot = 100e-6 # seconds, Estimated plasma exposure duration per shot
t_shot = 60e-6 # seconds, Estimated plasma exposure duration per shot

area_coupon = 0.00613 # m^2, surface area of coupon from CAD
area_pinch = np.pi * radius**2 # m^2, area using 3cm pinch radius
area_chord = np.pi * r_chord**2 # m^2, area using radius of spectroscopy chord

# Convert mass loss to erosion flux per shot [atoms / m^2 / s]
flux_net = mass_loss_shot / (m_C * area_chord * t_shot)

mass_loss_pmic0 = 11 # mg, overall mass loss of PMIC0
shots_pmic0 = 139 # number of high voltage shots on PMIC0

mass_per_area = mass_loss_pmic0 / area_coupon

# Mass loss rate (mg per m^2 per second) of pmic0
mass_rate_pmic0 = mass_per_area / (t_shot*shots_pmic0)

## Synthetic radial profile of mass loss
mass_rate_syn = np.ones([20,1])

# mass_rate_syn = mass_rate_pmic0*mass_rate_syn

# Generate 20 points centered around the peak (index 10) PARABOLIC
xs = np.linspace(-1, 1, 20)  # symmetric range from -1 to 1
xs = impa
ys = 1e11 - 1e9*xs**2                # parabolic shape, peak at x = 0
mass_rate_syn = ys


# Generate 20 points from -3 to 3 (for a typical Gaussian shape) GAUSSIAN
xs = impa
mu = 0        # Mean (center of the peak)
sigma = 2     # Standard deviation (controls width of peak)
A = 1e11

# Gaussian function
y_gauss = A * np.exp(- (xs - mu)**2 / (2 * sigma**2))

mass_rate_syn = y_gauss

# Get fwhm and function value at fwhm
fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
y_fwhm = 3.5e11 * np.exp(- (fwhm - mu)**2 / (2 * sigma**2))

# Getaverage y value
# Define Gaussian parameters
A = A       # Amplitude (peak value)
mu = mu      # Mean (center)
sigma = sigma   # Standard deviation

# Define the Gaussian function
def gaussian(x):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Define the integration range (e.g., μ ± 3σ)
# x_min = mu - 3 * sigma
x_min = 0
x_max = mu + 3 * sigma

# Compute the definite integral over the range
area, _ = quad(gaussian, x_min, x_max)

# Compute average y-value over this range
average_y = area / (x_max - x_min)

# Compute average yvalue divided by surface area
# average_y = area / area_coupon


# Plot to compare mass loss rates from S/XB and measured mass loss

fig4 = plt.figure()

### Log y plots

# Spectroscopic measured mass flux
# plt.semilogy(impa, flux_variable_sxb, 'bo-', label = 'S/XB')

# Synthetic mass flux
# plt.semilogy(impa, y_gauss, 'ko-', label = 'Synthetic')

### Normal plots

# Plot erosion flux
plt.errorbar(impa, flux_plot, flux_err_stack, fmt = 'k.', label = 'S/XB', capsize = 5)

# Plot equivalent net erosion flux
plt.axhline(flux_net[0], color = 'g', linestyle = '--',  label = 'Mass-loss')
# plt.axhline(flux_net[1], color = 'c', linestyle = '--',  label = 'Case II')
# plt.axhline(flux_net[2], color = 'm', linestyle = '--',  label = 'Case III')
# plt.axhline(flux_net[3], color = 'y', linestyle = '--',  label = 'Case IV')

# Plot theoretical erosion fluxes
flux_sput = 7.2e28
flux_sub = 3e31
plt.axhline(flux_sput, color = 'b', linestyle = ':',  label = 'Sputtering')
plt.axhline(flux_sub, color = 'r', linestyle = ':',  label = 'Sublimation')

# Change to log scale on y axis
plt.yscale('log',base=10) 


# plt.plot(impa, y_gauss, 'ko-', label = 'Synthetic')
# plt.axhline(y_fwhm, color = 'black', linestyle = '--',  label = 'Synthetic FWHM value')
# plt.axhline(average_y, color = 'black', linestyle = '--',  label = 'Synthetic average')


# plt.axvline(7.5, color = 'red', linestyle = '--')
# plt.axvline(-7.5, color = 'red', linestyle = '--')
# plt.title( "Carbon mass flux: " + str(shots[0]))
plt.xlabel('Radial position [mm]')
plt.ylabel('Erosion flux [atoms m$^{-2}$ s$^{-1}$]')
plt.ylim([1e28,1e32])
# plt.grid()
plt.legend(loc = 'upper right', fontsize = 6)

# Calculate redeposition rate?
redep = mass_rate_pmic0 / average_y

#%% More mass loss plots 

# fig5 = plt.figure()
# # plt.plot(impa, mass_loss[:,0], 'o-', label = 'Mass loss over gate width')
# # plt.plot(impa, mass_shot[:,0], 'o-', label = 'Mass loss per shot')

# plt.semilogy(impa, mass_loss[:,0], 'o-', label = 'Mass loss over gate width')
# plt.semilogy(impa, mass_shot[:,0], 'o-', label = 'Mass loss per shot, S/XB')
# plt.ylim([1e-10,1e3])
# plt.axhline(1e3*mass_change_shot, color = 'black', linestyle = '--',  label = 'Mass loss per shot, PMIC0')
# plt.axvline(7.5, color = 'red', linestyle = '--')
# plt.axvline(-7.5, color = 'red', linestyle = '--')
# plt.title( "Carbon mass loss : " + str(shots[0]))
# plt.xlabel('Radial Position (mm)')
# plt.ylabel('Mass [mg]')
# plt.ylim([1e-3,1e2])
# plt.legend()

# fig6 = plt.figure()
# plt.plot(impa, mass_pct[:,0], 'o')
# plt.axvline(7.5, color = 'red', linestyle = '--')
# plt.axvline(-7.5, color = 'red', linestyle = '--')
# plt.title( "Percentage of coupon mass eroded per shot: " + str(shots[0]))
# plt.xlabel('Radial Position (mm)')
# plt.ylabel('Percentage')
# # plt.legend()

# fig7 = plt.figure()
# plt.plot(impa, mass_shots[:,0], 'o')
# plt.axvline(7.5, color = 'red', linestyle = '--')
# plt.axvline(-7.5, color = 'red', linestyle = '--')
# plt.title( "Shots for 100% coupon erosion: " + str(shots[0]))
# plt.xlabel('Radial Position (mm)')
# plt.ylabel('Shots')
# plt.ylim([0,1e2])

#%% Scatter plots

# Plot intensity at one chord over gate delay time
chord = 10

intensity_out = np.zeros(len(shots))
fig4 = plt.figure()
for i in np.arange(len(shots)):
    intensity_out[i] = intensity[chord-1,i]
    plt.plot(gate_delay[i], intensity_out[i], 'bx')

plt.title( f"Intensity, Chord {chord}:")
plt.xlabel('Time [$\\mu$s]')
plt.ylabel('Counts [arb]')
# plt.ylim([0, 200])
# plt.legend()
plt.show()


#%% Plot flux at one chord over gate delay time

# Pre-allocate output arrays
flux_out = np.zeros(len(shots))
flux_err_out = np.zeros(len(shots))
flux_err_low_out = np.zeros(len(shots))
flux_err_hi_out = np.zeros(len(shots))

# Get flux values from single chord of different shots
# for i in np.arange(len(shots)):
#     flux_out[i] = A_fac[chord-1] * flux_variable_sxb[chord-1,i]
#     flux_err_out[i] = flux_err[chord-1,i]
#     flux_err_low_out[i] = flux_err_low[chord-1,i]
#     flux_err_hi_out[i] = flux_err_hi[chord-1,i]
    
#     flux_err_stack_single = np.vstack([flux_err_low[chord-1,i], flux_err_hi[chord-1,i]])
        
#     plt.errorbar(gate_delay[i], flux_out[i], flux_err_stack_single, fmt = 'k.-', capsize = 5)
    
# Get flux values from average of middle chord of different shots
for i in np.arange(len(shots)):
    flux_out[i] = np.mean([A_fac[chord-1] * flux_variable_sxb[chord-1,i], A_fac[chord] * flux_variable_sxb[chord,i]])
    flux_err_out[i] = np.mean([flux_err[chord-1,i], flux_err[chord,i]])
    flux_err_low_out[i] = np.mean([flux_err_low[chord-1,i], flux_err_low[chord,i]])
    flux_err_hi_out[i] = np.mean([flux_err_hi[chord-1,i], flux_err_hi[chord,i]])
    
    flux_err_stack_single = np.vstack([np.mean([flux_err_low[chord-1,i], flux_err_low[chord,i]]), np.mean([flux_err_hi[chord-1,i], flux_err_hi[chord,i]])])
        
    plt.errorbar(gate_delay[i], flux_out[i], flux_err_stack_single, fmt = 'k.-', capsize = 5)


# Sort flux and gate delay arrays in order of time
isort = np.argsort(gate_delay)
gate_delay_sort = gate_delay[isort]
flux_out_sort = flux_out[isort]

# Integrate the area under flux-time curve
flux_integral = cumulative_trapezoid(flux_out_sort,gate_delay_sort*1e-6,initial=0)
mass_sxb = flux_integral[-1] * area_pinch * m_C # Mass loss from SXB measurements in mg

# Plot theoretical erosion fluxes
flux_sput = 7.2e28
flux_sub = 3e31
flux_mean = np.mean(flux_out)
plt.axhline(flux_sput, color = 'b', linestyle = ':',  label = 'Max. sputtering')
plt.axhline(flux_sub, color = 'r', linestyle = ':',  label = 'Max. sublimation')
# plt.axhline(flux_mean, color = 'k', linestyle = ':',  label = 'S/XB average')
plt.axhline(flux_net[0], color = 'g', linestyle = ':',  label = 'Mass-loss')

# plt.title( f"Erosion Flux, Chord {chord}:")
# plt.text(60, 2300, str(shot[0]) + ' - ' + str(shot[-1]), fontsize = 8)
plt.text(0.01, 0.96, str(int(shots[0])) + ' - ' + str(int(shots[-1])), transform=plt.gca().transAxes, fontsize = 6)

plt.yscale('log',base=10) 

plt.xlabel('Time [$\\mu$s]')
plt.ylabel('Erosion Flux [C atoms / m$^2$ / s]')
plt.xlim([0, 120])
plt.ylim([1e27,1e32])
plt.legend(fontsize = 8)
plt.show()

# Plotting over max currents

chord = 10

ip10_max = np.zeros(len(shots))

for i in np.arange(len(shots)):
    
    ip_max = np.max(i_p[:,i])
    # plt.plot(ip_max, flux_variable_sxb[chord-1,i], 'bx')
    
    ic_max = np.max(i_c[:,i])
    # plt.plot(ic_max, flux_variable_sxb[chord-1,i], 'kx')
    
    ip10_max[i] = np.max(i_p10[:,i]/1e3)
    # plt.plot(ip10_max[i], flux_variable_sxb[chord-1,i], 'mx')
    
    flux_err_stack_single = np.vstack([flux_err_low[chord-1,i], flux_err_hi[chord-1,i]])

    plt.errorbar(ip10_max[i], flux_out[i], flux_err_stack_single, fmt = 'm.-', capsize = 5)

plt.yscale('log',base=10) 

    
plt.title( f"Erosion Flux, Chord {chord}:")
plt.xlabel('Current [kA]')
plt.ylabel('Erosion Flux [C atoms / m$^2$ / s]')
plt.ylim([1e27,1e32])
# plt.legend()
plt.show()

#%% Script to save as .npz file

if save == True:

    # Create a dictionary of specific arrays from the global scope
    campdict = dict(gate_delay = gate_delay, peak_currents = ip10_max, intensity = intensity_out, flux = flux_out, 
                    flux_err_low = flux_err_low_out, flux_err_hi = flux_err_hi_out)
    
    # Create a dictionary of all arrays 
    # campdict = {name: val for name, val in globals().items() if isinstance(val, np.ndarray)}

    # Save all to a single file
    np.savez(savename, **campdict)

#%% # How many Coulombs per shot? Per hour?

# zeros = np.where(currents<-0)
# zind = np.where(zeros[0]>10000)
# zero_ind = zeros[0][zind[0][0]]

# # Charge in coulombs per shot
# # Index range from t = 0 to end of current: 8193 to 10944
# chargecum = integrate.cumtrapz(currents[8192:zero_ind]*1000, t_ip[8192:zero_ind]*1e-6, initial=0)
# # charge = np.trapz(currents*1000, t_ip*1e-6)
# charge = np.trapz(currents[8192:zero_ind]*1000, t_ip[8192:zero_ind]*1e-6)

# # Get mass per coulomb value

# mpc = mass_loss*1000 / charge # Erosion rate in mg/C

# # Get average current if importing multiple shots
# # current_avg = np.mean(currents,0)

# fig2b = plt.figure()

# plt.plot(t_ip, currents, label = 'Compression Current')
# plt.plot(t_ip[8192:zero_ind], chargecum)
# plt.plot(t_ip[zero_ind],currents[zero_ind],'r*')
# plt.text(30, -40, 'Total charge per shot = ' + str(charge) + ' C')
# # plt.title( "Carbon erosion flux from nose cone")
# plt.xlabel('Time ($\\mu$s)')
# plt.ylabel('Current (kA)')
# plt.xlim([0, 120])
# plt.grid()
# plt.legend()
# plt.show()