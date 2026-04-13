# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:06:43 2025

@author: aqilk
"""

# Description:
# This is the main analysis code for the S/XB spectroscopy diagnostic, which measures the flux
# of eroded material along a line-of-sight to a plasma-facing component.
# This script takes spectroscopy measurements for one shot and calculates the inferred erosion flux
# Imports include the S/XB coefficients and absolute calibration data

from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
# import spe2py as spe
import spe_loader as sl

# Imported classes

# import read_sxb function
from read_sxb_class import read_sxb

# Function for importing pinch currents
# from get_mds import get_currents 

# Function for getting plasma parameters based on pinch current scaling
# from zpinch_scaling import pinch_scale 

# Function for extracting measured electron number density profiles from DHI
from read_dhi_mds import dhi_profiles

# Create an instance of the read_sxb_class for ion C-III
sxb_vals = read_sxb("C-III")

# Import array of SXB calibration lamp bin locations, bin widths, and counts
# NOTE: sxb_cal.npy file is stored in G:\Shared drives\Shumlak Lab\Users\Current\Aqil Khairi\Python and is produced by the 
sxb_cal = np.load('sxb_cal.npy') # load


#%% Import .spe files
def sxb_shot(shotnum, radius, plot = False, chordconfig = 'lateral',):
    
    print(f"Starting analysis for Shot: {shotnum}, ")

    spe_day = str(shotnum)[0:6]
    spe_shot = str(shotnum)[6:9]
    spe_path = 'G:\\Shared drives\\Shumlak Lab\\Diagnostics\\Spectroscopy\\S_XB\\Data\\Ops\\' + spe_day
    spe_filename = spe_day + '  ' + spe_shot + '.spe'
    spe_files = sl.load_from_files([spe_path + '\\' + spe_filename])
    
    # This is a list with a single element, which is the 1024x1024 Numpy array of spectral intensities
    frame = spe_files.data[0]

    #%% Get currents from the tree
    # currents = get_currents(shotnum)
  
    # # Get time base for plasma current:
    # t_ip = currents[0]

    # # Get plasma current
    # i_p = currents[1]
    
    # # Get P10 current
    # i_p10 = np.transpose(currents[2])
        
    # # Get P10 current time base
    # t_ip10 = np.transpose(currents[3])

    # # Get compression current
    # i_c = currents[5]
    
    # mds_dict = dict(t_ip = t_ip, i_p = i_p, t_ip10 = t_ip10, i_p10 = i_p10, i_c = i_c)
       
    #%% Extract metadata
            
    # Array of calibration wavelengths in nanometers
    wavecalnm = spe_files.wavelength

    # Gate delay in nanoseconds, as numpy array
    gate_delay = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['delay']])
    gate_delay = gate_delay.astype(float)
    
    # Gate width in nanoseconds, as numpy array
    gate_width = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['width']])
    gate_width = gate_width.astype(float)
    
    # Get center wavelength, as numpy array
    cwl = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Spectrometers.Spectrometer.Grating.CenterWavelength.cdata])
    cwl = cwl.astype(float)

    # Calculate wavelength resolution per pixel
    wavperpix = (wavecalnm[2] - wavecalnm[0]) / 1024
    
    # Extract the number of pixels of the frame
    pixels = np.arange(len(frame[0]))
    
    # Get wavelength values for x axis
    xcal = wavecalnm
    
    # Wavelength resolution per pixel
    pixelres = ( xcal[-1] - xcal[0]) / len(pixels)
    # pixelres = xcal/1024
    
    metadict = dict(wave_cal = wavecalnm, gate_delay = gate_delay, gate_width = gate_width, cwl = cwl)
    #%% Plot spectral image
    
    if plot == True:
    
        fig = plt.figure()
        
        # Plot 2d colormap using pcolormesh (faster than pcolor with easy customization)
        # flip = np.flipud(frame[0])
        plt.pcolormesh(xcal, pixels, frame[0], vmin = 0, vmax = 65000, cmap = 'plasma')
        plt.axvline(x = 229.687, color = 'white', lw = 1)
        plt.text(229, 50, 'Chord 1', color = 'white')
        plt.text(229, 1000, 'Chord 20', color = 'white')
        plt.text(230.2, 1000, "Pulse " + str(shotnum), fontsize = 8, color = 'white')

        plt.gca().invert_yaxis()
        
        # Color bar
        plt.colorbar()
        
        # plt.title( "C-III Intensity: " + str(shotnum) )
        plt.xlim([228,231])
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Spatial position [pixels]')
        plt.show() 
    
    #%% Find peak in horizontal slice
    
        # fig3 = plt.figure()
        
        # How is row 200 chosen? - This is just to visualize, and isnt used in any calcs. 
        nrow = 200
        
        # Extract single row from image
        row = frame[0][nrow]
        
        # peak find
        peaks, _ = find_peaks(row,prominence=2000)
        
        # Plot
        plt.plot(xcal, row)
        plt.plot(xcal[peaks],row[peaks],"x")
        plt.title( "C-III: " + str(shotnum) + " - Row " + str(nrow))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Counts')
        plt.show() 
    
    #%% Import bins from calibration file
    
    # bins_import = SXB_Cal.cal_bins()
    # bins_import = calibration.auto_binner()
    
    # Lower bin location in pixels
    bins_lower = sxb_cal[0]
    
    # Upper bin location in pixels
    bins_upper = sxb_cal[1]
    
    # Bin width in pixels
    cwl_peak_width = sxb_cal[2]
    
    #%% Average the intensities within each binned chord and find the average peak intensity
    
    # Pre-allocate averaged intensities array
    chord_avg = np.zeros([len(xcal),20])
    
    # Pre-allocate intensity array
    intensity = np.zeros(20)
    
    # Placeholder to track global y-limits
    int_max = float('-inf')

    # Store y-data to determine global limits later
    int_data_list = []

    # Loop through 20 plots
    for chord_num in range(20):

        # Make a list containing the upper and lower pixel location of the specified bin
        bins_width = [int(bins_lower[chord_num]),int(bins_upper[chord_num])]
        
        # Extract the intensity values from each row within this bin
        chord = frame[0][bins_width[0]:bins_width[1]]
        
        # Take average over the columns of binned rows
        chord_avg[:,chord_num] = np.mean(chord,0)
        
        # Finds the maximum value from the average intensity for each chord (maybe peak finind instead?)
        intensity[chord_num] = max(chord_avg[:,chord_num])
        
        int_data_list.append(intensity[chord_num])
        
        # Update global max
        int_max = max(int_max, intensity[chord_num])
        
    if plot == True:
        # Set up the figure and 4x5 grid of axes
        fig, ax = plt.subplots(4, 5, figsize=(15, 10))
        
        # Flatten to 1D array for easy indexing
        ax = ax.flatten()
        
        # Loop through 20 plots
        for chord_num in range(20):
            
            ax[chord_num].plot(xcal, chord_avg[:,chord_num])
            
            ax[chord_num].set_title('Chord ' + str(chord_num+1))
    
            fig.suptitle( "C-III Binned Chord Intensities: " + str(shotnum), fontsize = 30)
    
    if plot == True:

        # Set uniform y-axis limits after plotting
        for ii in ax:
            ii.set_ylim(0,int_max)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    #%% Subtracting background chord intensity from plasma
    
    # Pre-allocate array for background intensities
    int_bg = np.zeros([len(chord_avg),20])
    
    # Index of chords to extract as background
    # For lateral chords
    bg_chords = np.array([0, 1, 2, 3, 16, 17, 18, 19])
    
    # Set up the figure and 4x5 grid of axes
    fig, ax = plt.subplots(4, 5, figsize=(15, 10))
    
    # Flatten to 1D array for easy indexing
    ax = ax.flatten()
    
    # Pre-allocate intensity array
    bg_max = np.zeros(20)
    
    # Placeholder to track global y-limits
    int_max = float('-inf')
    
    # Store y-data to determine global limits later
    int_data_list = []
    
    # for j in np.arange(len(bg_chords)):
    for j in bg_chords:
    
        # Extract background chords into a new array
        int_bg[:,j] = chord_avg[:,j]
        
        # Finds the maximum value from the average intensity for each chord (maybe peak finind instead?)
        bg_max[j] = max(int_bg[:,j])
        
        int_data_list.append(bg_max[j])
        
        # Update global max
        int_max = max(int_max, bg_max[j])
        
        ax[j].plot(xcal, int_bg[:,j])
        
        ax[j].set_title('Chord ' + str(j+1))
    
        fig.suptitle( "C-III Plasma Background Intensity: " + str(shotnum), fontsize = 30)
        
    # Set uniform y-axis limits after plotting
    for ii in ax:
        ii.set_ylim(0,int_max)
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    plt.close()
    
    #%% Take average of the background chords
    
    # Extract first and last 4 chords which go through plasma and form into one array
    bg_only = np.concatenate((int_bg[:,0:4], int_bg[:,16:20]), axis=1)
    
    # Take the mean of these chord intensities
    int_bg_mean = np.mean(bg_only,1)
    int_bg_max = np.max(int_bg_mean)/2
    
    #  Pre-allocate new array of intensities with background subtraction. Subtract half of the intensity of the bg chords
    int_new = np.zeros_like(chord_avg)
    for jj in np.arange(20):
        int_new[:,jj] = chord_avg[:,jj] - (int_bg_mean/2)
        
    # Set up the figure and 4x5 grid of axes
    fig, ax = plt.subplots(4, 5, figsize=(15, 10))
    
    # Flatten to 1D array for easy indexing
    ax = ax.flatten()
    
    # Pre-allocate array to store chord peak values
    intensity_sub = np.zeros(20)
    
    for j in np.arange(20):
        
        # Finds the maximum value from the background subtracted intensity values
        peaks, _ = find_peaks(int_new[:,j], prominence = 500)
        # intensity_sub[j] = int_new[peaks[0],j]
        intensity_sub[j] = max(int_new[:,j])
        
        ax[j].plot(xcal, int_new[:,j], label = 'Subtracted')
        
        ax[j].plot(xcal, chord_avg[:,j], label = 'Raw')
        
        ax[j].set_title('Chord ' + str(j+1))
        
        ax[j].legend()
    
    fig.suptitle( "C-III Plasma Background Subtraction: " + str(shotnum), fontsize = 30)
        
    # Set uniform y-axis limits after plotting
    for ii in ax:
        ii.set_ylim(0,int_max)
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    plt.close()
    
    #%% Get S/XB coefficients using peak temperature and density
    
    #### S/XB values  ####
        
    # Peak plasma parmaeters from 2017 PoP
    Te_sxb = 1000 # eV
    ne_sxb = 2e23 # m^-3
    
    # Call the function get_sxb from the class instance
    sxb = sxb_vals.get_sxb(Te_sxb, ne_sxb) 

    # Optional display of S/XB coefficient plots for C-III:
    # fig1 = sxb_vals.plot_2d()
    # fig2 = sxb_vals.plot_3d()
    # fig3 = sxb_vals.plot_dens_param()
    
    #%% Get S/XB values and errors using n_e and T_e profiles from DHI on ZaP-HD shot 160524021
    # DHI data published in Shumlak, et al., Physics of Plasmas (2017).
    
    # Import output of dhi_profiles function as a tuple
    radius = radius # define radius from input of SXB_shot function
    dhi_mds = dhi_profiles(chordconfig, radius)
    
    # Array of radial positions from DHI
    r_dhi = dhi_mds[0]
    
    # Array of Btheta from DHI in Tesla
    # B_dhi = dhi_mds[2]
    
    # Array of densities in m^-3 from DHI
    ne_dhi = dhi_mds[1]
    
    # Array of electron temperatures in eV from DHI
    te_dhi = dhi_mds[3]
    
    # Array of number density error values in m^-3 from DHI
    ne_error_dhi = dhi_mds[4]
    
    # Array of electron temperature error from direct propagation[eV]
    te_error_dhi = dhi_mds[5]
    
    # Array of electron temperature error from difference to propagated hi and low profiles [eV]
    te_error_low_dhi = dhi_mds[6]
    te_error_hi_dhi = dhi_mds[7]


    # Radial chord positions to get S/XB values for depend on lateral or axial chord configuration
    
    if chordconfig == 'lateral':
        # Chord positions in cm
        # chord_pos = 0.1 * np.arange(-23.6/2, 23.6/2, 1.24)
        
        # Chord positions in m for LATERAL chords. COMMENT this for prescribed profiles.
        chord_pos = 1e-3 * np.linspace(-23.6/2, 23.6/2, 20)
        
        # Chord positions for prescribed lateral chords (all chords at nose cone radius).Uncomment for modified profiles
        r_c = r_dhi[0] # nose cone radius from nosecone_profiles
        chord_pos = r_c * np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        
        
    elif chordconfig == 'axial':
        # Chord positions in m for AXIAL chords. From nosecone_profile.py
        chord_pos = 1e-3 * np.array([21.44733208, 20.60827119, 19.66379088, 18.57321808, 17.32547784,
       15.9108877 , 14.23324798, 12.17906691,  9.57152072,  5.64817876,
        0.02674082,  0.02674082,  0.02674082,  0.02674082,  0.02674082,
        0.02674082,  0.02674082,  0.02674082,  0.02674082,  0.02674082])

    # Define a quadratic function for fitting the temperature up to radius a = 0.3cm
    def quad(x, A, B, C):
        y = A*x**2 + B*x + C
        return y
    
    # Fit the quadratic function to the T_e data up to radius a
    par_chord, var_chord = curve_fit(quad, r_dhi[0:27], te_dhi[0:27])
    
    # Evaluate the T_e fit at the radial chord positions
    te_dhi_chords = quad(np.abs(chord_pos), par_chord[0], par_chord[1], par_chord[2])
    
    # Evaluate the T_e fit at all radial positions
    te_fit_full = quad(r_dhi, par_chord[0], par_chord[1], par_chord[2])
    
    # Select only positive temperature values from fit
    for i in np.arange(0,20):
        if te_dhi_chords[i] <= 0:
            te_dhi_chords[i] = 0
        
    # # Plot the DHI T_e data and fit
    # f1 = plt.figure()
    # plt.plot(r_dhi*1e3, te_fit_full, label = 'T_e fit')
    # plt.plot(r_dhi*1e3, te_dhi, label = 'T_e from DHI 160524021')
    # plt.plot(chord_pos*1e3,te_dhi_chords, 'o', label = 'Temperature fit at chord locations') # Plot the fit to the temperature data
    # plt.title(f'Electron temperature data and fitting: {shotnum}')
    # plt.xlabel('Radius [mm]')
    # plt.ylabel('Electron temperature [eV]')
    # plt.xlim([0,10])
    # plt.ylim([-10,1100])
    # plt.grid()
    # plt.legend()
    
    ### Density fitting
    
    # Take fitted values for positive r and reflect on other side of r axis
    r_dhi_neg = -r_dhi
    r_dhi_2 = np.concatenate((r_dhi_neg[::-1], r_dhi), axis=0).flatten()
    
    ne_dhi_neg = ne_dhi[::-1]
    ne_dhi_2 = np.concatenate((ne_dhi_neg, ne_dhi), axis=0).flatten()

    # # Lorentzian function
    # def lorentzian(x, A, x0, gamma):
    #     return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2)

    # # Initial guess: [Amplitude, Center, FWHM]
    # p0 = [1e23,0,0.1]
    
    # # Get coefficients for lorenztian fit to density data
    # popt, pcov = curve_fit(lorentzian, r_dhi_2, ne_dhi_2, p0=p0)
    # A_fit, x0_fit, gamma_fit = popt
    
    # ne_dhi_fit = lorentzian(r_dhi_2, *popt)
    # ne_dhi_chords = lorentzian(chord_pos, *popt)
    
    # Gaussian
    
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Initial guess: [Amplitude, Center, Std Dev]
    p0 = [1e23, 0, 0.1]
    
    # Fit
    popt, pcov = curve_fit(gaussian, r_dhi_2, ne_dhi_2, p0=p0)
    A_fit, x0_fit, sigma_fit = popt
    
    # Fit curve
    ne_dhi_fit = gaussian(r_dhi_2, *popt)
    ne_dhi_chords = gaussian(chord_pos, *popt)        
    
    # # Get coefficients for 3o polynomial fit to density data
    # ne_fit_coeffs = np.polyfit(r_dhi, ne_dhi, 3)
    
    # # Evaluate density fit to all new radial positions
    # ne_dhi_fit = np.polyval(ne_fit_coeffs, r_dhi)
    
    # # Evaluate density fit at radial chord positions
    # ne_dhi_chords = np.polyval(ne_fit_coeffs, chord_pos)
    
    # # Take fitted values for positive r and reflect on other side of r axis
    # ne_dhi_chords[0:10] = ne_dhi_chords[20:9:-1]

    # # Plot the DHI n_e data and fit
    # f2 = plt.figure()
    # plt.plot(r_dhi_2*1e3, ne_dhi_2, label = 'n_e from DHI 160524021')
    # plt.plot(r_dhi_2*1e3, ne_dhi_fit, label = 'n_e fit')
    # plt.plot(chord_pos*1e3, ne_dhi_chords, 'o', label = 'n_e fit at chord locations')
    # plt.title(f'Electron density data and fitting: {shotnum}')
    # plt.xlabel('Radius [mm]')
    # plt.ylabel('Electron density [m$^3$]')
    # plt.xlim([0,10])
    # # plt.ylim([0,2e23])
    # plt.grid('both')
    # plt.legend(fontsize = 9)

    # Get S/XB values for Te and ne profiles from above:
    sxb_chords = np.zeros(20)
    
    for i in np.arange(20):
        sxb_chords[i] = sxb_vals.get_sxb(te_dhi_chords[i], ne_dhi_chords[i])

    # ## Plot the extracted density, temperature, and resulting sxb values with chord position
    # # Note that we are assuming the radial profiles are axisymmetric
    # fa = plt.figure()
    # plt.plot(chord_pos, ne_dhi_chords * 1e-17 * 1e-3, 'o-', label = 'Density points')
    # plt.plot(chord_pos, te_dhi_chords, 'o-', label = 'Temperature points')
    # plt.plot(chord_pos, sxb_chords * 1e1, 'o-', label = 'SXB')
    # plt.legend()
    # plt.show()
    
    #%% Get S/XB error values
    
    # Shorten by one element to match size of error arrays
    r_error = r_dhi[0:len(r_dhi)]
    
    # Shorten by one element to match size of error arrays
    # r_error = r_dhi[0:85]
    
    r_ind = np.zeros([len(chord_pos)])
    ne_error_dhi_chords = np.zeros([len(chord_pos)])
    te_error_dhi_chords = np.zeros([len(chord_pos)])
    
    # Preallocate array of interpolated density error values
    ne_err_interp_vals = np.zeros([len(chord_pos)])
    
    # Preallocate array of interpolated temperature errors, including low and hi

    te_err_interp_vals = np.zeros([len(chord_pos)])
    te_err_low_interp_vals = np.zeros([len(chord_pos)])
    te_err_hi_interp_vals = np.zeros([len(chord_pos)])
    
    # Interpolate density and temperature error values
    ne_err_interp = interp1d(r_error, ne_error_dhi)
    
    te_err_interp = interp1d(r_error, te_error_dhi)
    te_err_interp_low = interp1d(r_error, te_error_low_dhi)
    te_err_interp_hi = interp1d(r_error, te_error_hi_dhi)

    # Get error values at chord positions. Chord positions outside the radial range of interpolation are assigned to nan.
    for j in np.arange(len(chord_pos)):
        try:
            ne_err_interp_vals[j] = ne_err_interp(np.abs(chord_pos[j]))
            te_err_interp_vals[j] = te_err_interp(np.abs(chord_pos[j]))
            te_err_low_interp_vals[j] = te_err_interp_low(np.abs(chord_pos[j]))
            te_err_hi_interp_vals[j] = te_err_interp_hi(np.abs(chord_pos[j]))

        except ValueError:
            ne_err_interp_vals[j] = np.nan
            te_err_interp_vals[j] = np.nan
            te_err_low_interp_vals[j] = np.nan
            te_err_hi_interp_vals[j] = np.nan

            
    ne_error_dhi_chords = ne_err_interp_vals
    te_error_dhi_chords = te_err_interp_vals
    te_error_low_dhi_chords = te_err_low_interp_vals
    te_error_hi_dhi_chords = te_err_hi_interp_vals

    # Get S/XB values for Te and ne profiles from above:
    sxb_error_chords = np.zeros(20)
    sxb_error_low_chords = np.zeros(20)
    sxb_error_hi_chords = np.zeros(20)

    for i in np.arange(20):
        sxb_error_chords[i] = sxb_vals.get_sxb(te_error_dhi_chords[i], ne_error_dhi_chords[i])
        sxb_error_low_chords[i] = sxb_vals.get_sxb(te_error_low_dhi_chords[i], ne_error_dhi_chords[i])
        sxb_error_hi_chords[i] = sxb_vals.get_sxb(te_error_hi_dhi_chords[i], ne_error_dhi_chords[i])


    #%% Error propagation for S/XB coefficients
    
    # NOTE: Propagating density errors through Te calcs results in huge error bars. Using only density errors.
    
    sxb_all = np.zeros([len(ne_dhi), len(ne_dhi)])
    # sxb_axes = np.zeros([len(ne_dhi), len(ne_dhi)])

    for ii in np.arange(len(ne_error_dhi)):
        sxb_all[ii] = sxb_vals.get_sxb(te_dhi[ii], ne_dhi[ii])
    
    # Compute partial derivatives along each axis
    partial_te, partial_ne = np.gradient(sxb_all, axis=(0, 1))
    
    # Calculate error with product rule method:
    sxb_error = partial_te * te_error_dhi + partial_ne * ne_error_dhi    

#%% Get impact parameter defined by fiber bundle
    
    # The 20 chords of the wide fiber image across 23.6 mm across the plasma diameter
    impa = np.linspace(-23.6/2, 23.6/2, 20)

    #%% The absolute irradiance based on DH3 calibration [Watts/m^2/nm]
    
    # Input wavelength in nanometers. Absolute irradiance in uW/cm2/nm
    # Irr_abs = SXB_Cal.dh3cal(cwl)
    # Irr_abs = calibration.dh3cal(cwl)
    Irr_abs = 6.3525963157828755 # Value for 229.7 nm
    
    # Convert to units of W/m^2/nm. Do i need to multiply by wavelength? Two orders of magnitude
    Irr_abs_W = Irr_abs*1e-6*1e4*229.7
    
    #%% # Get the counts measured during absolute calibration with the DH3CAL.
    
    # Using the SXB calibration from 4/22/25
    
    # PI-MAX4 exposure time in seconds. 21 s maximum
    gate_cal = 21    
    
    # counts_cal = calibration.averaging()/gate_cal  # Counts/s obtained from calibration
    
    # Get the counts corresponding to 229.7nm from calibration analysis script, divide by gate width
    counts_cal = sxb_cal[3]/gate_cal
    
    #%% Calculate the impurity influx with S/XB
    
    # TO DO
    # Run analysis for the time scan and plot time evolution of values at each chord
    # What is the depth of field of the telescope?
    
    # Redeposition fraction (unused for now)
    Fc = 0.9999   
    
    h = 6.62e-34    # Planck's constant [Joule-seconds]
    wav = 229.7e-9 # Wavelength of study [meters]
    c = 3e8         # Speed of light in vacuum [m/s]
    f = c/wav      # Frequency of emission [1/meters]
    E_ph = h*f      # Photon energy of measured emission [Joules]
    
    
    # Get the counts measured from experiment and divide by the exposure time
    
    # Get experiment PI-MAX4 exposure time from .spe metadata. Convert nanoseconds to seconds
    gate_exp = gate_width[0]*1e-9
    # gate_exp = 50e-9
    
    # Counts / s obtained from experiment at target wavelength
    # counts_exp = intensity/gate_exp # With no background subtraction
    counts_exp = intensity_sub/gate_exp 
    
    # Get Poisson uncertainty for intensity counts (bg subtracted)
    intensity_sub_err = np.sqrt(intensity_sub)
    
    # Photon flux [photons/m^2/s]
    PhotonFlux = ((counts_exp/counts_cal)*Irr_abs_W) / (E_ph)
    
    photon_flux_err = ((intensity_sub_err/counts_cal)*Irr_abs_W) / (E_ph)
    
    Photons = PhotonFlux # Multiply by exposure time of ICCD to get total number of photons
    # print(Photons)
    
    # Erosion flux calculated using a single S/XB value corresponding to peak core Te and ne
    I = PhotonFlux
    # ErosionFlux = 4*np.pi*sxb*I*1e-4    #Erosion rate [atoms/cm^2/s]
    
    # Erosion rate [atoms/m^2/s] using single value
    ErosionFlux = 4*np.pi*sxb*I    

    # Include the redeposition factor
    # ErosionFlux = 4*(1/(1-Fc))*np.pi*sxb*I    #Erosion rate [atoms/m^2/s]
    
    # Use different S/XB values according to DHI data from PoP 2017
    
    ErosionFlux2 = np.zeros(20)
    for i in np.arange(20):
        ErosionFlux2[i] = 4*np.pi*sxb_chords[i]*I[i]    #Erosion rate [atoms/m^2/s]

    # print(ErosionFlux)
    
    # fig = plt.figure()
    # # xchord = np.arange(1,21)
    # # plt.plot(xchord, ErosionFlux, 'o')
    # plt.plot(impa, ErosionFlux2, 'o-')
    # plt.title( "Carbon erosion flux from nose cone: " + str(shotnum))
    # plt.xlabel('Radial Position (mm)')
    # # plt.xticks(np.arange(min(impa), max(impa)+1, 2.0))
    # # plt.ylabel('Flux [C atoms/cm$^2$/s]')
    # plt.ylabel('Flux [C atoms/m$^2$/s]')
    
    # # Plot vertical lines to represent representative pinch size
    # # plt.axvline(x = -3, color='g', linestyle='--', label='Ref 3 mm radius pinch')
    # # plt.axvline(x = 3, color='g', linestyle='--')
    
    # plt.grid()
    # plt.legend()
    # plt.show()
    
    #%% Propagation of uncertainty calcs

    # Get Poisson uncertainty for intensity counts (bg subtracted)
    intensity_sub_err = np.sqrt(intensity_sub)
    
    # intensity_unc = np.sqrt(intensity)
    
    # Need to get 'y' which is the propagated uncertainty of the s/xb coefficient
    # Uncertainty of the flux measurement
    # flux_err = ErosionFlux2 * np.sqrt((intensity_sub_err / intensity_sub)**2 + (ne_error_dhi_chords / ne_dhi_chords)**2)
    
    # Includes ne and Te
    # flux_err = ErosionFlux2 * np.sqrt((intensity_sub_err / intensity_sub)**2 + (ne_error_dhi_chords / ne_dhi_chords)**2 + (te_error_dhi_chords / te_dhi_chords)**2)

    # S/XB error and counting error
    flux_err = ErosionFlux2 * np.sqrt((photon_flux_err / PhotonFlux)**2 + (sxb_error_chords / sxb_chords)**2)
    flux_err_low = ErosionFlux2 * np.sqrt((photon_flux_err / PhotonFlux)**2 + (sxb_error_low_chords / sxb_chords)**2)
    flux_err_hi = ErosionFlux2 * np.sqrt((photon_flux_err / PhotonFlux)**2 + (sxb_error_hi_chords / sxb_chords)**2)


    sxbdict = dict(flux_constant_sxb = ErosionFlux, flux_variable_sxb = ErosionFlux2, intensity = intensity_sub, sxb_chords = sxb_chords, flux_err = flux_err, flux_err_low = flux_err_low, flux_err_hi = flux_err_hi, ne_err = ne_error_dhi_chords)
    return sxbdict, metadict#, mds_dict
