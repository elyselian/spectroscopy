# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 13:44:11 2025

@author: aqilk
"""

## to do
# Figure out why there is no metadata associated with spe files



### This is a basic plotter to view spectral data in .spe files. 
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
import spe_loader as sl # Note spe_loader is a method within spe2py package

#%% Import .spe files with spe_loader

# Select shot to analyze
# shotnum = [250620012, 250620023]
shotnum = [250424006]
# shotnum = [250620012]

col = np.zeros([1024,len(shotnum)])

for i in np.arange(len(shotnum)):
    
    spe_day = str(shotnum[i])[0:6]
    spe_shot = str(shotnum[i])[6:9]
    
    # Select ops or calibration path by commenting the unused path
    # Ops data path
    spe_path = 'G:\\Shared drives\\Shumlak Lab\\Diagnostics\\Spectroscopy\\S_XB\\Data\\Ops\\' + spe_day
    
    # Calibration data path
    # spe_path = 'G:\\Shared drives\\Shumlak Lab\\Diagnostics\\Spectroscopy\\S_XB\\Data\\Calibration\\' + spe_day
    
    spe_filename = spe_day + '  ' + spe_shot + '.spe'
        
    # Ops data path:
    # spe_path = 'G:\Shared drives\Shumlak Lab\Diagnostics\Spectroscopy\S_XB\Data\Ops\\250620'
    # spe_filename = '250620  012.spe'
    
    # Calibration data path:
    # spe_path = 'G:\Shared drives\Shumlak Lab\Diagnostics\Spectroscopy\S_XB\Data\Calibration\\250422'
    # spe_filename = '250422  004.spe'
    
    # Combine .spe filename and path 
    spe_files = sl.load_from_files([spe_path + '\\' + spe_filename])
    
    # For 250422 files, this only loads the first 3 rows, which seems to be due to a weird roi selection being read from the .spe. All of the intensity data is 
    # still recorded and accessible in lightfield and in the matlab load_spe function... see _Read_Data function in spe_loader.py

    # This is a list with a single element, which is the 1024x1024 Numpy array of spectral intensities
    frame = spe_files.data[0]
    
    # Pre-allocate the list to store arrays of intensities from .spe files
    intensity = frame[0]
    
    # Append each 1024x1024 array of intensities into a list of arrays
    # intensity.append(frame[0])   
    
    # Extract metadata:
    
    # Gate delay in nanoseconds
    gate_delay = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['delay']])
    gate_delay = gate_delay.astype(float)
    
    # Gate width in nanoseconds
    gate_width = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['width']])
    gate_width = gate_width.astype(float)
    
    # Get center wavelength
    cwl = np.array([spe_files.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Spectrometers.Spectrometer.Grating.CenterWavelength.cdata])
    cwl = cwl.astype(float)
    
    # Get wavelength calibration
    wave_cal = spe_files.wavelength
    xcal = wave_cal

    # Calculate wavelength resolution per pixel
    wavperpix = (wave_cal[2] - wave_cal[0]) / 1024
    
    # 
    def wavcalout(pixel):
        pixtowav = wave_cal[0] + wavperpix * pixel
        return pixtowav
    
    # Extract the number of pixels of the frame
    pixels = np.arange(len(frame[0]))
    
    # Get xml tree
    # footer = spe_files.xmltree(footer)
    
    # Get SpeFormat object to search
    # metadata = spe_files.footer.SpeFormat
    
    #%% Plot single spectral image

    fig = plt.figure()
    
    # Plot with imshow. Faster, less options. Origin in Lightfield is top left "upper".
    # plt.imshow(frame[0], cmap='plasma', origin='upper') 
    # plt.axvline(x = 512, color = 'red', lw = 1, linestyle = '--')
    
    # Plot with pcolor. Plot 2D colormap of spectrum (slower, but easy axis customization)
    # plt.pcolor(xcal,pixels,frame[0], cmap='plasma') 
    # plt.axvline(x = cwl, color = 'red', lw = 1, linestyle = '--')
    
    # Plot 2d colormap using pcolormesh (faster than pcolor with easy customization)
    # flip = np.flipud(frame[0])
    plt.pcolormesh(xcal, pixels, frame[0], vmin = 0, vmax = 65000, cmap = 'plasma')
    plt.axvline(x = cwl, color = 'red', lw = 1, linestyle = '--')
    plt.gca().invert_yaxis()
    
    # Color bar
    plt.colorbar()
    
    plt.title( "ICCD: " + str(shotnum[i]))
    
    # X labels
    # plt.xlabel('Pixels')
    plt.xlabel('Wavelength [nm]')
    
    
    plt.ylabel('Pixels [Impact parameter]')
    plt.show() 
    
    #%% View horizontal and vertical slice
    
    # If changing plot settings
    i = 0
    
    # The 20 chords of the wide fiber image across 23.6 mm across the plasma diameter
    impa = np.arange(-23.6/2,23.6/2,1.24)
    
    # Horizontal slice
    fig3 = plt.figure()
    nrow = 511
    row = frame[0][nrow]    # Extract single row from image
    
    # Plot horizontal slice
    plt.plot(xcal,row)
    plt.axvline(x = cwl, color = 'red', lw = 1, linestyle = '--')
    plt.title("ICCD " + str(shotnum[i]) + ": Row " + str(nrow))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    plt.show() 
    
    # Vertical slice
    fig4 = plt.figure()
    ncol = 511
    col[:,i] = frame[0][:,ncol]    # Extract single row from image
    
    # Plot vertical slice
    plt.plot(pixels,col[:,i])
    plt.title("ICCD " + str(shotnum[i]) + ": Column " + str(ncol))
    plt.xlabel('Pixels')
    plt.ylabel('Counts')
    plt.ylim([0,65000])
    plt.show() 

    #%% Auto binning procedure

    fig3 = plt.figure()

    # Row number
    nrow = 50

    # Column number
    ncol = 512

    # Extract single row from array of summed intensities
    row = intensity[nrow]

    # Extract single column from array of summed intensities
    col = intensity[:,ncol]

    # Take a vertical slice by using intensity data from middle column (ncol=511) and use Savitzky-Golay filter to smooth
    col_filt = savgol_filter(col, window_length=30, polyorder=2)

    # Find peaks in vertical slice. Peaks will be center of each horizontal chord
    peaks, _ = find_peaks(col_filt,prominence=1200)
    print(str(len(peaks))+ " chords identified")

    # Define region of interest (ROI) in wavelength space for auto-binner
    ROI = [229,231] 
    # ROI_pix = [np.abs(xcal-ROI[0]).argmin(),np.abs(xcal-ROI[1]).argmin()]   # Convert ROI to pixel space

    # Find peak widths
    widths = peak_widths(col_filt, peaks, rel_height=0.5)

    # Make array of peak widths
    peak_width = widths[0]

    # Calculate upper and lower bin locations
    bins_lower = peaks - (peak_width/2)
    bins_lower[0] = 0   # First value was negative so replcae it with 0

    bins_upper = peaks + (peak_width/2)

    # Plot filtered intensity data, peaks, widths, and bin locations
    fig = plt.figure()
    plt.plot(pixels, col_filt)
    plt.plot(peaks,col_filt[peaks],"x")
    plt.hlines(widths[1], pixels[widths[2].astype(int)], pixels[widths[3].astype(int)])

    for chord_num in np.arange(0,20):
        plt.axvline(x = bins_lower[chord_num], linewidth = '0.5', color = 'black', linestyle='--')
        plt.axvline(x = bins_upper[chord_num], linewidth = '0.5', color = 'black', linestyle = '--')
        
    plt.title( "Column " + str(ncol) + " Bins")
    plt.xlabel('Pixels')
    plt.ylabel('Counts')
    plt.xlim([-1,1023])
    plt.show()

    # Plot the spectrum, reference vertical slice, and resulting horizontal bins
    plt.imshow(intensity, cmap='plasma', origin='upper', vmin = 0)
    # plt.vlines(ncol, 0, 1000, color='red')
    # plt.axvline(x = ncol, linewidth = '0.5', color = 'red')

    for chord_num in np.arange(0,20):
        plt.axhline(bins_lower[chord_num], linewidth = '1', color = 'white', linestyle = '--')
        plt.axhline(bins_upper[chord_num], linewidth = '1', color = 'red', linestyle = '--')
        # plt.axhline(bins_lower[1], linewidth = '1', color = 'red')

    # plt.hlines(bins_lower,0,1023, color='white', linestyles='--')
    # plt.hlines(bins_upper,0,1023, color='white', linestyles='--')

    plt.colorbar()
    plt.title( "Calibration Bin Locations")
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.show()
    
    def cal_bins():
        y1 = bins_lower
        y2 = bins_upper
        y3 = peak_width
        return y1, y2, y3

    # Save bin locations, widths as .npy file
    # np.save('inst_fun', bins_lower, bins_upper, peak_width)
#%% Plot multiple shot slices
# fig5 = plt.figure()

# # Plot
# plt.plot(pixels,col[:,0], label = str(shotnum[0]))
# plt.plot(pixels,col[:,1], label = str(shotnum[1]))

# plt.title("ICCD " + ": Column " + str(ncol))
# plt.xlabel('Pixels')
# plt.ylabel('Counts')
# plt.ylim([0,65000])
# plt.legend()
# plt.show() 