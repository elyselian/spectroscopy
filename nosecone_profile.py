# -*- coding: utf-8 -*-
"""
Created on Wed May  7 20:23:10 2025

@author: aqilk
"""

# This code will plot the nominal surface profile of the nose cone for spectroscopic references.

import matplotlib.pyplot as plt
import numpy as np

# Define z position of interest in mm, with 0 at the n/c tip
zloc = 1 # mm

# Define the radius of the semicircle
r = 25.4 # mm

# On the nosecone, the radius ends at a Z distance of 0.504 in from the tip. If tip is at 1.0, radius ends at y = 1-0.504 in this plot
# r_end = 1 - 0.504 # in

# This is the angle the nose cone profile makes with the vertical axis in the plot
theta1 = 29.74*(np.pi/180)

# Create an array of angles from 0 to pi (for the upper semicircle)
theta = np.linspace(theta1, np.pi - theta1, 1000)

# Define point at end of radius section
x1 = r*np.cos(theta1)
y1 = r*np.sin(theta1) - 25.4

# Define point at end of straight section of profile. 1.99 inches is the length of this section before the next radius

x2 = x1 + 1.99 * np.sin(theta1) * 25.4 # mm
y2 = y1 - 1.99 * np.cos(theta1) * 25.4 # mm

x3 = r * np.cos(np.pi-theta1)
y3 = r*np.sin(np.pi-theta1) - 25.4

x4 = x3 - 1.99 * np.sin(theta1) * 25.4 # mm
y4 = y3 - 1.99 * np.cos(theta1) * 25.4 # mm

slope = (y2 - y1) / (x2 - x1)
slope2 = (y3 - y4) / (x3 - x4)

xrange = np.linspace(x1, x2,1000)
xrange2 = np.linspace(x4, x3, 1000)
line = slope * (xrange - x1) + y1

# line2 = -line
line2 = slope2 * (xrange2 - x3) + y3

# Calculate the x and y coordinates of the semicircle
x = r * np.cos(theta)
y = r * np.sin(theta) - 25.4

# Circle equation explicit
X = np.linspace(x3, x1, 500)

Y = np.sqrt(25.4**2 - X**2) - 25.4

# Vertical line equations for chords
chord1 = 0

# Coupon interface
zcoupon = (1 - 1.35)*25.4 - 25.4


# At each y location, for transverse aligned chords, the variation in angle of the surface is described by a circle, whose diameter
# is equal to the distance between opposite points

def get_diam(zloc):
    "zloc is the distance from the tip of the nose cone in mm that the chords are positioned"
    
    # This is the z position of the chords being evaluated, relative to 0 at the nose cone tip
    zpos = -zloc
    
    # This is the z position of the interface between the circular tip and the linear profile
    yrad = r*np.sin(theta1) - 25.4
    
    if zpos > yrad:
        # Evaluate cross-sectional diameter at the tip
        
        # Get the index where the equation of the circle (Y) is closest to the desired z position
        yind = np.abs(Y-zpos).argmin()
        
        # Get the x position corresponding to the above index and multiply by 2 to get the radius
        xdim = np.abs(2 * X[yind])
        
        print("zpos > yrad, Diameter is " + str(xdim) + " mm")

    else:
        # Evaluate cross-sectional diameter at the linear profile
        
        # Get the index where the equation of the line is closest to the desired z position
        yind = np.abs(line-zpos).argmin()
        
        # Get the x position corresponding to the above index and multiply by 2 to get the radius
        xdim = 2 * xrange[yind]
        
        print("Diameter is " + str(xdim) + " mm")

    return zpos, yind, xdim

# Plot the semicircle
fig = plt.figure()
# plt.plot(x, y, 'k', label='Semicircle')
# plt.plot(xrange[489], line[489], 'x')
# plt.plot(x1,y1,'ro')
# plt.plot(x2,y2, 'ro')
# plt.plot(x3,y3, 'ro')
# plt.plot(x4,y4, 'ro')

# Plot nose cone profiles
plt.plot(xrange, line, 'k')
plt.plot(xrange2, line2, 'k')

plt.plot(X,Y,'k')

# plt.hlines(np.sin(theta1), -1, 1)
plt.hlines(zcoupon, (-1.35 * 25.4), (1.35 * 25.4), color = 'black', linestyle='--', label = 'Coupon edge') # mm


# Set the aspect of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.title('Nose cone profile')
plt.grid()
plt.legend()

#%% Plot the circular profile at position of chords to determine angle of incidence of chord

# Create an array of angles from 0 to pi (for the upper semicircle)
th = np.linspace(0, np.pi, 1000)

# The radius of the cross-section at the z position of interest
r2 = 0.5 * get_diam(zloc)[2]

# Define the equations of the circle at this radius
x_cross = r2*np.cos(th)
y_cross = r2*np.sin(th)

# explicit circle 
# Circle equation explicit

X2 = np.linspace(-r2, r2, 50)

Y2 = np.sqrt((r2)**2 - X2**2)

fig2 = plt.figure()

# plot semicircle
plt.plot(x_cross,y_cross,'k')
# plt.plot(X2,Y2,'r')

# plot chords

# chord positions relative to the axis in mm
chord_pos = np.arange(-23.6/2, 23.6/2, 1.24)
# chord_pos = [-23.6/2]

angle_degrees = np.zeros(20)

for i in np.arange(len(chord_pos)):
# for i in [0]:

    # Find index where the value of chord position equals value of circle equation
    ind = np.abs(x_cross-chord_pos[i]).argmin()
    
    # Plot vertical lines that terminate on the circle
    plt.vlines(chord_pos[i], y_cross[ind], 2*np.max(y_cross))
    
    # get angle between chord and surface
    # chordline = np.array([y1[ind],2*np.max(y1)])
    
    # vector for vertical line
    chordline = np.array([0,1])
    
    # Get the slope of the semicircle using explicit form of circle equation Y2
    tangent = np.gradient(Y2, X2)
    
    # Find index where value of X values for explicit circle equation equal the chord positions
    j = np.abs(X2-chord_pos[i]).argmin()
    
    plt.plot(chord_pos[i],Y2[j],'bx')
    
    # Get the y-intercept of the line tangent to each intersection point (for plotting only)
    b = Y2[j]-tangent[j]*chord_pos[i]
    # plt.plot(X2,(tangent[j]*X2)+b)
    
    # Vector for tangent line
    tangline = np.array([tangent[j],1])
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(chordline, tangline)
    magnitude1 = np.linalg.norm(chordline)
    magnitude2 = np.linalg.norm(tangline)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees[i] = np.degrees(angle_radians)
    
    
# Set the aspect of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Nose cone cross-section with chords')
# plt.xlim([-10, 0])
plt.ylim([0, 15])
# plt.legend()
plt.show()


#%% Plot the zy cross-section to get angle change along z

# Plot nose cone profiles

# This is the angle the nose cone profile makes with the vertical axis in the plot
theta_zy = (90-29.74)*(np.pi/180)

# Define point at end of radius section
x1_zy = r*np.cos(theta_zy) - 25.4
y1_zy = r*np.sin(theta_zy)

# Define point at end of straight section of profile. 1.99 inches is the length of this section before the next radius

x2_zy = x1_zy - 1.99 * np.sin(theta_zy) * 25.4 # mm
y2_zy = y1_zy + 1.99 * np.cos(theta_zy) * 25.4 # mm

# Point on negative z side of semicircle
x3_zy = r * np.cos(theta_zy) - 25.4
y3_zy = r*np.sin(-theta_zy)

#  Point at negative z side of straight section
x4_zy = x3_zy - 1.99 * np.sin(theta_zy) * 25.4 # mm
y4_zy = y3_zy - 1.99 * np.cos(theta_zy) * 25.4 # mm

slope_zy = (y2_zy - y1_zy) / (x2_zy - x1_zy)
slope2_zy = (y3_zy - y4_zy) / (x3_zy - x4_zy)

xrange_zy = np.linspace(x1_zy, x2_zy,1000)
xrange2_zy = np.linspace(x4_zy, x3_zy, 1000)
line_zy = slope_zy * (xrange_zy - x1_zy) + y1_zy

line2_zy = -line_zy
line2_zy = slope2_zy * (xrange2_zy - x3_zy) + y3_zy

# Positive linear segment
plt.plot(xrange_zy, line_zy, 'k')

# Negative linear segment
plt.plot(xrange2_zy, line2_zy, 'k')

#Plot ploints 
# plt.plot(x1_zy,y1_zy,'ro')
# plt.plot(x2_zy,y2_zy,'ro')
# plt.plot(x3_zy,y3_zy,'ro')
# plt.plot(x4_zy,y4_zy,'ro')

# Plot semicircle
# Create an array of angles from 0 to pi (for the upper semicircle)
# th = np.linspace(np.pi/2, -np.pi/2, 1000)
th = np.linspace(theta_zy, -theta_zy, 1000)

# Calculate the x and y coordinates of the semicircle
x_zy = r * np.cos(th) - 25.4
y_zy = r * np.sin(th)

# Arc
plt.plot(x_zy, y_zy, 'k')

# Combine arc and linear segment
y_side = np.append(line_zy, y_zy[0:500])
x_side = np.append(xrange_zy, x_zy[0:500])

# Check combination
# fig = plt.figure()
# plt.plot(x_side, y_side, 'b')
# plt.plot(x[0:500], y[0:500])
# plt.plot(xrange, line)

# plot chords

# chord positions relative to the axis in mm
chord_pos = np.arange(-23.6/2, 23.6/2, 1.24)

angle_degrees = np.zeros(20)

# Pre-allocate array of radial positions for axial sxb calcs
chord_pos_ax = np.zeros(20)
    
for i in np.arange(len(chord_pos)):
# for i in [0]:

    # Find index where the value of chord position equals value in array of profile x points
    ind = np.abs(x_side-chord_pos[i]).argmin()
    
    # Plot chords as vertical lines that terminate on the profile
    plt.vlines(chord_pos[i], y_side[ind], max(y_side))
    
    # Store y coordinates as radial positions for sxb, in mm
    chord_pos_ax[i] = y_side[ind]
    
    # get angle between chord and surface
    
    # vector for vertical line
    chordline = np.array([0,1])
    
    # Get the slope of the profile by numerical differentiation
    tangent = np.gradient(x_side, y_side)
    
    # Find index where value of X values for explicit circle equation equal the chord positions
    j = np.abs(x_side-chord_pos[i]).argmin()
    plt.plot(chord_pos[i], y_side[ind],'bx')
    
    # Get the y-intercept of the line tangent to each intersection point (for plotting only)
    b = chord_pos[i]-tangent[j]*y_side[j]
    # plt.plot(x_side,(tangent[j]*x_side)+b)
    
    # Vector for tangent line
    tangline = np.array([tangent[j],1])
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(chordline, tangline)
    magnitude1 = np.linalg.norm(chordline)
    magnitude2 = np.linalg.norm(tangline)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees[i] = np.degrees(angle_radians)

# Plot edge of coupon
plt.vlines(zcoupon, (-1.35 * 25.4), (1.35 * 25.4), color = 'black', linestyle='--', label = 'Coupon edge') # mm

# Set the aspect of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('z [mm]')
plt.ylabel('y [mm]')
plt.title('Nose cone zy cross-section with chords')
plt.ylim([-10, 50])
plt.grid()
# plt.legend()

#%% Plot the zy cross-section to get angle change along z

# # Create an array of angles from 0 to pi (for the upper semicircle)
# th = np.linspace(0, np.pi, 1000)

# # Plot nose cone profiles

# # Positive linear segment
# plt.plot(xrange, line, 'k')

# # Negative linear segment
# plt.plot(xrange2, line2, 'k')

# # Arc
# plt.plot(x, y, 'k')

# # Combine arc and linear segment
# y_side = np.append(line2, y[0:500])
# x_side = np.append(xrange2, -x[0:500])

# fig = plt.figure()
# plt.plot(x_side, y_side, 'b')
# plt.plot(x[0:500], y[0:500])
# plt.plot(xrange, line)

# # plot chords

# # chord positions relative to the axis in mm
# chord_pos = np.arange(-23.6/2, 23.6/2, 1.24)
# # chord_pos = [-23.6/2]

# angle_degrees = np.zeros(20)
    
    
# for i in np.arange(len(chord_pos)):
# # for i in [0]:

#     # Find index where the value of chord position equals value in array of profile y points
#     ind = np.abs(y_side-chord_pos[i]).argmin()
    
#     # Plot chords as horizontal lines that terminate on the profile
#     plt.hlines(chord_pos[i], min(x_side), x_side[ind])
#     # plt.plot(x_side[ind], chord_pos[i],'x')
    
#     # get angle between chord and surface
    
#     # vector for horizontal line
#     chordline = np.array([1,0])
    
#     # Get the slope of the profile by numerical differentiation
#     tangent = np.gradient(y_side, x_side)
    
#     # Find index where value of X values for explicit circle equation equal the chord positions
#     j = np.abs(y_side-chord_pos[i]).argmin()
#     plt.plot(x_side[j], chord_pos[i],'x')
    
#     # Get the y-intercept of the line tangent to each intersection point (for plotting only)
#     b = chord_pos[i]-tangent[j]*x_side[j]
#     # plt.plot(x_side,(tangent[j]*x_side)+b)
    
#     # Vector for tangent line
#     tangline = np.array([tangent[j],1])
    
#     # Calculate the dot product and magnitudes of the vectors
#     dot_product = np.dot(chordline, tangline)
#     magnitude1 = np.linalg.norm(chordline)
#     magnitude2 = np.linalg.norm(tangline)
    
#     # Calculate the cosine of the angle
#     cos_angle = dot_product / (magnitude1 * magnitude2)
    
#     # Calculate the angle in radians and then convert to degrees
#     angle_radians = np.arccos(cos_angle)
#     angle_degrees[i] = np.degrees(angle_radians)
    
    
# # Set the aspect of the plot to be equal
# plt.gca().set_aspect('equal')

# # Add labels and title
# plt.xlabel('y [mm]')
# plt.ylabel('z (mm)')
# plt.title('Nose cone zy cross-section with chords')
# plt.grid()
# # plt.legend()

#%% Plot chords on the nose cone

# Plot the semicircle
fig3 = plt.figure()
# plt.plot(x, y, label='Semicircle')
# plt.plot(xrange[489], line[489], 'x')
# plt.plot(x1,y1,'ro')
# plt.plot(x2,y2, 'ro')
# plt.plot(x3,y3, 'ro')
# plt.plot(x4,y4, 'ro')

# Plot nose cone profiles
plt.plot(xrange, line, 'k')
plt.plot(xrange2, line2, 'k')

# Plot semi circle
# plt.plot(x, y, 'k')
plt.plot(X,Y,'k')


# plt.hlines(np.sin(theta1), -1, 1)
# Plot edge of coupon
# plt.hlines(zcoupon, (-1.35 * 25.4), (1.35 * 25.4), color = 'black', linestyle='--', label = 'Coupon edge') # mm

# Note: Markersize is to scale, assuming dpi=144 (use fig3.get_dpi) and converting to dots per micron for 800 micron chord diameter

# # Lateral chords
for i in np.arange(len(chord_pos)):
    ind = np.abs(x1-chord_pos[i]).argmin()
    # plt.plot(chord_pos[i], -zloc,'b.', markersize = 2.27, fillstyle = 'none')
    plt.plot(chord_pos[i], -zloc,'b.', markersize = 10)

plt.text(-4, -2, 'Chord 20 $\longrightarrow$ Chord 1',bbox=dict(facecolor='white', edgecolor='black'))
# plt.text(5, -2, 'Chord 1',bbox=dict(facecolor='white', edgecolor='black'))

# Axial chords, top down
# for i in np.arange(len(chord_pos)):
#     plt.plot(0, chord_pos[i],'b.', markersize = 2.27, fillstyle = 'none')

# # Axial chords, side
# for i in np.arange(len(chord_pos)):
#     plt.axhline(chord_pos[i])
#     plt.plot(0, chord_pos[i],'b.')



# Set the aspect of the plot to be equal
# plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('Impact parameter [mm]')
plt.ylabel('z [mm]')
# plt.title('Nose cone profile with chords')
# plt.grid()
# plt.legend()
plt.xlim([-12, 12])
plt.ylim([-3, 2])

# plt.xlim([-11.8, 11.8])


## Find angles between surface and chords

# chord1 = np.array([])

#%% Plot spot size of circle and ellipse


# # Create an array of angles from 0 to pi (for the upper semicircle)
# th2 = np.linspace(0, 2*np.pi, 1000)

# # The radius of the circular spot size made by 800 micron diameter chord
# r_circ = 400 # 400 microns is 0.4 mm

# # Define the equations of the circle at this radius
# x_circ = r_circ*np.cos(th2)
# y_circ = r_circ*np.sin(th2)

# # explicit circle 
# # Circle equation explicit

# X_circ = np.linspace(-r_circ, r_circ, 50)

# Y_circ = np.sqrt((r_circ)**2 - X2**2)

# ang = 60 * (np.pi/180) # angle of surface normal with chord, radians

# # Area of circle
# A_circ = np.pi * r_circ**2

# # Area of ellipse
# A_ell = np.pi * r_circ**2 * np.cos(np.radians(angle_degrees))

# # Calculate area factor
# A_fac = A_ell / A_circ

# fig4 = plt.figure()

# # Set the aspect of the plot to be equal
# plt.gca().set_aspect('equal')

# # plot semicircle
# plt.plot(x_circ,y_circ,'k')
# plt.xlabel('x (mm)')
# plt.ylabel('z (mm)')
# plt.title('Nose cone profile with chords')
# plt.grid()

#%% Halo calculations

# # Create an array of angles from 0 to pi (for the upper semicircle)
# th = np.linspace(0, np.pi, 1000)

# # The radius of the halo plasma in mm
# r_halo = 12

# # Equation of halo circle

# X_halo = np.linspace(-r_halo, r_halo, 500)

# Y_halo = np.sqrt((r_halo)**2 - X_halo**2)

# # The radius of the cross-section at the z position of interest
# r2 = 0.5 * get_diam(zloc)[2]

# # Circle equation explicit

# X2 = np.linspace(-r2, r2, 500)

# Y2 = np.sqrt((r2)**2 - X2**2)

# chord_pos = np.arange(-23.6/2, 23.6/2, 1.24)

# def Ycirc(x, r):
#     "x is the domain for values of the circle in explicit equation form. r is the radius of the circle being used."
#     Y2 = np.sqrt((r)**2 - x**2)
#     return Y2

# # Half of path length of each chord through the halo
# r_chord = np.zeros([len(chord_pos)])

# fig = plt.figure(figsize=(5,5))

# for ii in np.arange(len(chord_pos)):
#     r_chord[ii] = Ycirc(chord_pos[ii], r_halo)
    
#     plt.plot(chord_pos[ii], r_chord[ii], 'x')
#     plt.vlines(chord_pos[ii], -r_chord[ii], r_chord[ii])

# plt.plot(X2, Y2, 'k', label = 'Nose cone')
# plt.plot(X2,-Y2, 'k')

# plt.plot(X_halo, Y_halo, 'r--', label = 'Halo')
# plt.plot(X_halo, -Y_halo, 'r--')


# plt.ylim([-15, 15])
# plt.xlim([-15,15])
# plt.legend()

# # Get path lengths through halo for each chord
# halo_chord = 2 * r_chord

# # Half of path length each chord passes through the nose cone cross-section
# r_thru = np.zeros([len(chord_pos)])

# for ii in np.arange(len(chord_pos)):
#     r_thru[ii] = Ycirc(chord_pos[ii], r2)

    
#     plt.plot(chord_pos[ii], r_thru[ii], 'x')
#     plt.vlines(chord_pos[ii], -r_thru[ii], r_thru[ii], 'red')
    
# r_thru[np.isnan(r_thru)] = 0

# # Subtract the path length through nose cone c-s from the overall path length through halo
# halo_path_r = halo_chord - 2*r_thru

# # Get proportion of diameter of chord path length through halo:
# halo_path_prop = halo_path_r/(2 * r_halo)

# # Take half of the path lengths that go through the nose cone c-s

# halo_path = halo_path_r

# for ii in np.arange(4,16):
#     halo_path[ii] = 0.5 * halo_path_r[ii]
    
# # for ii in np.arange(len(chord_pos)):
#     # plt.vlines(chord_pos[ii], r_thru[ii], r_thru[ii] + halo_path[ii], 'green')

