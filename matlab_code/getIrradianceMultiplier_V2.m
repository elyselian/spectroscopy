function irr = getIrradianceMultiplier_V2(row_intensity)

% row_intensity = fibers(1).rowintensity;
% getIrradianceFromFile calculates the irradiance multiplier matrix from a single .spe file.
%
% Input:
%   speFilePath - full path to a .spe file
%
% Output:
%   irr - irradiance multiplier matrix (units: µW/cm²/count)

% Load the SPE file
% sp = loadSPE(speFilePath);
% sp = loadSPE('/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Calibration/250422/250422  014.spe');

% % Extract intensity matrix
% intensity_matrix = sp.int; 
% 
% % Low-pass filter to reduce noise
% filtered_matrix = lowpass(intensity_matrix, 0.01, 1);
% 
% % Find row with maximum mean intensity
% row_means = mean(filtered_matrix, 2);
% [~, max_row_index] = max(row_means);

% Load calibration data
lamp_uv_data = readmatrix('/Users/elyselian/Downloads/DH3_Plus_Calibration_UV - Sheet1.csv');
wavelength_query = linspace(226.654, 232.726, 1024); 
wavelength = lamp_uv_data(:,1); % nm
irradiance_per_nm = lamp_uv_data(:,2); % µW/cm²/nm

% Convert calibration to match pixels
irradiance_q = interp1(wavelength, irradiance_per_nm, wavelength_query, 'spline') .* wavelength_query;

% Get counts from the brightest row
pixel = 1:1024;
% count_q = filtered_matrix(max_row_index, :);

% Fit counts to irradiance curve
% cal_curve = @(x) interp1(pixel, irradiance_q, x, 'spline');
% model = @(params, x_data) params(1) * cal_curve(x_data) + params(2);
% initial_guess = [2, 10];
% params_fit = lsqcurvefit(model, initial_guess, pixel, count_q);

% Get fitted counts
% count_fitted = model(params_fit, pixel);
% 
% p = polyfit(pixel, row_intensity, 5);
% count_fitted = polyval(p, pixel);
count_fitted = row_intensity;

% Compute irradiance multiplier
irr = irradiance_q ./ count_fitted;

%% 

% figure
% yyaxis left
% plot(pixel, count_q, 'b')
% hold on
% plot(pixel, count_fitted, 'k')
% hold off
% yyaxis right
% plot(pixel, irradiance_q)
% ylim([1400 1650])
