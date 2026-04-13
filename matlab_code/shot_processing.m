clear 
close all
clc

load('/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Code/fiber_regions.mat');

folderPath = '/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops';

speFiles = dir(fullfile(folderPath, '2504*', '*.spe'));

speFiles = speFiles(~contains({speFiles.name}, '-raw'));
speFiles = speFiles(~contains({speFiles.name}, 'b'));
speFiles = speFiles(~contains({speFiles.name}, 't'));

%%

fileData(length(speFiles)) = struct();

for i = 1:length(speFiles)
    fileName = speFiles(i).name;
    fullFilePath = fullfile(speFiles(i).folder, fileName);

    try
        [fileData(i).Intensity, ~, param] = loadSPE(fullFilePath);
        fileData(i).exposure = param.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse.Attributes.width;
        % intensity = imStruct.int;
        fileData(i).FileName = fileName;
        % fileData(i).FullPath = fullFilePath;
        fileData(i).Intensity = fileData(i).Intensity';

    catch ME
        warning('Error loading %s: %s', fileName, ME.message);
        fileData(i).FileName = fileName;
        fileData(i).FullPath = fullFilePath;
        fileData(i).Error = ME.message;
    end
end

%% 

clear activeMask bw regionValues 

averageIrradiance = zeros(10,1);

pixel = 1:1024;

% set default axis font size
set(groot,'defaultAxesFontSize',16)

% set path to shot lightfield files
% addpath(genpath('/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Users/Current/Elyse Lian/2025 ICCD'), '-begin') 

% initialize cell
sp = cell(1,1);

% import electron density and temperature profiles 
data = readmatrix('/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Users/Current/Elyse Lian/electron_density_temperature_profiles.csv');
radius = data(:,1);
electron_temp = data(:,3)*1e3;
electron_density = data(:,2);

% divide radial space into 1024 divisions for pixel space mapping
radius_pixel = linspace(0, 1.18, 1024/2)'; 

% interpolation for electron temp and density data 
electron_temp = interp1(radius, electron_temp, radius_pixel, 'spline');
electron_density = interp1(radius, electron_density, radius_pixel, 'spline');

% construct the radially symmetric profiles of radius, temp and density. negative radius simply denotes the "left side"
radius_pixel_profile = [-flip(radius_pixel); radius_pixel];
electron_temp_profile = [flip(electron_temp); electron_temp];
electron_density_profile = [flip(electron_density); electron_density];

% sxb_coefficients function give the sxb coefficient from a given electron temperature and density, this is done using spline interpolation
sxb = sxb_coefficients(electron_temp_profile, electron_density_profile);
sxb(isnan(sxb)) = 0;

%%

for file = 1:numel(fileData)
    shotImage = fileData(file).Intensity;
    brightnessThreshold = 0.4*max(shotImage);
    exposure = str2double(fileData(file).exposure) * 1e-9;
    gate_ratio = 20.999 / exposure;
    % centroid = zeros(numel(fibers),2);

    for i = 1: numel(fibers)
        bbox = round(fibers(i).BoundingBox);
        x1 = max(bbox(1), 1);
        y1 = max(bbox(2), 1);
        x2 = min(x1 + bbox(3) - 1, size(shotImage, 2));
        y2 = min(y1 + bbox(4) - 1, size(shotImage, 1));
        
        % Crop the irradiance and shot image to the bounding box
        irradiancepcount = fibers(i).irradiancepcount;
        irradianceImage = gate_ratio*shotImage.*repmat(irradiancepcount, 1024,1);
        localIrradiance = irradianceImage(y1:y2, x1:x2);
        localShot = shotImage(y1:y2, x1:x2);
    
        % Region mask from fiber struct
        % regionMask = localShot;  % logical matrix the size of cropped region
        bw = mat2gray(localShot);
        bw = bw.*(bw > 0.1*max(bw, [], "all"));
       
        bw = imbinarize(bw, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.3);
        
        % Remove small blobs
        bw = bwareaopen(bw, 70);
    
        % Label connected regions
        % labeledImage = logical(bw);
    
        % Apply brightness threshold to define the "active" area
        activeMask = logical(bw);

        % Average irradiance within the active region
        regionValues = localIrradiance.*activeMask;
        
        averageIrradiance(i) = mean(regionValues, "all");

        % centroid(i,:) = fibers(i).CentroidIndex;
    end
    % 
    % fileData(file).CentroidIndex = centroid;
    fileData(file).averageIrradiance = averageIrradiance;
    fileData(file).photonFlux = 4 * pi * averageIrradiance / (6.63e-34 * (3e8 / 229.7e-9));
    
end


%%
centroid = zeros(numel(fibers),2);
for i = 1:numel(fibers)
    centroid(i,:) = fibers(i).CentroidIndex;
end

idx = centroid(:,2);

sxb(isnan(sxb)) = 0;

radius_fiber = radius_pixel_profile(idx);

%%

erosion_rate = sxb(idx).*fileData(89).photonFlux; 

figure
plot(radius_fiber, erosion_rate, '-o')


%%
figure;
imshow(localShot, [], 'InitialMagnification', 'fit');
colormap(gray);
title(sprintf('Fiber %d - Active Region Overlay', i));
hold on;

% Create a red overlay with alpha where activeMask is true
redOverlay = cat(3, ones(size(activeMask)), zeros(size(activeMask)), zeros(size(activeMask)));

h = imshow(redOverlay);
set(h, 'AlphaData', 0.4 * activeMask);  % 0.4 = transparency level

hold off;

%% 
figure
imagesc(fileData(54).Intensity)
colorbar 
bw = mat2gray(fileData(i).Intensity);
bw = bw.*(bw > 0.1*max(bw, [], "all"));

bw = imbinarize(bw, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.3);
bw = bwareaopen(bw, 70);

figure
imagesc(bw)

%%


