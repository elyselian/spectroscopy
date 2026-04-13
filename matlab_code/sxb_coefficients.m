function sxb_value = sxb_coefficients(Tempq, Densityq)

% Tempq = electron_temp;
% Densityq = electron_density;

% import and clean up data
adas_data = readmatrix('/Users/elyselian/Downloads/sxb coefficients - Sheet1.csv');
adas_data = transpose(adas_data(:,2:end));

% sort all data
temperature = adas_data(:,1); %eV
density = adas_data(:,2); %cm^-3
sxb = zeros([24 24]);
for i = 3:7
    sxb(:,19+(i-2)) = adas_data(:,i);
end

log_density = log10(density);

Densitylogq = log10(Densityq);
sxb_value = griddata(log_density, temperature, sxb, Densitylogq, Tempq);

end