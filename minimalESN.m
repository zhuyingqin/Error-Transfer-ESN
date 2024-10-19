% A minimalistic Echo State Network (ESN) demo with wind speed data
% Adapted from Mantas Lukosevicius 2012-2018
% http://mantas.info
% Optimized and adapted for wind speed prediction

clc;
clear;

%% Load and preprocess data
trainLen = 500;    % Length of training dataset
testLen = 100;     % Length of test dataset
initLen = 30;      % Initial reservoir state initialization length

% Load wind speed data
data_wind = readtable("dataset/wtbdata_cleaned123.csv");
data_wind = table2array(data_wind(2:end, 4));
data_wind = data_wind(~isnan(data_wind));

% Normalize data
data_mean = mean(data_wind(1:10000));
data_std = std(data_wind(1:10000));
data = normalize(data_wind);
data = diff(data);
data = seriesDecomp(data, 3);

%% Generate the ESN reservoir
inSize = 1;        % Number of input nodes
outSize = 1;       % Number of output nodes
resSize = 200;     % Number of reservoir nodes
leakingRate = 0.99;% Leaking rate

% Initialize weights
rng(42);           % Set random seed for reproducibility
Win = (rand(resSize, 1+inSize) - 0.5) * 1;  % Input weights
W = rand(resSize, resSize) - 0.5;           % Reservoir weights

% Normalize and set spectral radius
disp('Computing spectral radius...');
opt.disp = 0;
rhoW = abs(eigs(W, 1, 'LM', opt));
W = W * (0.99 / rhoW);
disp('done.');

%% Collect states
X = zeros(1+inSize+resSize, trainLen-initLen);
Yt = data(initLen+2:trainLen+1)';  % Target output

% Run the reservoir with the data and collect states
x = zeros(resSize, 1);
for t = 1:trainLen
    u = data(t);
    x = (1-leakingRate) * x + leakingRate * tanh(Win*[1; u] + W*x);
    if t > initLen
        X(:, t-initLen) = [1; u; x];
    end
end

%% Train the output weights
reg = 1e-3;  % Regularization coefficient
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))';

%% Test the trained ESN
Y = zeros(outSize, testLen);
u = data(trainLen+1);
for t = 1:testLen 
    x = (1-leakingRate)*x + leakingRate*tanh(Win*[1; u] + W*x);
    y = Wout * [1; u; x];
    Y(:, t) = y;
    u = data(trainLen+t+1);  % Teacher forcing
end

%% Denormalize and compute error metrics
output = Y * data_std + data_mean;
ytest = data(trainLen+2:trainLen+testLen+1)' * data_std + data_mean;

mae = mean(abs(ytest - output));
rmse = sqrt(mean((ytest - output).^2));

disp(['MAE = ', num2str(mae)]);
disp(['RMSE = ', num2str(rmse)]);

%% Plot results
figure;
plot(ytest, 'b', 'LineWidth', 2);
hold on;
plot(output, 'r', 'LineWidth', 2);
legend('Actual', 'Predicted');
title('ESN Wind Speed Prediction');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;
