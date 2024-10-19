clc;
clear;

%% Load and preprocess data
trainLen = 500;    % Training data length
testLen = 300;     % Test data length
initLen = 30;      % Initial run length

% Load and preprocess data for wind turbine 1
[data1, data_mean1, data_std1] = loadAndPreprocessData('dataset/wtbdata_cleaned73.csv');

% Load and preprocess data for wind turbine 2
[data2, data_mean2, data_std2] = loadAndPreprocessData('dataset/wtbdata_cleaned2.csv');

% Select a subset of data for both turbines
data = data1(1:6000);
data_new = data2(1:6000);

%% Generate the ESN reservoir
inputSize = 1;         % Number of input nodes
outputSize = 1;        % Number of output nodes
reservoirSize = 200;   % Number of reservoir nodes
leakingRate = 0.1;     % Leaking rate

% Initialize input weights and reservoir weights
inputWeights = (rand(reservoirSize, 1 + inputSize) - 0.5) * 1;
reservoirWeights = rand(reservoirSize, reservoirSize) - 0.5;

% Normalize reservoir weights
disp('Computing spectral radius...');
opt.disp = 0;
spectralRadius = abs(eigs(reservoirWeights, 1, 'LM', opt));
reservoirWeights = reservoirWeights * (0.99 / spectralRadius);
disp('done.');

%% Run the reservoir with the data and collect states
state = zeros(reservoirSize, 1);
state_new = zeros(reservoirSize, 1);
P = eye(reservoirSize + inputSize + 1) / 1;
forgettingFactor = 0.99;
outputWeights = zeros(1 + inputSize + reservoirSize, 1);
outputs = zeros(trainLen, 1);
outputs_new = zeros(trainLen, 1);

for t = 1:trainLen
    [state, state_new, outputWeights, P, output, output_new] = updateReservoirState(data(t), data_new(t), state, state_new, ...
        inputWeights, reservoirWeights, leakingRate, outputWeights, P, forgettingFactor, t, data, data_new);
    outputs(t) = output;
    outputs_new(t) = output_new;
end

%% Evaluate performance
[mae1, rmse1] = evaluatePerformance(data(2:trainLen+1), outputs, data_mean1, data_std1);
[mae2, rmse2] = evaluatePerformance(data_new(2:trainLen+1), outputs_new, data_mean2, data_std2);

% Display results
disp(['MAE1 = ', num2str(mae1), ', RMSE1 = ', num2str(rmse1)]);
disp(['MAE2 = ', num2str(mae2), ', RMSE2 = ', num2str(rmse2)]);

%% Helper functions
function [data, dataMean, dataStd] = loadAndPreprocessData(filename)
    rawData = readtable(filename);
    data = table2array(rawData(2:end, 4));
    data = data(~isnan(data));
    dataMean = mean(data(1:10000));
    dataStd = std(data(1:10000));
    data = normalize(data);
    data = diff(data);
    data = seriesDecomp(data, 3);
end

function [state, state_new, outputWeights, P, output, output_new] = updateReservoirState(input, input_new, state, state_new, ...
    inputWeights, reservoirWeights, leakingRate, outputWeights, P, forgettingFactor, t, data, data_new)
    
    % Update reservoir states
    state = (1 - leakingRate) * state + leakingRate * tanh(inputWeights * [1; input] + reservoirWeights * state);
    state_new = (1 - leakingRate) * state_new + leakingRate * tanh(inputWeights * [1; input_new] + reservoirWeights * state_new);
    
    % Combine states
    combinedState = [1; input; state];
    combinedState_new = [1; input_new; state_new];
    
    % RLS update
    PIn = P * [combinedState combinedState_new];
    denok = forgettingFactor * eye(2) + [combinedState combinedState_new]' * PIn;
    K = PIn / denok;
    
    % Compute output and error
    output = combinedState' * outputWeights;
    output_new = combinedState_new' * outputWeights;
    error = data(t + 1) - output;
    error_new = data_new(t + 1) - output_new;
    
    % Update output weights
    if t >= 0  % Enable transfer learning
        outputWeights = outputWeights + K * [error; error_new];
    else
        outputWeights = outputWeights + K * (error_new + error_new);
    end
    
    % Update P matrix
    P = P / forgettingFactor - K * PIn' / forgettingFactor;
end

function [mae, rmse] = evaluatePerformance(actual, predicted, dataMean, dataStd)
    output = predicted * dataStd + dataMean;
    ytest = actual * dataStd + dataMean;
    mae = mean(abs(ytest - output));
    rmse = sqrt(mean((ytest - output).^2));
end
