clc;
clear;

%% Load and preprocess data
trainLen = 900;    % Training data length
testLen = 300;     % Test data length
initLen = 30;      % Initial run length

% Load and preprocess data for wind turbine 1
data_wind1 = loadAndPreprocessData('dataset/wtbdata_cleaned123.csv');
[data1, data_mean1, data_std1] = normalizeAndDifferentiate(data_wind1);

% Load and preprocess data for wind turbine 2
data_wind2 = loadAndPreprocessData('dataset/wtbdata_cleaned122.csv');
[data2, data_mean2, data_std2] = normalizeAndDifferentiate(data_wind2);

% Select a subset of data for both turbines
data = data1(1:6000);
data_new = data2(1:6000);

%% Generate the ESN reservoir
inputSize = 1;     % Number of input nodes
outputSize = 1;    % Number of output nodes
reservoirSize = 200; % Number of reservoir nodes
leakingRate = 0.99; % Leaking rate

% Initialize input weights and reservoir weights
inputWeights = (rand(reservoirSize, 1 + inputSize) - 0.5) * 1;
reservoirWeights = rand(reservoirSize, reservoirSize) - 0.5;

% Normalize reservoir weights
disp('Computing spectral radius...');
opt.disp = 0;
spectralRadius = abs(eigs(reservoirWeights, 1, 'LM', opt));
reservoirWeights = reservoirWeights * (0.99 / spectralRadius);
disp('done.');

%% Run the reservoir and collect states
targetData1 = data(initLen+2:trainLen+1)';
targetData2 = data_new(initLen+2:trainLen+1)';

[stateMatrix1, stateMatrix2] = collectReservoirStates(data, data_new, inputWeights, reservoirWeights, leakingRate, trainLen, initLen, inputSize, reservoirSize);

%% Train the output weights
regularization = 1e-8;  % Regularization coefficient
outputWeights = calculateOutputWeights(stateMatrix1, stateMatrix2, targetData1, targetData2, regularization);

%% Generate predictions
[predictions1, predictions2] = generatePredictions(data, data_new, inputWeights, reservoirWeights, outputWeights, leakingRate, trainLen, testLen, inputSize, reservoirSize);

%% Evaluate performance
[mae1, rmse1] = evaluatePerformance(predictions1, data, trainLen, testLen, data_mean1, data_std1);
[mae2, rmse2] = evaluatePerformance(predictions2, data_new, trainLen, testLen, data_mean2, data_std2);

% Display results
disp(['MAE1 = ', num2str(mae1), ', RMSE1 = ', num2str(rmse1)]);
disp(['MAE2 = ', num2str(mae2), ', RMSE2 = ', num2str(rmse2)]);

%% Helper functions
function data = loadAndPreprocessData(filename)
    rawData = readtable(filename);
    data = table2array(rawData(2:end, 4));
    data = data(~isnan(data));
end

function [normalizedData, dataMean, dataStd] = normalizeAndDifferentiate(data)
    dataMean = mean(data(1:10000));
    dataStd = std(data(1:10000));
    normalizedData = normalize(data);
    normalizedData = diff(normalizedData);
    normalizedData = seriesDecomp(normalizedData, 3);
end

function [stateMatrix1, stateMatrix2] = collectReservoirStates(data, data_new, inputWeights, reservoirWeights, leakingRate, trainLen, initLen, inputSize, reservoirSize)
    stateMatrix1 = zeros(1 + inputSize + reservoirSize, trainLen - initLen);
    stateMatrix2 = zeros(1 + inputSize + reservoirSize, trainLen - initLen);
    state1 = zeros(reservoirSize, 1);
    state2 = zeros(reservoirSize, 1);
    
    for t = 1:trainLen
        u1 = data(t);
        u2 = data_new(t);
        state1 = (1 - leakingRate) * state1 + leakingRate * tanh(inputWeights * [1; u1] + reservoirWeights * state1);
        state2 = (1 - leakingRate) * state2 + leakingRate * tanh(inputWeights * [1; u2] + reservoirWeights * state2);
        
        if t > initLen
            stateMatrix1(:, t - initLen) = [1; u1; state1];
            stateMatrix2(:, t - initLen) = [1; u2; state2];
        end
    end
end

function outputWeights = calculateOutputWeights(stateMatrix1, stateMatrix2, targetData1, targetData2, regularization)
    combinedStates = stateMatrix1 * stateMatrix1' + stateMatrix2 * stateMatrix2';
    combinedTargets = stateMatrix1 * targetData1' + stateMatrix2 * targetData2';
    outputWeights = (regularization * eye(size(combinedStates)) + combinedStates) \ combinedTargets;
end

function [predictions1, predictions2] = generatePredictions(data, data_new, inputWeights, reservoirWeights, outputWeights, leakingRate, trainLen, testLen, inputSize, reservoirSize)
    predictions1 = zeros(1, testLen);
    predictions2 = zeros(1, testLen);
    state1 = zeros(reservoirSize, 1);
    state2 = zeros(reservoirSize, 1);
    
    for t = 1:testLen
        u1 = data(trainLen + t);
        u2 = data_new(trainLen + t);
        state1 = (1 - leakingRate) * state1 + leakingRate * tanh(inputWeights * [1; u1] + reservoirWeights * state1);
        state2 = (1 - leakingRate) * state2 + leakingRate * tanh(inputWeights * [1; u2] + reservoirWeights * state2);
        predictions1(t) = outputWeights' * [1; u1; state1];
        predictions2(t) = outputWeights' * [1; u2; state2];
    end
end

function [mae, rmse] = evaluatePerformance(predictions, data, trainLen, testLen, dataMean, dataStd)
    output = predictions * dataStd + dataMean;
    ytest = data(trainLen+2:trainLen+testLen+1)' * dataStd + dataMean;
    mae = mean(abs(ytest - output));
    rmse = sqrt(mean((ytest - output).^2));
end
