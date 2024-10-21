clc;
clear;

%% Load and preprocess data
trainLen = 1000;    % Training data length
testLen  = 300;     % Test data length
initLen  = 50;      % Initial run length

% Load and preprocess data for wind turbine 1
data_wind1 = loadAndPreprocessData('dataset/wtbdata_cleaned123.csv');
[data1, data_mean1, data_std1] = normalizeAndDifferentiate(data_wind1);

% Load and preprocess data for wind turbine 2
data_wind2 = loadAndPreprocessData('dataset/wtbdata_cleaned2.csv');
[data2, data_mean2, data_std2] = normalizeAndDifferentiate(data_wind2);

% Select a subset of data for both turbines
data     = data1(1:4000, :);
data_new = data2(1:6000, :);

%% Generate the ESN reservoir
inputSize = 2;     % Number of input nodes
outputSize = 1;    % Number of output nodes
reservoirSize = 200; % Number of reservoir nodes
leakingRate = 0.8; % Leaking rate

%% Run multiple times and calculate average metrics
num_runs   = 20;
mae1_runs  = zeros(1, num_runs);
rmse1_runs = zeros(1, num_runs);
mae2_runs  = zeros(1, num_runs);
rmse2_runs = zeros(1, num_runs);
time_runs  = zeros(1, num_runs);

for run = 1:num_runs
    tic; % Start timing

    % Initialize input weights and reservoir weights
    inputWeights = (rand(reservoirSize, 1 + inputSize) - 0.5) * 1;
    reservoirWeights = rand(reservoirSize, reservoirSize) - 0.5;

    % Normalize reservoir weights
    opt.disp = 0;
    spectralRadius = abs(eigs(reservoirWeights, 1, 'LM', opt));
    reservoirWeights = reservoirWeights * (0.9 / spectralRadius);

    %% Run the reservoir and collect states
    targetData1 = data(initLen+2:trainLen+1);
    targetData2 = data_new(initLen+2:trainLen+1);

    [stateMatrix1, stateMatrix2] = collectReservoirStates(data, data_new, ...
        inputWeights, reservoirWeights, leakingRate, trainLen, initLen, ...
        inputSize, reservoirSize);

    %% Train the output weights
    regularization = 1e-5;  % Regularization coefficient
    outputWeights = calculateOutputWeights(stateMatrix1, stateMatrix2, targetData1, targetData2, regularization);

    %% Generate predictions
    [predictions1, predictions2] = generatePredictions(data, data_new, inputWeights, reservoirWeights, outputWeights, leakingRate, trainLen, testLen, inputSize, reservoirSize);

    %% Evaluate performance
    [mae1, rmse1] = evaluatePerformance(predictions1, data, trainLen, testLen, data_mean1, data_std1);
    [mae2, rmse2] = evaluatePerformance(predictions2, data_new, trainLen, testLen, data_mean2, data_std2);
    
    mae1_runs(run) = mae1;
    rmse1_runs(run) = rmse1;
    mae2_runs(run) = mae2;
    rmse2_runs(run) = rmse2;
    
    time_runs(run) = toc; % End timing
    
    fprintf('Run %d: MAE1 = %.4f, RMSE1 = %.4f, MAE2 = %.4f, RMSE2 = %.4f, Time = %.4f seconds\n', ...
            run, mae1, rmse1, mae2, rmse2, time_runs(run));
end

%% Calculate and display average metrics
avg_mae1 = mean(mae1_runs);
avg_rmse1 = mean(rmse1_runs);
avg_mae2 = mean(mae2_runs);
avg_rmse2 = mean(rmse2_runs);
avg_time = mean(time_runs);

disp('----------------------------');
disp(['Average MAE1 = ', num2str(avg_mae1)]);
disp(['Average RMSE1 = ', num2str(avg_rmse1)]);
disp(['Average MAE2 = ', num2str(avg_mae2)]);
disp(['Average RMSE2 = ', num2str(avg_rmse2)]);
disp(['Average Runtime = ', num2str(avg_time), ' seconds']);

%% Plot results (using the last run)
figure;
subplot(2,1,1);
plot(predictions1, 'r', 'LineWidth', 2);
hold on;
plot(data(trainLen+2:trainLen+testLen+1), 'b', 'LineWidth', 2);
legend('Predicted', 'Actual');
title('Wind Turbine 1 Prediction');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;

subplot(2,1,2);
plot(predictions2, 'r', 'LineWidth', 2);
hold on;
plot(data_new(trainLen+2:trainLen+testLen+1), 'b', 'LineWidth', 2);
legend('Predicted', 'Actual');
title('Wind Turbine 2 Prediction');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;

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

function [stateMatrix1, stateMatrix2] = collectReservoirStates(data, ...
    data_new, inputWeights, reservoirWeights, leakingRate, trainLen, ...
    initLen, inputSize, reservoirSize)

    stateMatrix1 = zeros(1 + reservoirSize, trainLen - initLen);
    stateMatrix2 = zeros(1 + reservoirSize, trainLen - initLen);
    state1 = zeros(reservoirSize, 1);
    state2 = zeros(reservoirSize, 1);
    
    for t = 1:trainLen
        u1 = data(t, :);
        u2 = data_new(t, :);
        state1 = (1 - leakingRate) * state1 + leakingRate ...
                    * tanh(inputWeights * [1 u1]' + reservoirWeights * state1);
        state2 = (1 - leakingRate) * state2 + leakingRate ...
                    * tanh(inputWeights * [1 u2]' + reservoirWeights * state2);
        
        if t > initLen
            stateMatrix1(:, t - initLen) = [1 state1'];
            stateMatrix2(:, t - initLen) = [1 state2'];
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
        u1 = data(trainLen + t, :);
        u2 = data_new(trainLen + t, :);
        state1 = (1 - leakingRate) * state1 + leakingRate ...
                  * tanh(inputWeights * [1 u1]' + reservoirWeights * state1);
        state2 = (1 - leakingRate) * state2 + leakingRate ...
                  * tanh(inputWeights * [1 u2]' + reservoirWeights * state2);
        predictions1(t) = outputWeights' * [1 state1']';
        predictions2(t) = outputWeights' * [1 state2']';
    end
end

function [mae, rmse] = evaluatePerformance(predictions, data, trainLen, testLen, dataMean, dataStd)
    output = predictions * dataStd + dataMean;
    ytest = data(trainLen+2:trainLen+testLen+1)' * dataStd + dataMean;
    output = output(30:end);
    ytest  = ytest(30:end);
    mae = mean(abs(ytest - output'));
    rmse = sqrt(mean((ytest - output').^2));
end
