%% Initialization
clear; clc; close all;

%% Load and preprocess data
trainLength = 1000;  % Training data length
testLength = 300;   % Test data length
predictionInterval = 1;  % Prediction interval

% Load data
rawData = readtable("dataset/wtbdata_cleaned1.csv");
windData = table2array(rawData(2:end, 4));
windData = windData(~isnan(windData));

% Normalize data
dataMean = mean(windData(1:10000));
dataStd = std(windData(1:10000));
normalizedData = (windData - dataMean) / dataStd;

% Preprocess data
processedData = diff(normalizedData);
processedData = seriesDecomp(processedData, 3);

% Prepare input and target data
inputData = processedData(1:end-1);
targetData = processedData(2:end);

% Split data into training and testing sets
xTrain = inputData(1:trainLength)';
yTrain = targetData(predictionInterval:trainLength + predictionInterval - 1)';

xTest = inputData(trainLength + 1:trainLength + testLength)';
yTest = targetData(trainLength + predictionInterval:trainLength + testLength + predictionInterval - 1)';

%% Run multiple times and calculate average metrics
num_runs = 20;
mae_runs = zeros(1, num_runs);
rmse_runs = zeros(1, num_runs);
time_runs = zeros(1, num_runs);

for run = 1:num_runs
    tic; % Start timing
    
    %% Build BP neural network
    hiddenLayerSize = 100;
    net = newff(xTrain, yTrain, hiddenLayerSize);

    % Set network parameters
    net.trainParam.epochs = 10000;  % Maximum training epochs
    net.trainParam.lr = 0.001;      % Learning rate
    net.trainParam.goal = 0.00001;  % Training goal (minimum error)
    net.trainFcn = 'traingd';       % Training function (gradient descent)

    %% Train BP neural network
    net = train(net, xTrain, yTrain);

    %% Test BP neural network
    % Make predictions on test set
    yPredTest = sim(net, xTest);

    % Denormalize predictions and actual values
    yPredTestDenorm = yPredTest * dataStd + dataMean;
    yTestDenorm = yTest * dataStd + dataMean;

    % Evaluate performance
    mae_runs(run) = mean(abs(yTestDenorm - yPredTestDenorm));
    rmse_runs(run) = sqrt(mean((yTestDenorm - yPredTestDenorm).^2));
    
    time_runs(run) = toc; % End timing
    
    fprintf('Run %d: MAE = %.4f, RMSE = %.4f, Time = %.4f seconds\n', run, mae_runs(run), rmse_runs(run), time_runs(run));
end

%% Calculate and display average metrics
avg_mae = mean(mae_runs);
avg_rmse = mean(rmse_runs);
avg_time = mean(time_runs);

disp('----------------------------');
disp(['Average MAE = ', num2str(avg_mae)]);
disp(['Average RMSE = ', num2str(avg_rmse)]);
disp(['Average Runtime = ', num2str(avg_time), ' seconds']);

%% Plot results (using the last run)
figure;
plot(yTestDenorm, 'b', 'LineWidth', 2);
hold on;
plot(yPredTestDenorm, 'r', 'LineWidth', 2);
legend('Actual', 'Predicted');
title('BP Neural Network Wind Speed Prediction');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;
