clc; clear;

%% Load and preprocess data
trainLength = 500;
testLength = 100;
initLength = 30;

% Load data from file
try
    rawData = readtable("dataset/wtbdata_cleaned1.csv");
    data = table2array(rawData(2:end, 4));
    data = data(~isnan(data));
catch
    error('Unable to load the file. Please ensure "dataset\wtbdata_cleaned1.csv" exists in the correct directory.');
end

% Normalize the data
dataMean = mean(data(1:10000));
dataStd = std(data(1:10000));
data = (data - dataMean) / dataStd;

% Prepare input and target data
inputData = data(1:end-1);
targetData = data(2:end);
predictionInterval = 1;

% Split data into training and testing sets
xTrain = inputData(1:trainLength)';
yTrain = targetData(1 + predictionInterval:trainLength + predictionInterval)';

xTest = inputData(trainLength + 1:trainLength + testLength)';
yTest = targetData(trainLength + 1 + predictionInterval:trainLength + testLength + predictionInterval)';

%% Define LSTM network structure
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.95, ...
    'Verbose', 0, ...
    'Plots', 'none');

%% Run multiple times and calculate average metrics
num_runs = 20;
mae_runs = zeros(1, num_runs);
rmse_runs = zeros(1, num_runs);
time_runs = zeros(1, num_runs);

for run = 1:num_runs
    tic; % Start timing
    
    %% Train the model
    net = trainNetwork(xTrain, yTrain, layers, options);

    %% Make predictions on test set
    net = resetState(net);
    numTimeStepsTest = size(xTest, 2);
    yPredTest = zeros(numTimeStepsTest, 1);

    for i = 1:numTimeStepsTest
        [net, yPredTest(i)] = predictAndUpdateState(net, xTest(:, i), 'ExecutionEnvironment', 'cpu');
    end

    %% Evaluate performance
    mae_runs(run) = mean(abs(yTest - yPredTest'));
    rmse_runs(run) = sqrt(mean((yTest - yPredTest').^2));
    
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
plot(yTest, 'b', 'LineWidth', 2);
hold on;
plot(yPredTest, 'r', 'LineWidth', 2);
legend('Actual', 'Predicted');
title('LSTM Network Performance');
xlabel('Time Step');
ylabel('Normalized Wind Speed');
grid on;
