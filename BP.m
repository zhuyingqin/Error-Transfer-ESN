%% Initialization
clear; clc; close all;

%% Load and preprocess data
trainLength = 500;  % Training data length
testLength = 100;   % Test data length
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

% Save the trained model
save('bp_model.mat', 'net');

%% Test BP neural network
% Compute training error
yPredTrain = sim(net, xTrain);
mseTrain = mean((yPredTrain - yTrain).^2);

% Make predictions on test set
yPredTest = sim(net, xTest);

% Denormalize predictions and actual values
yPredTestDenorm = yPredTest * dataStd + dataMean;
yTestDenorm = yTest * dataStd + dataMean;

% Evaluate performance
mae = mean(abs(yTestDenorm - yPredTestDenorm));
rmse = sqrt(mean((yTestDenorm - yPredTestDenorm).^2));

% Display results
disp(['MAE = ', num2str(mae)]);
disp(['RMSE = ', num2str(rmse)]);
