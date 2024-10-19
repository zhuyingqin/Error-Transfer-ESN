clc;
clear;

% Set up parameters
trainLen = 500;                    % Length of training dataset
testLen = 100;                     % Length of test dataset
initLen = 0;                       % Initial length to discard

% Load and preprocess data
data_wind = readtable("dataset/wtbdata_cleaned123.csv");
data_wind = data_wind(2:end, 4);
data_wind = table2array(data_wind);
index = isnan(data_wind);
data_wind = data_wind(~index);

% Calculate mean and standard deviation
data_mean = mean(data_wind(1:10000));
data_std = std(data_wind(1:10000));

% Normalize data
data = normalize(data_wind);
data = diff(data);
data = seriesDecomp(data, 3);

% Prepare input and target data
inputs = data(1:trainLen)';
targets = data(initLen+2:trainLen+1)'; 

%% Create a RBF Network
goal = 0.000;   % Mean squared error goal (default = 0.0)
spread = 3;     % Spread of radial basis functions (default = 1.0)
MN = 200;       % Maximum number of neurons 
DF = 10;        % Number of neurons to add between displays (default = 25)

% Create and train the RBF network
net = newrb(inputs, targets, goal, spread, MN, DF);
view(net)

% Prepare test data
xtest = data(trainLen+1:trainLen+testLen)';
ytest = data(trainLen+2:trainLen+testLen+1)';

% Generate network output
output = net(xtest);

% Denormalize output and test data
output = output * data_std + data_mean;
ytest = ytest * data_std + data_mean;

% Calculate error
error = ytest - output;

% Calculate performance metrics
nrmse = sqrt((sum(((ytest-output).^2) / var(ytest))) * (1/length(ytest)));
disp(['NRMSE = ', num2str(nrmse)]);

mae = mae(ytest, output);
disp(['MAE = ', num2str(mae)]);

rmse = sqrt(mean((ytest-output).^2));
disp(['RMSE = ', num2str(rmse)]);

% Plot results
figure;
plot(ytest, 'b', 'LineWidth', 2);
hold on;
plot(output, 'r', 'LineWidth', 2);
legend('Actual', 'Predicted');
title('RBF Network Performance');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;
