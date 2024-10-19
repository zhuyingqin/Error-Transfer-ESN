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

% Prepare test data
xtest = data(trainLen+1:trainLen+testLen)';
ytest = data(trainLen+2:trainLen+testLen+1)';

%% Run multiple times and calculate average metrics
num_runs = 20;
nrmse_runs = zeros(1, num_runs);
mae_runs = zeros(1, num_runs);
rmse_runs = zeros(1, num_runs);
time_runs = zeros(1, num_runs);

for run = 1:num_runs
    tic; % Start timing
    
    %% Create a RBF Network
    goal = 0.000;   % Mean squared error goal (default = 0.0)
    spread = 3;     % Spread of radial basis functions (default = 1.0)
    MN = 200;       % Maximum number of neurons 
    DF = 10;        % Number of neurons to add between displays (default = 25)

    % Create and train the RBF network
    net = newrb(inputs, targets, goal, spread, MN, DF);

    % Generate network output
    output = net(xtest);

    % Denormalize output and test data
    output = output * data_std + data_mean;
    ytest_denorm = ytest * data_std + data_mean;

    % Calculate error
    error = ytest_denorm - output;

    % Calculate performance metrics
    nrmse_runs(run) = sqrt((sum(((ytest_denorm-output).^2) / var(ytest_denorm))) * (1/length(ytest_denorm)));
    mae_runs(run) = mean(abs(error));
    rmse_runs(run) = sqrt(mean(error.^2));
    
    time_runs(run) = toc; % End timing
    
    fprintf('Run %d: NRMSE = %.4f, MAE = %.4f, RMSE = %.4f, Time = %.4f seconds\n', ...
            run, nrmse_runs(run), mae_runs(run), rmse_runs(run), time_runs(run));
end

%% Calculate and display average metrics
avg_nrmse = mean(nrmse_runs);
avg_mae = mean(mae_runs);
avg_rmse = mean(rmse_runs);
avg_time = mean(time_runs);

disp('----------------------------');
disp(['Average NRMSE = ', num2str(avg_nrmse)]);
disp(['Average MAE = ', num2str(avg_mae)]);
disp(['Average RMSE = ', num2str(avg_rmse)]);
disp(['Average Runtime = ', num2str(avg_time), ' seconds']);

%% Plot results (using the last run)
figure;
plot(ytest_denorm, 'b', 'LineWidth', 2);
hold on;
plot(output, 'r', 'LineWidth', 2);
legend('Actual', 'Predicted');
title('RBF Network Performance');
xlabel('Time Step');
ylabel('Wind Speed');
grid on;
