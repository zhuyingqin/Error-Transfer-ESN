function [output_matrix] = test_transfor(data, data_new, washout, i)
    %% Load data
    trainLen = length(data) - 1;
    data = data(1:end, i);
    data_new = data_new(1:end, i);

    %% Generate ESN reservoir
    inSize = 2;                                 % Number of input nodes
    resSize = 100;                              % Number of reservoir nodes
    leakingRate = 0.9;                          % Leaking rate

    %% Initialize weight matrices
    Win = (rand(resSize, 1+inSize) - 0.5) * 1;  % Input weights
    W = rand(resSize, resSize) - 0.5;           % Reservoir internal weights

    %% Normalize and set spectral radius
    opt.disp = 0;
    rhoW = abs(eigs(W, 1, 'LM', opt));          % Calculate spectral radius
    W = W * (1.25 / rhoW);                      % Adjust spectral radius

    %% Run reservoir and collect states
    x = zeros(resSize, 1);                      % Initialize reservoir state [Target domain A]
    
    % Initialize RLS parameters
    P = eye(resSize + inSize + 1) / 1e-4;       % Initialize P matrix
    lambda = 0.99;                              % Forgetting factor
    output_weight = zeros(resSize + inSize + 1, 2);  % Initialize output weights
    
    output_matrix = zeros(2, trainLen);         % Pre-allocate output matrix

    for t = 1:trainLen
        u = [data(t), data_new(t)];
        % Update reservoir state
        x = (1-leakingRate) * x + leakingRate * tanh(Win * [1; u'] + W * x);
        
        % Construct extended state vector
        X = [1; u'; x];
        
        % RLS update
        if t >= washout
            % Compute output
            output_matrix(:, t) = output_weight' * X;
            
            % Compute error
            e = [data(t+1); data_new(t+1)] - output_matrix(:, t);
            
            % Update weights
            k = P * X / (lambda + X' * P * X);
            output_weight = output_weight + k * e';
            P = (P - k * X' * P) / lambda;
        else
            output_matrix(:, t) = [0; 0];
        end
    end
end
