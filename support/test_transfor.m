function [output_seq] = test_transfor(data, data_new, washout, i, lengthl)
    %% Load data
    trainLen = length(data) - 1;
    data = data(1:end, i);
    data_new = data_new(1:end, i);

    %% Generate ESN reservoir
    inSize = 2;                                 % Number of input nodes
    resSize = 200;                              % Number of reservoir nodes
    leakingRate = 0.1;                          % Leaking rate

    %% Initialize weight matrices
    Win = (rand(resSize, 1+inSize) - 0.5) * 1;  % Input weights
    W = rand(resSize, resSize) - 0.5;           % Reservoir internal weights

    %% Normalize and set spectral radius
    opt.disp = 0;
    rhoW = abs(eigs(W, 1, 'LM', opt));          % Calculate spectral radius
    W = W * (0.99 / rhoW);                      % Adjust spectral radius

    %% Run reservoir and collect states
    x = zeros(resSize, 1);                      % Initialize reservoir state [Target domain A]
    P = eye(resSize + inSize + 1) / 1;          % Initialize P matrix
    lambda = 0.99;                              % Forgetting factor
    output_weight = zeros(1 + inSize + resSize, 1);  % Initialize output weights
    u = [data(1), data_new(1)];                 % Initialize input

    output_seq = zeros(1, trainLen);            % Pre-allocate output sequence
    output_seq_new = zeros(1, trainLen);        % Pre-allocate new output sequence
    error = zeros(1, trainLen);                 % Pre-allocate error sequence
    error_new = zeros(1, trainLen);             % Pre-allocate new error sequence

    for t = 1:trainLen
        % Update reservoir state
        x = (1-leakingRate)*x + leakingRate*tanh(Win*[1; u'] + W*x);
        
        % Construct extended state vector
        X = [1; u'; x];
        
        % Calculate output and error
        output_seq(t) = X' * output_weight;
        output_seq_new(t) = output_seq(t);  % In this implementation, both outputs are the same
        
        error(t) = data(t+1) - output_seq(t);
        error_new(t) = data_new(t+1) - output_seq_new(t);
        
        % Update weights (online learning)
        PX = P * X;
        denom = lambda + X' * PX;
        k = PX / denom;
        
        if t >= washout  % Use washout parameter
            output_weight = output_weight + k * (error(t) + error_new(t));
        end
        
        % Update P matrix
        P = (P - k * PX') / lambda;
        
        % Update input
        if mod(t, 100) == 0
            u = [data(t+1), data_new(t+1)];
        else
            u = [output_seq(t), output_seq_new(t)];
        end
    end
end
