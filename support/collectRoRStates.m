function[final_states,individual] = collectRoRStates(individual, ...
                                    input_sequence, input_columns, config)

states = zeros(size(input_sequence,1), individual.nodes(1));

% Equation: x(n) = f(Win*u(n) + S)
% Add input_columns as a parameter to the function
input_sequence = input_sequence(:, input_columns);
for n = 2: size(input_sequence,1) 
    states(n,:) = (1 - individual.leak_rate(1)) * states(n, :)' ...
        + individual.leak_rate(1) ...
        * tanh((individual.input_weights{1} * individual.input_scaling(1) ...
        * ([individual.bias_node input_sequence(n, :)])') ...
        + (individual.W{1, 1} * states(n-1, :)'));
end


% concat all states for output weights
final_states = [];
for i= 1:config.num_reservoirs
    final_states = [final_states states];
    
    %assign last state variable
    individual.last_state = states(end,:);
end

% concat input states
if config.add_input_states == 1
    final_states = [final_states input_sequence];
end

if size(input_sequence,1) == 2
    final_states = final_states(end, :); % remove washout
else
    final_states = final_states(config.wash_out+1:end, :); % remove washout
end
