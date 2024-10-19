function[final_states,individual] = collectRoRStates(individual,input_sequence,config)

% if single input entry, add previous state
% 如果是单个输入条目，请添加先前状态
if size(input_sequence,1) == 2
    input_sequence = [zeros(size(input_sequence)); input_sequence];
end

for i= 1:config.num_reservoirs
    if size(input_sequence,1) == 2
        states{i} = individual.last_state{i};
    else
        states{i} = zeros(size(input_sequence,1),individual.nodes(i));%状态矩阵
    end
    x{i} = zeros(size(input_sequence,1),individual.nodes(i));%
end

% preassign activation function calls
% 预分配激活功能调用【未启用】 
if config.multi_activ               % size(individual.activ_Fcn,2) > 1
    for i= 1:config.num_reservoirs
        for p = 1:length(config.activ_list)
            index{i,p} = findActiv({individual.activ_Fcn{i,:}},config.activ_list{p});
        end
    end
end

%equation: x(n) = f(Win*u(n) + S) 储层状态
for n = 2:size(input_sequence,1) % 输入u(n)
    
    for i= 1:config.num_reservoirs
        
        for k= 1:config.num_reservoirs %储备池个数
            % x{i}(n,:) = x{i}(n,:) + ((individual.W{i,k}*individual.W_scaling(i,k))*states{k}(n-1,:)')';
            x{i}(n,:) = x{i}(n,:) + (individual.W{i,k}*states{k}(n-1,:)')';
        end
        
        if config.multi_activ
            for p = 1:length(config.activ_list)
                if config.evolve_feedback_weights
                    % states{i}(n,index{i,p}) = config.activ_list{p}(((individual.input_weights{i}(index{i,p},:)*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])')+ x{i}(n,index{i,p})' + individual.feedback_weights*states{i}(n-1,:)*individual.output_weights(1:end-config.add_input_states,:));
                    states{i}(n,index{i,p}) = config.activ_list{p}(((individual.input_weights{i}(index{i,p},:)*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])')+ x{i}(n,index{i,p})' + individual.feedback_weights * input_sequence(n,:));
                else
                    states{i}(n,index{i,p}) = config.activ_list{p}(((individual.input_weights{i}(index{i,p},:)*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])')+ x{i}(n,index{i,p})');
                end
            end
        else
            if config.evolve_feedback_weights
                % states{i}(n,:) = individual.activ_Fcn{i}(((individual.input_weights{i}*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])') + x{i}(n,:)'+ (individual.feedback_scaling*individual.feedback_weights)*states{i}(n-1,:)*individual.output_weights(1:end-config.add_input_states,:));
                states{i}(n,:) = individual.activ_Fcn{i}(((individual.input_weights{i}*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])') + x{i}(n,:)'+ individual.feedback_weights * input_sequence(n,:));
                % states{i}(n,:) = config.activ_list{p}(((individual.input_weights{i}(index{i,p},:)*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])')+ x{i}(n,index{i,p})' + individual.feedback_weights * input_sequence(n,:));
            else
                % in = ((individual.input_weights{i}*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])');
                states{i}(n,:) = individual.activ_Fcn{i}(((individual.input_weights{i}*individual.input_scaling(i))*([individual.bias_node input_sequence(n,:)])')+ x{i}(n,:)');
            end
        end        
    end
end

% get leak states 获得泄漏率
if config.leak_on
    states = getLeakStates(states,individual,input_sequence,config);
end


% concat all states for output weights
final_states = [];
for i= 1:config.num_reservoirs
    final_states = [final_states states{i}];
    
    %assign last state variable
    individual.last_state{i} = states{i}(end,:);
end

% concat input states
if config.add_input_states == 1
    % 添加
    final_states = [final_states input_sequence];
    % final_states = [final_states circshift(input_sequence,1)];
    % final_states = [final_states circshift(input_sequence,2)];
end

if size(input_sequence,1) == 2
    final_states = final_states(end,:); % remove washout
else
    final_states = final_states(config.wash_out+1:end,:); % remove washout
end