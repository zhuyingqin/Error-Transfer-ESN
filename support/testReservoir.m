function [individual, test_states, test_sequence] = testReservoir(individual, config)
config.wash_out = 30;
washout = config.wash_out;
train_states  = config.assessFcn(individual, config.train_input_sequence, ...
                                [1,2], config);
val_states    = config.assessFcn(individual, config.val_input_sequence,   ...
                                [1,2], config);
train_states1 = config.assessFcn(individual, config.train_input_sequence, ...
                                [3,4], config);
val_states1   = config.assessFcn(individual, config.val_input_sequence,   ...
                                [3,4], config);
% 状态矩阵输出的

%% if W_out are evolved instead of trained

% load output.mat;
% Find best reg parameter
reg_train_error = [];
reg_val_error   = [];
reg_weights     = [];
reg_param       = [10e-1 10e-3 10e-4 10e-5 10e-6 10e-7];
error_collect   = [];
error_collect1  = [];
train_length    = 500;

for i = 1:length(reg_param)
    % Train: tanspose is inversed compared to equation 
    output_weights  = config.train_output_sequence(config.wash_out + 1:end, 1)'    ...
                      * train_states  * ((train_states' * train_states ...
                      + reg_param(i)  * eye(size(train_states' * train_states)))   ...
                      \ eye(size(train_states, 2)));
    output_weights1 = config.train_output_sequence(config.wash_out + 1:end, 2)'    ...
                      * train_states1 * ((train_states1' * train_states1 ...
                      + reg_param(i)  * eye(size(train_states1' * train_states1))) ...
                      \ eye(size(train_states1, 2)));

    %% 
    output_train_sequence  = train_states  * output_weights';
    output_train_sequence1 = train_states1 * output_weights1';
 
    %% Error tracker (parameters can be arbitrarily set)
    % Calculate training error: target value - predicted value
    error_trace   = config.train_output_sequence(config.wash_out + 1:end, 1) ... 
                    - output_train_sequence;  
    error_trace1  = config.train_output_sequence(config.wash_out + 1:end, 2) ...
                    - output_train_sequence1;
    
    % figure(3);
    % plot(config.train_output_sequence(config.wash_out + 1:end, 1));
    % hold on;
    % plot(output_train_sequence);
    % Shift errors to the right by one value
    error_trace   = [0; error_trace(1:end-1)];
    error_trace1  = [0; error_trace1(1:end-1)];
          
    % Error compensation
    output_error_sequence = test_transfor(error_trace, error_trace1, ...
                                          0, 1);
    output_error_sequence = output_error_sequence(:, 1:train_length)';
    
    % Apply error compensation
    output_train_sequence = output_train_sequence(1:train_length, :)... 
                            + output_error_sequence;
      
    % Calculate post-compensation NMSE error
    reg_train_error(i,:) = calculateError(output_train_sequence, ...
                        config.train_output_sequence(config.wash_out + 1 : ...
                        config.wash_out + train_length, 1), config);
            
    %% Validation set evaluation
    % Calculate validation set output
    output_val_sequence = val_states * output_weights';
    % Calculate validation set error
    reg_val_error(i,:)  = calculateError(output_val_sequence, config.val_output_sequence, config);
    % Save weights
    reg_weights(i,:,:)  = output_weights';
end
    [~, reg_indx] = min(sum(reg_val_error, 2));
    individual.train_error = sum(reg_train_error(reg_indx, :));
    individual.output_weights = reshape(reg_weights(reg_indx, :, :), size(reg_weights, 2), size(reg_weights, 3)); % reshape重新调整矩阵的行数，列数等
    % remove NaNs
    individual.output_weights(isnan(individual.output_weights)) = 0;


%% Evaluate on test data
test_states    = config.assessFcn(individual, config.test_input_sequence, [1,2], config);
test_sequence  = test_states * individual.output_weights;
test_states1   = config.assessFcn(individual, config.test_input_sequence, [3,4], config);
test_sequence1 = test_states1 * output_weights1';
test_length    = 300;

error_trace_test   = config.test_output_sequence(config.wash_out+1:end, 1) - test_sequence;
error_trace_test1  = config.test_output_sequence(config.wash_out+1:end, 2) - test_sequence1;

error_trace_test   = [0; error_trace_test(1:end-1)];
error_trace_test1  = [0; error_trace_test1(1:end-1)];

%% Compensation
% Use test_transfor function to calculate error compensation sequence
output_error_sequence_test = test_transfor(error_trace_test, error_trace_test1, ...
                                           0, 1);
% Extract required length of error compensation sequence and transpose
output_error_sequence_test = output_error_sequence_test(1, 1:test_length)';

% Apply error compensation to original test sequence
test_sequence = test_sequence(1:test_length,:) + output_error_sequence_test;

% Calculate final error: difference between true output and predicted output
test_error = config.test_output_sequence(config.wash_out + 1:config.wash_out + test_length, 1) - test_sequence;

y_mean   = mean(config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length, 1));
TSS      = sum((config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length, 1) - y_mean).^2);
RSS      = sum((config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length, 1) - test_sequence).^2);
err_r2   = 1 - RSS / TSS;

reg_test_error = calculateError(test_sequence, ...
                     config.test_output_sequence(config.wash_out + 1: ...
                     config.wash_out + test_length, 1), config);

individual.test_error    = reg_test_error;
individual.test_sequence = test_sequence;
individual.err_r2        = err_r2;
% individual.mae           = reg_test_mae;

% 替换原有的 plotComparisonAndError 调用
% plotComparisonAndError(config, test_length, test_sequence, ...
%      test_sequence, error_trace_test, test_error)

% figure(7);
% plot(config.test_output_sequence(config.wash_out+1:config.wash_out+test_length),'Marker','o','MarkerIndices',1:10:100,'MarkerFaceColor',[0 0 1],'linewidth',1.2,'Color',[0 0 1]);
% hold on;
% plot(test_sequence,'r','linewidth',1.2);
% hold on;
% xlabel('Time');
% ylabel('Wspd(m/s)');
% legend("target","output");
% grid on;

end


function plotComparisonAndError(config, train_length, output_train_sequence, output_train_origin, error_trace, error_trace_new)
    %% 补偿后与原始值的比较
    % figure;
    % plot(config.train_output_sequence(config.wash_out+1:train_length+config.wash_out,:),'Marker','o','MarkerIndices',1:20:2000,'MarkerFaceColor',[0 0 1],'linewidth',1.2,'Color',[0 0 1]);
    % hold on;
    % plot(output_train_sequence(1:train_length,:),'r','linewidth',1.2);
    % plot(output_train_origin(1:train_length,:),'-','MarkerIndices',1:100:train_length,'linewidth',1);
    % hold on;
    % % xlim([0,1000]);
    % xlabel('Time');
    % ylabel('amplitude');
    % legend("target","output");
    % grid on;
    % hold off;

    %% 补偿前后误差对比
    figure(4);
    plot(abs(error_trace(1:train_length)), 'bo', 'MarkerIndices', 1:train_length);
    hold on
    % mean_error = mean(abs(error_trace(1:train_length)));
    plot(abs(error_trace_new(1:train_length)), 'r*', 'MarkerIndices', 1:train_length, 'MarkerFaceColor', 'r');         % 求解ETRC的训练误差
    % plot([1,train_length],[mean_error,mean_error], 'LineWidth', 2.5); 
    % xlim([0,1500])
    xlabel('Time');
    ylabel('training error');
    hold off
end



