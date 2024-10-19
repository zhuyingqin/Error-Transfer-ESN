function [individual, test_states, test_sequence] = testReservoir(individual, config)
config.wash_out = 30;
washout = config.wash_out;
train_states  = config.assessFcn(individual, config.train_input_sequence, config);
val_states    = config.assessFcn(individual, config.val_input_sequence, config);
train_states1 = config.assessFcn1(individual, config.train_input_sequence, config);
% val_states1   = config.assessFcn1(individual, config.val_input_sequence, config);

%% if W_out are evolved instead of trained

% load output.mat;
% Find best reg parameter
reg_train_error = [];
reg_val_error   = [];
reg_weights     = [];
reg_param       = [10e-3 10e-5 10e-7 10e-9 10e-11 10e-13];
error_collect   = [];
error_collect1  = [];
train_length    = 500;
% [m, n] = size(train_states);

for i = 1:length(reg_param)
    % Train: tanspose is inversed compared to equation 
    output_weights  = config.train_output_sequence(config.wash_out+1:end,:)' ...
                      * train_states  * inv(train_states' * train_states ...
                      + reg_param(i) * eye(size(train_states' * train_states)));
    output_weights1 = config.train_output_sequence(config.wash_out+1:end,:)' ...
                      * train_states1 * inv(train_states1' * train_states1 ...
                      + reg_param(i) * eye(size(train_states1'*train_states1)));

    %% 
    output_train_sequence  = train_states * output_weights';
    output_train_sequence1 = train_states1 * output_weights1';
 
    %% 误差跟踪器（该跟踪器的参数可以任意拟定）
    % 计算训练误差 目标值-预测值
    error_trace   = config.train_output_sequence(config.wash_out+1:end, :) - output_train_sequence;   
    error_trace1  = config.train_output_sequence(config.wash_out+1:end, :) - output_train_sequence1; 
    
    % 计算NMRSE误差
    err_rc(i,:) = calculateNMRSE(output_train_sequence(1 : train_length, :), ...
        config.train_output_sequence(config.wash_out + 1 : train_length + config.wash_out, :), ...
        config.train_output_sequence(config.wash_out + 1 : train_length + 1 + config.wash_out));

    % 计算NMSE误差
    reg_train_error_rc(i,:) = calculateError(output_train_sequence(1:train_length, :), ...
        config.train_output_sequence(config.wash_out + 1 : config.wash_out + train_length, :), config);

    % 计算MAE误差
    mae_rc(i,:) = mae(config.train_output_sequence(config.wash_out+1:end,:), output_train_sequence);
   
    % 收集误差
    error_collect  = [error_collect, error_trace];
    error_collect1 = [error_collect1, error_trace1];
   
    % 误差补偿
    output_error_sequence = test_transfor(error_collect1, error_collect, washout, i, train_length);     % 拿到补偿后的误差
    output_error_sequence = output_error_sequence(:,1:train_length)';
    
    % 应用误差补偿
    output_train_sequence = output_train_sequence(1:train_length,:) + output_error_sequence;        % 得到补偿后的预测值的结果
    
    % 反归一化处理
    output_train_sequence = output_train_sequence * config.data_std + config.data_mean;
    config.train_output_sequence(config.wash_out + 1 : config.wash_out + train_length, :) = ...
        config.train_output_sequence(config.wash_out + 1 : config.wash_out+train_length, :) ...
        * config.data_std + config.data_mean;
    
    % 计算补偿后的NMRSE误差
    reg_train_error_nmrse(i,:) = calculateNMRSE(output_train_sequence, ...
        config.train_output_sequence(config.wash_out+1 : train_length + config.wash_out, :), ...
        config.train_output_sequence(config.wash_out+1 : train_length + 1 + config.wash_out));
    
    % 计算补偿后的NMSE误差
    reg_train_error(i,:) = calculateError(output_train_sequence, ...
        config.train_output_sequence(config.wash_out + 1 : config.wash_out + train_length, :), config);
    
    % 计算补偿后的MSE误差
    reg_train_error_mse(i,:) = mean((output_train_sequence-config.train_output_sequence(config.wash_out+1:config.wash_out+train_length,:)).^2);

    % 计算补偿后的MAE误差
    reg_train_error_mae(i,:) = mae(output_train_sequence,config.train_output_sequence(config.wash_out+1:config.wash_out+train_length,:));
        
    %% 验证集评估
    % 计算验证集输出
    output_val_sequence = val_states * output_weights';
    % 计算验证集误差
    reg_val_error(i,:)  = calculateError(output_val_sequence, config.val_output_sequence, config);
    % 保存权重
    reg_weights(i,:,:)  = output_weights';
end
    [~, reg_indx] = min(sum(reg_val_error, 2));
    individual.train_error = sum(reg_train_error(reg_indx, :));
    individual.output_weights = reshape(reg_weights(reg_indx, :, :), size(reg_weights, 2), size(reg_weights, 3)); % reshape重新调整矩阵的行数，列数等
    % remove NaNs
    individual.output_weights(isnan(individual.output_weights)) = 0;


%% Evaluate on test data
test_states    = config.assessFcn(individual, config.test_input_sequence, config);
test_sequence  = test_states * individual.output_weights;
test_states1   = config.assessFcn(individual, config.test_input_sequence, config);
test_sequence1 = test_states1 * individual.output_weights;
test_length = 289;

error_trace_test_true   = config.test_output_sequence(config.wash_out+1:end,:) - test_sequence;
error_trace_test_true1  = config.test_output_sequence(config.wash_out+1:end,:) - test_sequence1;

% error_trace_test = filter([1/4 1/4 1/4 1/4],1,error_trace_test);
error_trace_test  = smoothdata(error_trace_test_true,'gaussian', 10); 
error_trace_test1 = smoothdata(error_trace_test_true1,'gaussian', 10); 
 
%% 补偿
output_error_sequence_test = test_transfor(error_trace_test,error_trace_test1,washout,1,test_length);
output_error_sequence_test = output_error_sequence_test(:,1:test_length)';
test_sequence = test_sequence(1:test_length,:) + output_error_sequence_test(1:test_length);
test_sequence = test_sequence*config.data_std+config.data_mean;
config.test_output_sequence(config.wash_out+1:config.wash_out+test_length) = config.test_output_sequence(config.wash_out+1:config.wash_out+test_length)*config.data_std+config.data_mean;
test_error = config.test_output_sequence(config.wash_out+1:config.wash_out+test_length) - test_sequence;%最终的误差值

absolute_err = abs(test_error);
err_mae  = mean(absolute_err);
err_rmse = sqrt(mean((config.test_output_sequence(config.wash_out+1:config.wash_out+test_length)-test_sequence).^2));
err_mse  = mean((config.test_output_sequence(config.wash_out+1:config.wash_out+test_length)-test_sequence).^2);
errors   = abs((config.test_output_sequence(config.wash_out+1:config.wash_out+test_length) - test_sequence) ./ config.test_output_sequence(config.wash_out+1:config.wash_out+test_length));
err_mape = mean(errors) * 100;
y_mean   = mean(config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length));
TSS      = sum((config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length) - y_mean).^2);
RSS      = sum((config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length) - test_sequence).^2);
err_r2   = 1 - RSS / TSS;
err_nmrse = calculateNMRSE(test_sequence, ...
    config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length), ...
    config.test_output_sequence(config.wash_out+1 : config.wash_out + test_length));

reg_test_error = sum(calculateError(test_sequence, ...
    config.test_output_sequence(config.wash_out+1 : config.wash_out+test_length), config));

individual.test_error    = reg_test_error;
individual.test_sequence = test_sequence;
individual.err_r2        = err_r2;
individual.err_rmse      = err_rmse;

%   figure(7);
%   plot(config.test_output_sequence(config.wash_out+1:config.wash_out+test_length),'Marker','o','MarkerIndices',1:10:100,'MarkerFaceColor',[0 0 1],'linewidth',1.2,'Color',[0 0 1]);
%   hold on;
%   plot(test_sequence,'r','linewidth',1.2);
%   hold on;
%   xlabel('Time');
%   ylabel('Wspd(m/s)');
%   legend("target","output");
%   grid on;

end


function plotComparisonAndError(config, train_length, output_train_sequence, output_train_origin, error_trace, error_trace_new)
    %% 补偿后与原始值的比较
    figure;
    plot(config.train_output_sequence(config.wash_out+1:train_length+config.wash_out,:),'Marker','o','MarkerIndices',1:20:2000,'MarkerFaceColor',[0 0 1],'linewidth',1.2,'Color',[0 0 1]);
    hold on;
    plot(output_train_sequence(1:train_length,:),'r','linewidth',1.2);
    plot(output_train_origin(1:train_length,:),'-','MarkerIndices',1:100:train_length,'linewidth',1);
    hold on;
    % xlim([0,1000]);
    xlabel('Time');
    ylabel('amplitude');
    legend("target","output");
    grid on;
    hold off;

    %% 补偿前后误差对比
    figure(4);
    plot(abs(error_trace(1:train_length)),'bo','MarkerIndices',1:10:train_length);
    hold on
    mean_error = mean(abs(error_trace(1:train_length)));
    plot(abs(error_trace_new(1:train_length)),'r*','MarkerIndices',1:9:train_length,'MarkerFaceColor','r');         % 求解ETRC的训练误差
    plot([1,train_length],[mean_error,mean_error], 'LineWidth', 2.5); 
    % xlim([0,1500])
    xlabel('Time');
    ylabel('training error');
    hold off
end


function nmrse = calculateNMRSE(predicted, actual, actual_for_var)
    % Calculate the Normalized Root Mean Square Error (NMRSE)
    %
    % Input parameters:
    % predicted: Vector of predicted values
    % actual: Vector of actual values
    % actual_for_var: Vector of actual values used for variance calculation (usually the same as 'actual')
    %
    % Output:
    % nmrse: Normalized Root Mean Square Error

    squared_error = sum((predicted - actual).^2);
    variance = var(actual_for_var);
    n = length(actual_for_var);
    nmrse = sqrt(squared_error / (variance * n));
end
