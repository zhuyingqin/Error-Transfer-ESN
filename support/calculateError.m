function [err] = calculateError(system_output,desired_output,config)

if size(system_output,1) == size(desired_output,1)
    config.wash_out = 0;
elseif size(system_output,1) > size(desired_output,1)
    system_output = system_output(config.wash_out+1:end,:);
else
    desired_output = desired_output(config.wash_out+1:end,:);
end

% final measured error type
switch(config.err_type)
    
    case 'mae'
        err = desired_output - system_output;
        
        % Then take the "absolute" value of the "error".
        absolute_err = abs(err);
        
        % Finally take the "mean" of the "absoluteErr".
        err = mean(absolute_err);
        
    case 'mase'
        err = (desired_output-system_output);
        
        % Then take the "absolute" value of the "error".
        absolute_err = abs(err).*abs(desired_output);
        
        % Finally take the "mean" of the "absoluteErr".
        err = mean(absolute_err);
        
        
    case 'rmse'
        err = sqrt(mean((desired_output-system_output).^2));
        
    case 'crossEntropy'
        [~,p] = max(system_output,[],2);
        tp = zeros(size(system_output));
        for i = 1:length(tp)
            tp(i,p(i)) =1;
        end
        %err = -(sum(desiredOutput*log(systemOutput)'+(1-desiredOutput)*log(1-systemOutput)')/size(desiredOutput,1));
        err = (sum(diag(-desired_output'*log(system_output))-diag((1-desired_output')*log(1-system_output)))/size(desired_output,1));
    
    case 'NRMSE'
        err= sqrt((sum((desired_output-system_output).^2)/(var(desired_output)))*(1/length(desired_output)));
        %err = compute_NRMSE(systemOutput,desiredOutput);
        %err = goodnessOfFit(systemOutput,desiredOutput,type);
        
    case 'NSE'
        err= sum((desired_output-system_output).^2)/sum(desired_output.^2);
        
    case 'NRMSE_henon'
        err = sqrt(mean((system_output-desired_output).^2))/(max(max(desired_output))-min(min(desired_output))); %Rodan paper
        
    case 'NMSE'
        err= mean((desired_output-system_output).^2)/var(desired_output);
                  
    case 'MSE'
        err = mean((desired_output-system_output).^2);
                
    otherwise
        err = computeNRMSE(system_output,desired_output);
end

if isnan(err)
    err = 1;
end