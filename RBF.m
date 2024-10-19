clc;
clear;
trainLen = 500;                    % 训练数据集长度
testLen = 100;                     % 测试数据集长度
initLen = 0;  
data_wind=readtable("dataset\wtbdata_cleaned123.csv");
data_wind=data_wind(2:end,4);
data_wind=table2array(data_wind);
index=isnan(data_wind);
data_wind=data_wind(~index);
data_mean=mean(data_wind(1:10000));
data_std=std(data_wind(1:10000));
if length(data_wind)>=10000
     data=data_wind;
    else
     data=data_wind;
end
data=normalize(data);
data=diff(data);
data=seriesDecomp(data,3);
inputs=data(1:trainLen)';
targets = data(initLen+2:trainLen+1)'; 
%% Create a RBF Network
goal = 0.000;   % Mean squared error goal (default = 0.0)
spread  =  3;   % Spread of radial basis functions (default = 1.0)
MN = 100;       % Maximum number of neurons 
DF = 10;        % Number of neurons to add between displays (default = 25)
net  = newrb(inputs,targets,goal,spread,MN,DF);
view(net)

xtest= data(trainLen+1:trainLen+testLen)';
ytest=data(trainLen+2:trainLen+testLen+1)';
output=net(xtest);
output=output*data_std+data_mean;
ytest=ytest*data_std+data_mean;

error=ytest-output;

nrmse=sqrt((sum(((ytest-output).^2)/(var(ytest)))*(1/length(ytest))));
disp( ['NRMSE = ', num2str(nrmse )] );
mae = mae(ytest,output);
disp( ['MAE = ', num2str(mae)] );
rmse= sqrt(mean((ytest-output).^2));
disp( ['RMSE = ', num2str(rmse )] );
