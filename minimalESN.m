% A minimalistic sparse Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Octave/Matlab.
% by Mantas Lukosevicius 2012-2018
% http://mantas.info
% 中文注释以及部分代码优化：朱应钦
% data： 2019/12/29
% 参考论文：A Practical Guide to Applying Echo State Networks
% 群文件ESN结构设计中可以找到相关文献
clc;
clear;

%% load the data 【数据加载】
trainLen = 500;                    % 训练数据集长度
testLen = 100;                     % 测试数据集长度
initLen = 30;  
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
% N = 10000;
% 
% for k=1:N
%     data(k)= sin(0.2*k)+sin(0.311*k)+cos(k);
%     data = data';
% end


%% generate the ESN reservoir         【生成ESN的储藏层】
inSize = 1;                                 % 输入节点
outSize = 1;                                % 输出节点
resSize = 3;                                % 池内节点数
a = 0.3;                                    % leaking rate【泄露率】
rand( 'seed', 42 );                         % 随机种子
Win = (rand(resSize,1+inSize)-0.5) .* 1;    % 初始化输入
% dense W:                                  % 密集的W
W = rand(resSize,resSize)-0.5;              % 池内权值
% sparse W:                                 % 稀疏的W
% W = sprand(resSize,resSize,0.01);
% W_mask = (W~=0); 
% W(W_mask) = (W(W_mask)-0.5);

%% normalizing and setting spectral radius
disp 'Computing spectral radius...';        % 计算谱半径
opt.disp = 0;                               % 
rhoW = abs(eigs(W,1,'LM',opt));             % 谱半径W的最大特征值的绝对值
disp 'done.'                                % 结束
W = W .* ( 1.25 /rhoW);                     % 

% allocated memory for the design (collected states) matrix
% 矩阵内存分配
X = zeros(1+inSize+resSize,trainLen-initLen);   %  
% set the corresponding target matrix directly
% 设置相应的目标矩阵
Yt = data(initLen+2:trainLen+1)';               % 训练集

%% run the reservoir with the data and collect X
%  更新池并收集X
x = zeros(resSize,1);                           % 初始化储藏层矩阵
for t = 1:trainLen
	u = data(t);                                % 输入值
	x = (1-a)*x  + a*tanh( Win*[1;u] + W*x );   % 公式（27.3）
	if t > initLen                              % 大于初始化时
		X(:,t-initLen) = [1;u;x];               % 收集初始化后的[1;u;x]
	end
end

%% train the output by ridge regression【通过岭回归训练输出】
reg = 1e-8;      % regularization coefficient【正则化系数】
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; % 公式27.9
% eye返回单位向量
%% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );   % 公式（27.3）
	y = Wout*[1;u;x];                          % 公式（27.4)
	Y(:,t) = y;                                % 
	% generative mode:
	% u = y;
	% this would be a predictive mode:
    % 这将是一种预测模式
	u = data(trainLen+t+1);
end
%% 收集误差曲线
output=Y*data_std+data_mean;
ytest=data(trainLen+2:trainLen+testLen+1)'*data_std+data_mean;

mae = mae(ytest,output);
disp( ['MAE = ', num2str(mae)] );
rmse= sqrt(mean((ytest-output).^2));
disp( ['RMSE = ', num2str(rmse )] );

