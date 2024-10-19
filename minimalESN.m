% A minimalistic sparse Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Octave/Matlab.
% by Mantas Lukosevicius 2012-2018
% http://mantas.info
% ����ע���Լ����ִ����Ż�����Ӧ��
% data�� 2019/12/29
% �ο����ģ�A Practical Guide to Applying Echo State Networks
% Ⱥ�ļ�ESN�ṹ����п����ҵ��������
clc;
clear;

%% load the data �����ݼ��ء�
trainLen = 500;                    % ѵ�����ݼ�����
testLen = 100;                     % �������ݼ�����
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


%% generate the ESN reservoir         ������ESN�Ĵ��ز㡿
inSize = 1;                                 % ����ڵ�
outSize = 1;                                % ����ڵ�
resSize = 3;                                % ���ڽڵ���
a = 0.3;                                    % leaking rate��й¶�ʡ�
rand( 'seed', 42 );                         % �������
Win = (rand(resSize,1+inSize)-0.5) .* 1;    % ��ʼ������
% dense W:                                  % �ܼ���W
W = rand(resSize,resSize)-0.5;              % ����Ȩֵ
% sparse W:                                 % ϡ���W
% W = sprand(resSize,resSize,0.01);
% W_mask = (W~=0); 
% W(W_mask) = (W(W_mask)-0.5);

%% normalizing and setting spectral radius
disp 'Computing spectral radius...';        % �����װ뾶
opt.disp = 0;                               % 
rhoW = abs(eigs(W,1,'LM',opt));             % �װ뾶W���������ֵ�ľ���ֵ
disp 'done.'                                % ����
W = W .* ( 1.25 /rhoW);                     % 

% allocated memory for the design (collected states) matrix
% �����ڴ����
X = zeros(1+inSize+resSize,trainLen-initLen);   %  
% set the corresponding target matrix directly
% ������Ӧ��Ŀ�����
Yt = data(initLen+2:trainLen+1)';               % ѵ����

%% run the reservoir with the data and collect X
%  ���³ز��ռ�X
x = zeros(resSize,1);                           % ��ʼ�����ز����
for t = 1:trainLen
	u = data(t);                                % ����ֵ
	x = (1-a)*x  + a*tanh( Win*[1;u] + W*x );   % ��ʽ��27.3��
	if t > initLen                              % ���ڳ�ʼ��ʱ
		X(:,t-initLen) = [1;u;x];               % �ռ���ʼ�����[1;u;x]
	end
end

%% train the output by ridge regression��ͨ����ع�ѵ�������
reg = 1e-8;      % regularization coefficient������ϵ����
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; % ��ʽ27.9
% eye���ص�λ����
%% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );   % ��ʽ��27.3��
	y = Wout*[1;u;x];                          % ��ʽ��27.4)
	Y(:,t) = y;                                % 
	% generative mode:
	% u = y;
	% this would be a predictive mode:
    % �⽫��һ��Ԥ��ģʽ
	u = data(trainLen+t+1);
end
%% �ռ��������
output=Y*data_std+data_mean;
ytest=data(trainLen+2:trainLen+testLen+1)'*data_std+data_mean;

mae = mae(ytest,output);
disp( ['MAE = ', num2str(mae)] );
rmse= sqrt(mean((ytest-output).^2));
disp( ['RMSE = ', num2str(rmse )] );

