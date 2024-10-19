% Separation Metrics and Kernel Quality
function [metrics, Ec_select, M] = getMetrics(individual,config)

scurr = rng;
temp_seed = scurr.Seed;

% set parameters
metrics = [];
config.reg_param = 10e-6;
config.wash_out = 0;
metrics_type =  config.metrics;
num_timesteps = round(individual.total_units*1.5) + config.wash_out; % input should be twice the size of network + wash out 300
MC_num_timesteps = 500 + config.wash_out*2;
n_input_units = individual.n_input_units;

for metric_item = 1:length(config.metrics) 
    
    rng(1,'twister');
    
    switch metrics_type{metric_item}
        
        % kernel rank
        case 'KR'
            
            % define input signal
           
            ui = 2*rand(num_timesteps,n_input_units)-1;  % 
            
            input_sequence = repmat(ui(:,1),1,n_input_units); %
            
            % rescale for each reservoir
        
            input_sequence =input_sequence.*config.scaler;
            
            % kernel matrix - pick 'to' at halfway point
            M = config.assessFcn(individual,input_sequence,config);
            
            %catch errors
            M(isnan(M)) = 0;
            M(isinf(M)) = 0;
            
          %% Kernal Quality
            s = svd(M);
            
            tmp_rank_sum = 0;
            full_rank_sum = 0;
            e_rank = 1;
            for i = 1:length(s)
                full_rank_sum = full_rank_sum + s(i);
                while (tmp_rank_sum < full_rank_sum * 0.99)
                    tmp_rank_sum = tmp_rank_sum + s(e_rank);
                    e_rank= e_rank+1;
                end
            end
            
            kernel_rank = e_rank-1;
                   
            metrics = [metrics kernel_rank];
            
            % Genralization Rank
        case 'GR'
            % define input signal
            input_sequence = 0.5 + 0.1*rand(num_timesteps,n_input_units)-0.05;
            
            % rescale for each reservoir
            input_sequence =input_sequence.*config.scaler;
            
            %collect states
            G = config.assessFcn(individual,input_sequence,config);
            
            %catch errors
            G(isnan(G)) = 0;
            G(isinf(G)) = 0;
            
            % get rank of matrix
            s = svd(G);
            
            %claculate effective rank
            tmp_rank_sum = 0;
            full_rank_sum = 0;
            e_rank = 1;
            for i = 1:length(s)
                full_rank_sum = full_rank_sum +s(i);
                while (tmp_rank_sum < full_rank_sum * 0.99)
                    tmp_rank_sum = tmp_rank_sum + s(e_rank);
                    e_rank= e_rank+1;
                end
            end
            gen_rank = e_rank-1;
            
            metrics = [metrics gen_rank];
            
            % LE measure
        case 'LE'
            seed = 1;
            LE = lyapunovExponent(individual,config,seed);
            metrics = [metrics LE];
            
            % Entropy measure
        case 'entropy'
            
            data_length = num_timesteps;%individual.total_units*2 + config.wash_out;%400;
            input_sequence = ones(data_length,n_input_units).*config.scaler;
            
            X = config.assessFcn(individual,input_sequence,config);
            C = X'*X;
            
            X_eig = eig(C);
            
            normX_eig = X_eig./sum(X_eig);
            
            H = -sum(normX_eig.*log2(normX_eig));
            
            entropy = real(H/log2(size(X,2)));
            
            entropy(isnan(entropy)) = 0;
            metrics = [metrics entropy*100];
            
            % linear memory capacity
        case 'linearMC'
            
            % measure MC multiple times
            mc_seed = 1;
            temp_MC = testMC(individual,config,mc_seed,MC_num_timesteps);
            %
            temp_MC_1 = testMC1(individual,config,mc_seed,MC_num_timesteps);
            MC = mean(temp_MC);
            MC_1=mean(temp_MC_1);
            metrics = [metrics MC MC_1];

            
            % quadratic memory capacity (nonlinear) 
        case 'quadMC'
            
            quad_MC = quadraticMC(individual,config,1,MC_num_timesteps);
            
            metrics = [metrics quad_MC];
            
            % cross-memory capacity (nonlinear) 
        case 'crossMC'
            
            cross_MC = crossMC(individual,config,1,MC_num_timesteps);
            
            metrics = [metrics cross_MC];
            
            % separation property
        case 'separation'
            
            data_length = num_timesteps;%individual.total_units*4 + config.wash_out*2;%400;
            
            u1 = (rand(data_length,n_input_units)-1).*config.scaler;
            u2 = (rand(data_length,n_input_units)).*config.scaler;
            
            D= norm(u1-u2);
            
            X1 = config.assessFcn(individual,u1,config);
            
            X2 = config.assessFcn(individual,u2,config);
            
            sep = norm(X1 - X2)/D;
            metrics = [metrics sep];  
            
        case 'mutalInformation'
            % Data：2021/10/11 （作者：朱应钦  Z alert）
            % 邮箱：xacker@foxmail.com
            % 定义互信息：衡量神经元池输出信号的相关性，定义平均相关度指标 
            data_length = individual.total_units*4 + config.wash_out*2; % 400;
            
            u = (rand(data_length,n_input_units)-1).*config.scaler;
            
            X = config.assessFcn(individual,u,config);
            
            MI_A = [];
            for i = 1:config.num_nodes
                MI = [];
                for j = 1:config.num_nodes
                    MI(j) = ami(X(:,i), X(:,j));
                end
                MI_A= [MI_A; MI];
            end
            MI_mean = mean(mean(abs(MI_A)));
            metrics = [metrics MI_mean];
        
        case 'Correlation' 
            % Data：2021/10/11 （作者：朱应钦  Z alert）
            % 邮箱：xacker@foxmail.com
            % 相关系数：衡量神经元池输出信号的相关性，定义平均相关度指标
            % data_length = individual.total_units*4 + config.wash_out*2;
            % u = (rand(data_length,n_input_units)-1).*config.scaler;
            X = config.assessFcn(individual,individual.test_sequence,config);
            relate_A = [];
            for i = 1:config.num_nodes
                relate = [];
                for j = 1:config.num_nodes
                    relate(j) = corr(X(:,i), X(:,j),'type','Pearson');
                end
                relate_A = [relate_A; relate];
            end
            relate_mean = mean(mean(abs(relate_A)));
            metrics = [metrics relate_mean];
            
        case 'Corr_data'
            % Data：2021/10/13 （作者：朱应钦  Z alert）
            % 邮箱：xacker@foxmail.com
            % 自相关系数：衡量数据与过去时间点的相关系数
            [c,lags] = xcorr(config.train_input_sequence, 50, 'normalized');
%             for i = 1:50
%                 relate(i) = corr(config.train_input_sequence(i+1:300-50+i,1),config.train_input_sequence(1:300-50,1));
%             end
        case 'Multifractal'
            % Data：2021/10/13 （作者：朱应钦  Z alert）
            % 邮箱：xacker@foxmail.com
            [dh1,h1,cp1,tauq1] = dwtleader(config.train_input_sequence);
            
        case 'transferEntropy'
            TE = transferEntropy(X, Y, W, varargin);
            
        case 'connectivity' 
            metrics = [metrics individual.connectivity];
      
        case 'deviationlineartity'
            % Data：2021/04/09 （作者：朱应钦  Z alert）
            % 邮箱：xacker@foxmail.com
            % 定义线性偏差：根据输入频率能量与其他所有频率能量之间的比值
            % Step1： 创建100组单输入信号，频率从0.01至0.5Hz（采用特定频率正弦信号）
            % config.wash_out = 1;      % 此部分的丢弃会导致全局问题（于2021年10月18日发现）
            score = [];                 % 线性偏离
            score_nonlinear = [];       % 非线性分数
            time = 1;                   % 
            % 记录每一个神经元提取当前输入频率的能量值
            Ec_fqall = zeros(50,config.num_nodes);  % 初始化矩阵大小 50*nodes
            % plot(digraph(individual.W{1,1}))
            % mean(archive([find(archive(:,3)< 0.98)],3)) % 用于计算平均线性偏离
            for f = 0.01:0.01:0.63      % 信号频率（从0.01-0.5 每次步进0.01）
              %% STEP1 :生成正弦波
                phase = 0;         % 相移为0
                Fs = 3;            % 采样率     Sampling frequency
                T = 1/Fs;          % 采样周期   Sampling period
                L = 301;           % 信号长度   Length of signal
                t = (0:L-1)*T;     % 时间矩阵   Time vector
                winv = (sin(2*pi*f*t + phase))'; % 生成不同频率的单点信号（100*1）
                % winv = (sin(0.2*pi*t)+sin(0.311*t*pi))';
              %% STEP2 ：将频率为f的正弦波输入，并计算其状态矩阵
                % 计算状态矩阵(不优雅)
                config.teacher_forcing = 0;       % 
                x = config.assessFcn(individual,winv,config); % 计算状态矩阵(100*201)
                config.teacher_forcing = 0;       % 
%                 for i = 1:1:size(x,2)
%                     plot(x(:,i))
%                     hold on
%                 end
                % 绘制特定状态函数的框
                % set(groot,'defaultAxesLineStyleOrder','remove','defaultAxesColorOrder','remove');
                % h2 = axes('Position',[0.15 0.15 0.3 0.3]);
                % axes(h2);
                % plot(x(:,2));                
                % hold off
                % heatmap(individual.W{1,1})
               
              %% STEP3 ： 傅里叶变换计算能量总和
                dc_cmp = mean(winv);              % 输入信号的直流分量
                % n = 2 ^ nextpow2(size(x,1));    % 快速傅里叶变换，添零满足傅里叶变换长度Qq
                Ec = abs(fft(x,size(x,1),1)/size(x,1));   % 2：沿行进行计算 1：沿列进行计算 100*201 按列进行运行提取每个神经元的获取频率信息  
                % 新版本（计算每一个神经元不同频率的能量值）2021.10.14
                Ec_fqall(time,:) = Ec(time+1,:);          % 计算当前频率下，获取特征能量值的多少
                % 原版本（计算平均的方式）
                Ec_all = mean(Ec,2);                      % 2：沿行进行计算 1：沿列进行计算 状态矩阵信号强度 按行提取每组状态矩阵的平均能量
                Ec_all = Ec_all(1:size(x,1)/2+1);         % 根据傅里叶变换对称的特点折中
                Ec_all(:,2:end-1) = 2*Ec_all(:,2:end-1);  % 根据傅里叶变换对称的特点取2倍
              %% STEP4 ： 提取输入频率能量，计算线性偏离度，并将其存入数组
                time =  time+1;                           % 提取特定频率信号强度位置
                E_total = sum(Ec_all(2:end))-dc_cmp;      % 状态矩阵总能量
                Ec_all = Ec_all/E_total;                  % 
                score = [score 1-Ec_all(time)];           % 
              %% STEP5 ： 计算其他分量占比
                loc = Ec_all > 0.0001;
                loc(time) = 0; loc(1) = 0;        % 清除直流分量（1） 以及线性分量（time）
                score_nonlinear = [score_nonlinear sum(Ec_all(loc))];
%                 if score >=0.98     % 若计算得出线性偏离较差的情况，可以重新生成网络
%                     break
%                 end
              %% 用于调试观察其图像的函数 
%                 Y = fft(winv);
%                 P2 = abs(Y/L);
%                 P1 = P2(1:L/2+1);
%                 P1(2:end-1) = 2*P1(2:end-1);
%                 plot(Fs*(0:(L/2))/L,Ec_all)
%                 hold on
%                 plot(Fs*(0:(L/2))/L,P1)
%                 hold off
            end
            Ec_nodes = Ec_fqall > mean(Ec_fqall,2);                  % 计算所有频率的提取能量的平均值并提取大于平均值的节点
            Ec_select = find(sum(Ec_nodes,1) > 10);                  % 筛选能够准确获取出频率信号的节点(值越大说明范围越广)
            score = mean(score);
            if score <= 0
                score = 0;
            end
            metrics = [metrics score];
        
       %% 根据输入信号得频率特性设计网络储层
        case 'deviationlineartity_s'
            config.teacher_forcing = 0;       % 
            x = config.assessFcn(individual,config.train_input_sequence,config); % 计算状态矩阵
            config.teacher_forcing = 0;       % 
            % 分析输入信号的频率分布(MSO2 33 50)
            Y = fft(config.train_input_sequence);
            P2 = abs(Y/size(config.train_input_sequence,1));
            P1 = P2(1:size(config.train_input_sequence,1)/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            % 分析状态矩阵的频率分布
            Ec = abs(fft(x,size(x,1),1)/size(x,1));   % 2：沿行进行计算 1：沿列进行计算 100*201 按列进行运行提取每个神经元的获取频率信息  
            % 原版本（计算平均的方式）
            Ec_all = mean(Ec,2);                      % 2：沿行进行计算 1：沿列进行计算 状态矩阵信号强度 按行提取每组状态矩阵的平均能量
            Ec_all = Ec_all(1:size(x,1)/2+1);         % 根据傅里叶变换对称的特点折中
            Ec_all(:,2:end-1) = 2*Ec_all(:,2:end-1);  % 根据傅里叶变换对称的特点取2倍
            % 针对特定频率信号制定状态矩阵（前期先根据 33 和 50 来预测）
            Ec_select = find(Ec(33,:) > 0.8*mean(Ec(33,:)));
            Ec_select = unique([Ec_select find(Ec(50,:) > 0.8*mean(Ec(50,:)))]);
            Ec_select = unique([Ec_select find(Ec(68,:) > 0.8*mean(Ec(68,:)))]);
            Ec_select = unique([Ec_select find(Ec(82,:) > 0.8*mean(Ec(82,:)))]);
            Ec_select = unique([Ec_select find(Ec(101,:) > 0.8*mean(Ec(101,:)))]);
            metrics = [metrics []];
            % Ec_select = [];      
    end
end

rng(temp_seed,'twister');