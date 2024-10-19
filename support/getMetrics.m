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
            % Data��2021/10/11 �����ߣ���Ӧ��  Z alert��
            % ���䣺xacker@foxmail.com
            % ���廥��Ϣ��������Ԫ������źŵ�����ԣ�����ƽ����ض�ָ�� 
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
            % Data��2021/10/11 �����ߣ���Ӧ��  Z alert��
            % ���䣺xacker@foxmail.com
            % ���ϵ����������Ԫ������źŵ�����ԣ�����ƽ����ض�ָ��
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
            % Data��2021/10/13 �����ߣ���Ӧ��  Z alert��
            % ���䣺xacker@foxmail.com
            % �����ϵ���������������ȥʱ�������ϵ��
            [c,lags] = xcorr(config.train_input_sequence, 50, 'normalized');
%             for i = 1:50
%                 relate(i) = corr(config.train_input_sequence(i+1:300-50+i,1),config.train_input_sequence(1:300-50,1));
%             end
        case 'Multifractal'
            % Data��2021/10/13 �����ߣ���Ӧ��  Z alert��
            % ���䣺xacker@foxmail.com
            [dh1,h1,cp1,tauq1] = dwtleader(config.train_input_sequence);
            
        case 'transferEntropy'
            TE = transferEntropy(X, Y, W, varargin);
            
        case 'connectivity' 
            metrics = [metrics individual.connectivity];
      
        case 'deviationlineartity'
            % Data��2021/04/09 �����ߣ���Ӧ��  Z alert��
            % ���䣺xacker@foxmail.com
            % ��������ƫ���������Ƶ����������������Ƶ������֮��ı�ֵ
            % Step1�� ����100�鵥�����źţ�Ƶ�ʴ�0.01��0.5Hz�������ض�Ƶ�������źţ�
            % config.wash_out = 1;      % �˲��ֵĶ����ᵼ��ȫ�����⣨��2021��10��18�շ��֣�
            score = [];                 % ����ƫ��
            score_nonlinear = [];       % �����Է���
            time = 1;                   % 
            % ��¼ÿһ����Ԫ��ȡ��ǰ����Ƶ�ʵ�����ֵ
            Ec_fqall = zeros(50,config.num_nodes);  % ��ʼ�������С 50*nodes
            % plot(digraph(individual.W{1,1}))
            % mean(archive([find(archive(:,3)< 0.98)],3)) % ���ڼ���ƽ������ƫ��
            for f = 0.01:0.01:0.63      % �ź�Ƶ�ʣ���0.01-0.5 ÿ�β���0.01��
              %% STEP1 :�������Ҳ�
                phase = 0;         % ����Ϊ0
                Fs = 3;            % ������     Sampling frequency
                T = 1/Fs;          % ��������   Sampling period
                L = 301;           % �źų���   Length of signal
                t = (0:L-1)*T;     % ʱ�����   Time vector
                winv = (sin(2*pi*f*t + phase))'; % ���ɲ�ͬƵ�ʵĵ����źţ�100*1��
                % winv = (sin(0.2*pi*t)+sin(0.311*t*pi))';
              %% STEP2 ����Ƶ��Ϊf�����Ҳ����룬��������״̬����
                % ����״̬����(������)
                config.teacher_forcing = 0;       % 
                x = config.assessFcn(individual,winv,config); % ����״̬����(100*201)
                config.teacher_forcing = 0;       % 
%                 for i = 1:1:size(x,2)
%                     plot(x(:,i))
%                     hold on
%                 end
                % �����ض�״̬�����Ŀ�
                % set(groot,'defaultAxesLineStyleOrder','remove','defaultAxesColorOrder','remove');
                % h2 = axes('Position',[0.15 0.15 0.3 0.3]);
                % axes(h2);
                % plot(x(:,2));                
                % hold off
                % heatmap(individual.W{1,1})
               
              %% STEP3 �� ����Ҷ�任���������ܺ�
                dc_cmp = mean(winv);              % �����źŵ�ֱ������
                % n = 2 ^ nextpow2(size(x,1));    % ���ٸ���Ҷ�任���������㸵��Ҷ�任����Qq
                Ec = abs(fft(x,size(x,1),1)/size(x,1));   % 2�����н��м��� 1�����н��м��� 100*201 ���н���������ȡÿ����Ԫ�Ļ�ȡƵ����Ϣ  
                % �°汾������ÿһ����Ԫ��ͬƵ�ʵ�����ֵ��2021.10.14
                Ec_fqall(time,:) = Ec(time+1,:);          % ���㵱ǰƵ���£���ȡ��������ֵ�Ķ���
                % ԭ�汾������ƽ���ķ�ʽ��
                Ec_all = mean(Ec,2);                      % 2�����н��м��� 1�����н��м��� ״̬�����ź�ǿ�� ������ȡÿ��״̬�����ƽ������
                Ec_all = Ec_all(1:size(x,1)/2+1);         % ���ݸ���Ҷ�任�ԳƵ��ص�����
                Ec_all(:,2:end-1) = 2*Ec_all(:,2:end-1);  % ���ݸ���Ҷ�任�ԳƵ��ص�ȡ2��
              %% STEP4 �� ��ȡ����Ƶ����������������ƫ��ȣ��������������
                time =  time+1;                           % ��ȡ�ض�Ƶ���ź�ǿ��λ��
                E_total = sum(Ec_all(2:end))-dc_cmp;      % ״̬����������
                Ec_all = Ec_all/E_total;                  % 
                score = [score 1-Ec_all(time)];           % 
              %% STEP5 �� ������������ռ��
                loc = Ec_all > 0.0001;
                loc(time) = 0; loc(1) = 0;        % ���ֱ��������1�� �Լ����Է�����time��
                score_nonlinear = [score_nonlinear sum(Ec_all(loc))];
%                 if score >=0.98     % ������ó�����ƫ��ϲ�����������������������
%                     break
%                 end
              %% ���ڵ��Թ۲���ͼ��ĺ��� 
%                 Y = fft(winv);
%                 P2 = abs(Y/L);
%                 P1 = P2(1:L/2+1);
%                 P1(2:end-1) = 2*P1(2:end-1);
%                 plot(Fs*(0:(L/2))/L,Ec_all)
%                 hold on
%                 plot(Fs*(0:(L/2))/L,P1)
%                 hold off
            end
            Ec_nodes = Ec_fqall > mean(Ec_fqall,2);                  % ��������Ƶ�ʵ���ȡ������ƽ��ֵ����ȡ����ƽ��ֵ�Ľڵ�
            Ec_select = find(sum(Ec_nodes,1) > 10);                  % ɸѡ�ܹ�׼ȷ��ȡ��Ƶ���źŵĽڵ�(ֵԽ��˵����ΧԽ��)
            score = mean(score);
            if score <= 0
                score = 0;
            end
            metrics = [metrics score];
        
       %% ���������źŵ�Ƶ������������索��
        case 'deviationlineartity_s'
            config.teacher_forcing = 0;       % 
            x = config.assessFcn(individual,config.train_input_sequence,config); % ����״̬����
            config.teacher_forcing = 0;       % 
            % ���������źŵ�Ƶ�ʷֲ�(MSO2 33 50)
            Y = fft(config.train_input_sequence);
            P2 = abs(Y/size(config.train_input_sequence,1));
            P1 = P2(1:size(config.train_input_sequence,1)/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            % ����״̬�����Ƶ�ʷֲ�
            Ec = abs(fft(x,size(x,1),1)/size(x,1));   % 2�����н��м��� 1�����н��м��� 100*201 ���н���������ȡÿ����Ԫ�Ļ�ȡƵ����Ϣ  
            % ԭ�汾������ƽ���ķ�ʽ��
            Ec_all = mean(Ec,2);                      % 2�����н��м��� 1�����н��м��� ״̬�����ź�ǿ�� ������ȡÿ��״̬�����ƽ������
            Ec_all = Ec_all(1:size(x,1)/2+1);         % ���ݸ���Ҷ�任�ԳƵ��ص�����
            Ec_all(:,2:end-1) = 2*Ec_all(:,2:end-1);  % ���ݸ���Ҷ�任�ԳƵ��ص�ȡ2��
            % ����ض�Ƶ���ź��ƶ�״̬����ǰ���ȸ��� 33 �� 50 ��Ԥ�⣩
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