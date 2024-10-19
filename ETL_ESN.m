clear 
close all

rng(1,'twister');% rng(seed,generator) 

%% Setup   
config.parallel = 0;                        % use parallel toolbox  

% start paralllel pool if empty
% if isempty(gcp) && config.parallel
%     parpool('local',4,'IdleTimeout', Inf);  % create parallel pool
% end

% type of network to evolve
config.res_type  = 'RoR';           
config.num_nodes = 200;                  
config = selectReservoirType(config);         
config.ngrc = 0;
%% Evolutionary parameters
config.num_tests    = 1;                     % num of tests/runs
config.pop_size     = 10;                  % initail population size.
config.error_to_check = 'test';              % train&val&test'
%% Task parameters
config.discrete = 0;             % select '1' for binary input for discrete systems
config.nbits    = 16;            % only applied if config.discrete = 1; if wanting to convert data for binary/discrete systems
config.dataset  = 'wind';      
%% 
% prediction parameters
config.get_prediction_data = 1;             % collect task performances after experiment. Variables below are applied if '1'.
config.task_list = {'MSO12'}; % tasks to assess

% get any additional params. This might include:
% details on reservoir structure, extra task variables, etc.
config = getAdditionalParameters(config);
config = getAdditionalParameters1(config);

% get dataset information 
config = selectDataset(config);

% Novelty search parameters
config.k_neighbours = 10;                    % how many neighbours to check, e.g 10-15 is a good rule-of-thumb
config.p_min_start = sqrt(sum(config.num_nodes));   % sum(config.num_nodes)/10;                     % novelty threshold. In general start low. Reduce or increase depending on network size.
config.p_min_check = 100;                    % change novelty threshold dynamically after "p_min_check" generations.

%% general params
config.gen_print = 50;                       % after 'gen_print' generations print task performance and show any plots
config.start_time = datestr(now, 'HH:MM:SS');
config.save_gen = inf;                       % save data at generation = save_gen

% Only necessary if wanting to parallelise the microGA algorithm
config.multi_offspring = 1;                  % multiple tournament selection and offspring in one cycle
config.num_sync_offspring = 4;               % length of cycle/synchronisation step

%% Run experiments
total_time = 0;
for tests = 1:config.num_tests
    config.input_scaling = 0.5 * tests;   
    clearvars -except config tests figure1 figure2 quality database_history pred_dataset total_time
    
    warning('off','all')
    fprintf('\n Test: %d  ',tests);
    fprintf('Processing genotype......... %s \n',datestr(now, 'HH:MM:SS'))
    test_start_time = tic;
    
    % update random seed
    % rng(tests,'twister');
    
    % create population of reservoirs
    population = config.createFcn(config);
    % Evaluate population and assess novelty
    if config.parallel
        ppm = ParforProgMon('Initial population: ', config.pop_size);
        parfor pop_indx = 1:config.pop_size
            warning('off','all')
            population(pop_indx) = config.testFcn(population(pop_indx),config);
            ppm.increment();
        end
    else
        for pop_indx = 1:config.pop_size
            tic
            population(pop_indx) = config.testFcn(population(pop_indx),config);
            iteration_time = toc;
            et(pop_indx) = iteration_time;
            fprintf('\n i = %d, took: %.4f\n',pop_indx,iteration_time);
        end
    end
    
    test_end_time = toc(test_start_time);
    total_time = total_time + test_end_time;
    fprintf('\nTest %d completed in %.2f seconds\n', tests, test_end_time);
end
config.finish_time = datestr(now, 'HH:MM:SS');

% Find the minimum test error
min_test_error = Inf;
for i = 1:length(population)
    if isfield(population(i), 'err_rmse') && population(i).err_rmse < min_test_error
        min_test_error = population(i).err_rmse;
    end
end

fprintf('Minimum test error: %.6f\n', min_test_error);
fprintf('Total execution time: %.2f seconds\n', total_time);
fprintf('Average time per test: %.2f seconds\n', total_time / config.num_tests);

% Calculate and display time statistics for individual iterations
mean_iteration_time = mean(et);
std_iteration_time = std(et);
min_iteration_time = min(et);
max_iteration_time = max(et);

fprintf('\nIteration time statistics:\n');
fprintf('Mean: %.4f seconds\n', mean_iteration_time);
fprintf('Standard deviation: %.4f seconds\n', std_iteration_time);
fprintf('Minimum: %.4f seconds\n', min_iteration_time);
fprintf('Maximum: %.4f seconds\n', max_iteration_time);

% Plot histogram of iteration times
figure;
histogram(et);
title('Histogram of Iteration Times');
xlabel('Time (seconds)');
ylabel('Frequency');
