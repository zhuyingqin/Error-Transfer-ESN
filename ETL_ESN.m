clear 
close all

rng(1,'twister');% rng(seed,generator) 

%% Setup   
config.parallel = 0;                        % use parallel toolbox  

% start paralllel pool if empty
if isempty(gcp) && config.parallel
    parpool('local',4,'IdleTimeout', Inf);  % create parallel pool
end

% type of network to evolve
config.res_type  = 'RoR';           
config.num_nodes = 200;                  
config = selectReservoirType(config);         
config.ngrc = 0;
%% Evolutionary parameters
config.num_tests    = 1;                     % num of tests/runs
config.pop_size     = 100;                    % initail population size.
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
for tests = 1:config.num_tests
    config.input_scaling = 0.5 * tests;   
    clearvars -except config tests figure1 figure2 quality database_history pred_dataset
    
    warning('off','all')
    fprintf('\n Test: %d  ',tests);
    fprintf('Processing genotype......... %s \n',datestr(now, 'HH:MM:SS'))
    tic
    
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
            toc
            et(pop_indx) = toc;
            fprintf('\n i = %d, took: %.4f\n',pop_indx,toc);
        end
    end
    
end
config.finish_time = datestr(now, 'HH:MM:SS');

% Find the minimum test error
min_test_error = Inf;
for i = 1:length(population)
    if isfield(population(i), 'test_error') && population(i).test_error < min_test_error
        min_test_error = population(i).test_error;
    end
end

fprintf('Minimum test error: %.6f\n', min_test_error);
