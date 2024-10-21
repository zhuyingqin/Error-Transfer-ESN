%% Select Data Script: Generate task data sets and split data
function [config] = selectDataset(config)
    wash_out = 50;
    rng(1, 'twister');

    switch config.dataset
        case 'wind' % good task
            [config, input_sequence, output_sequence] = prepareWindData(config);
        otherwise
            error('Unsupported dataset: %s', config.dataset);
    end

    % Split datasets
    [train_input_sequence, val_input_sequence, test_input_sequence] = ...
            split_train_test3way(input_sequence, config.train_fraction, ...
            config.val_fraction, config.test_fraction);

    [train_output_sequence, val_output_sequence, test_output_sequence] = ...
            split_train_test3way(output_sequence, config.train_fraction, ...
            config.val_fraction, config.test_fraction);

    % Update config structure
    config.train_input_sequence  = train_input_sequence;
    config.train_output_sequence = train_output_sequence;
    config.val_input_sequence    = val_input_sequence;
    config.val_output_sequence   = val_output_sequence;
    config.test_input_sequence   = test_input_sequence;
    config.test_output_sequence  = test_output_sequence;
    config.wash_out  = wash_out;
end

function [config, input_sequence, output_sequence] = prepareWindData(config)
    config.err_type = 'mae';
    config.train_fraction = 0.8;
    config.val_fraction   = 0.1;
    config.test_fraction  = 0.1;
    ahead = 1;

    % Read wind speed data (original dataset)
    wind_file = fullfile('dataset', 'wtbdata_cleaned1.csv');
    tran_file = fullfile('dataset', 'wtbdata_cleaned123.csv');

    try
        data_wind = readtable(wind_file);
        data_wind = data_wind{2:end, 4};  % Extract wind speed column
        data_wind(isnan(data_wind)) = [];

        % Read temperature data (transfer dataset)
        data_tran = readtable(tran_file);
        data_tran = data_tran{2:end, 4};  % Extract temperature column
        data_tran(isnan(data_tran)) = [];
    catch ME
        error('Error reading data files: %s\nMake sure the files exist in the correct location.', ME.message);
    end

    % Ensure both datasets have the same length
    min_length = min(length(data_wind), length(data_tran));
    data_wind = data_wind(1: 5000);
    data_tran = data_tran(1: 5000);

    data_wind_Decomp = seriesDecomp(data_wind, 3);
    data_tran_Decomp = seriesDecomp(data_tran, 7);
    
    % Combine wind speed and temperature data
    data_comb = [data_wind_Decomp, data_tran_Decomp];
    data = [data_wind, data_tran];
    
    % Apply series decomposition
    sequence_length = size(data, 1);
    input_sequence  = data_comb(1:sequence_length - ahead, :);
    output_sequence = data(ahead + 1:sequence_length, :);
end
