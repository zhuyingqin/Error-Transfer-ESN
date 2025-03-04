# Echo State Network (ESN) for Wind Speed Prediction

This project implements an Echo State Network (ESN) for wind speed prediction using MATLAB. It includes evolutionary optimization techniques to improve the network's performance.

## Features

- Implementation of Reservoir Computing using Echo State Networks (ESN)
- Evolutionary optimization of ESN parameters
- Support for parallel processing to speed up computations
- Adaptive error compensation technique
- Visualization of training and test results

## Main Components

- `ETL_ESN.m`: Main script for running experiments
- `testReservoir.m`: Function for testing individual reservoirs
- `test_transfor.m`: Implementation of the ESN and error compensation

## Key Parameters

The following parameters in `ETL_ESN.m` can be adjusted to customize the experiment:

- `config.parallel`: Set to 1 to use parallel processing, 0 otherwise
- `config.res_type`: Type of reservoir ('RoR' in this case)
- `config.num_nodes`: Number of nodes in the reservoir (default: 200)
- `config.num_tests`: Number of tests/runs to perform
- `config.pop_size`: Initial population size for evolutionary optimization
- `config.error_to_check`: Error metric to use ('test' in this case)
- `config.dataset`: Dataset to use ('wind' in this case)
- `config.input_scaling`: Input scaling factor (adjusted in each test)
- `config.k_neighbours`: Number of neighbours to check for novelty search
- `config.p_min_start`: Initial novelty threshold
- `config.p_min_check`: Generation count to dynamically change novelty threshold

Adjust these parameters based on your specific requirements and computational resources.

## Recent Updates

- Added functionality to output the minimum test error across the population
- Implemented error compensation technique in `test_transfor.m`
- Enhanced visualization of results using `plotComparisonAndError` function

## Usage

1. Ensure all required MATLAB toolboxes are installed
2. Set the desired configuration parameters in `ETL_ESN.m`
3. Run `ETL_ESN.m` to start the experiment

## Results

After running the experiment, the script will output the minimum test error achieved across all individuals in the population. This provides a quick insight into the best performance achieved by the evolved ESNs.

## Future Work

- Further optimization of ESN parameters
- Integration with other machine learning techniques
- Expansion to other time series prediction tasks

## Citation

If you use this code in your research, please cite:

```
@article{zhu2025real,
  title={Real-time Error Compensation Transfer Learning with Echo State Networks for Enhanced Wind Power Prediction},
  author={Zhu, Yingqin and Liu, Yue and Wang, Nan and Zhang, ZhaoZhao and Li, YuanQiang},
  journal={Applied Energy},
  volume={379},
  pages={124893},
  year={2025},
  publisher={Elsevier}
}
```

## Contributors

[Your Name/Team Name]

## License

[Specify your license here]
