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

## Contributors

[Your Name/Team Name]

## License

[Specify your license here]
