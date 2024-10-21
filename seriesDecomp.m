function result = seriesDecomp(x, kernelSize)
    % seriesDecomp - Time series decomposition function
    % x - Input time series
    % kernelSize - Window size for moving average
    % Returns a matrix containing the detrended series (residual) and moving average
    % result: first column is residual, second column is movingMean

    % Ensure x is a column vector
    x = x(:);

    % Calculate moving average
    movingMean = movmean(x, kernelSize);

    % Calculate residual (original series - moving average)
    residual = x - movingMean;

    % Combine results into a matrix
    result = [residual, movingMean];
end
