function [residual] = seriesDecomp(x, kernelSize)
    % seriesDecomp - 时间序列分解函数
    % x - 输入的时间序列
    % kernelSize - 移动平均的窗口大小
    % 返回去趋势化后的序列 (residual) 和移动平均 (movingMean)

    % 计算移动平均值
    movingMean = movmean(x, kernelSize);

    % 计算剩余部分（原序列 - 移动平均）
    residual = x - movingMean;
end
