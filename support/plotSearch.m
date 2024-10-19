function plotSearch(database, gen,config)

all_behaviours = reshape([database.behaviours],length(database(1).behaviours),length(database))';

% Add specific parameter to observe here
% Example: plot a particular parameter:
% param = [database.leak_rate]';
% param = 1:length(all_behaviours);
% param = [database.connectivity]';
param = [database.test_error]';

% set(1,'currentFigure',config.figure_array(0))
title(strcat('Gen:',num2str(gen)))
% v = 1:length(config.metrics);
% C = nchoosek(v,2);
% 
% if size(C,1) > 3
%     num_plot_x = ceil(size(C,1)/2);
%     num_plot_y = 2;
% else
%     num_plot_x = 3;
%     num_plot_y = 1;
% end

% for i = 1:size(C,1)
%     subplot(num_plot_x,num_plot_y,i)
%     scatter(all_behaviours(:,C(i,1)),all_behaviours(:,C(i,2)),20,param,'filled')
%     
%     % Replace with desired parameter:
%     % scatter(all_behaviours(:,C(i,1)),all_behaviours(:,C(i,2)),20,lr,'filled')
%     
%     xlabel(config.metrics(C(i,1)))
%     ylabel(config.metrics(C(i,2)))
%     colormap('jet')
% end
scatter3(all_behaviours(:,1),all_behaviours(:,2),all_behaviours(:,3),20,param,'filled')%以KR,GR,MC为三个指标，参数为测试误差

colorbar
% caxis([0 1]);
drawnow
end