%% ----------------------------- About -------------------------- %
% To assess the correlation between the clinical improvement and the ATM averages ON/OFF ratio using robust linear regression, as well as the correlation of the clinical improvement with subgroups of edges (cortex_cortex,STN-cortex, STN-STN).
%% ----------------------------- Authors -------------------------- %
% Hasnae AGOURAM
% Date: 13/06/2024
% License: BSD (3-clause)


%% Matlab code for correlation of Clinical Improvement with ATM averages ON/OFF ratio %%
% X corresponds to the ATM averages ON/OFF ratio % and y corresponds to the Clinical Improvement %
% Fit robust linear regression model
[b, stats] = robustfit(X, y,'bisquare',4.685);

% Display the results
disp('Robust Linear Regression Results:');
fprintf('Intercept: %.4f\n', b(1));
fprintf('Slope: %.4f\n', b(2));
fprintf('Standard Errors:\n');
disp(stats.se');
fprintf('t-Statistics:\n');
disp(stats.t');
fprintf('p-Values:\n');
disp(stats.p');

% Find outliers
outliers_ind = find(abs(stats.resid) > stats.mad_s);

% Visualize the data and the robust regression line
figure;
scatter(X, y, 'filled');
hold on;
plot(X(outliers_ind), y(outliers_ind), 'mo', 'LineWidth', 2);
plot(X, b(1) + b(2) * X, 'g');
hold off;
xlabel('ATM averages ON/OFF ratio %');   
ylabel('Clinical Improvement %');
title("Full ATM analysis");
legend_handle = legend('Data', 'Outliers', 'Robust Regression Line', 'Location', 'northwest');
set(legend_handle, 'FontSize', 7);   
grid on;
axis square;

% Display slope and p-value on the plot
slope_text = sprintf('Slope: %.4f', b(2));
p_value_text = sprintf('p-Value: %.4f', stats.p(2));
ylim([60 95])
xL=xlim;
yL=ylim;
text(0.99*xL(2),0.98*yL(2),{slope_text, p_value_text}, 'FontSize', 7 ,'BackgroundColor', 'white', 'EdgeColor', 'black','HorizontalAlignment','right','VerticalAlignment','top');
filename = '/home/user/Bureau/code_linear_regression/figure_paper/panelA.jpg';
resolution = '-r600';
print(gcf, filename, '-djpeg', resolution);

% Matlab code for correlation with subgroups of edges (cortex_cortex, STN-cortex, STN-STN)
%%%%%%%%%%%CORTEX-CORTEX%%%%%

% X corresponds to the Cortex-Cortex edges ON/OFF ratio % and y corresponds to the Clinical Improvement %

% Fit robust linear regression model
[b, stats] = robustfit(X, y,'bisquare',4.685);

% Display the results
disp('Robust Linear Regression Results:');
fprintf('Intercept: %.4f\n', b(1));
fprintf('Slope: %.4f\n', b(2));
fprintf('Standard Errors:\n');
disp(stats.se');
fprintf('t-Statistics:\n');
disp(stats.t');
fprintf('p-Values:\n');
disp(stats.p');

% Find outliers
outliers_ind = find(abs(stats.resid) > stats.mad_s);

% Visualize the data and the robust regression line
figure;
scatter(X, y, 'filled');
hold on;
plot(X(outliers_ind), y(outliers_ind), 'mo', 'LineWidth', 2);
plot(X, b(1) + b(2) * X, 'g');
hold off;
xlabel('Cortex-Cortex edges ON/OFF ratio %');   
ylabel('Clinical Improvement %');
title("Cortex-Cortex analysis");
legend_handle = legend('Data', 'Outliers', 'Robust Regression Line', 'Location', 'northwest');
set(legend_handle, 'FontSize', 7);   
grid on;
axis square;

% Display slope and p-value on the plot
slope_text = sprintf('Slope: %.4f', b(2));
p_value_text = sprintf('p-Value: %.4f', stats.p(2));
xL=xlim;
yL=ylim;
text(0.99*xL(2),0.98*yL(2),{slope_text, p_value_text}, 'FontSize', 7 ,'BackgroundColor', 'white', 'EdgeColor', 'black','HorizontalAlignment','right','VerticalAlignment','top');
filename = '/home/user/Bureau/code_linear_regression/figure_paper/panelB.jpg';
resolution = '-r600';
print(gcf, filename, '-djpeg', resolution);

%%%%%%%%%%%STN-CORTEX%%%%%
% X corresponds to the STN-Cortex edges ON/OFF ratio % and y corresponds to the Clinical Improvement %

% Fit robust linear regression model
[b, stats] = robustfit(X, y,'bisquare',4.685);

% Display the results
disp('Robust Linear Regression Results:');
fprintf('Intercept: %.4f\n', b(1));
fprintf('Slope: %.4f\n', b(2));
fprintf('Standard Errors:\n');
disp(stats.se');
fprintf('t-Statistics:\n');
disp(stats.t');
fprintf('p-Values:\n');
disp(stats.p');

% Find outliers
outliers_ind = find(abs(stats.resid) > stats.mad_s);

% Visualize the data and the robust regression line
figure;
scatter(X, y, 'filled');
hold on;
plot(X(outliers_ind), y(outliers_ind), 'mo', 'LineWidth', 2);
plot(X, b(1) + b(2) * X, 'g');
hold off;
xlabel('STN-Cortex edges ON/OFF ratio %');
ylabel('Clinical Improvement %');
title("STN-Cortex analysis");
legend_handle = legend('Data', 'Outliers', 'Robust Regression Line', 'Location', 'northwest');
set(legend_handle, 'FontSize', 6);   
grid on;
axis square;
% Display slope and p-value on the plot
slope_text = sprintf('Slope: %.4f', b(2));
p_value_text = sprintf('p-Value: %.4f', stats.p(2));
ylim([60 95])
xlim([100 180])
xL=xlim;
yL=ylim;
text(0.99*xL(2),0.98*yL(2),{slope_text, p_value_text}, 'FontSize', 7 ,'BackgroundColor', 'white', 'EdgeColor', 'black','HorizontalAlignment','right','VerticalAlignment','top');
filename = '/home/user/Bureau/code_linear_regression/figure_paper/panelC.jpg';
resolution = '-r600';
print(gcf, filename, '-djpeg', resolution);

%%%%%%%%%%%STN-STN%%%%%
% X corresponds to the STN-STN edges ON/OFF ratio % and y corresponds to the Clinical Improvement %
 
% Fit robust linear regression model
[b, stats] = robustfit(X, y,'bisquare',4.685);

% Display the results
disp('Robust Linear Regression Results:');
fprintf('Intercept: %.4f\n', b(1));
fprintf('Slope: %.4f\n', b(2));
fprintf('Standard Errors:\n');
disp(stats.se');
fprintf('t-Statistics:\n');
disp(stats.t');
fprintf('p-Values:\n');
disp(stats.p');

% Find outliers
outliers_ind = find(abs(stats.resid) > stats.mad_s);

% Visualize the data and the robust regression line
figure;
scatter(X, y, 'filled');
hold on;
plot(X(outliers_ind), y(outliers_ind), 'mo', 'LineWidth', 2);
plot(X, b(1) + b(2) * X, 'g');
hold off;
xlabel('STN-STN edges ON/OFF ratio %');
ylabel('Clinical Improvement %');
title("STN-STN analysis");
legend_handle = legend('Data', 'Outliers', 'Robust Regression Line', 'Location', 'northwest');
set(legend_handle, 'FontSize', 6);   
grid on;
axis square;
% Display slope and p-value on the plot
slope_text = sprintf('Slope: %.4f', b(2));
p_value_text = sprintf('p-Value: %.4f', stats.p(2));
ylim([60 95])
xlim([80 180])
xL=xlim;
yL=ylim;
text(0.99*xL(2),0.98*yL(2),{slope_text, p_value_text}, 'FontSize', 7 ,'BackgroundColor', 'white', 'EdgeColor', 'black','HorizontalAlignment','right','VerticalAlignment','top');
filename = '/home/user/Bureau/code_linear_regression/figure_paper/panelD.jpg';
resolution = '-r600';
print(gcf, filename, '-djpeg', resolution);


