clc;
clear all;
close all;
tic

stock_data = readtable('new.csv', 'VariableNamingRule', 'preserve');  
% Extract relevant features
prediction = {'Open', 'Close'};
features = {'High', 'Low', 'Volume', 'compound', 'RSI', 'SMA'};
actual_price = table2array(stock_data(:, prediction));
stock_data_matrix = table2array(stock_data(:, features));

rows_used = 1748;  

% Column-wise Z-score Normalization (Limit Till rows_used)

means_actual = mean(actual_price(1:rows_used, :), 1);  % Mean per column
stds_actual = std(actual_price(1:rows_used, :), 1);    % Std per column
actual_price(1:rows_used, :) = (actual_price(1:rows_used, :) - means_actual) ./ stds_actual;

means_features = mean(stock_data_matrix(1:rows_used, :), 1);
stds_features = std(stock_data_matrix(1:rows_used, :), 1);
stock_data_matrix(1:rows_used, :) = (stock_data_matrix(1:rows_used, :) - means_features) ./ stds_features;
%  PCA for First 3 Features (Limit Till rows_used)
pca_features = stock_data_matrix(1:rows_used, 1:3);
[pca_coeff, score, latent] = pca(pca_features);
disp('PCA Coefficient Matrix (coeff):');
disp(pca_coeff);
disp('Eigenvalue Matrix (latent):');
disp(latent);

node1 = score(:, 1);
node2 = stock_data_matrix(1:rows_used, 5);
node3 = stock_data_matrix(1:rows_used, 6);
node4 = stock_data_matrix(1:rows_used, 4);
nodes = [node1, node2, node3, node4];

filename = 'nodes_data.csv';
headers = {'Node1', 'Node2', 'Node3', 'Node4'};
T = array2table(nodes, 'VariableNames', headers);
writetable(T, filename);

M = 4;  % Number of input features
N = 4;  % Number of nodes
window_size = 7;  % Window for past data
total_data_pt = rows_used;
num_samples = total_data_pt - window_size;
nodes_input = zeros(num_samples, window_size * N);
nodes_target = zeros(num_samples, 1);

% Create sliding windows for LMS input
for i = 1:num_samples
    input_window = nodes(i:i+window_size-1, :);
    nodes_input(i, :) = reshape(input_window', 1, []);  % Flatten to 1 row
    nodes_target(i, :) = actual_price(i+window_size, 1);  % Next day's Open price
end

metropolis = [0  0.33 0.33 0.34;
              0.33 0  0.34 0.33;
              0.33 0.34 0  0.33;
              0.34 0.33 0.33 0];
metropolis = metropolis ./ sum(metropolis, 2);


N_EXPER = 1;
N_ITER = num_samples;  % Limited iterations to avoid errors beyond data range
rng(0);
w_in = randn(4, 1);  % Initial weight initialization
step_size = 0.004 * ones(N, 1);  % Same step size for all nodes
MSE = zeros(N, N_ITER);
EMSE = zeros(N, N_ITER);
MSE1 = MSE; EMSE1 = EMSE;

[MSE, EMSE, w] = diffusion_LMS(step_size, nodes, nodes_target, w_in, window_size, N, N_ITER, metropolis, MSE, EMSE);
%[MSE1, EMSE1, MSD1, w1] = d22_LMS_noncop_fed(step_size, nodes, nodes_target, w_in, window_size, N, N_ITER, MSE1, EMSE1, MSD1);
[MSE1, EMSE1, w1] = diffusion_LMS_noncop(step_size, nodes,nodes_target, w_in, M, N, N_ITER, metropolis, MSE1, EMSE1);

% Final Weight Calculations
final_weights = mean(w, 2);
final_weights = final_weights(end - N + 1:end);
weight_changes = final_weights - w_in;
%final_weights_normalized = (w_in + weight_changes);

% Reverse Z-Score Normalization
nodes_means = mean(nodes(1:rows_used, :), 1);
nodes_stds = std(nodes(1:rows_used, :), 1);
nodes_means_expanded = repmat(nodes_means, 1, window_size);
nodes_stds_expanded = repmat(nodes_stds, 1, window_size);

nodes_input_original = (nodes_input(end, :) .* nodes_stds_expanded) + nodes_means_expanded;
nodes_input_original_matrix = reshape(nodes_input_original, N, window_size);
nodes_input_original_avg = mean(nodes_input_original_matrix, 2);

final_weights = (final_weights .* nodes_stds') + nodes_means';
predicted_open_price = final_weights' * nodes_input_original_avg;
real_predicted_price = (predicted_open_price * stds_actual(1)) + means_actual(1);


disp('Predicted Open Price for Next Day (Actual Scale) using d11_LMS:');
disp(real_predicted_price);
disp('Final Normalized Weights:');
disp(['[', num2str(final_weights', ' %.4f;'), ']']);

% Without HW LMS Prediction
final_weights1 = mean(w1, 2);
final_weights1 = final_weights1(end - N + 1:end);
weight_changes1 = final_weights1 - w_in;
%final_weights_normalized1 = (w_in + weight_changes1);

predicted_open_price1 = final_weights1' * nodes_input_original_avg;
real_predicted_price1 = (predicted_open_price1 * stds_actual(1)) + means_actual(1);

disp('Predicted Open Price for Next Day (Actual Scale) using d222_LMS:');
disp(real_predicted_price1);
disp('Final Normalized Weights:');
disp(['[', num2str(final_weights1', ' %.4f;'), ']']);

%Testing Data
% Extract Test Data
test_rows_start = rows_used + 1;
test_rows_end = size(stock_data_matrix, 1);
test_means_features = mean(stock_data_matrix(test_rows_start:test_rows_end, :), 1);
test_stds_features = std(stock_data_matrix(test_rows_start:test_rows_end, :), 1);
test_means_actual = mean(actual_price(test_rows_start:test_rows_end, 1), 1);
test_stds_actual = std(actual_price(test_rows_start:test_rows_end, 1), 1);
test_features = (stock_data_matrix(test_rows_start:test_rows_end, :) - test_means_features) ./ test_stds_features;
test_actual_price = (actual_price(test_rows_start:test_rows_end, 1) - test_means_actual) / test_stds_actual;

%  Apply PCA on Test Features Separately

test_pca_features = test_features(:, 1:3);
[pca_coeff_test, score_test, latent_test] = pca(test_pca_features); 
node1_test = score_test(:, 1); % Use test PCA transformation

% Select other nodes
node2_test = test_features(:, 5);
node3_test = test_features(:, 6);
node4_test = test_features(:, 4);
test_nodes = [node1_test, node2_test, node3_test, node4_test];

% Prepare test data and reverse normalize it
test_nodes_means = mean(test_nodes, 1);
test_nodes_stds = std(test_nodes, 1);
test_nodes_original = (test_nodes .* test_nodes_stds) + test_nodes_means;
%  Predict Open Prices for Test Set
num_test_samples = size(test_nodes_original, 1);
predicted_test_prices = zeros(num_test_samples, 1);
for i = 1:num_test_samples
    predicted_test_prices(i) = final_weights' * test_nodes_original(i, :)';
end

real_predicted_test_prices = (predicted_test_prices * test_stds_actual(1)) + test_means_actual(1);
test_actual_price_adjusted = (test_actual_price * test_stds_actual(1)) + test_means_actual(1);  % Reverse normalize actual price
error_percentage = abs(test_actual_price_adjusted - real_predicted_test_prices) ./ test_actual_price_adjusted * 100;

%  Compute Overall Accuracy
mean_error = mean(error_percentage); % Average error percentage
accuracy = 100 - mean_error; % Accuracy in %
disp('--- d11_LMS_fed (Cooperative) Testing Results ---');
disp(['Mean Error Percentage: ', num2str(mean_error, '%.2f'), '%']);
disp(['Overall Model Accuracy: ', num2str(accuracy, '%.2f'), '%']);

% Compute MAE (Mean Absolute Error)
mae = mean(abs(test_actual_price_adjusted - real_predicted_test_prices));

% Compute RMSE (Root Mean Squared Error)
rmse = sqrt(mean((test_actual_price_adjusted - real_predicted_test_prices).^2));

% Compute R-squared (Coefficient of Determination)
ss_total = sum((test_actual_price_adjusted - mean(test_actual_price_adjusted)).^2);
ss_residual = sum((test_actual_price_adjusted - real_predicted_test_prices).^2);
r_squared = 1 - (ss_residual / ss_total);
disp(['Mean Absolute Error (MAE): ', num2str(mae, '%.4f')]);
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse, '%.4f')]);
disp(['R-Squared (R²): ', num2str(r_squared, '%.4f')]);

% Predict Test Prices using final_weights_normalized1 (Non-Cooperative LMS)
predicted_test_prices1 = zeros(num_test_samples, 1);

for i = 1:num_test_samples
    predicted_test_prices1(i) = final_weights1' * test_nodes_original(i, :)';
end

real_predicted_test_prices1 = (predicted_test_prices1 * test_stds_actual(1)) + test_means_actual(1);

% Compute Error Metrics for final_weights_normalized1
error_percentage1 = abs(test_actual_price_adjusted - real_predicted_test_prices1) ./ test_actual_price_adjusted * 100;
mean_error1 = mean(error_percentage1);
accuracy1 = 100 - mean_error1;

% Compute MAE
mae1 = mean(abs(test_actual_price_adjusted - real_predicted_test_prices1));

% Compute RMSE
rmse1 = sqrt(mean((test_actual_price_adjusted - real_predicted_test_prices1).^2));

% Compute R-Squared (R²)
ss_total1 = sum((test_actual_price_adjusted - mean(test_actual_price_adjusted)).^2);
ss_residual1 = sum((test_actual_price_adjusted - real_predicted_test_prices1).^2);
r_squared1 = 1 - (ss_residual1 / ss_total1);
disp('--- d222_LMS (Non-Cooperative) Testing Results ---');
disp(['Mean Error Percentage: ', num2str(mean_error1, '%.2f'), '%']);
disp(['Overall Model Accuracy: ', num2str(accuracy1, '%.2f'), '%']);
disp(['Mean Absolute Error (MAE): ', num2str(mae1, '%.4f')]);
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse1, '%.4f')]);
disp(['R-Squared (R²): ', num2str(r_squared1, '%.4f')]);
disp(['Final Test Day Predicted Open Price (With HW): ', num2str(real_predicted_test_prices(end-5), '%.4f')]);
disp(['Final Test Day Predicted Open Price (Without HW): ', num2str(real_predicted_test_prices1(end-5), '%.4f')]);
% Plot Actual vs Predicted Prices for d11_LMS_fed and d222_LMS
figure;
plot(test_actual_price_adjusted, 'b', 'LineWidth', 1.5, 'DisplayName', 'Actual Price');
hold on;
plot(real_predicted_test_prices, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Price (d11_LMS)');
plot(real_predicted_test_prices1, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Price (d222_LMS)');
xlabel('Sample Index', 'FontSize', 16);
ylabel('Open Price (Actual Scale)', 'FontSize', 16);
title('Actual vs Predicted Open Prices (Cooperative vs Non-Cooperative)', 'FontSize', 16);
legend('show');
grid on;
hold off;

figure;
plot(1:N_ITER, 10*log10(EMSE), 'r', 1:N_ITER, 10*log10(EMSE1), 'b');
xlabel('Iteration i', 'FontSize', 16);
ylabel('Steady-state EMSE [dB]', 'FontSize', 16);
grid on;
labels = {'Node 1', 'Node 2', 'Node 3', 'Node 4'};

figure;
plot(10*log10(MSE(1, :)));
xlabel('Node 1');
ylabel('Steady-state MSE [dB]', 'FontSize', 16);
grid on;
figure;
plot(10*log10(MSE(2, :)));
xlabel('Node 2');
ylabel('Steady-state MSE [dB]', 'FontSize', 16);
grid on;
figure;
plot(10*log10(MSE(3, :)));
xlabel('Node 3');
ylabel('Steady-state MSE [dB]', 'FontSize', 16);
grid on;
figure;
plot(10*log10(MSE(4, :)));
xlabel('Node 4');
ylabel('Steady-state MSE [dB]', 'FontSize', 16);
grid on;

figure;
plot(1:N_ITER, 10*log10(MSE), 'r', 1:N_ITER, 10*log10(MSE1), 'b');
xlabel('Iteration i', 'FontSize', 16);
ylabel('Steady-state MSE [dB]', 'FontSize', 16);
grid on;

toc