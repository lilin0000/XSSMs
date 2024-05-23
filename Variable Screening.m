% Set global random seed
rng(123);

% Load data from CSV file
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1); % Assuming header in first row

% Split data into features (X) and target variable (y)
X = data(:, 4:end);
y = data(:, 3);

% Split data into training, validation, and test sets (5:3:2)
[trainInd, remainingInd] = dividerand(size(X, 1), 0.5, 0.5); % 划分训练集和剩下的数据
[valInd, testInd] = dividerand(length(remainingInd), 0.6, 0.4); % 从剩下的数据划分验证集和测试集

X_train = X(trainInd, :);
y_train = y(trainInd);
X_val = X(remainingInd(valInd), :);
y_val = y(remainingInd(valInd));
X_test = X(remainingInd(testInd), :);
y_test = y(remainingInd(testInd));

% Perform Decision Tree Regression
model = fitrtree(X_train, y_train);

% Predictions on training, validation, and test sets
y_train_pred = predict(model, X_train);
y_val_pred = predict(model, X_val);
y_test_pred = predict(model, X_test);

% Calculate evaluation metrics
correlation_train = corr(y_train, y_train_pred);
r2_train = correlation_train^2;
bias_train = mean(y_train_pred - y_train);
ubrmse_train = sqrt(sum((y_train - y_train_pred).^2) / length(y_train));
mae_train = mean(abs(y_train - y_train_pred));

%correlation_val = corr(y_val, y_val_pred);
%r2_val = correlation_val^2;
%bias_val = mean(y_val_pred - y_val);
%ubrmse_val = sqrt(sum((y_val - y_val_pred).^2) / length(y_val));
%mae_val = mean(abs(y_val - y_val_pred));

correlation_test = corr(y_test, y_test_pred);
r2_test = correlation_test^2;
bias_test = mean(y_test_pred - y_test);
ubrmse_test = sqrt(sum((y_test - y_test_pred).^2) / length(y_test));
mae_test = mean(abs(y_test - y_test_pred));

% Display results
%fprintf('Training Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_train, correlation_train, bias_train, ubrmse_train, mae_train);

%fprintf('\nValidation Set Metrics:\n');
%fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    %r2_val, correlation_val, bias_val, ubrmse_val, mae_val);

%fprintf('\nTest Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_test, correlation_test, bias_test, ubrmse_test, mae_test);





% Set global random seed
rng(123);

% Load data from CSV file
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1); % Assuming header in first row

% Split data into features (X) and target variable (y)
X = data(:, 4:end);
y = data(:, 3);

% Split data into training, validation, and test sets (5:3:2)
[trainInd, remainingInd] = dividerand(size(X, 1), 0.5, 0.5); % 划分训练集和剩下的数据
[valInd, testInd] = dividerand(length(remainingInd), 0.6, 0.4); % 从剩下的数据划分验证集和测试集

X_train = X(trainInd, :);
y_train = y(trainInd);
X_val = X(remainingInd(valInd), :);
y_val = y(remainingInd(valInd));
X_test = X(remainingInd(testInd), :);
y_test = y(remainingInd(testInd));

% Create a feedforward neural network
hiddenLayerSize = 500; % Number of neurons in the hidden layer
net = feedforwardnet(hiddenLayerSize);

% Train the neural network
net = train(net, X_train', y_train');

% Predictions on training, validation, and test sets
y_train_pred = net(X_train');
y_val_pred = net(X_val');
y_test_pred = net(X_test');

% Calculate evaluation metrics
correlation_train = corr(y_train, y_train_pred');
r2_train = correlation_train^2;
bias_train = mean(y_train_pred' - y_train);
ubrmse_train = sqrt(sum((y_train - y_train_pred').^2) / length(y_train));
mae_train = mean(abs(y_train - y_train_pred'));

%correlation_val = corr(y_val, y_val_pred);
%r2_val = correlation_val^2;
%bias_val = mean(y_val_pred - y_val);
%ubrmse_val = sqrt(sum((y_val - y_val_pred).^2) / length(y_val));
%mae_val = mean(abs(y_val - y_val_pred));

correlation_test = corr(y_test, y_test_pred');
r2_test = correlation_test^2;
bias_test = mean(y_test_pred' - y_test);
ubrmse_test = sqrt(sum((y_test - y_test_pred').^2) / length(y_test));
mae_test = mean(abs(y_test - y_test_pred'));

% Display results
%fprintf('Training Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_train, correlation_train, bias_train, ubrmse_train, mae_train);

%fprintf('\nValidation Set Metrics:\n');
%fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    %r2_val, correlation_val, bias_val, ubrmse_val, mae_val);

%fprintf('\nTest Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_test, correlation_test, bias_test, ubrmse_test, mae_test);





% Set global random seed
rng(123);

% Load data from CSV file
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1); % Assuming header in first row

% Split data into features (X) and target variable (y)
X = data(:, 4:end);
y = data(:, 3);

% Split data into training, validation, and test sets (5:3:2)
[trainInd, remainingInd] = dividerand(size(X, 1), 0.5, 0.5); % 划分训练集和剩下的数据
[valInd, testInd] = dividerand(length(remainingInd), 0.6, 0.4); % 从剩下的数据划分验证集和测试集

X_train = X(trainInd, :);
y_train = y(trainInd);
X_val = X(remainingInd(valInd), :);
y_val = y(remainingInd(valInd));
X_test = X(remainingInd(testInd), :);
y_test = y(remainingInd(testInd));

% Perform Gradient Boosting Regression with default parameters
model = fitrensemble(X_train, y_train);

% Predictions on training, validation, and test sets
y_train_pred = predict(model, X_train);
y_val_pred = predict(model, X_val);
y_test_pred = predict(model, X_test);

% Calculate evaluation metrics
correlation_train = corr(y_train, y_train_pred);
r2_train = correlation_train^2;
bias_train = mean(y_train_pred - y_train);
ubrmse_train = sqrt(sum((y_train - y_train_pred).^2) / length(y_train));
mae_train = mean(abs(y_train - y_train_pred));

%correlation_val = corr(y_val, y_val_pred);
%r2_val = correlation_val^2;
%bias_val = mean(y_val_pred - y_val);
%ubrmse_val = sqrt(sum((y_val - y_val_pred).^2) / length(y_val));
%mae_val = mean(abs(y_val - y_val_pred));

correlation_test = corr(y_test, y_test_pred);
r2_test = correlation_test^2;
bias_test = mean(y_test_pred - y_test);
ubrmse_test = sqrt(sum((y_test - y_test_pred).^2) / length(y_test));
mae_test = mean(abs(y_test - y_test_pred));

% Display results
%fprintf('Training Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_train, correlation_train, bias_train, ubrmse_train, mae_train);

%fprintf('\nValidation Set Metrics:\n');
%fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    %r2_val, correlation_val, bias_val, ubrmse_val, mae_val);

%fprintf('\nTest Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_test, correlation_test, bias_test, ubrmse_test, mae_test);





% Set global random seed
rng(123);

% Load data from CSV file
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1); % Assuming header in first row

% Split data into features (X) and target variable (y)
X = data(:, 4:end);
y = data(:, 3);

% Split data into training, validation, and test sets (5:3:2)
[trainInd, remainingInd] = dividerand(size(X, 1), 0.5, 0.5); % 划分训练集和剩下的数据
[valInd, testInd] = dividerand(length(remainingInd), 0.6, 0.4); % 从剩下的数据划分验证集和测试集

X_train = X(trainInd, :);
y_train = y(trainInd);
X_val = X(remainingInd(valInd), :);
y_val = y(remainingInd(valInd));
X_test = X(remainingInd(testInd), :);
y_test = y(remainingInd(testInd));

% Perform Random Forest Regression with default parameters
numTrees = 100; % Number of trees in the forest
model = TreeBagger(numTrees, X_train, y_train, 'Method', 'regression');

% Predictions on training, validation, and test sets
y_train_pred = predict(model, X_train);
y_val_pred = predict(model, X_val);
y_test_pred = predict(model, X_test);

% Calculate evaluation metrics
correlation_train = corr(y_train, y_train_pred);
r2_train = correlation_train^2;
bias_train = mean(y_train_pred - y_train);
ubrmse_train = sqrt(sum((y_train - y_train_pred).^2) / length(y_train));
mae_train = mean(abs(y_train - y_train_pred));

%correlation_val = corr(y_val, y_val_pred);
%r2_val = correlation_val^2;
%bias_val = mean(y_val_pred - y_val);
%ubrmse_val = sqrt(sum((y_val - y_val_pred).^2) / length(y_val));
%mae_val = mean(abs(y_val - y_val_pred));

correlation_test = corr(y_test, y_test_pred);
r2_test = correlation_test^2;
bias_test = mean(y_test_pred - y_test);
ubrmse_test = sqrt(sum((y_test - y_test_pred).^2) / length(y_test));
mae_test = mean(abs(y_test - y_test_pred));

% Display results
%fprintf('Training Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_train, correlation_train, bias_train, ubrmse_train, mae_train);

%fprintf('\nValidation Set Metrics:\n');
%fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    %r2_val, correlation_val, bias_val, ubrmse_val, mae_val);

%fprintf('\nTest Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_test, correlation_test, bias_test, ubrmse_test, mae_test);





% Set global random seed
rng(123);

% Load data from CSV file
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1); % Assuming header in first row

% Split data into features (X) and target variable (y)
X = data(:, 4:end);
y = data(:, 3);

% Split data into training, validation, and test sets (5:3:2)
[trainInd, remainingInd] = dividerand(size(X, 1), 0.5, 0.5); % 划分训练集和剩下的数据
[valInd, testInd] = dividerand(length(remainingInd), 0.6, 0.4); % 从剩下的数据划分验证集和测试集

X_train = X(trainInd, :);
y_train = y(trainInd);
X_val = X(remainingInd(valInd), :);
y_val = y(remainingInd(valInd));
X_test = X(remainingInd(testInd), :);
y_test = y(remainingInd(testInd));

% Create a cell array of regression models
models = {
    'DecisionTree', ...
    'NeuralNetwork', ...
    'GradientBoosting', ...
    'RandomForest'
};

% Initialize an empty cell array to store individual model predictions
model_predictions = cell(size(models));

% Train individual regression models and make predictions
for i = 1:length(models)
    switch models{i}
        case 'DecisionTree'
            model = fitrtree(X_train, y_train);
        case 'GradientBoosting'
            model = fitrensemble(X_train, y_train);
        case 'NeuralNetwork'
            hiddenLayerSize = 500; % Number of neurons in the hidden layer
            net = feedforwardnet(hiddenLayerSize);
            net = train(net, X_train', y_train');
        case 'RandomForest'
            numTrees = 100;
            model = TreeBagger(numTrees, X_train, y_train, 'Method', 'regression');
    end
    y_train_pred = predict(model, X_train);
    y_val_pred = predict(model, X_val);
    y_test_pred = predict(model, X_test);
    
    model_predictions{i} = y_val_pred; % Store validation set predictions
end

% Perform majority voting to get the final prediction
y_val_pred = mode(cell2mat(model_predictions), 2);

% Calculate evaluation metrics for training set
correlation_train = corr(y_train, y_train_pred);
r2_train = correlation_train^2;
bias_train = mean(y_train_pred - y_train);
ubrmse_train = sqrt(sum((y_train - y_train_pred).^2) / length(y_train));
mae_train = mean(abs(y_train - y_train_pred));

% Calculate evaluation metrics for validation set
%correlation_val = corr(y_val, y_val_pred);
%r2_val = correlation_val^2;
%bias_val = mean(y_val_pred - y_val);
%ubrmse_val = sqrt(sum((y_val - y_val_pred).^2) / length(y_val));
%mae_val = mean(abs(y_val - y_val_pred));

% Calculate evaluation metrics for test set
correlation_test = corr(y_test, y_test_pred);
r2_test = correlation_test^2;
bias_test = mean(y_test_pred - y_test);
ubrmse_test = sqrt(sum((y_test - y_test_pred).^2) / length(y_test));
mae_test = mean(abs(y_test - y_test_pred));

% Display results for training, validation, and test sets
%fprintf('Training Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_train, correlation_train, bias_train, ubrmse_train, mae_train);

%fprintf('\nValidation Set Metrics:\n');
%fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    %r2_val, correlation_val, bias_val, ubrmse_val, mae_val);

%fprintf('\nTest Set Metrics:\n');
fprintf(' %.5f,  %.5f,  %.5f, %.5f, %.5f\n', ...
    r2_test, correlation_test, bias_test, ubrmse_test, mae_test);
