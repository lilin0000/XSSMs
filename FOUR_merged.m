% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
tree_depths = [1:1000]; % 尝试不同的树深度
mse_cv = zeros(size(tree_depths));

for i = 1:length(tree_depths)
    mdl = fitrtree(X_train, y_train, 'MaxNumSplits', tree_depths(i), 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树深度
[min_mse, best_depth_idx] = min(mse_cv);
best_depth = tree_depths(best_depth_idx);

% 打印最优树深度
fprintf('最优树深度：%d\n', best_depth);

% 使用最优参数重新训练模型
best_mdl = fitrtree(X_train, y_train, 'MaxNumSplits', best_depth);

% 用新数据进行预测
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

newFeatures = new_data(:, 3:end);

% 使用随机森林回归模型进行预测
predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), predictions];

variableNames = {'longititude', 'latitude', 'dt_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\dt_b_1k.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 最大值标准化旧数据
data(:, 4:end) = (data(:, 4:end) - min(data(:, 4:end))) ./ (max(data(:, 4:end)) - min(data(:, 4:end)));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
tree_depths = [1:1000]; % 尝试不同的树深度
mse_cv = zeros(size(tree_depths));

for i = 1:length(tree_depths)
    mdl = fitrtree(X_train, y_train, 'MaxNumSplits', tree_depths(i), 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树深度
[min_mse, best_depth_idx] = min(mse_cv);
best_depth = tree_depths(best_depth_idx);

% 打印最优树深度
fprintf('最优树深度：%d\n', best_depth);

% 使用最优参数重新训练模型
best_mdl = fitrtree(X_train, y_train, 'MaxNumSplits', best_depth);

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 最大值标准化新数据
new_data(:, 3:end) = (new_data(:, 3:end) - min(new_data(:, 3:end))) ./ (max(new_data(:, 3:end)) - min(new_data(:, 3:end)));

newFeatures = new_data(:, 3:end);

% 使用随机森林回归模型进行预测
predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), predictions];

variableNames = {'longititude', 'latitude', 'dt_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\dt_b_1k_MMS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 标准差标准化旧数据
data(:, 4:end) = (data(:, 4:end) - mean(data(:, 4:end))) ./ std(data(:, 4:end));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
tree_depths = [1:1000]; % 尝试不同的树深度
mse_cv = zeros(size(tree_depths));

for i = 1:length(tree_depths)
    mdl = fitrtree(X_train, y_train, 'MaxNumSplits', tree_depths(i), 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树深度
[min_mse, best_depth_idx] = min(mse_cv);
best_depth = tree_depths(best_depth_idx);

% 打印最优树深度
fprintf('最优树深度：%d\n', best_depth);

% 使用最优参数重新训练模型
best_mdl = fitrtree(X_train, y_train, 'MaxNumSplits', best_depth);

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 标准差标准化新数据
new_data(:, 3:end) = (new_data(:, 3:end) - mean(new_data(:, 3:end))) ./ std(new_data(:, 3:end));

newFeatures = new_data(:, 3:end);

% 使用随机森林回归模型进行预测
predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), predictions];

variableNames = {'longititude', 'latitude', 'dt_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\dt_b_1k_SS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
hidden_layer_sizes = [100,200,300,400,500]; % 尝试不同的隐藏层大小
mse_cv = zeros(size(hidden_layer_sizes));

for i = 1:length(hidden_layer_sizes)
    hidden_size = hidden_layer_sizes(i);
    
    % 创建神经回归网络模型
    net = fitnet(hidden_size);
    
    % 使用交叉验证训练并计算均方误差
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_indices;
    net.divideParam.valInd = test_indices;
    net.trainParam.showWindow = false;
    
    net = train(net, X_train', y_train');
    y_pred = net(X_test');
    
    mse_cv(i) = mean((y_test' - y_pred).^2);
end

% 找到最小交叉验证误差对应的隐藏层大小
[min_mse, best_hidden_size_idx] = min(mse_cv);
best_hidden_size = hidden_layer_sizes(best_hidden_size_idx);

% 打印最优隐藏层大小
fprintf('最优隐藏层大小：%d\n', best_hidden_size);

% 使用最优参数重新训练神经回归网络模型
best_net = fitnet(best_hidden_size);
best_net = train(best_net, X_train', y_train');

% 用新数据进行预测
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);
newFeatures = new_data(:, 3:end);

% 使用训练好的神经回归网络模型进行预测
new_predictions = best_net(newFeatures');

predictedData = [new_data(:, 1:2), new_predictions'];

variableNames = {'longititude', 'latitude', 'nr_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\nr_b_1k.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['神经回归预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 最大值标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - min(data(:, 4:end))) ./ (max(data(:, 4:end)) - min(data(:, 4:end)));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
hidden_layer_sizes = [100,200,300,400,500]; % 尝试不同的隐藏层大小
mse_cv = zeros(size(hidden_layer_sizes));

for i = 1:length(hidden_layer_sizes)
    hidden_size = hidden_layer_sizes(i);
    
    % 创建神经回归网络模型
    net = fitnet(hidden_size);
    
    % 使用交叉验证训练并计算均方误差
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_indices;
    net.divideParam.valInd = test_indices;
    net.trainParam.showWindow = false;
    
    net = train(net, X_train', y_train');
    y_pred = net(X_test');
    
    mse_cv(i) = mean((y_test' - y_pred).^2);
end

% 找到最小交叉验证误差对应的隐藏层大小
[min_mse, best_hidden_size_idx] = min(mse_cv);
best_hidden_size = hidden_layer_sizes(best_hidden_size_idx);

% 打印最优隐藏层大小
fprintf('最优隐藏层大小：%d\n', best_hidden_size);

% 使用最优参数重新训练神经回归网络模型
best_net = fitnet(best_hidden_size);
best_net = train(best_net, X_train', y_train');

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 最大值标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - min(new_data(:, 3:end))) ./ (max(new_data(:, 3:end)) - min(new_data(:, 3:end)));

newFeatures = new_data(:, 3:end);

% 使用训练好的神经回归网络模型进行预测
new_predictions = best_net(newFeatures');

predictedData = [new_data(:, 1:2), new_predictions'];

variableNames = {'longititude', 'latitude', 'nr_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\nr_b_1k_MMS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['神经回归预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 标准差标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - mean(data(:, 4:end))) ./ std(data(:, 4:end));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
hidden_layer_sizes = [100,200,300,400,500]; % 尝试不同的隐藏层大小
mse_cv = zeros(size(hidden_layer_sizes));

for i = 1:length(hidden_layer_sizes)
    hidden_size = hidden_layer_sizes(i);
    
    % 创建神经回归网络模型
    net = fitnet(hidden_size);
    
    % 使用交叉验证训练并计算均方误差
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_indices;
    net.divideParam.valInd = test_indices;
    net.trainParam.showWindow = false;
    
    net = train(net, X_train', y_train');
    y_pred = net(X_test');
    
    mse_cv(i) = mean((y_test' - y_pred).^2);
end

% 找到最小交叉验证误差对应的隐藏层大小
[min_mse, best_hidden_size_idx] = min(mse_cv);
best_hidden_size = hidden_layer_sizes(best_hidden_size_idx);

% 打印最优隐藏层大小
fprintf('最优隐藏层大小：%d\n', best_hidden_size);

% 使用最优参数重新训练神经回归网络模型
best_net = fitnet(best_hidden_size);
best_net = train(best_net, X_train', y_train');

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 标准差标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - mean(new_data(:, 3:end))) ./ std(new_data(:, 3:end));

newFeatures = new_data(:, 3:end);

% 使用训练好的神经回归网络模型进行预测
new_predictions = best_net(newFeatures');

predictedData = [new_data(:, 1:2), new_predictions'];

variableNames = {'longititude', 'latitude', 'nr_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\nr_b_1k_SS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['神经回归预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500,600,700,800,900,1000]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = fitensemble(X_train, y_train, 'LSBoost', num_trees(i), 'Tree', 'type', 'regression', 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练梯度提升回归模型
best_mdl = fitensemble(X_train, y_train, 'LSBoost', best_num_trees, 'Tree', 'type', 'regression');

% 用新数据进行预测
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);
newFeatures = new_data(:, 3:end);

% 使用训练好的梯度提升回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'gb_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\gb_b_1k.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 最大值标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - min(data(:, 4:end))) ./ (max(data(:, 4:end)) - min(data(:, 4:end)));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500,600,700,800,900,1000]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = fitensemble(X_train, y_train, 'LSBoost', num_trees(i), 'Tree', 'type', 'regression', 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练梯度提升回归模型
best_mdl = fitensemble(X_train, y_train, 'LSBoost', best_num_trees, 'Tree', 'type', 'regression');

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 最大值标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - min(new_data(:, 3:end))) ./ (max(new_data(:, 3:end)) - min(new_data(:, 3:end)));

newFeatures = new_data(:, 3:end);

% 使用训练好的梯度提升回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'gb_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\gb_b_1k_MMS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 标准差标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - mean(data(:, 4:end))) ./ std(data(:, 4:end));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500,600,700,800,900,1000]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = fitensemble(X_train, y_train, 'LSBoost', num_trees(i), 'Tree', 'type', 'regression', 'CrossVal', 'on');
    mse_cv(i) = kfoldLoss(mdl);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练梯度提升回归模型
best_mdl = fitensemble(X_train, y_train, 'LSBoost', best_num_trees, 'Tree', 'type', 'regression');

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 标准差标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - mean(new_data(:, 3:end))) ./ std(new_data(:, 3:end));

newFeatures = new_data(:, 3:end);

% 使用训练好的梯度提升回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'gb_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\gb_b_1k_SS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = TreeBagger(num_trees(i), X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel',true));
    predictions = predict(mdl, X_test);
    mse_cv(i) = mean((str2double(predictions) - y_test).^2);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练随机森林回归模型
best_mdl = TreeBagger(best_num_trees, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel',true));

% 用新数据进行预测
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);
newFeatures = new_data(:, 3:end);

% 使用训练好的梯度提升回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'rf_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\rf_b_1k.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 最大值标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - min(data(:, 4:end))) ./ (max(data(:, 4:end)) - min(data(:, 4:end)));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = TreeBagger(num_trees(i), X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));
    predictions = predict(mdl, X_test);
    mse_cv(i) = mean((str2double(predictions) - y_test).^2);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练随机森林回归模型
best_mdl = TreeBagger(best_num_trees, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 最大值标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - min(new_data(:, 3:end))) ./ (max(new_data(:, 3:end)) - min(new_data(:, 3:end)));

newFeatures = new_data(:, 3:end);

% 使用训练好的随机森林回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'rf_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\rf_b_1k_MMS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 标准差标准化旧数据的特征部分
data(:, 4:end) = (data(:, 4:end) - mean(data(:, 4:end))) ./ std(data(:, 4:end));

% 划分数据为训练集和测试集（70% 训练集，30% 测试集）
rng(123); % 设置随机种子以保证可重复性
train_ratio = 0.7;
num_samples = size(data, 1);
num_train = round(train_ratio * num_samples);
indices = randperm(num_samples);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

X_train = data(train_indices, 4:end);
y_train = data(train_indices, 3);
X_test = data(test_indices, 4:end);
y_test = data(test_indices, 3);

% 使用交叉验证选择最优参数
num_trees = [100,200,300,400,500]; % 尝试不同的树的数量
mse_cv = zeros(size(num_trees));

for i = 1:length(num_trees)
    mdl = TreeBagger(num_trees(i), X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));
    predictions = predict(mdl, X_test);
    mse_cv(i) = mean((str2double(predictions) - y_test).^2);
end

% 找到最小交叉验证误差对应的树的数量
[min_mse, best_num_trees_idx] = min(mse_cv);
best_num_trees = num_trees(best_num_trees_idx);

% 打印最优树的数量
fprintf('最优树的数量：%d\n', best_num_trees);

% 使用最优参数重新训练随机森林回归模型
best_mdl = TreeBagger(best_num_trees, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 标准差标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - mean(new_data(:, 3:end))) ./ std(new_data(:, 3:end));

newFeatures = new_data(:, 3:end);

% 使用训练好的随机森林回归模型进行预测
new_predictions = predict(best_mdl, newFeatures);

predictedData = [new_data(:, 1:2), new_predictions];

variableNames = {'longititude', 'latitude', 'rf_b_1k'};
predictedDataTable = array2table(predictedData, 'VariableNames', variableNames);

% 定义输出文件路径
outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\rf_b_1k_SS.csv';

% 将表格数据写入 CSV 文件
writetable(predictedDataTable, outputFilePath);

disp(['预测结果已保存至: ', outputFilePath]);

