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

DT = fitrtree(X_train, y_train, 'MaxNumSplits', 921);
NR = fitnet(100)
NR = train(NR, X_train', y_train');
GB = fitensemble(X_train, y_train, 'LSBoost', 1000, 'Tree', 'type', 'regression');
RF = TreeBagger(100, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));

% 针对每个模型进行预测
y_pred_DT = predict(DT, X_test);
y_pred_NR = NR(X_test')';
y_pred_GB = predict(GB, X_test);
y_pred_RF = predict(RF, X_test);

% 将每个模型的预测结果合并为一个矩阵
all_predictions = [y_pred_DT, y_pred_NR, y_pred_GB, y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
ensemble_predictions = mean(all_predictions, 2);

% 计算投票回归器的均方误差
mse_ensemble = mean((y_test - ensemble_predictions).^2);

fprintf('投票回归器均方误差：%f\n', mse_ensemble);

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);
newFeatures = new_data(:, 3:end);

% 针对每个模型进行预测
new_y_pred_DT = predict(DT, newFeatures);
new_y_pred_NR = NR(newFeatures')';
new_y_pred_GB = predict(GB, newFeatures);
new_y_pred_RF = predict(RF, newFeatures);

% 将每个模型的预测结果合并为一个矩阵
new_all_predictions = [new_y_pred_DT, new_y_pred_NR, new_y_pred_GB, new_y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
new_ensemble_predictions = mean(new_all_predictions, 2);

% 合并新数据的前两列和投票回归器的预测结果
new_predictedData = [new_data(:, 1:2), new_ensemble_predictions];

% 定义新数据的列名
new_variableNames = {'longititude', 'latitude', 'vr_b_1k'};

% 创建包含预测结果的新数据表
new_predictedDataTable = array2table(new_predictedData, 'VariableNames', new_variableNames);

% 定义输出文件路径
new_outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\vr_b_1k.csv';

% 将新数据表写入 CSV 文件
writetable(new_predictedDataTable, new_outputFilePath);

disp(['预测结果已保存至: ', new_outputFilePath]);





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

DT = fitrtree(X_train, y_train, 'MaxNumSplits', 921);
NR = fitnet(100);
NR = train(NR, X_train', y_train');
GB = fitensemble(X_train, y_train, 'LSBoost', 1000, 'Tree', 'type', 'regression');
RF = TreeBagger(100, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));

% 针对每个模型进行预测
y_pred_DT = predict(DT, X_test);
y_pred_NR = NR(X_test')';
y_pred_GB = predict(GB, X_test);
y_pred_RF = predict(RF, X_test);

% 将每个模型的预测结果合并为一个矩阵
all_predictions = [y_pred_DT, y_pred_NR, y_pred_GB, y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
ensemble_predictions = mean(all_predictions, 2);

% 计算投票回归器的均方误差
mse_ensemble = mean((y_test - ensemble_predictions).^2);

fprintf('投票回归器均方误差：%f\n', mse_ensemble);

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 最大值标准化新数据的特征部分
new_data(:, 3:end) = (new_data(:, 3:end) - min(new_data(:, 3:end))) ./ (max(new_data(:, 3:end)) - min(new_data(:, 3:end)));

newFeatures = new_data(:, 3:end);

% 针对每个模型进行预测
new_y_pred_DT = predict(DT, newFeatures);
new_y_pred_NR = NR(newFeatures')';
new_y_pred_GB = predict(GB, newFeatures);
new_y_pred_RF = predict(RF, newFeatures);

% 将每个模型的预测结果合并为一个矩阵
new_all_predictions = [new_y_pred_DT, new_y_pred_NR, new_y_pred_GB, new_y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
new_ensemble_predictions = mean(new_all_predictions, 2);

% 合并新数据的前两列和投票回归器的预测结果
new_predictedData = [new_data(:, 1:2), new_ensemble_predictions];

% 定义新数据的列名
new_variableNames = {'longititude', 'latitude', 'vr_b_1k'};

% 创建包含预测结果的新数据表
new_predictedDataTable = array2table(new_predictedData, 'VariableNames', new_variableNames);

% 定义输出文件路径
new_outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\vr_b_1k_MMS.csv';

% 将新数据表写入 CSV 文件
writetable(new_predictedDataTable, new_outputFilePath);

disp(['预测结果已保存至: ', new_outputFilePath]);





% 导入数据
data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_9k.csv', 1, 0);

% 标准差标准化旧数据的特征部分
data(:, 4:end) = zscore(data(:, 4:end));

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

DT = fitrtree(X_train, y_train, 'MaxNumSplits', 921);
NR = fitnet(100);
NR = train(NR, X_train', y_train');
GB = fitensemble(X_train, y_train, 'LSBoost', 1000, 'Tree', 'type', 'regression');
RF = TreeBagger(100, X_train, y_train, 'Method', 'regression', 'Options', statset('UseParallel', true));

% 针对每个模型进行预测
y_pred_DT = predict(DT, X_test);
y_pred_NR = NR(X_test')';
y_pred_GB = predict(GB, X_test);
y_pred_RF = predict(RF, X_test);

% 将每个模型的预测结果合并为一个矩阵
all_predictions = [y_pred_DT, y_pred_NR, y_pred_GB, y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
ensemble_predictions = mean(all_predictions, 2);

% 计算投票回归器的均方误差
mse_ensemble = mean((y_test - ensemble_predictions).^2);

fprintf('投票回归器均方误差：%f\n', mse_ensemble);

% 导入新数据
new_data = csvread('D:\Users\lily\Desktop\Sample data\Variable Sorting\merged_1k.csv', 1, 0);

% 标准差标准化新数据的特征部分
new_data(:, 3:end) = zscore(new_data(:, 3:end));

newFeatures = new_data(:, 3:end);

% 针对每个模型进行预测
new_y_pred_DT = predict(DT, newFeatures);
new_y_pred_NR = NR(newFeatures')';
new_y_pred_GB = predict(GB, newFeatures);
new_y_pred_RF = predict(RF, newFeatures);

% 将每个模型的预测结果合并为一个矩阵
new_all_predictions = [new_y_pred_DT, new_y_pred_NR, new_y_pred_GB, new_y_pred_RF];

% 计算投票回归器的预测结果（平均预测值）
new_ensemble_predictions = mean(new_all_predictions, 2);

% 合并新数据的前两列和投票回归器的预测结果
new_predictedData = [new_data(:, 1:2), new_ensemble_predictions];

% 定义新数据的列名
new_variableNames = {'longititude', 'latitude', 'vr_b_1k'};

% 创建包含预测结果的新数据表
new_predictedDataTable = array2table(new_predictedData, 'VariableNames', new_variableNames);

% 定义输出文件路径
new_outputFilePath = 'D:\Users\lily\Desktop\Sample data\results\vr_b_1k_SS.csv';

% 将新数据表写入 CSV 文件
writetable(new_predictedDataTable, new_outputFilePath);

disp(['预测结果已保存至: ', new_outputFilePath]);
