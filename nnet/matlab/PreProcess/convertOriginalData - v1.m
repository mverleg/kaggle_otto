% ========================================================
% convert the given data from Kaggle to format that is possible for training

% TRAINING SET
[M T]= xlsread('train.csv');

% ignore first column
M1 = M(:, 2:end);
csvwrite('XTrain.csv', M(:, 2:end));     % X_training
clear M M1;


% skip first row, get last column
T1 = T(2:end,end);

% convert text to number
T2 = zeros(size(T1));
Class = unique(T1);
for i=1:size(Class,1)
    T2(ismember(T1,Class(i))) = i;
end

% % test 
% for i=1:size(Class,1)    
%     idx1 = ismember(T1,Class(i));
%     idx2 = (T2==i);
%     fprintf('%d %d\n', i, isequal(idx1, idx2));    
% end
% clear idx1 idx2;

csvwrite('YTrain.csv', T2);     % Y_training
clear T1 T2;


% process test data

[M T]= xlsread('test.csv');

% ignore first column
M1 = M(:, 2:end);
csvwrite('XTest.csv', M1);
clear M M1 T;
