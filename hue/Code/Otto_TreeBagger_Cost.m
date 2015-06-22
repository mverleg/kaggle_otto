function Otto_TreeBagger_Cost(pathIn, K)
% train TreeBagger on train set and evaluate it on test set
% input:
%   pathIn      path where train and test set are stored
%   K           pair of data for significance testing 

    addpath('Utils');
    
    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end
    if ~exist('K', 'var')
        K = 6;
    end    
    
    % create pool for parallel
    numWorkers = 30;
	poolObj = createPool(numWorkers); % Invoke workers    
    
    options = statset('UseParallel',1);    
    nTree = 600;
    fprintf('TreeBagger numTree: %d\n', nTree);
    
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    AccuracyS = zeros(K,1);     % for shuffle data
    LogLossS = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\tAccuracyS\tLogLossS\n');
    
%     C = [
%          0     4     3     1     1     4     1     3     4;
%          1     0     3     1     1     1     1     1     1;
%          1     2     0     1     1     2     1     1     1;
%          1     5     3     0     1     4     1     3     2;
%          1     4     3     1     0     4     1     3     2;
%          1     1     1     1     1     0     1     1     1;
%          1     5     3     1     1     4     0     3     2;
%          1     2     1     1     1     2     1     0     1;
%          1     3     2     1     1     3     1     2     0;
%          ];
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);        

        C = getCostFunction(trainData)
        
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options, 'Cost', C);
        [labels,score] = predict(model, testData(:,1:end-1));    
        labels = str2num(cell2mat(labels));
        Accuracy(k) = mean(labels == testData(:,end));
        LogLoss(k) = logLoss(score', convertLabel(testData(:,end)));        

        % Test with shuffle train data
        trainData = suffleData(trainData);    
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options, 'Cost', C);
        [labels,score] = predict(model, testData(:,1:end-1));    
        labels = str2num(cell2mat(labels));
        AccuracyS(k) = mean(labels == testData(:,end));
        LogLossS(k) = logLoss(score', convertLabel(testData(:,end)));        
        
        fprintf('\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n', Accuracy(k), LogLoss(k), AccuracyS(k), LogLossS(k));
    end

    fprintf('Mean: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss), mean(AccuracyS), mean(LogLossS));
    fprintf('Std: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', std(Accuracy), std(LogLoss), std(AccuracyS), std(LogLossS));    

    delete(poolObj);
    
end
