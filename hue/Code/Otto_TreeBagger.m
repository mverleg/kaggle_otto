function Otto_TreeBagger(pathIn, K)
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
    AccuracyS = zeros(K,1);
    LogLossS = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\tAccuracyS\tLogLossS\n');
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);        
        
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);
        [labels,score] = predict(model, testData(:,1:end-1));    
        labels = str2num(cell2mat(labels));
        Accuracy(k) = mean(labels == testData(:,end));
        LogLoss(k) = logLoss(score', convertLabel(testData(:,end)));        

        % Test with suffle train data
        trainData = suffleData(trainData);    
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);
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
