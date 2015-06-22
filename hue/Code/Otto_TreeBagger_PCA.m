function Otto_TreeBagger_PCA(pathIn, K)
% test with PCA
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
    fprintf('TreeBagger PCA numTree: %d\n', nTree);    
    
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\n');
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);
        [trainData, testData] = getPCA(trainData, testData);
        
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);
        [labels,score] = predict(model, testData(:,1:end-1));    
        labels = str2num(cell2mat(labels));
        Accuracy(k) = mean(labels == testData(:,end));
        LogLoss(k) = logLoss(score', convertLabel(testData(:,end)));        
    
        fprintf('\t%6.2f\t%6.2f\n', Accuracy(k), LogLoss(k));
    end

    fprintf('Mean: %6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss));
    fprintf('Std: %6.2f\t%6.2f\n', std(Accuracy), std(LogLoss));
  
    delete(poolObj);
end