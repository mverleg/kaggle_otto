function Otto_TreeBagger_Features23(pathIn, K, fileFeatures)
% train TreeBagger on train set and evaluate it on test set
% input:
%   pathIn      path where train and test set are stored
%   K           pair of data for significance testing 
%   fileFeatures    name of file which is the result of feature selection,
%                   0 if the corresponding feature should be removed

    addpath('Utils');
    
    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end
    if ~exist('K', 'var')
        K = 6;
    end    
    if ~exist('fileFeatures', 'var')
        fileFeatures = 'features.csv';
    end    
    
    features = csvread(fileFeatures);
    
    % create pool for parallel
    numWorkers = 30;
	poolObj = createPool(numWorkers); % Invoke workers    
    
    options = statset('UseParallel',1);    
    nTree = 600;
    fprintf('TreeBagger numTree: %d\n', nTree);
    
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\n');
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);        
        [testClassSet, ~] = separateClasses(testData);
        [trainClassSet, ~] = separateClasses(trainData);
        testData = [testClassSet{2} ; testClassSet{3}];
        trainData = [trainClassSet{2} ; trainClassSet{3}];
        testData = testData(:, logical([features 1]));
        trainData = trainData(:, logical([features 1]));
        
        model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);
        [labels,score] = predict(model, testData(:,1:end-1));    
        labels = str2num(cell2mat(labels));
        Accuracy(k) = mean(labels == testData(:,end));
%         LogLoss(k) = logLoss(score', convertLabel(testData(:,end)));        

        fprintf('\t%6.2f\t%6.2f\n', Accuracy(k), 0);
    end

    fprintf('Mean: %6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss));
    fprintf('Std: %6.2f\t%6.2f\n', std(Accuracy), std(LogLoss));    

    delete(poolObj);
    
end
