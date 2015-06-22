function Otto_SVM(pathIn, K)
% train SVM on train set and evaluate it on test set
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
    fprintf('SVM\n');
    
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    AccuracyS = zeros(K,1);     % for shuffle data
    LogLossS = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\tAccuracyS\tLogLossS\n');
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);        
        
        model = fitcecoc(trainData(:,1:end-1), trainData(:,end), 'Coding', 'onevsone', 'Options', options, 'FitPosterior', 1);
        [labels,NegLoss,PBScore,Posterior] = predict(model, testData(:,1:end-1));    
        Accuracy(k) = mean(labels == testData(:,end));
        LogLoss(k) = logLoss(Posterior', convertLabel(testData(:,end)));        

        % Test with shuffle train data
        trainData = suffleData(trainData);    
        model = fitcecoc(trainData(:,1:end-1), trainData(:,end), 'Coding', 'onevsone', 'Options', options, 'FitPosterior', 1);
        [labels,NegLoss,PBScore,Posterior] = predict(model, testData(:,1:end-1));    
        Accuracy(k) = mean(labels == testData(:,end));
        LogLoss(k) = logLoss(Posterior', convertLabel(testData(:,end)));        
        
        fprintf('\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n', Accuracy(k), LogLoss(k), AccuracyS(k), LogLossS(k));
    end

    fprintf('Mean: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss), mean(AccuracyS), mean(LogLossS));
    fprintf('Std: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', std(Accuracy), std(LogLoss), std(AccuracyS), std(LogLossS));    

    delete(poolObj);
    
end
